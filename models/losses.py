"""
Loss Functions for Monocular 3D Object Detection

Implements multi-task loss components:
1. Focal Loss - Heatmap center detection (handles class imbalance)
2. Depth Loss - Bin classification + residual regression
3. Dimension Loss - L1 regression for 3D box dimensions
4. Rotation Loss - Multi-bin angle classification + residual
5. 2D Offset Loss - Sub-pixel center localization

All losses extract predictions at sparse object locations using indices tensor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class FocalLoss(nn.Module):
    """Weighted MSE for heatmap with positive sample emphasis"""
    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super(FocalLoss, self).__init__()
        self.pos_weight = 100.0
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        
        mse = (pred - target) ** 2
        neg_weights = torch.pow(1 - target, 4)
        
        pos_loss = (mse * pos_inds).sum()
        neg_loss = (mse * neg_weights * neg_inds).sum()
        
        num_pos = pos_inds.sum().clamp(min=1.0)
        loss = (self.pos_weight * pos_loss + neg_loss) / num_pos
        
        return loss


class DepthLoss(nn.Module):
    """
    Depth estimation loss with bin classification + residual regression
    """
    def __init__(self, num_bins: int = 6, bin_weight: float = 1.0, residual_weight: float = 0.1):
        super(DepthLoss, self).__init__()
        self.num_bins = num_bins
        self.bin_weight = bin_weight
        self.residual_weight = residual_weight
    
    def forward(
        self,
        pred_bins: torch.Tensor,
        pred_residuals: torch.Tensor,
        target_bins: torch.Tensor,
        target_residuals: torch.Tensor,
        indices: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_bins: (B, num_bins, H, W) bin logits
            pred_residuals: (B, num_bins, H, W) residual predictions
            target_bins: (B, max_objects) ground truth bin indices
            target_residuals: (B, max_objects) ground truth residuals
            indices: (B, max_objects, 2) [y, x] heatmap locations
            mask: (B, max_objects) valid object mask
        
        Returns:
            Dictionary with 'bin_loss', 'residual_loss', 'total_loss'
        """
        batch_size, max_objects = target_bins.shape
        device = pred_bins.device
        
        bin_loss = torch.tensor(0.0, device=device)
        residual_loss = torch.tensor(0.0, device=device)
        
        num_valid = mask.sum()
        if num_valid == 0:
            return {
                'bin_loss': bin_loss,
                'residual_loss': residual_loss,
                'total_loss': bin_loss
            }
        
        # Extract predictions at object locations
        pred_bins_at_locs = []
        pred_residuals_at_locs = []
        
        for b in range(batch_size):
            y_coords = indices[b, :, 0].long()
            x_coords = indices[b, :, 1].long()
            
            # Clamp coordinates to valid range
            y_coords = torch.clamp(y_coords, 0, pred_bins.shape[2] - 1)
            x_coords = torch.clamp(x_coords, 0, pred_bins.shape[3] - 1)
            
            pred_bins_b = pred_bins[b, :, y_coords, x_coords].t()  # (max_objects, num_bins)
            pred_residuals_b = pred_residuals[b, :, y_coords, x_coords].t()
            
            pred_bins_at_locs.append(pred_bins_b)
            pred_residuals_at_locs.append(pred_residuals_b)
        
        pred_bins_gathered = torch.stack(pred_bins_at_locs)  # (B, max_objects, num_bins)
        pred_residuals_gathered = torch.stack(pred_residuals_at_locs)
        
        # Flatten
        pred_bins_flat = pred_bins_gathered.reshape(-1, self.num_bins)
        pred_residuals_flat = pred_residuals_gathered.reshape(-1, self.num_bins)
        target_bins_flat = target_bins.reshape(-1)
        target_residuals_flat = target_residuals.reshape(-1)
        mask_flat = mask.reshape(-1)
        
        valid_idx = mask_flat > 0
        
        if valid_idx.sum() > 0:
            # Bin classification loss
            bin_loss = F.cross_entropy(
                pred_bins_flat[valid_idx],
                target_bins_flat[valid_idx],
                reduction='mean'
            )
            
            # Residual regression loss (supervised by ground truth bins)
            target_bins_valid = target_bins_flat[valid_idx].long()
            pred_residuals_selected = pred_residuals_flat[valid_idx].gather(
                1, target_bins_valid.unsqueeze(1)
            ).squeeze(1)
            
            residual_loss = F.l1_loss(
                pred_residuals_selected,
                target_residuals_flat[valid_idx],
                reduction='mean'
            )
        
        total_loss = self.bin_weight * bin_loss + self.residual_weight * residual_loss
        
        return {
            'bin_loss': bin_loss,
            'residual_loss': residual_loss,
            'total_loss': total_loss
        }


class DimensionLoss(nn.Module):
    """
    3D dimension regression loss (height, width, length)
    """
    def __init__(self, dim_mean: torch.Tensor = None):
        super(DimensionLoss, self).__init__()
        
        if dim_mean is None:
            # nuScenes vehicle statistics: [h, w, l]
            dim_mean = torch.tensor([2.0, 2.0, 4.9], dtype=torch.float32)
        
        self.register_buffer('dim_mean', dim_mean)
    
    def forward(
        self,
        pred_dims: torch.Tensor,
        target_dims: torch.Tensor,
        indices: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_dims: (B, 3, H, W) predicted dimensions
            target_dims: (B, max_objects, 3) ground truth dimensions
            indices: (B, max_objects, 2) [y, x] heatmap locations
            mask: (B, max_objects) valid object mask
        
        Returns:
            Scalar loss value
        """
        num_valid = mask.sum()
        if num_valid == 0:
            return torch.tensor(0.0, device=pred_dims.device)
        
        batch_size, max_objects = target_dims.shape[:2]
        
        # Extract predictions at object locations
        pred_dims_at_locs = []
        for b in range(batch_size):
            y_coords = indices[b, :, 0].long()
            x_coords = indices[b, :, 1].long()
            
            # Clamp coordinates
            y_coords = torch.clamp(y_coords, 0, pred_dims.shape[2] - 1)
            x_coords = torch.clamp(x_coords, 0, pred_dims.shape[3] - 1)
            
            pred_dims_b = pred_dims[b, :, y_coords, x_coords].t()  # (max_objects, 3)
            pred_dims_at_locs.append(pred_dims_b)
        
        pred_dims_gathered = torch.stack(pred_dims_at_locs)  # (B, max_objects, 3)
        
        # Normalize by mean dimensions
        pred_dims_norm = pred_dims_gathered / self.dim_mean.view(1, 1, 3)
        target_dims_norm = target_dims / self.dim_mean.view(1, 1, 3)
        
        # Flatten and mask
        pred_dims_flat = pred_dims_norm.reshape(-1, 3)
        target_dims_flat = target_dims_norm.reshape(-1, 3)
        mask_flat = mask.reshape(-1)
        
        valid_idx = mask_flat > 0
        
        loss = F.l1_loss(
            pred_dims_flat[valid_idx],
            target_dims_flat[valid_idx],
            reduction='mean'
        )
        
        return loss


class RotationLoss(nn.Module):
    """
    Rotation angle loss with multi-bin classification + residual regression
    """
    def __init__(self, num_bins: int = 8, bin_weight: float = 1.0, residual_weight: float = 0.1):
        super(RotationLoss, self).__init__()
        self.num_bins = num_bins
        self.bin_weight = bin_weight
        self.residual_weight = residual_weight
    
    def forward(
        self,
        pred_bins: torch.Tensor,
        pred_residuals: torch.Tensor,
        target_bins: torch.Tensor,
        target_residuals: torch.Tensor,
        indices: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_bins: (B, num_bins, H, W) bin logits
            pred_residuals: (B, num_bins, H, W) residual predictions
            target_bins: (B, max_objects) ground truth bin indices
            target_residuals: (B, max_objects) ground truth residuals
            indices: (B, max_objects, 2) [y, x] heatmap locations
            mask: (B, max_objects) valid object mask
        
        Returns:
            Dictionary with 'bin_loss', 'residual_loss', 'total_loss'
        """
        batch_size, max_objects = target_bins.shape
        device = pred_bins.device
        
        bin_loss = torch.tensor(0.0, device=device)
        residual_loss = torch.tensor(0.0, device=device)
        
        num_valid = mask.sum()
        if num_valid == 0:
            return {
                'bin_loss': bin_loss,
                'residual_loss': residual_loss,
                'total_loss': bin_loss
            }
        
        # Extract predictions at object locations
        pred_bins_at_locs = []
        pred_residuals_at_locs = []
        
        for b in range(batch_size):
            y_coords = indices[b, :, 0].long()
            x_coords = indices[b, :, 1].long()
            
            # Clamp coordinates
            y_coords = torch.clamp(y_coords, 0, pred_bins.shape[2] - 1)
            x_coords = torch.clamp(x_coords, 0, pred_bins.shape[3] - 1)
            
            pred_bins_b = pred_bins[b, :, y_coords, x_coords].t()
            pred_residuals_b = pred_residuals[b, :, y_coords, x_coords].t()
            
            pred_bins_at_locs.append(pred_bins_b)
            pred_residuals_at_locs.append(pred_residuals_b)
        
        pred_bins_gathered = torch.stack(pred_bins_at_locs)
        pred_residuals_gathered = torch.stack(pred_residuals_at_locs)
        
        # Flatten
        pred_bins_flat = pred_bins_gathered.reshape(-1, self.num_bins)
        pred_residuals_flat = pred_residuals_gathered.reshape(-1, self.num_bins)
        target_bins_flat = target_bins.reshape(-1)
        target_residuals_flat = target_residuals.reshape(-1)
        mask_flat = mask.reshape(-1)
        
        valid_idx = mask_flat > 0
        
        if valid_idx.sum() > 0:
            # Bin classification loss
            bin_loss = F.cross_entropy(
                pred_bins_flat[valid_idx],
                target_bins_flat[valid_idx],
                reduction='mean'
            )
            
            # Residual regression loss
            target_bins_valid = target_bins_flat[valid_idx].long()
            pred_residuals_selected = pred_residuals_flat[valid_idx].gather(
                1, target_bins_valid.unsqueeze(1)
            ).squeeze(1)
            
            residual_loss = F.l1_loss(
                pred_residuals_selected,
                target_residuals_flat[valid_idx],
                reduction='mean'
            )
        
        total_loss = self.bin_weight * bin_loss + self.residual_weight * residual_loss
        
        return {
            'bin_loss': bin_loss,
            'residual_loss': residual_loss,
            'total_loss': total_loss
        }


class OffsetLoss(nn.Module):
    """
    2D center offset loss for sub-pixel localization
    """
    def __init__(self):
        super(OffsetLoss, self).__init__()
    
    def forward(
        self,
        pred_offset: torch.Tensor,
        target_offset: torch.Tensor,
        indices: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_offset: (B, 2, H, W) predicted offsets [Δu, Δv]
            target_offset: (B, max_objects, 2) ground truth offsets
            indices: (B, max_objects, 2) [y, x] heatmap locations
            mask: (B, max_objects) valid object mask
        
        Returns:
            Scalar loss value
        """
        num_valid = mask.sum()
        if num_valid == 0:
            return torch.tensor(0.0, device=pred_offset.device)
        
        batch_size, max_objects = target_offset.shape[:2]
        
        # Extract predictions at object locations
        pred_offset_at_locs = []
        for b in range(batch_size):
            y_coords = indices[b, :, 0].long()
            x_coords = indices[b, :, 1].long()
            
            # Clamp coordinates
            y_coords = torch.clamp(y_coords, 0, pred_offset.shape[2] - 1)
            x_coords = torch.clamp(x_coords, 0, pred_offset.shape[3] - 1)
            
            pred_offset_b = pred_offset[b, :, y_coords, x_coords].t()  # (max_objects, 2)
            pred_offset_at_locs.append(pred_offset_b)
        
        pred_offset_gathered = torch.stack(pred_offset_at_locs)  # (B, max_objects, 2)
        
        # Flatten and mask
        pred_offset_flat = pred_offset_gathered.reshape(-1, 2)
        target_offset_flat = target_offset.reshape(-1, 2)
        mask_flat = mask.reshape(-1)
        
        valid_idx = mask_flat > 0
        
        loss = F.l1_loss(
            pred_offset_flat[valid_idx],
            target_offset_flat[valid_idx],
            reduction='mean'
        )
        
        return loss


class DetectionLoss(nn.Module):
    """
    Combined multi-task loss for monocular 3D object detection
    """
    def __init__(
        self,
        heatmap_weight: float = 1.0,
        depth_weight: float = 1.0,
        dimension_weight: float = 0.1,
        rotation_weight: float = 0.5,
        offset_weight: float = 0.1,
        num_depth_bins: int = 6,
        num_rotation_bins: int = 8
    ):
        super(DetectionLoss, self).__init__()
        
        self.heatmap_weight = heatmap_weight
        self.depth_weight = depth_weight
        self.dimension_weight = dimension_weight
        self.rotation_weight = rotation_weight
        self.offset_weight = offset_weight
        
        # Loss modules
        self.focal_loss = FocalLoss(alpha=2.0, beta=4.0)
        self.depth_loss = DepthLoss(num_bins=num_depth_bins)
        self.dimension_loss = DimensionLoss()
        self.rotation_loss = RotationLoss(num_bins=num_rotation_bins)
        self.offset_loss = OffsetLoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss
        
        Args:
            predictions: Dictionary from model forward pass
            targets: Dictionary from dataset (must include 'indices')
        
        Returns:
            Dictionary containing all loss components and total loss
        """
        # Heatmap loss (dense)
        loss_heatmap = self.focal_loss(
            predictions['heatmap'],
            targets['heatmap']
        )
        
        # Depth loss (sparse)
        depth_losses = self.depth_loss(
            predictions['depth_bin'],
            predictions['depth_residual'],
            targets['depth_bin'],
            targets['depth_residual'],
            targets['indices'],
            targets['mask']
        )
        
        # Dimension loss (sparse)
        loss_dimension = self.dimension_loss(
            predictions['dimensions'],
            targets['dimensions'],
            targets['indices'],
            targets['mask']
        )
        
        # Rotation loss (sparse)
        rotation_losses = self.rotation_loss(
            predictions['rotation_bin'],
            predictions['rotation_residual'],
            targets['rotation_bin'],
            targets['rotation_residual'],
            targets['indices'],
            targets['mask']
        )
        
        # Offset loss (sparse)
        loss_offset = self.offset_loss(
            predictions['offset_2d'],
            targets['offset_2d'],
            targets['indices'],
            targets['mask']
        )
        
        # Total loss
        total_loss = (
            self.heatmap_weight * loss_heatmap +
            self.depth_weight * depth_losses['total_loss'] +
            self.dimension_weight * loss_dimension +
            self.rotation_weight * rotation_losses['total_loss'] +
            self.offset_weight * loss_offset
        )
        
        return {
            'total_loss': total_loss,
            'loss_heatmap': loss_heatmap,
            'loss_depth_bin': depth_losses['bin_loss'],
            'loss_depth_residual': depth_losses['residual_loss'],
            'loss_dimension': loss_dimension,
            'loss_rotation_bin': rotation_losses['bin_loss'],
            'loss_rotation_residual': rotation_losses['residual_loss'],
            'loss_offset': loss_offset
        }


def test_losses():
    """Test loss computation with dummy data"""
    print("=" * 60)
    print("Testing Multi-Task Loss Functions")
    print("=" * 60)
    
    batch_size = 4
    height, width = 180, 320
    max_objects = 50
    
    # Dummy predictions (from network)
    # Dummy predictions (from network) - ENABLE GRADIENTS
    predictions = {
        'heatmap': torch.rand(batch_size, 1, height, width).requires_grad_(True),
        'depth_bin': torch.randn(batch_size, 6, height, width).requires_grad_(True),
        'depth_residual': torch.randn(batch_size, 6, height, width).requires_grad_(True),
        'dimensions': torch.randn(batch_size, 3, height, width).requires_grad_(True),
        'rotation_bin': torch.randn(batch_size, 8, height, width).requires_grad_(True),
        'rotation_residual': torch.randn(batch_size, 8, height, width).requires_grad_(True),
        'offset_2d': torch.randn(batch_size, 2, height, width).requires_grad_(True),
    }
    
    # Dummy targets (from dataset)
    targets = {
        'heatmap': torch.rand(batch_size, 1, height, width),
        'depth_bin': torch.randint(0, 6, (batch_size, max_objects)),
        'depth_residual': torch.randn(batch_size, max_objects),
        'dimensions': torch.abs(torch.randn(batch_size, max_objects, 3)),
        'rotation_bin': torch.randint(0, 8, (batch_size, max_objects)),
        'rotation_residual': torch.randn(batch_size, max_objects),
        'offset_2d': torch.randn(batch_size, max_objects, 2),
        'indices': torch.stack([
            torch.randint(0, height, (batch_size, max_objects)),
            torch.randint(0, width, (batch_size, max_objects))
        ], dim=2),
        'mask': torch.zeros(batch_size, max_objects)
    }
    
    # Set some objects as valid (first 5 per batch)
    targets['mask'][:, :5] = 1.0
    
    print(f"\nBatch size: {batch_size}")
    print(f"Feature map size: {height}×{width}")
    print(f"Max objects per image: {max_objects}")
    print(f"Valid objects per image: {int(targets['mask'][0].sum())}")
    
    # Initialize loss
    criterion = DetectionLoss()
    
    print("\nComputing losses...")
    losses = criterion(predictions, targets)
    
    print("\n" + "=" * 60)
    print("Loss Components:")
    print("=" * 60)
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:25s}: {value.item():.6f}")
    
    print("\n" + "=" * 60)
    print("Gradient Flow Test:")
    print("=" * 60)
    print(f"  Total loss requires_grad: {losses['total_loss'].requires_grad}")
    
    # Test backward pass
    losses['total_loss'].backward()
    print(f"  ✓ Backward pass successful")
    
    # Check gradients exist
    has_grads = sum(1 for p in criterion.parameters() if p.grad is not None)
    print(f"  ✓ Parameters with gradients: {has_grads}")
    
    print("\n" + "=" * 60)
    print("✓ All loss functions operational")
    print("=" * 60)


if __name__ == "__main__":
    test_losses()