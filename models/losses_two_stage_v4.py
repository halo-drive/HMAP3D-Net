"""Loss Functions for V4 - Stable Depth Loss (No Uncertainty)

Key Changes from V3:
1. Removed log_variance from depth loss (stability fix)
2. Added gradient clipping awareness
3. Balanced loss weights for better convergence
4. Removed IoU loss (was disabled anyway)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DepthLossV4(nn.Module):
    """Stable depth loss (NO uncertainty term)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_depth, gt_depth):
        """
        Args:
            pred_depth: [N, 2] [depth, offset]
            gt_depth: [N] ground truth depth
        
        Returns:
            Smooth L1 loss on depth
        """
        # Combine depth + offset
        depth = pred_depth[:, 0] + pred_depth[:, 1]
        
        # Simple smooth L1 loss (stable, no exploitation possible)
        loss = F.smooth_l1_loss(depth, gt_depth)
        
        return loss


class DimensionLossV4(nn.Module):
    """Dimension loss with smooth L1"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_dims, gt_dims):
        """
        Args:
            pred_dims: [N, 3] predicted [h, w, l]
            gt_dims: [N, 3] ground truth [h, w, l]
        """
        return F.smooth_l1_loss(pred_dims, gt_dims)


class RotationLossV4(nn.Module):
    """Rotation loss with bin classification + residual (24 bins)"""
    def __init__(self, num_bins=24):
        super().__init__()
        self.num_bins = num_bins
        self.bin_size = 2 * np.pi / num_bins
    
    def forward(self, pred_rotation, gt_rotation):
        """
        Args:
            pred_rotation: tuple of (bins [N, 24], residuals [N, 24])
            gt_rotation: [N] rotation in radians [-π, π]
        """
        pred_bins, pred_res = pred_rotation
        
        # Convert rotation to bin index
        gt_rotation_shifted = (gt_rotation + np.pi) % (2 * np.pi)
        gt_bin = (gt_rotation_shifted / self.bin_size).long()
        gt_bin = torch.clamp(gt_bin, 0, self.num_bins - 1)
        
        # Bin classification loss
        loss_bin = F.cross_entropy(pred_bins, gt_bin)
        
        # Residual regression loss (only for correct bin)
        gt_res = gt_rotation_shifted - (gt_bin.float() + 0.5) * self.bin_size
        pred_res_selected = pred_res[torch.arange(len(gt_bin)), gt_bin]
        loss_res = F.smooth_l1_loss(pred_res_selected, gt_res)
        
        return loss_bin + loss_res


class ForegroundLossV4(nn.Module):
    """Foreground classification loss (auxiliary)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_fg, gt_fg):
        """
        Args:
            pred_fg: [N, 2] foreground logits
            gt_fg: [N] ground truth (all 1s for foreground objects)
        """
        gt_fg_long = gt_fg.long()
        return F.cross_entropy(pred_fg, gt_fg_long)


class Total3DLossV4(nn.Module):
    """Combined loss for V4 with balanced weights"""
    def __init__(self, loss_weights=None):
        super().__init__()
        
        if loss_weights is None:
            # Balanced weights based on V3 analysis
            loss_weights = {
                'depth': 1.0,        # Stable with new loss
                'dimension': 1.0,    # Working well
                'rotation': 1.2,     # Slightly higher for orientation accuracy
                'foreground': 0.2    # Auxiliary, lower weight
            }
        self.loss_weights = loss_weights
        
        self.depth_loss = DepthLossV4()
        self.dimension_loss = DimensionLossV4()
        self.rotation_loss = RotationLossV4(num_bins=24)
        self.foreground_loss = ForegroundLossV4()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with keys [depth, dimensions, rotation, foreground]
            targets: dict with keys [depth, dimensions, rotation, foreground]
        
        Returns:
            total_loss, loss_dict
        """
        loss_depth = self.depth_loss(predictions['depth'], targets['depth'])
        loss_dim = self.dimension_loss(predictions['dimensions'], targets['dimensions'])
        loss_rot = self.rotation_loss(predictions['rotation'], targets['rotation'])
        loss_fg = self.foreground_loss(predictions['foreground'], targets['foreground'])
        
        total_loss = (
            self.loss_weights['depth'] * loss_depth +
            self.loss_weights['dimension'] * loss_dim +
            self.loss_weights['rotation'] * loss_rot +
            self.loss_weights['foreground'] * loss_fg
        )
        
        loss_dict = {
            'loss_depth': loss_depth.item(),
            'loss_dimension': loss_dim.item(),
            'loss_rotation': loss_rot.item(),
            'loss_foreground': loss_fg.item()
        }
        
        return total_loss, loss_dict


class GradientMonitor:
    """Monitor gradients during training for debugging"""
    def __init__(self, model):
        self.model = model
        self.grad_norms = {}
    
    def compute_grad_norms(self):
        """Compute gradient norms for each parameter group"""
        self.grad_norms.clear()
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.grad_norms[name] = grad_norm
        
        return self.grad_norms
    
    def get_max_grad_norm(self):
        """Get maximum gradient norm across all parameters"""
        if not self.grad_norms:
            self.compute_grad_norms()
        
        if self.grad_norms:
            return max(self.grad_norms.values())
        return 0.0
    
    def check_nan_gradients(self):
        """Check if any gradients are NaN"""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    return True, name
        return False, None