"""Loss Functions for V3 - Rotation-Only (No Direction Loss)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DepthLoss(nn.Module):
    """Depth loss with uncertainty"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_depth, gt_depth):
        """
        Args:
            pred_depth: [N, 3] [depth, log_var, offset]
            gt_depth: [N] ground truth depth
        """
        depth = pred_depth[:, 0] + pred_depth[:, 2]
        log_var = pred_depth[:, 1]
        
        loss = torch.exp(-log_var) * torch.abs(depth - gt_depth) + log_var
        return loss.mean()


class DimensionLoss(nn.Module):
    """Dimension loss (smooth L1)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_dims, gt_dims):
        """
        Args:
            pred_dims: [N, 3] predicted [h, w, l]
            gt_dims: [N, 3] ground truth [h, w, l]
        """
        return F.smooth_l1_loss(pred_dims, gt_dims)


class RotationLoss(nn.Module):
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
        
        # Residual regression loss
        gt_res = gt_rotation_shifted - (gt_bin.float() + 0.5) * self.bin_size
        pred_res_selected = pred_res[torch.arange(len(gt_bin)), gt_bin]
        loss_res = F.smooth_l1_loss(pred_res_selected, gt_res)
        
        return loss_bin + loss_res


class IoULoss(nn.Module):
    """IoU prediction loss (disabled in V3)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_iou, pred_boxes_3d=None, gt_boxes_3d=None):
        """Always returns 0 (IoU disabled)"""
        return torch.tensor(0.0, device=pred_iou.device)


class ForegroundLoss(nn.Module):
    """Foreground classification loss (auxiliary)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_fg, gt_fg):
        """
        Args:
            pred_fg: [N, 2] foreground logits
            gt_fg: [N] ground truth (all 1s for foreground)
        """
        gt_fg_long = gt_fg.long()
        return F.cross_entropy(pred_fg, gt_fg_long)


class Total3DLossV3(nn.Module):
    """Combined loss for V3 (no direction loss)"""
    def __init__(self, loss_weights=None):
        super().__init__()
        
        if loss_weights is None:
            loss_weights = {
                'depth': 1.0,
                'dimension': 1.0,
                'rotation': 1.5,
                'iou': 0.0,
                'foreground': 0.3
            }
        self.loss_weights = loss_weights
        
        self.depth_loss = DepthLoss()
        self.dimension_loss = DimensionLoss()
        self.rotation_loss = RotationLoss(num_bins=24)
        self.iou_loss = IoULoss()
        self.foreground_loss = ForegroundLoss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with keys [depth, dimensions, rotation, iou, foreground]
            targets: dict with keys [depth, dimensions, rotation, boxes_3d, foreground]
        
        Returns:
            total_loss, loss_dict
        """
        loss_depth = self.depth_loss(predictions['depth'], targets['depth'])
        loss_dim = self.dimension_loss(predictions['dimensions'], targets['dimensions'])
        loss_rot = self.rotation_loss(predictions['rotation'], targets['rotation'])
        loss_iou = self.iou_loss(predictions['iou'])
        loss_fg = self.foreground_loss(predictions['foreground'], targets['foreground'])
        
        total_loss = (
            self.loss_weights['depth'] * loss_depth +
            self.loss_weights['dimension'] * loss_dim +
            self.loss_weights['rotation'] * loss_rot +
            self.loss_weights['iou'] * loss_iou +
            self.loss_weights['foreground'] * loss_fg
        )
        
        loss_dict = {
            'loss_depth': loss_depth.item(),
            'loss_dimension': loss_dim.item(),
            'loss_rotation': loss_rot.item(),
            'loss_iou': loss_iou.item(),
            'loss_foreground': loss_fg.item()
        }
        
        return total_loss, loss_dict
