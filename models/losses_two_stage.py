"""Loss Functions for Two-Stage 3D Detection"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DepthLoss(nn.Module):
    """Depth loss with uncertainty weighting"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_depth, pred_log_var, gt_depth):
        precision = torch.exp(-pred_log_var)
        loss = torch.abs(pred_depth - gt_depth) * precision + pred_log_var
        return loss.mean()


class DimensionLoss(nn.Module):
    """3D dimension loss with relative size awareness"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_dims, gt_dims):
        l1_loss = F.l1_loss(pred_dims, gt_dims)
        pred_log = torch.log(pred_dims + 1e-6)
        gt_log = torch.log(gt_dims + 1e-6)
        log_loss = F.l1_loss(pred_log, gt_log)
        return l1_loss + 0.5 * log_loss


class RotationLoss(nn.Module):
    """Rotation loss with bin classification + residual"""
    def __init__(self, num_bins=12):
        super().__init__()
        self.num_bins = num_bins
        self.bin_size = 2 * np.pi / num_bins
    
    def forward(self, pred_bins, pred_residuals, gt_rotation):
        gt_rotation_norm = (gt_rotation + np.pi) % (2 * np.pi)
        gt_bin = (gt_rotation_norm / self.bin_size).long()
        gt_bin = torch.clamp(gt_bin, 0, self.num_bins - 1)
        bin_center = (gt_bin.float() + 0.5) * self.bin_size
        gt_residual = gt_rotation_norm - bin_center
        
        bin_loss = F.cross_entropy(pred_bins, gt_bin)
        pred_res_selected = pred_residuals[torch.arange(len(gt_bin)), gt_bin]
        residual_loss = F.smooth_l1_loss(pred_res_selected, gt_residual)
        
        return bin_loss + residual_loss


class Total3DLoss(nn.Module):
    """Combined loss for 3D detection"""
    def __init__(self, loss_weights=None):
        super().__init__()
        
        if loss_weights is None:
            loss_weights = {'depth': 1.0, 'dimension': 1.0, 'rotation': 1.0}
        self.loss_weights = loss_weights
        
        self.depth_loss = DepthLoss()
        self.dimension_loss = DimensionLoss()
        self.rotation_loss = RotationLoss()
    
    def forward(self, predictions, targets):
        pred_depth = predictions['depth']
        pred_dims = predictions['dimensions']
        pred_rot_bins, pred_rot_res = predictions['rotation']
        
        gt_depth = targets['depth']
        gt_dims = targets['dimensions']
        gt_rotation = targets['rotation']
        
        loss_depth = self.depth_loss(pred_depth[:, 0], pred_depth[:, 1], gt_depth)
        loss_dims = self.dimension_loss(pred_dims, gt_dims)
        loss_rot = self.rotation_loss(pred_rot_bins, pred_rot_res, gt_rotation)
        
        total_loss = (
            self.loss_weights['depth'] * loss_depth +
            self.loss_weights['dimension'] * loss_dims +
            self.loss_weights['rotation'] * loss_rot
        )
        
        loss_dict = {
            'loss_depth': loss_depth.item(),
            'loss_dimension': loss_dims.item(),
            'loss_rotation': loss_rot.item(),
            'loss_total': total_loss.item()
        }
        
        return total_loss, loss_dict