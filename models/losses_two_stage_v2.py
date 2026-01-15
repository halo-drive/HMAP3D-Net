"""Loss Functions for Two-Stage 3D Detection V2

Enhanced with:
1. IoU prediction loss
2. Direction classification loss
3. Auxiliary foreground/background loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_3d_iou(boxes_pred, boxes_gt):
    """
    Compute 3D IoU between predicted and ground truth boxes
    Simplified axis-aligned approximation for speed
    
    Args:
        boxes_pred: [N, 7] predicted boxes [x, y, z, h, w, l, ry]
        boxes_gt: [N, 7] ground truth boxes
    
    Returns:
        [N] IoU values
    """
    def box_volume(boxes):
        return boxes[:, 3] * boxes[:, 4] * boxes[:, 5]  # h * w * l
    
    vol_pred = box_volume(boxes_pred)
    vol_gt = box_volume(boxes_gt)
    
    # Compute overlap in each dimension (axis-aligned approximation)
    # This ignores rotation for speed - good enough for training signal
    
    # X overlap (lateral)
    x_min_pred = boxes_pred[:, 0] - boxes_pred[:, 5] / 2  # x - l/2
    x_max_pred = boxes_pred[:, 0] + boxes_pred[:, 5] / 2
    x_min_gt = boxes_gt[:, 0] - boxes_gt[:, 5] / 2
    x_max_gt = boxes_gt[:, 0] + boxes_gt[:, 5] / 2
    
    x_overlap = torch.clamp(
        torch.min(x_max_pred, x_max_gt) - torch.max(x_min_pred, x_min_gt),
        min=0
    )
    
    # Y overlap (vertical)
    y_min_pred = boxes_pred[:, 1] - boxes_pred[:, 3] / 2
    y_max_pred = boxes_pred[:, 1] + boxes_pred[:, 3] / 2
    y_min_gt = boxes_gt[:, 1] - boxes_gt[:, 3] / 2
    y_max_gt = boxes_gt[:, 1] + boxes_gt[:, 3] / 2
    
    y_overlap = torch.clamp(
        torch.min(y_max_pred, y_max_gt) - torch.max(y_min_pred, y_min_gt),
        min=0
    )
    
    # Z overlap (depth)
    z_min_pred = boxes_pred[:, 2] - boxes_pred[:, 4] / 2
    z_max_pred = boxes_pred[:, 2] + boxes_pred[:, 4] / 2
    z_min_gt = boxes_gt[:, 2] - boxes_gt[:, 4] / 2
    z_max_gt = boxes_gt[:, 2] + boxes_gt[:, 4] / 2
    
    z_overlap = torch.clamp(
        torch.min(z_max_pred, z_max_gt) - torch.max(z_min_pred, z_min_gt),
        min=0
    )
    
    intersection = x_overlap * y_overlap * z_overlap
    union = vol_pred + vol_gt - intersection
    
    iou = intersection / (union + 1e-6)
    return iou


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


class DirectionLoss(nn.Module):
    """Direction loss with EXTREME settings to prevent mode collapse"""
    def __init__(self, alpha=0.75, gamma=3.0):  # EXTREME PARAMETERS
        super().__init__()
        self.alpha = alpha  # Increased from 0.25
        self.gamma = gamma  # Increased from 2.0
    
    def forward(self, pred_direction_logits, gt_rotation):
        """
        Args:
            pred_direction_logits: [N, 2] logits for [0-180°, 180-360°]
            gt_rotation: [N] rotation in radians [-π, π]
        """
        # Convert rotation to direction class
        gt_direction = (gt_rotation < 0).long()
        
        # Compute class weights (inverse frequency)
        n_class_0 = (gt_direction == 0).sum().float()
        n_class_1 = (gt_direction == 1).sum().float()
        total = n_class_0 + n_class_1
        
        if n_class_1 > 0:
            weight_0 = total / (2.0 * n_class_0 + 1e-6)
            weight_1 = total / (2.0 * n_class_1 + 1e-6)
        else:
            weight_0 = 1.0
            weight_1 = 1.0
        
        weights = torch.tensor([weight_0, weight_1], 
                              device=pred_direction_logits.device, 
                              dtype=pred_direction_logits.dtype)
        
        # Weighted focal loss
        ce_loss = F.cross_entropy(pred_direction_logits, gt_direction, 
                                  weight=weights, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        loss = (focal_weight * ce_loss).mean()
        
        return loss
    

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


class IoULoss(nn.Module):
    """IoU prediction loss"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_iou, pred_boxes_3d, gt_boxes_3d):
        """
        Args:
            pred_iou: [N] predicted IoU scores
            pred_boxes_3d: [N, 7] predicted 3D boxes
            gt_boxes_3d: [N, 7] ground truth 3D boxes
        """
        # Compute actual 3D IoU
        actual_iou = compute_3d_iou(pred_boxes_3d, gt_boxes_3d)
        
        # L2 loss between predicted and actual IoU
        loss = F.mse_loss(pred_iou, actual_iou)
        return loss


class ForegroundLoss(nn.Module):
    """Auxiliary foreground/background classification loss"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_fg_logits, gt_boxes_exist):
        """
        Args:
            pred_fg_logits: [N, 2] logits for [background, foreground]
            gt_boxes_exist: [N] binary labels (1 if valid object, 0 if background)
        """
        loss = F.cross_entropy(pred_fg_logits, gt_boxes_exist.long())
        return loss


class Total3DLossV2(nn.Module):
    """Combined loss for enhanced 3D detection"""
    def __init__(self, loss_weights=None):
        super().__init__()
        
        if loss_weights is None:
            loss_weights = {
                'depth': 1.0,
                'dimension': 1.0,
                'direction': 2.0,     # Increased from 0.5 to fight mode collapse
                'rotation': 1.0,
                'iou': 1.0,
                'foreground': 0.3
            }
        self.loss_weights = loss_weights
        
        self.depth_loss = DepthLoss()
        self.dimension_loss = DimensionLoss()
        self.direction_loss = DirectionLoss(alpha=0.25, gamma=2.0)  # Focal loss
        self.rotation_loss = RotationLoss()
        self.iou_loss = IoULoss()
        self.foreground_loss = ForegroundLoss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with keys:
                - depth: [N, 3] [depth, log_var, offset]
                - dimensions: [N, 3] [h, w, l]
                - direction: [N, 2] direction logits
                - rotation: tuple of ([N, 12], [N, 12]) bins and residuals
                - iou: [N] predicted IoU scores
                - foreground: [N, 2] foreground logits
            targets: dict with keys:
                - depth: [N] ground truth depths
                - dimensions: [N, 3] ground truth dimensions
                - rotation: [N] ground truth rotations
                - boxes_3d: [N, 7] ground truth 3D boxes (for IoU computation)
                - foreground: [N] ground truth foreground labels
        """
        pred_depth = predictions['depth']
        pred_dims = predictions['dimensions']
        pred_direction_logits = predictions['direction']
        pred_rot_bins, pred_rot_res = predictions['rotation']
        pred_iou = predictions['iou']
        pred_fg_logits = predictions['foreground']
        
        gt_depth = targets['depth']
        gt_dims = targets['dimensions']
        gt_rotation = targets['rotation']
        gt_boxes_3d = targets.get('boxes_3d')
        gt_foreground = targets.get('foreground', torch.ones_like(gt_depth).long())
        
        # Compute individual losses
        loss_depth = self.depth_loss(pred_depth[:, 0], pred_depth[:, 1], gt_depth)
        loss_dims = self.dimension_loss(pred_dims, gt_dims)
        loss_direction = self.direction_loss(pred_direction_logits, gt_rotation)
        loss_rot = self.rotation_loss(pred_rot_bins, pred_rot_res, gt_rotation)
        loss_fg = self.foreground_loss(pred_fg_logits, gt_foreground)
        
        # IoU loss (requires predicted 3D boxes)
        if gt_boxes_3d is not None and 'pred_boxes_3d' in predictions:
            loss_iou = self.iou_loss(pred_iou, predictions['pred_boxes_3d'], gt_boxes_3d)
        else:
            loss_iou = torch.tensor(0.0, device=pred_depth.device)
        
        # Weighted sum
        total_loss = (
            self.loss_weights['depth'] * loss_depth +
            self.loss_weights['dimension'] * loss_dims +
            self.loss_weights['direction'] * loss_direction +
            self.loss_weights['rotation'] * loss_rot +
            self.loss_weights['iou'] * loss_iou +
            self.loss_weights['foreground'] * loss_fg
        )
        
        loss_dict = {
            'loss_depth': loss_depth.item(),
            'loss_dimension': loss_dims.item(),
            'loss_direction': loss_direction.item(),
            'loss_rotation': loss_rot.item(),
            'loss_iou': loss_iou.item() if isinstance(loss_iou, torch.Tensor) else 0.0,
            'loss_foreground': loss_fg.item(),
            'loss_total': total_loss.item()
        }
        
        return total_loss, loss_dict