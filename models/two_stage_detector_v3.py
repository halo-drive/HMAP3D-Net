"""Two-Stage 3D Object Detector V3 - Rotation-Only (No Direction Head)

Architecture:
- Stage 1: YOLOv3 for 2D detection
- Stage 2: 3D parameter regression with intrinsics conditioning
  - Intrinsics encoder
  - Depth head (intrinsics-conditioned)
  - Dimension head (intrinsics-conditioned)
  - Rotation head (24 bins, full 360°)
  - Foreground head (auxiliary)
  
REMOVED: Direction head (caused mode collapse)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class IntrinsicsEncoder(nn.Module):
    """Encode camera intrinsics into learnable features"""
    def __init__(self, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.ReLU()
        )
    
    def forward(self, K):
        """
        Args:
            K: [B, 3, 3] intrinsic matrices
        Returns:
            [B, output_dim] intrinsics features
        """
        fx = K[:, 0, 0].unsqueeze(1)
        fy = K[:, 1, 1].unsqueeze(1)
        cx = K[:, 0, 2].unsqueeze(1)
        cy = K[:, 1, 2].unsqueeze(1)
        
        intrinsics_vec = torch.cat([fx, fy, cx, cy], dim=1)
        return self.encoder(intrinsics_vec)


class DepthHead(nn.Module):
    """Predict depth with uncertainty (intrinsics-conditioned)"""
    def __init__(self, input_dim=2048, intrinsics_dim=128):
        super().__init__()
        combined_dim = input_dim + intrinsics_dim
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
    
    def forward(self, roi_features, intrinsics_features):
        """
        Returns:
            [N, 3]: [depth, log_variance, offset]
        """
        combined = torch.cat([roi_features, intrinsics_features], dim=1)
        return self.fc(combined)


class DimensionHead(nn.Module):
    """Predict 3D dimensions (intrinsics-conditioned)"""
    def __init__(self, input_dim=2048, intrinsics_dim=128):
        super().__init__()
        combined_dim = input_dim + intrinsics_dim
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 3)
        )
    
    def forward(self, roi_features, intrinsics_features):
        """
        Returns:
            [N, 3]: [height, width, length]
        """
        combined = torch.cat([roi_features, intrinsics_features], dim=1)
        return self.fc(combined)


class RotationHead(nn.Module):
    """Predict rotation with bin classification + residual (24 bins for full 360°)"""
    def __init__(self, input_dim=2048):
        super().__init__()
        self.num_bins = 24
        
        self.bin_fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_bins)
        )
        
        self.res_fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_bins)
        )
    
    def forward(self, roi_features):
        """
        Returns:
            bins: [N, 24] bin logits
            residuals: [N, 24] residual values
        """
        bins = self.bin_fc(roi_features)
        residuals = self.res_fc(roi_features)
        return bins, residuals


class IoUHead(nn.Module):
    """Predict 3D IoU quality score"""
    def __init__(self, input_dim=2048):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, roi_features):
        return self.fc(roi_features)


class ForegroundHead(nn.Module):
    """Binary foreground/background classification (auxiliary)"""
    def __init__(self, input_dim=2048):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    
    def forward(self, roi_features):
        return self.fc(roi_features)


class TwoStage3DDetector(nn.Module):
    """V3: Two-stage 3D detector with intrinsics conditioning, no direction head"""
    def __init__(self, num_classes=3, active_classes=['Car']):
        super().__init__()
        self.num_classes = num_classes
        self.active_classes = active_classes
        
        # Stage 1: 2D detector (YOLOv3)
        self.detector_2d = None  # Not needed during training (use gt_boxes_2d)
        
        # Feature extractor for RoI
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        roi_feat_dim = 2048
        
        # Intrinsics encoder
        self.intrinsics_encoder = IntrinsicsEncoder(output_dim=128)
        
        # Stage 2 heads
        self.depth_head = DepthHead(roi_feat_dim, intrinsics_dim=128)
        self.dimension_head = DimensionHead(roi_feat_dim, intrinsics_dim=128)
        self.rotation_head = RotationHead(roi_feat_dim)
        self.iou_head = IoUHead(roi_feat_dim)
        self.foreground_head = ForegroundHead(roi_feat_dim)
    
    def forward(self, images, intrinsics, gt_boxes_2d=None):
        """
        Args:
            images: [B, 3, H, W]
            intrinsics: [B, 3, 3] camera intrinsics
            gt_boxes_2d: list of [N, 4] 2D boxes (for training)
        
        Returns:
            predictions dict
        """
        batch_size = images.shape[0]
        
        # Encode intrinsics
        intrinsics_features = self.intrinsics_encoder(intrinsics)
        
        # Stage 1: 2D detection
        if gt_boxes_2d is not None:
            boxes_2d_batch = gt_boxes_2d
        else:
            results = self.detector_2d(images)
            boxes_2d_batch = []
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    boxes = r.boxes.xyxy
                    boxes_2d_batch.append(boxes)
                else:
                    boxes_2d_batch.append(torch.zeros((0, 4), device=images.device))
        
        # Extract features
        features = self.feature_extractor(images)
        
        # Stage 2: Process each RoI
        all_depth = []
        all_dims = []
        all_rotation = []
        all_iou = []
        all_fg = []
        
        for i in range(batch_size):
            boxes_2d = boxes_2d_batch[i]
            if len(boxes_2d) == 0:
                continue
            
            # RoI pooling
            feat_map = features[i:i+1]
            h, w = feat_map.shape[2:]
            img_h, img_w = images.shape[2:]
            
            roi_features_list = []
            for box in boxes_2d:
                x1, y1, x2, y2 = box
                x1_n = (x1 / img_w * w).long().clamp(0, w-1)
                x2_n = (x2 / img_w * w).long().clamp(0, w-1)
                y1_n = (y1 / img_h * h).long().clamp(0, h-1)
                y2_n = (y2 / img_h * h).long().clamp(0, h-1)
                
                roi_feat = feat_map[:, :, y1_n:y2_n+1, x1_n:x2_n+1]
                roi_feat = self.avgpool(roi_feat).squeeze()
                roi_features_list.append(roi_feat)
            
            if len(roi_features_list) == 0:
                continue
            
            roi_features = torch.stack(roi_features_list)
            
            # Expand intrinsics features for all RoIs
            intrinsics_feat_expanded = intrinsics_features[i:i+1].expand(len(roi_features), -1)
            
            # Predict 3D parameters
            pred_depth = self.depth_head(roi_features, intrinsics_feat_expanded)
            pred_dims = self.dimension_head(roi_features, intrinsics_feat_expanded)
            pred_rot = self.rotation_head(roi_features)
            pred_iou = self.iou_head(roi_features)
            pred_fg = self.foreground_head(roi_features)
            
            all_depth.append(pred_depth)
            all_dims.append(pred_dims)
            all_rotation.append(pred_rot)
            all_iou.append(pred_iou)
            all_fg.append(pred_fg)
        
        return {
            'boxes_2d': boxes_2d_batch,
            'depth': all_depth,
            'dimensions': all_dims,
            'rotation': all_rotation,
            'iou': all_iou,
            'foreground': all_fg
        }


def build_model_v3(active_classes=['Car']):
    """Build V3 model (rotation-only, no direction head)"""
    model = TwoStage3DDetector(num_classes=len(active_classes), active_classes=active_classes)
    return model
