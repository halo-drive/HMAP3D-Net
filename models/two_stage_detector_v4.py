"""Two-Stage 3D Object Detector V4 - Domain-Adaptive with Normalized Intrinsics

Major Changes from V3:
1. Normalized intrinsics conditioning (resolution-invariant)
2. Removed log_variance from depth head (stability fix)
3. Multi-resolution support via dynamic image size detection
4. Simplified architecture for better generalization

Architecture:
- Stage 1: YOLOv3/YOLOv8 for 2D detection
- Stage 2: 3D parameter regression with NORMALIZED intrinsics
  - Intrinsics encoder (normalized: fx/w, fy/h, cx/w, cy/h)
  - Depth head (direct regression, no uncertainty)
  - Dimension head (intrinsics-conditioned)
  - Rotation head (24 bins, full 360°)
  - Foreground head (auxiliary)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class NormalizedIntrinsicsEncoder(nn.Module):
    """Encode NORMALIZED camera intrinsics (resolution-invariant)"""
    def __init__(self, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.ReLU()
        )
    
    def forward(self, K, image_size):
        """
        Args:
            K: [B, 3, 3] intrinsic matrices
            image_size: (height, width) tuple
        Returns:
            [B, output_dim] normalized intrinsics features
        """
        h, w = image_size
        
        # Extract intrinsics
        fx = K[:, 0, 0]
        fy = K[:, 1, 1]
        cx = K[:, 0, 2]
        cy = K[:, 1, 2]
        
        # CRITICAL: Normalize by image dimensions (resolution-invariant)
        fx_norm = fx / w
        fy_norm = fy / h
        cx_norm = cx / w
        cy_norm = cy / h
        
        intrinsics_vec = torch.stack([fx_norm, fy_norm, cx_norm, cy_norm], dim=1)
        return self.encoder(intrinsics_vec)


class DepthHeadV4(nn.Module):
    """Predict depth with offset (NO uncertainty - stability fix)"""
    def __init__(self, input_dim=2048, intrinsics_dim=128):
        super().__init__()
        combined_dim = input_dim + intrinsics_dim
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)  # [depth, offset] - NO log_variance
        )
    
    def forward(self, roi_features, intrinsics_features):
        """
        Returns:
            [N, 2]: [depth, offset]
        """
        combined = torch.cat([roi_features, intrinsics_features], dim=1)
        return self.fc(combined)


class DimensionHeadV4(nn.Module):
    """Predict 3D dimensions (intrinsics-conditioned)"""
    def __init__(self, input_dim=2048, intrinsics_dim=128):
        super().__init__()
        combined_dim = input_dim + intrinsics_dim
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3)
        )
    
    def forward(self, roi_features, intrinsics_features):
        """
        Returns:
            [N, 3]: [height, width, length]
        """
        combined = torch.cat([roi_features, intrinsics_features], dim=1)
        return self.fc(combined)


class RotationHeadV4(nn.Module):
    """Predict rotation with bin classification + residual (24 bins)"""
    def __init__(self, input_dim=2048):
        super().__init__()
        self.num_bins = 24
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.bin_fc = nn.Linear(256, self.num_bins)
        self.res_fc = nn.Linear(256, self.num_bins)
    
    def forward(self, roi_features):
        """
        Returns:
            bins: [N, 24] bin logits
            residuals: [N, 24] residual values
        """
        shared_feat = self.shared(roi_features)
        bins = self.bin_fc(shared_feat)
        residuals = self.res_fc(shared_feat)
        return bins, residuals


class ForegroundHeadV4(nn.Module):
    """Binary foreground/background classification (auxiliary)"""
    def __init__(self, input_dim=2048):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, roi_features):
        return self.fc(roi_features)


class TwoStage3DDetectorV4(nn.Module):
    """V4: Domain-adaptive two-stage 3D detector with normalized intrinsics"""
    def __init__(self, num_classes=3, active_classes=['Car']):
        super().__init__()
        self.num_classes = num_classes
        self.active_classes = active_classes
        
        # Stage 1: 2D detector (not needed during training with gt_boxes_2d)
        self.detector_2d = None
        
        # Feature extractor for RoIs
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        roi_feat_dim = 2048
        
        # Normalized intrinsics encoder
        self.intrinsics_encoder = NormalizedIntrinsicsEncoder(output_dim=128)
        
        # Stage 2 heads
        self.depth_head = DepthHeadV4(roi_feat_dim, intrinsics_dim=128)
        self.dimension_head = DimensionHeadV4(roi_feat_dim, intrinsics_dim=128)
        self.rotation_head = RotationHeadV4(roi_feat_dim)
        self.foreground_head = ForegroundHeadV4(roi_feat_dim)
    
    def forward(self, images, intrinsics, gt_boxes_2d=None):
        """
        Args:
            images: [B, 3, H, W] - any resolution
            intrinsics: [B, 3, 3] camera intrinsics (for current image size)
            gt_boxes_2d: list of [N, 4] 2D boxes (for training)
        
        Returns:
            predictions dict
        """
        batch_size = images.shape[0]
        image_h, image_w = images.shape[2:4]
        
        # Encode normalized intrinsics (resolution-invariant)
        intrinsics_features = self.intrinsics_encoder(intrinsics, (image_h, image_w))
        
        # Stage 1: 2D detection
        if gt_boxes_2d is not None:
            boxes_2d_batch = gt_boxes_2d
        else:
            if self.detector_2d is None:
                raise ValueError("2D detector not loaded. Provide gt_boxes_2d or load detector.")
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
        all_fg = []
        
        for i in range(batch_size):
            boxes_2d = boxes_2d_batch[i]
            if len(boxes_2d) == 0:
                continue
            
            # RoI pooling
            feat_map = features[i:i+1]
            h, w = feat_map.shape[2:]
            
            roi_features_list = []
            for box in boxes_2d:
                x1, y1, x2, y2 = box
                x1_n = (x1 / image_w * w).long().clamp(0, w-1)
                x2_n = (x2 / image_w * w).long().clamp(0, w-1)
                y1_n = (y1 / image_h * h).long().clamp(0, h-1)
                y2_n = (y2 / image_h * h).long().clamp(0, h-1)
                
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
            pred_fg = self.foreground_head(roi_features)
            
            all_depth.append(pred_depth)
            all_dims.append(pred_dims)
            all_rotation.append(pred_rot)
            all_fg.append(pred_fg)
        
        return {
            'boxes_2d': boxes_2d_batch,
            'depth': all_depth,
            'dimensions': all_dims,
            'rotation': all_rotation,
            'foreground': all_fg
        }


def build_model_v4(active_classes=['Car']):
    """
    Build V4 model with normalized intrinsics
    
    Args:
        active_classes: List of class names
    
    Returns:
        TwoStage3DDetectorV4 model
    """
    model = TwoStage3DDetectorV4(
        num_classes=len(active_classes),
        active_classes=active_classes
    )
    return model