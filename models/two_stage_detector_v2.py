"""Two-Stage 3D Detection V2 - Intrinsics-Conditioned + PointPillars Heads

Key Improvements:
1. Intrinsics-conditioned depth/dimension prediction
2. IoU prediction head for quality estimation
3. Direction classifier for orientation disambiguation
4. Auxiliary foreground/background head
5. Intrinsics augmentation during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from torchvision.models import resnet50
from ultralytics import YOLO
import numpy as np
import cv2

COCO_TO_KITTI = {
    0: 'Pedestrian',
    1: 'Cyclist',
    2: 'Car',
    3: 'Cyclist',
    5: 'Car',
    7: 'Car',
}


class IntrinsicsEncoder(nn.Module):
    """Encode camera intrinsics into a feature vector"""
    def __init__(self, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 64),  # [fx, fy, cx, cy]
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.ReLU()
        )
    
    def forward(self, intrinsics_batch):
        """
        Args:
            intrinsics_batch: [N, 3, 3] or [N, 4] camera intrinsics
        Returns:
            [N, output_dim] encoded features
        """
        if intrinsics_batch.dim() == 3:
            # Extract [fx, fy, cx, cy] from 3x3 matrix
            fx = intrinsics_batch[:, 0, 0]
            fy = intrinsics_batch[:, 1, 1]
            cx = intrinsics_batch[:, 0, 2]
            cy = intrinsics_batch[:, 1, 2]
            intrinsics_vec = torch.stack([fx, fy, cx, cy], dim=1)
        else:
            intrinsics_vec = intrinsics_batch
        
        return self.encoder(intrinsics_vec)


class IntrinsicsConditionedDepthHead(nn.Module):
    """Depth prediction conditioned on camera intrinsics"""
    def __init__(self, in_channels=2048, intrinsics_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels + intrinsics_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # [depth, log_var, offset]
        )
    
    def forward(self, roi_features, intrinsics_features):
        """
        Args:
            roi_features: [N, in_channels] RoI features
            intrinsics_features: [N, intrinsics_dim] encoded intrinsics
        Returns:
            depth, log_var, offset
        """
        # Concatenate RoI features with intrinsics encoding
        combined = torch.cat([roi_features, intrinsics_features], dim=1)
        
        out = self.fc(combined)
        depth = F.relu(out[:, 0]) + 0.1
        log_var = out[:, 1]
        offset = out[:, 2]
        
        return depth, log_var, offset


class IntrinsicsConditionedDimensionHead(nn.Module):
    """Dimension prediction conditioned on camera intrinsics"""
    def __init__(self, in_channels=2048, intrinsics_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels + intrinsics_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3)  # [h, w, l]
        )
    
    def forward(self, roi_features, intrinsics_features):
        combined = torch.cat([roi_features, intrinsics_features], dim=1)
        dims = F.relu(self.fc(combined)) + 0.1
        return dims


class DirectionHead(nn.Module):
    """Binary direction classifier: front (0-180°) vs back (180-360°)"""
    def __init__(self, in_channels=2048):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        return self.fc(x)


class RotationHead(nn.Module):
    """Rotation prediction with bin classification + residual"""
    def __init__(self, in_channels=2048, num_bins=12):
        super().__init__()
        self.num_bins = num_bins
        self.bin_size = 2 * 3.14159 / num_bins
        
        self.fc_bin = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_bins)
        )
        
        self.fc_res = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_bins)
        )
    
    def forward(self, x):
        bin_logits = self.fc_bin(x)
        residuals = torch.tanh(self.fc_res(x)) * (self.bin_size / 2)
        return bin_logits, residuals


class IoUPredictionHead(nn.Module):
    """Predict 3D IoU between predicted box and ground truth"""
    def __init__(self, in_channels=2048):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x).squeeze(-1)


class ForegroundHead(nn.Module):
    """Auxiliary head: classify if RoI contains foreground object"""
    def __init__(self, in_channels=2048):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        return self.fc(x)


class TwoStage3DDetectorV2(nn.Module):
    """Enhanced two-stage detector with intrinsics conditioning"""
    
    def __init__(self, yolo_model='yolov8x.pt', freeze_2d=True, active_classes=None):
        super().__init__()
        
        if active_classes is None:
            active_classes = ['Car', 'Pedestrian', 'Cyclist']
        self.active_classes = set(active_classes)
        
        self.coco_to_kitti_map = {}
        self.valid_coco_classes = []
        for coco_id, kitti_class in COCO_TO_KITTI.items():
            if kitti_class in self.active_classes:
                self.valid_coco_classes.append(coco_id)
                self.coco_to_kitti_map[coco_id] = kitti_class
        
        print(f"Active KITTI classes: {self.active_classes}")
        print(f"Mapped COCO classes: {self.valid_coco_classes}")
        
        # Load YOLO
        yolo = YOLO(yolo_model)
        yolo.model.eval()
        for param in yolo.model.parameters():
            param.requires_grad = False
        
        object.__setattr__(self, '_yolo_model', yolo)
        
        # Feature extractor
        resnet = resnet50(pretrained=True)
        self.feature_backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # RoI Align
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1.0/32, sampling_ratio=2)
        
        # Intrinsics encoder (shared across all heads)
        self.intrinsics_encoder = IntrinsicsEncoder(output_dim=128)
        
        # Main 3D regression heads (intrinsics-conditioned)
        self.depth_head = IntrinsicsConditionedDepthHead(in_channels=2048, intrinsics_dim=128)
        self.dimension_head = IntrinsicsConditionedDimensionHead(in_channels=2048, intrinsics_dim=128)
        
        # PointPillars-style heads
        self.direction_head = DirectionHead(in_channels=2048)
        self.rotation_head = RotationHead(in_channels=2048, num_bins=12)
        self.iou_head = IoUPredictionHead(in_channels=2048)
        
        # Auxiliary head
        self.foreground_head = ForegroundHead(in_channels=2048)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def extract_2d_boxes(self, images):
        """Extract 2D boxes with class tracking"""
        batch_size = images.shape[0]
        boxes_list = []
        scores_list = []
        classes_list = []
        
        yolo = object.__getattribute__(self, '_yolo_model')
        
        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        for i in range(batch_size):
            # Denormalize
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = img * IMAGENET_STD + IMAGENET_MEAN
            img = (img * 255).clip(0, 255).astype('uint8')
            
            # Upscale for better YOLO detection
            h, w = img.shape[:2]
            scale_factor = 3
            target_h, target_w = h * scale_factor, w * scale_factor
            img_upscaled = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            # Run YOLO
            results = yolo.predict(img_upscaled, verbose=False, conf=0.2)
            
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu()
                boxes[:, [0, 2]] /= scale_factor
                boxes[:, [1, 3]] /= scale_factor
                
                scores = results[0].boxes.conf.cpu()
                coco_classes = results[0].boxes.cls.cpu().numpy().astype(int)
                
                keep_mask = torch.tensor([cls in self.valid_coco_classes for cls in coco_classes])
                conf_mask = scores > 0.3
                keep_mask = keep_mask & conf_mask
                
                boxes = boxes[keep_mask]
                scores = scores[keep_mask]
                coco_classes_filtered = coco_classes[keep_mask.numpy()]
                
                kitti_classes = [self.coco_to_kitti_map[coco_id] for coco_id in coco_classes_filtered]
            else:
                boxes = torch.zeros((0, 4))
                scores = torch.zeros((0,))
                kitti_classes = []
            
            boxes_list.append(boxes)
            scores_list.append(scores)
            classes_list.append(kitti_classes)
        
        return boxes_list, scores_list, classes_list
    
    def forward(self, images, intrinsics, gt_boxes_2d=None, gt_classes=None):
        """
        Args:
            images: [B, 3, H, W]
            intrinsics: [B, 3, 3] or [B, 4] camera intrinsics
            gt_boxes_2d: Optional ground truth 2D boxes for training
            gt_classes: Optional ground truth classes
        """
        batch_size = images.shape[0]
        device = images.device
        
        # Extract 2D boxes
        if gt_boxes_2d is not None:
            boxes_list = [gt_boxes_2d[i] for i in range(batch_size)]
            scores_list = [torch.ones(len(boxes_list[i]), device=device) for i in range(batch_size)]
            if gt_classes is not None:
                classes_list = gt_classes
            else:
                classes_list = [[] for _ in range(batch_size)]
        else:
            with torch.no_grad():
                boxes_list, scores_list, classes_list = self.extract_2d_boxes(images)
        
        # Extract features
        features = self.feature_backbone(images)
        
        # Encode intrinsics once for the batch
        intrinsics_encoded = self.intrinsics_encoder(intrinsics)
        
        all_depths = []
        all_dims = []
        all_directions = []
        all_rotations = []
        all_ious = []
        all_foregrounds = []
        
        for i in range(batch_size):
            boxes_2d = boxes_list[i].to(device)
            
            if len(boxes_2d) == 0:
                all_depths.append(torch.zeros((0, 3), device=device))
                all_dims.append(torch.zeros((0, 3), device=device))
                all_directions.append(torch.zeros((0, 2), device=device))
                all_rotations.append((torch.zeros((0, 12), device=device), torch.zeros((0, 12), device=device)))
                all_ious.append(torch.zeros((0,), device=device))
                all_foregrounds.append(torch.zeros((0, 2), device=device))
                continue
            
            # RoI Align
            rois = torch.cat([
                torch.full((len(boxes_2d), 1), i, device=device),
                boxes_2d
            ], dim=1)
            
            roi_features = self.roi_align(features, rois)
            roi_features = self.avgpool(roi_features).squeeze(-1).squeeze(-1)
            
            # Repeat intrinsics encoding for each RoI
            num_rois = len(boxes_2d)
            intrinsics_feat = intrinsics_encoded[i:i+1].expand(num_rois, -1)
            
            # Intrinsics-conditioned predictions
            depth, depth_log_var, depth_offset = self.depth_head(roi_features, intrinsics_feat)
            dims = self.dimension_head(roi_features, intrinsics_feat)
            
            # PointPillars-style heads
            direction_logits = self.direction_head(roi_features)
            rot_bins, rot_res = self.rotation_head(roi_features)
            iou_pred = self.iou_head(roi_features)
            
            # Auxiliary head
            fg_logits = self.foreground_head(roi_features)
            
            all_depths.append(torch.stack([depth, depth_log_var, depth_offset], dim=1))
            all_dims.append(dims)
            all_directions.append(direction_logits)
            all_rotations.append((rot_bins, rot_res))
            all_ious.append(iou_pred)
            all_foregrounds.append(fg_logits)
        
        return {
            'boxes_2d': boxes_list,
            'scores_2d': scores_list,
            'classes': classes_list,
            'depth': all_depths,
            'dimensions': all_dims,
            'direction': all_directions,
            'rotation': all_rotations,
            'iou': all_ious,
            'foreground': all_foregrounds
        }


def build_model_v2(active_classes=None):
    """Build improved model with intrinsics conditioning"""
    if active_classes is None:
        active_classes = ['Car']
    
    model = TwoStage3DDetectorV2(
        yolo_model='yolov8x.pt',
        freeze_2d=True,
        active_classes=active_classes
    )
    return model
