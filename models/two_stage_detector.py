"""Two-Stage 3D Detection - With Anti-Overfitting Regularization"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from torchvision.models import resnet50
from ultralytics import YOLO
import numpy as np


COCO_TO_KITTI = {
    0: 'Pedestrian',
    1: 'Cyclist',
    2: 'Car',
    3: 'Cyclist',
    5: 'Car',
    7: 'Car',
}


class Depth3DHead(nn.Module):
    def __init__(self, in_channels=2048):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased from 0.3
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Added extra dropout
            nn.Linear(256, 3)
        )
    
    def forward(self, x):
        out = self.fc(x)
        depth = F.relu(out[:, 0]) + 0.1
        log_var = out[:, 1]
        offset = out[:, 2]
        return depth, log_var, offset


class Dimension3DHead(nn.Module):
    def __init__(self, in_channels=2048):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased from 0.3
            nn.Linear(512, 3)
        )
    
    def forward(self, x):
        dims = F.relu(self.fc(x)) + 0.1
        return dims


class Rotation3DHead(nn.Module):
    def __init__(self, in_channels=2048, num_bins=12):
        super().__init__()
        self.num_bins = num_bins
        self.bin_size = 2 * 3.14159 / num_bins
        
        self.fc_bin = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased from 0.3
            nn.Linear(512, num_bins)
        )
        
        self.fc_res = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased from 0.3
            nn.Linear(512, num_bins)
        )
    
    def forward(self, x):
        bin_logits = self.fc_bin(x)
        residuals = torch.tanh(self.fc_res(x)) * (self.bin_size / 2)
        return bin_logits, residuals


class TwoStage3DDetector(nn.Module):
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
        
        # 3D Regression Heads (with increased dropout)
        self.depth_head = Depth3DHead(in_channels=2048)
        self.dimension_head = Dimension3DHead(in_channels=2048)
        self.rotation_head = Rotation3DHead(in_channels=2048, num_bins=12)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def extract_2d_boxes(self, images):
        """Extract 2D boxes with class tracking"""
        batch_size = images.shape[0]
        boxes_list = []
        scores_list = []
        classes_list = []
        
        yolo = object.__getattribute__(self, '_yolo_model')
        
        for i in range(batch_size):
            img = images[i].cpu().numpy().transpose(1, 2, 0) * 255
            img = img.astype('uint8')
            
            results = yolo.predict(img, verbose=False)
            
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu()
                scores = results[0].boxes.conf.cpu()
                coco_classes = results[0].boxes.cls.cpu().numpy().astype(int)
                
                keep_mask = torch.tensor([cls in self.valid_coco_classes for cls in coco_classes])
                conf_mask = scores > 0.5
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
    
    def forward(self, images, intrinsics=None, gt_boxes_2d=None, gt_classes=None):
        batch_size = images.shape[0]
        device = images.device
        
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
        
        features = self.feature_backbone(images)
        
        all_depths = []
        all_dims = []
        all_rotations = []
        
        for i in range(batch_size):
            boxes_2d = boxes_list[i].to(device)
            
            if len(boxes_2d) == 0:
                all_depths.append(torch.zeros((0, 3), device=device))
                all_dims.append(torch.zeros((0, 3), device=device))
                all_rotations.append((torch.zeros((0, 12), device=device), torch.zeros((0, 12), device=device)))
                continue
            
            rois = torch.cat([
                torch.full((len(boxes_2d), 1), i, device=device),
                boxes_2d
            ], dim=1)
            
            roi_features = self.roi_align(features, rois)
            roi_features = self.avgpool(roi_features).squeeze(-1).squeeze(-1)
            
            depth, depth_log_var, depth_offset = self.depth_head(roi_features)
            dims = self.dimension_head(roi_features)
            rot_bins, rot_res = self.rotation_head(roi_features)
            
            all_depths.append(torch.stack([depth, depth_log_var, depth_offset], dim=1))
            all_dims.append(dims)
            all_rotations.append((rot_bins, rot_res))
        
        return {
            'boxes_2d': boxes_list,
            'scores_2d': scores_list,
            'classes': classes_list,
            'depth': all_depths,
            'dimensions': all_dims,
            'rotation': all_rotations
        }


def build_model(active_classes=None):
    if active_classes is None:
        active_classes = ['Car']
    
    model = TwoStage3DDetector(
        yolo_model='yolov8x.pt', 
        freeze_2d=True,
        active_classes=active_classes
    )
    return model
