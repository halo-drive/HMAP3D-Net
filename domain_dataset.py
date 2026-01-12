"""Domain-Specific Dataset Loader for Fine-Tuning on 10 Frames"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import torch
from torch.utils.data import Dataset


class DomainDataset(Dataset):
    """
    Custom dataset for domain-specific fine-tuning
    Uses KITTI format but with small number of samples
    """
    
    CLASSES = ['Car', 'Pedestrian', 'Cyclist']
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (384, 1280),
        filter_classes: List[str] = ['Car'],
        min_height: int = 25,
        val_split: float = 0.2  # Use 20% for validation (2 out of 10 frames)
    ):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_h, self.image_w = image_size
        self.filter_classes = set(filter_classes)
        self.min_height = min_height
        
        self.image_dir = self.root_dir / 'training' / 'image_2'
        self.label_dir = self.root_dir / 'training' / 'label_2'
        self.calib_dir = self.root_dir / 'training' / 'calib'
        
        # Get all available indices
        image_files = sorted(list(self.image_dir.glob('*.png')))
        all_indices = [int(f.stem) for f in image_files]
        
        # Split into train/val
        np.random.seed(42)
        shuffled_indices = np.array(all_indices)
        np.random.shuffle(shuffled_indices)
        
        split_idx = max(1, int(len(shuffled_indices) * (1 - val_split)))
        
        if split == 'train':
            self.indices = shuffled_indices[:split_idx].tolist()
        else:
            self.indices = shuffled_indices[split_idx:].tolist()
        
        print(f"Domain {split} split: {len(self.indices)} images (from {len(all_indices)} total)")
    
    def __len__(self):
        return len(self.indices)
    
    def load_calibration(self, idx: int) -> Dict:
        """Load calibration file"""
        calib_file = self.calib_dir / f'{idx:06d}.txt'
        
        calib = {}
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                calib[key] = np.array([float(x) for x in value.split()])
        
        P2 = calib['P2'].reshape(3, 4)
        K = P2[:3, :3]
        
        return {'K': K, 'P2': P2}
    
    def load_labels(self, idx: int) -> List[Dict]:
        """Load 3D object labels"""
        label_file = self.label_dir / f'{idx:06d}.txt'
        
        objects = []
        with open(label_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(' ')
                
                cls = parts[0]
                if cls not in self.filter_classes:
                    continue
                
                obj = {
                    'class': cls,
                    'truncated': float(parts[1]),
                    'occluded': int(parts[2]),
                    'alpha': float(parts[3]),
                    'bbox_2d': np.array([float(x) for x in parts[4:8]]),
                    'dimensions': np.array([float(x) for x in parts[8:11]]),  # [h,w,l]
                    'location': np.array([float(x) for x in parts[11:14]]),   # [x,y,z]
                    'rotation_y': float(parts[14])
                }
                
                box_h = obj['bbox_2d'][3] - obj['bbox_2d'][1]
                if box_h < self.min_height:
                    continue
                
                objects.append(obj)
        
        return objects
    
    def __getitem__(self, idx: int) -> Dict:
        img_idx = self.indices[idx]
        
        img_path = self.image_dir / f'{img_idx:06d}.png'
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        calib = self.load_calibration(img_idx)
        K = calib['K'].copy()
        
        objects = self.load_labels(img_idx)
        
        # Resize image
        scale_x = self.image_w / orig_w
        scale_y = self.image_h / orig_h
        image = cv2.resize(image, (self.image_w, self.image_h))
        
        # Adjust intrinsics
        K[0, :] *= scale_x
        K[1, :] *= scale_y
        
        # Process annotations
        boxes_2d = []
        boxes_3d = []
        labels = []
        
        for obj in objects:
            # Scale 2D box
            box_2d = obj['bbox_2d'].copy()
            box_2d[0::2] *= scale_x
            box_2d[1::2] *= scale_y
            
            # Your labels are already in center format (corrected by label editor)
            # Just use location and dimensions directly
            loc = obj['location'].copy()
            dims = obj['dimensions']  # [h, w, l]
            
            box_3d = np.concatenate([loc, dims, [obj['rotation_y']]])
            
            label = self.CLASSES.index(obj['class'])
            
            boxes_2d.append(box_2d)
            boxes_3d.append(box_3d)
            labels.append(label)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if len(boxes_2d) > 0:
            boxes_2d = torch.from_numpy(np.stack(boxes_2d)).float()
            boxes_3d = torch.from_numpy(np.stack(boxes_3d)).float()
            labels = torch.from_numpy(np.array(labels)).long()
        else:
            boxes_2d = torch.zeros((0, 4), dtype=torch.float32)
            boxes_3d = torch.zeros((0, 7), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        
        return {
            'image': image_tensor,
            'boxes_2d': boxes_2d,
            'boxes_3d': boxes_3d,
            'labels': labels,
            'intrinsics': torch.from_numpy(K).float(),
            'img_idx': img_idx
        }
