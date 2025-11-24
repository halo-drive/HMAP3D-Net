"""KITTI 3D Object Detection Dataset Loader - CORRECTED"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import torch
from torch.utils.data import Dataset


class KITTI3DDataset(Dataset):
    """
    KITTI 3D Object Detection Dataset
    
    KITTI Format:
    - location (x,y,z): y is BOTTOM CENTER of 3D box, not geometric center
    - dimensions (h,w,l): height, width, length in meters
    - rotation_y: rotation around Y-axis in camera frame
    """
    
    CLASSES = ['Car', 'Pedestrian', 'Cyclist']
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (384, 1280),
        filter_classes: List[str] = ['Car'],
        min_height: int = 25,
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
        
        # Train/val split (80/20)
        np.random.seed(42)
        all_indices = list(range(7481))
        np.random.shuffle(all_indices)
        split_idx = int(0.8 * len(all_indices))
        
        if split == 'train':
            self.indices = all_indices[:split_idx]
        else:
            self.indices = all_indices[split_idx:]
        
        print(f"KITTI {split} split: {len(self.indices)} images")
    
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
                    'location': np.array([float(x) for x in parts[11:14]]),   # [x,y_bottom,z]
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
            
            # CRITICAL FIX: Convert KITTI location to box center
            # KITTI: location_y is BOTTOM center
            # We need: geometric center
            loc = obj['location'].copy()
            dims = obj['dimensions']  # [h, w, l]
            
            # Adjust y to geometric center (shift up by h/2)
            loc[1] = loc[1] - dims[0] / 2.0
            
            # Now loc is [x_center, y_center, z_center]
            box_3d = np.concatenate([loc, dims, [obj['rotation_y']]])
            
            label = self.CLASSES.index(obj['class'])
            
            boxes_2d.append(box_2d)
            boxes_3d.append(box_3d)
            labels.append(label)
        
        # Convert to tensors - OPTIMIZED (no list warning)
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


def test_dataset():
    """Test dataset loading"""
    import matplotlib.pyplot as plt
    
    dataset = KITTI3DDataset(
        root_dir='/media/ashwin-benchdev/eb2a8889-3ec9-411a-826e-816cf9759b02/KITTI-3D',
        split='train',
        filter_classes=['Car']
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    
    print(f"Image shape: {sample['image'].shape}")
    print(f"Number of objects: {len(sample['boxes_2d'])}")
    print(f"Boxes 3D shape: {sample['boxes_3d'].shape}")
    print(f"Sample 3D box: {sample['boxes_3d'][0]}")
    
    image = sample['image'].permute(1, 2, 0).numpy()
    boxes_2d = sample['boxes_2d'].numpy()
    
    plt.figure(figsize=(12, 6))
    plt.imshow(image)
    
    for box in boxes_2d:
        x1, y1, x2, y2 = box
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'g-', linewidth=2)
    
    plt.title(f'Sample with {len(boxes_2d)} objects')
    plt.axis('off')
    plt.savefig('kitti_sample_fixed.png', dpi=150, bbox_inches='tight')
    print("Saved: kitti_sample_fixed.png")


if __name__ == "__main__":
    test_dataset()
