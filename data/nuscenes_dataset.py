"""
nuScenes Monocular 3D Object Detection Dataset Loader

Handles:
- Image loading from multi-blob directory structure
- Coordinate transformations: global/ego → camera frame
- Ground truth generation: heatmap, depth bins, rotation bins, dimensions
- Augmentation: flip, color jitter, scale (with intrinsics adjustment)
- Camera intrinsic scaling for 1600×900 → 1280×720 resize

Critical operations:
- Quaternion-based rotation matrix computation
- Multi-stage coordinate transformation pipeline
- Gaussian heatmap generation for object centers
- Depth/rotation discretization into bins + residual encoding
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from pyquaternion import Quaternion
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from collections import defaultdict
import random


class NuScenesMonocular3D(Dataset):
    """
    nuScenes dataset for monocular 3D object detection
    
    Returns samples at 1280×720 resolution with full ground truth:
    - Image: (3, 720, 1280) RGB tensor
    - Heatmap: (1, 180, 320) Gaussian peaks at object centers
    - Depth targets: bin indices + residuals
    - Dimension targets: (h, w, l) in meters
    - Rotation targets: bin indices + residuals
    - 2D offset targets: sub-pixel localization
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        input_size: Tuple[int, int] = (720, 1280),
        output_stride: int = 4,
        depth_bins: List[float] = [0, 10, 20, 35, 50, 70, 200],
        num_rotation_bins: int = 8,
        max_objects: int = 50,
        augment: bool = True,
    ):
        """
        Args:
            root_dir: Path to NuScenes-Full-Dataset directory
            split: 'train' or 'val' (80/20 split of trainval)
            input_size: (H, W) network input resolution
            output_stride: Downsampling factor for heatmap
            depth_bins: Bin boundaries in meters
            num_rotation_bins: Number of angular bins (45° each for 8 bins)
            max_objects: Maximum objects per image
            augment: Enable data augmentation
        """
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.input_h, self.input_w = input_size
        self.output_stride = output_stride
        self.output_h = self.input_h // output_stride
        self.output_w = self.input_w // output_stride
        
        self.depth_bins = np.array(depth_bins, dtype=np.float32)
        self.num_depth_bins = len(depth_bins) - 1
        self.depth_bin_centers = (self.depth_bins[:-1] + self.depth_bins[1:]) / 2
        
        self.num_rotation_bins = num_rotation_bins
        self.rotation_bin_size = 2 * np.pi / num_rotation_bins
        
        self.max_objects = max_objects
        self.augment = augment
        
        # Paths
        self.dataroot = self.root_dir / "NuScenes" / "trainval"
        self.annotation_root = self.root_dir / "nuscenes_prepared" / "v1.0-trainval"
        
        # Load annotation tables
        print(f"Loading nuScenes annotations for {split} split...")
        self._load_annotations()
        self._build_lookups()
        self._filter_cam_front_samples()
        self._create_split()
        
        print(f"Dataset initialized: {len(self.sample_indices)} samples")
        
    def _load_annotations(self):
        """Load all required JSON annotation tables"""
        tables = [
            'sample', 'sample_data', 'sample_annotation',
            'calibrated_sensor', 'ego_pose', 'instance',
            'category', 'sensor'
        ]
        
        for table in tables:
            filepath = self.annotation_root / f"{table}.json"
            with open(filepath, 'r') as f:
                setattr(self, table, json.load(f))
    
    def _build_lookups(self):
        """Build lookup dictionaries for fast access"""
        self.instance_lookup = {inst['token']: inst for inst in self.instance}
        self.category_lookup = {cat['token']: cat for cat in self.category}
        self.calib_lookup = {c['token']: c for c in self.calibrated_sensor}
        self.ego_lookup = {e['token']: e for e in self.ego_pose}
        self.sample_data_lookup = {sd['token']: sd for sd in self.sample_data}
        
        # Sample annotations grouped by sample_token
        self.sample_annotations = defaultdict(list)
        for ann in self.sample_annotation:
            self.sample_annotations[ann['sample_token']].append(ann)
        
        # Find CAM_FRONT sensor
        self.cam_front_sensor = next(
            s for s in self.sensor if s['channel'] == 'CAM_FRONT'
        )
        
        # Vehicle category tokens
        self.vehicle_categories = set()
        for cat in self.category:
            if any(v in cat['name'] for v in ['vehicle.car', 'vehicle.truck', 
                                               'vehicle.bus', 'vehicle.construction',
                                               'vehicle.trailer', 'vehicle.motorcycle']):
                self.vehicle_categories.add(cat['token'])
    
    def _filter_cam_front_samples(self):
        """Build index of keyframe CAM_FRONT samples"""
        self.cam_front_samples = []
        
        for sd in self.sample_data:
            calib = self.calib_lookup.get(sd['calibrated_sensor_token'])
            if (calib and 
                calib['sensor_token'] == self.cam_front_sensor['token'] and
                sd.get('is_key_frame', False)):
                self.cam_front_samples.append(sd)
        
        print(f"Found {len(self.cam_front_samples)} CAM_FRONT keyframes")
    
    def _create_split(self):
        """Create train/val split (80/20)"""
        np.random.seed(42)
        num_samples = len(self.cam_front_samples)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        split_idx = int(0.8 * num_samples)
        
        if self.split == 'train':
            self.sample_indices = indices[:split_idx]
        else:
            self.sample_indices = indices[split_idx:]
    
    def __len__(self):
        return len(self.sample_indices)
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to 3×3 rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
    
    def _transform_to_camera_frame(
        self, 
        point_global: np.ndarray,
        ego_pose: Dict,
        calib_sensor: Dict
    ) -> np.ndarray:
        """
        Transform point from global coordinates to camera frame
        
        Pipeline: Global → Ego → Camera
        
        Args:
            point_global: (3,) array [x, y, z] in global frame
            ego_pose: ego_pose record (vehicle in global frame)
            calib_sensor: calibrated_sensor record (sensor in ego frame)
        
        Returns:
            (3,) array [x, y, z] in camera frame
        """
        point_global = np.array(point_global, dtype=np.float32)
        
        # Global → Ego transformation
        ego_translation = np.array(ego_pose['translation'], dtype=np.float32)
        ego_rotation = self._quaternion_to_rotation_matrix(ego_pose['rotation'])
        
        # Ego → Camera transformation
        sensor_translation = np.array(calib_sensor['translation'], dtype=np.float32)
        sensor_rotation = self._quaternion_to_rotation_matrix(calib_sensor['rotation'])
        
        # Apply transformations
        point_ego = ego_rotation.T @ (point_global - ego_translation)
        point_camera = sensor_rotation.T @ (point_ego - sensor_translation)
        
        return point_camera
    
    def _load_image(self, filename: str) -> np.ndarray:
        """
        Load image from blob directories
        
        Args:
            filename: Relative path like "samples/CAM_FRONT/xxx.jpg"
        
        Returns:
            (H, W, 3) RGB image as uint8 numpy array
        """
        # Search across all blob directories
        blob_dirs = sorted([
            d for d in self.dataroot.iterdir()
            if d.is_dir() and 'trainval' in d.name and 'blobs' in d.name
        ])
        
        for blob_dir in blob_dirs:
            img_path = blob_dir / filename
            if img_path.exists():
                # Load as RGB
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
        
        raise FileNotFoundError(f"Image not found: {filename}")
    
    def _encode_depth(self, depth: float) -> Tuple[int, float]:
        """
        Encode depth into bin index + residual
        
        Args:
            depth: Distance in meters
        
        Returns:
            bin_idx: Index of depth bin (0 to num_bins-1)
            residual: Continuous offset within bin
        """
        # Find bin
        bin_idx = np.searchsorted(self.depth_bins[1:-1], depth, side='right')
        bin_idx = np.clip(bin_idx, 0, self.num_depth_bins - 1)
        
        # Compute residual (distance from bin center)
        bin_center = self.depth_bin_centers[bin_idx]
        residual = depth - bin_center
        
        return int(bin_idx), float(residual)
    
    def _encode_rotation(self, yaw: float) -> Tuple[int, float]:
        """
        Encode rotation angle into bin index + residual
        
        Args:
            yaw: Rotation angle in radians [-π, π]
        
        Returns:
            bin_idx: Index of rotation bin (0 to num_bins-1)
            residual: Angle residual within bin
        """
        # Normalize to [0, 2π]
        yaw = yaw % (2 * np.pi)
        
        # Find bin
        bin_idx = int(yaw / self.rotation_bin_size)
        bin_idx = np.clip(bin_idx, 0, self.num_rotation_bins - 1)
        
        # Compute residual (distance from bin center)
        bin_center = (bin_idx + 0.5) * self.rotation_bin_size
        residual = yaw - bin_center
        
        return int(bin_idx), float(residual)
    
    def _generate_gaussian_heatmap(
        self,
        heatmap: np.ndarray,
        center: Tuple[int, int],
        radius: int
    ):
        """
        Draw 2D Gaussian at center location (in-place modification)
        
        Args:
            heatmap: (H, W) array to modify
            center: (x, y) center in heatmap coordinates
            radius: Gaussian radius in pixels
        """
        diameter = 2 * radius + 1
        gaussian = np.zeros((diameter, diameter), dtype=np.float32)
        
        # Generate 2D Gaussian
        sigma = diameter / 6
        for i in range(diameter):
            for j in range(diameter):
                x = i - radius
                y = j - radius
                gaussian[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # Get bounding box in heatmap
        x, y = center
        height, width = heatmap.shape
        
        left = max(0, x - radius)
        right = min(width, x + radius + 1)
        top = max(0, y - radius)
        bottom = min(height, y + radius + 1)
        
        # Get corresponding region in Gaussian
        g_left = max(0, radius - x)
        g_right = g_left + (right - left)
        g_top = max(0, radius - y)
        g_bottom = g_top + (bottom - top)
        
        # Apply Gaussian (take max to handle overlapping objects)
        heatmap[top:bottom, left:right] = np.maximum(
            heatmap[top:bottom, left:right],
            gaussian[g_top:g_bottom, g_left:g_right]
        )
    
    def _compute_gaussian_radius(self, bbox_size: Tuple[float, float]) -> int:
        """
        Compute Gaussian radius based on object size
        Uses formula from CenterNet paper
        
        Args:
            bbox_size: (width, height) of 2D bounding box in pixels
        
        Returns:
            radius: Gaussian radius in pixels
        """
        width, height = bbox_size
        
        # Formula ensures at least one pixel overlap with IoU > 0.7
        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - 0.7) / (1 + 0.7)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        radius1 = (b1 + sq1) / 2
        
        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - 0.7) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        radius2 = (b2 + sq2) / 2
        
        a3 = 4 * 0.7
        b3 = -2 * 0.7 * (height + width)
        c3 = (0.7 - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        radius3 = (b3 + sq3) / 2
        
        return max(0, int(min(radius1, radius2, radius3)))
    
    def _augment_image(
        self,
        image: np.ndarray,
        intrinsics: np.ndarray,
        annotations: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Apply data augmentation
        
        Args:
            image: (H, W, 3) RGB image
            intrinsics: (3, 3) camera intrinsic matrix
            annotations: List of annotation dicts in camera frame
        
        Returns:
            image: Augmented image
            intrinsics: Adjusted intrinsics
            annotations: Adjusted annotations
        """
        K = intrinsics.copy()
        anns = [ann.copy() for ann in annotations]
        
        # Random horizontal flip
        if self.augment and random.random() < 0.5:
            image = np.fliplr(image).copy()
            
            # Flip intrinsics (cx adjustment)
            K[0, 2] = image.shape[1] - K[0, 2]
            
            # Flip annotations
            for ann in anns:
                # Flip X coordinate in camera frame
                ann['location'][0] = -ann['location'][0]
                # Flip rotation
                ann['rotation_y'] = -ann['rotation_y']
        
        # Color jittering
        if self.augment:
            # Convert to PIL for torchvision transforms
            image_pil = TF.to_pil_image(image)
            
            # Random brightness, contrast, saturation
            if random.random() < 0.5:
                image_pil = TF.adjust_brightness(image_pil, random.uniform(0.8, 1.2))
            if random.random() < 0.5:
                image_pil = TF.adjust_contrast(image_pil, random.uniform(0.8, 1.2))
            if random.random() < 0.5:
                image_pil = TF.adjust_saturation(image_pil, random.uniform(0.8, 1.2))
            
            image = np.array(image_pil)
        
        return image, K, anns
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get training sample
        
        Returns:
            Dictionary containing:
            - 'image': (3, 720, 1280) RGB tensor
            - 'heatmap': (1, 180, 320) target heatmap
            - 'depth_bin': (max_objects,) bin indices
            - 'depth_residual': (max_objects,) residuals
            - 'dimensions': (max_objects, 3) [h, w, l]
            - 'rotation_bin': (max_objects,) bin indices
            - 'rotation_residual': (max_objects,) residuals
            - 'offset_2d': (max_objects, 2) [Δu, Δv]
            - 'indices': (max_objects, 2) [y, x] heatmap locations
            - 'mask': (max_objects,) valid object mask
            - 'intrinsics': (3, 3) adjusted camera matrix
        """
        sample_idx = self.sample_indices[idx]
        sample_data = self.cam_front_samples[sample_idx]
        
        # Load image
        image = self._load_image(sample_data['filename'])  # (900, 1600, 3)
        
        # Get camera calibration
        calib_sensor = self.calib_lookup[sample_data['calibrated_sensor_token']]
        ego_pose = self.ego_lookup[sample_data['ego_pose_token']]
        
        # Camera intrinsics (native 1600×900)
        K = np.array(calib_sensor['camera_intrinsic'], dtype=np.float32)
        
        # Get annotations for this sample
        sample_token = sample_data['sample_token']
        annotations_global = self.sample_annotations.get(sample_token, [])
        
        # Transform annotations to camera frame
        annotations_camera = []
        for ann in annotations_global:
            # Filter vehicles only
            instance = self.instance_lookup.get(ann['instance_token'])
            if not instance or instance['category_token'] not in self.vehicle_categories:
                continue
            
            # Transform location to camera frame
            location_global = ann['translation']
            location_camera = self._transform_to_camera_frame(
                location_global, ego_pose, calib_sensor
            )
            
            # Check if object is in front of camera
            if location_camera[2] <= 0:
                continue
            
            # nuScenes size format: [width, length, height]
            # Our format: [height, width, length]
            size = ann['size']
            dimensions = np.array([size[2], size[0], size[1]], dtype=np.float32)
            
            rotation_quat = Quaternion(ann['rotation'])
            ego_quat = Quaternion(ego_pose['rotation'])
            rotation_ego = ego_quat.inverse * rotation_quat

            # Extract yaw in ego frame
            R = rotation_ego.rotation_matrix
            yaw_ego = np.arctan2(R[1, 0], R[0, 0])

            # Transform location to ego frame
            location_global = np.array(ann['translation'], dtype=np.float32)
            ego_translation = np.array(ego_pose['translation'], dtype=np.float32)
            ego_rotation = self._quaternion_to_rotation_matrix(ego_pose['rotation'])

            location_ego = ego_rotation.T @ (location_global - ego_translation)

            R = rotation_ego.rotation_matrix
            yaw_ego = np.arctan2(R[1, 0], R[0, 0])

            annotations_camera.append({
                'location': location_ego,
                'location_camera': location_camera,
                'dimensions': dimensions,
                'rotation_y': yaw_ego
            })
            
        
        # Apply augmentation
        image, K, annotations_camera = self._augment_image(
            image, K, annotations_camera
        )
        
        # Resize image: 1600×900 → 1280×720
        scale_x = self.input_w / image.shape[1]
        scale_y = self.input_h / image.shape[0]
        image = cv2.resize(image, (self.input_w, self.input_h))
        
        # Adjust intrinsics for resize
        K[0, :] *= scale_x  # fx, cx
        K[1, :] *= scale_y  # fy, cy
        
        # Initialize ground truth tensors
        heatmap = np.zeros((self.output_h, self.output_w), dtype=np.float32)
        
        depth_bins = np.zeros(self.max_objects, dtype=np.int64)
        depth_residuals = np.zeros(self.max_objects, dtype=np.float32)
        dimensions = np.zeros((self.max_objects, 3), dtype=np.float32)
        rotation_bins = np.zeros(self.max_objects, dtype=np.int64)
        rotation_residuals = np.zeros(self.max_objects, dtype=np.float32)
        offsets_2d = np.zeros((self.max_objects, 2), dtype=np.float32)
        indices = np.zeros((self.max_objects, 2), dtype=np.int64)
        mask = np.zeros(self.max_objects, dtype=np.float32)
        
        # Process annotations
        num_objects = min(len(annotations_camera), self.max_objects)
        
        for i, ann in enumerate(annotations_camera[:self.max_objects]):
            location = ann['location']
            dims = ann['dimensions']
            yaw = ann['rotation_y']
            
            # Project to image plane
            location_cam = ann['location_camera']
            x_cam, y_cam, z_cam = location_cam
            u = (K[0, 0] * x_cam / z_cam) + K[0, 2]
            v = (K[1, 1] * y_cam / z_cam) + K[1, 2]
            
            # Check if center is visible
            if u < 0 or u >= self.input_w or v < 0 or v >= self.input_h:
                continue
            
            # Map to heatmap coordinates
            u_hm = u / self.output_stride
            v_hm = v / self.output_stride
            
            # Integer location in heatmap
            u_hm_int = int(u_hm)
            v_hm_int = int(v_hm)
            
            if u_hm_int >= self.output_w or v_hm_int >= self.output_h:
                continue
            
            # Encode targets
            depth_bin, depth_res = self._encode_depth(z_cam)
            rot_bin, rot_res = self._encode_rotation(yaw)
            
            # Store targets
            depth_bins[i] = depth_bin
            depth_residuals[i] = depth_res
            dimensions[i] = dims
            rotation_bins[i] = rot_bin
            rotation_residuals[i] = rot_res
            offsets_2d[i] = [u_hm - u_hm_int, v_hm - v_hm_int]
            indices[i] = [v_hm_int, u_hm_int]
            mask[i] = 1.0
            
            # Generate Gaussian heatmap
            # Estimate 2D box size for radius computation
            # Approximate: project 3D box corners
            bbox_w = dims[1] * K[0, 0] / z_cam  # width in pixels
            bbox_h = dims[0] * K[1, 1] / z_cam  # height in pixels
            bbox_w_hm = bbox_w / self.output_stride
            bbox_h_hm = bbox_h / self.output_stride
            
            radius = self._compute_gaussian_radius((bbox_w_hm, bbox_h_hm))
            radius = max(0, int(radius))
            
            self._generate_gaussian_heatmap(heatmap, (u_hm_int, v_hm_int), radius)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0)
        
        return {
            'image': image_tensor,
            'heatmap': heatmap_tensor,
            'depth_bin': torch.from_numpy(depth_bins),
            'depth_residual': torch.from_numpy(depth_residuals),
            'dimensions': torch.from_numpy(dimensions),
            'rotation_bin': torch.from_numpy(rotation_bins),
            'rotation_residual': torch.from_numpy(rotation_residuals),
            'offset_2d': torch.from_numpy(offsets_2d),
            'indices': torch.from_numpy(indices),
            'mask': torch.from_numpy(mask),
            'intrinsics': torch.from_numpy(K)
        }


def test_dataset():
    """Test dataset loading and visualization"""
    import matplotlib.pyplot as plt
    
    root_dir = "/media/ashwin-benchdev/eb2a8889-3ec9-411a-826e-816cf9759b02/NuScenes-Full-Dataset"
    
    dataset = NuScenesMonocular3D(
        root_dir=root_dir,
        split='train',
        augment=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load sample
    sample = dataset[0]
    
    print("\nSample contents:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:20s}: {tuple(value.shape)} {value.dtype}")
    
    # Visualize
    image = sample['image'].permute(1, 2, 0).numpy()
    heatmap = sample['heatmap'][0].numpy()
    mask = sample['mask'].numpy()
    
    num_objects = int(mask.sum())
    print(f"\nNumber of objects: {num_objects}")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='hot')
    axes[1].set_title(f'Heatmap ({num_objects} objects)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/ashwin-benchdev/NetAsh3D/outputs/dataset_sample.png', dpi=150)
    print("\nVisualization saved to outputs/dataset_sample.png")


if __name__ == "__main__":
    test_dataset()
