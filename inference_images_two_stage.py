"""
Image Inference for Two-Stage 3D Detector
Outputs side-by-side comparison: Original | With 3D Boxes
"""

import os
import sys
import argparse
from pathlib import Path
import json

import torch
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models.two_stage_detector import build_model
from fisheye_utils import FisheyeCamera


CLASS_COLORS = {
    'Car': (0, 255, 0),
    'Pedestrian': (255, 165, 0),
    'Cyclist': (0, 191, 255)
}


class ImageInferenceEngine:
    def __init__(self, checkpoint_path, fisheye_config=None, device='cuda', 
                 use_intrinsics_normalization=True, dim_scale=None):
        """
        Initialize inference engine
        
        Args:
            checkpoint_path: Path to model checkpoint
            fisheye_config: Dict or path to fisheye camera config (None for regular camera)
            device: 'cuda' or 'cpu'
            use_intrinsics_normalization: Normalize intrinsics to KITTI scale
            dim_scale: Optional dimension scaling factor (e.g., 0.3)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.use_fisheye = fisheye_config is not None
        self.use_intrinsics_normalization = use_intrinsics_normalization
        self.dim_scale = dim_scale
        
        # Setup camera
        if self.use_fisheye:
            print("\nInitializing fisheye camera...")
            if isinstance(fisheye_config, str):
                with open(fisheye_config, 'r') as f:
                    fisheye_config = json.load(f)
            
            self.fisheye_camera = FisheyeCamera(fisheye_config)
            self.intrinsics_original = self.fisheye_camera.get_undistorted_intrinsics()
            self.image_shape = (fisheye_config['height'], fisheye_config['width'])
        else:
            print("\nUsing regular camera (no fisheye correction)")
            # Default KITTI intrinsics if not using fisheye
            self.intrinsics_original = np.array([
                [721.5377, 0, 609.5593],
                [0, 721.5377, 172.854],
                [0, 0, 1]
            ], dtype=np.float32)
            self.image_shape = (375, 1242)
        
        # Normalize intrinsics if requested
        if use_intrinsics_normalization:
            print("\nNormalizing intrinsics to KITTI scale...")
            self.intrinsics_normalized = self.normalize_intrinsics_to_kitti(
                self.intrinsics_original,
                self.image_shape
            )
            print(f"  Original fx: {self.intrinsics_original[0,0]:.1f}")
            print(f"  Normalized fx: {self.intrinsics_normalized[0,0]:.1f}")
            print(f"  Scale factor: {self.intrinsics_normalized[0,0]/self.intrinsics_original[0,0]:.3f}")
            
            self.intrinsics = self.intrinsics_normalized
        else:
            self.intrinsics = self.intrinsics_original
        
        # Load model
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'classes' in checkpoint:
            self.classes = checkpoint['classes']
            print(f"Loaded trained classes: {self.classes}")
        else:
            self.classes = ['Car']
            print(f"Warning: Using default classes: {self.classes}")
        
        self.model = build_model(active_classes=self.classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
        
        # Normalization params
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        if dim_scale is not None:
            print(f"\nUsing dimension scaling: {dim_scale}")
    
    def normalize_intrinsics_to_kitti(self, K_yours, image_shape):
        """Normalize intrinsics to KITTI scale"""
        fx_kitti = 721.5377
        width_kitti = 1242
        
        fx_yours = K_yours[0, 0]
        height_yours, width_yours = image_shape
        
        ratio_kitti = fx_kitti / width_kitti
        ratio_yours = fx_yours / width_yours
        scale_factor = ratio_kitti / ratio_yours
        
        K_normalized = K_yours.copy()
        K_normalized[0, 0] *= scale_factor
        K_normalized[1, 1] *= scale_factor
        K_normalized[0, 2] *= scale_factor
        K_normalized[1, 2] *= scale_factor
        
        return K_normalized
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Undistort if fisheye
        if self.use_fisheye:
            image = self.fisheye_camera.undistort(image)
        
        # Store original for visualization
        image_original = image.copy()
        h_orig, w_orig = image.shape[:2]
        
        # Resize to model input size
        target_h, target_w = 384, 1280
        image_resized = cv2.resize(image, (target_w, target_h))
        
        # Normalize
        image_norm = image_resized.astype(np.float32) / 255.0
        image_norm = (image_norm - self.mean) / self.std
        
        # To tensor
        image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).unsqueeze(0).float()
        
        # Scale factors
        scale_x = w_orig / target_w
        scale_y = h_orig / target_h
        
        return image_tensor, image_original, (scale_x, scale_y)
    
    @torch.no_grad()
    def predict_image(self, image):
        """Run inference on single image"""
        # Preprocess
        image_tensor, image_original, scales = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # Create scaled intrinsics for model input
        K_scaled = self.intrinsics.copy()
        scale_x, scale_y = scales
        K_scaled[0, 0] /= scale_x
        K_scaled[1, 1] /= scale_y
        K_scaled[0, 2] /= scale_x
        K_scaled[1, 2] /= scale_y
        
        intrinsics_tensor = torch.from_numpy(K_scaled).float().to(self.device)
        
        # Inference
        predictions = self.model(image_tensor, intrinsics_tensor.unsqueeze(0), gt_boxes_2d=None)
        
        # Decode predictions
        boxes_3d, scores, class_names = self.decode_predictions(predictions, scales)
        
        return boxes_3d, scores, class_names, image_original
    
    def decode_predictions(self, predictions, scales):
        """Decode predictions to 3D boxes"""
        boxes_2d = predictions['boxes_2d'][0]
        depth_pred = predictions['depth'][0]
        dims_pred = predictions['dimensions'][0]
        rot_bins, rot_res = predictions['rotation'][0]
        scores = predictions['scores_2d'][0]
        class_names = predictions['classes'][0]
        
        if len(boxes_2d) == 0:
            return np.zeros((0, 7)), np.zeros(0), []
        
        # Decode depth
        depth = depth_pred[:, 0].cpu().numpy()
        depth_offset = depth_pred[:, 2].cpu().numpy()
        depth = depth + depth_offset
        
        # Decode rotation
        rot_bin_idx = torch.argmax(rot_bins, dim=1).cpu().numpy()
        bin_size = 2 * np.pi / 12
        rotation = (rot_bin_idx + 0.5) * bin_size
        rot_res_selected = rot_res[torch.arange(len(rot_bin_idx)), torch.from_numpy(rot_bin_idx)]
        rotation = rotation + rot_res_selected.cpu().numpy()
        rotation = (rotation + np.pi) % (2 * np.pi) - np.pi
        
        # Decode dimensions
        dims = dims_pred.cpu().numpy()
        
        # Apply dimension scaling if provided
        if self.dim_scale is not None:
            dims = dims * self.dim_scale
        
        # Use intrinsics for unprojection
        scale_x, scale_y = scales
        K = self.intrinsics.copy()
        
        # Scale boxes back to original image coordinates
        boxes_2d_np = boxes_2d.cpu().numpy()
        boxes_2d_np[:, [0, 2]] *= scale_x
        boxes_2d_np[:, [1, 3]] *= scale_y
        
        # Unproject to 3D
        x_2d = (boxes_2d_np[:, 0] + boxes_2d_np[:, 2]) / 2
        y_2d = boxes_2d_np[:, 3]
        
        x_cam = (x_2d - K[0, 2]) * depth / K[0, 0]
        y_cam_bottom = (y_2d - K[1, 2]) * depth / K[1, 1]
        z_cam = depth
        
        y_cam = y_cam_bottom - dims[:, 0] / 2.0
        
        boxes_3d = np.stack([x_cam, y_cam, z_cam, dims[:, 0], dims[:, 1], dims[:, 2], rotation], axis=1)
        scores_np = scores.cpu().numpy()
        
        return boxes_3d, scores_np, class_names


def project_3d_box(box_3d, K):
    """Project 3D box to 2D image coordinates"""
    x, y, z, h, w, l, ry = box_3d
    
    # Define 8 corners
    x_corners = [l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2]
    y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    z_corners = [w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2]
    
    corners_3d = np.array([x_corners, y_corners, z_corners])
    
    # Rotation matrix
    R = np.array([
        [np.cos(ry), 0, -np.sin(ry)],
        [0, 1, 0],
        [np.sin(ry), 0, np.cos(ry)]
    ])
    corners_3d = R @ corners_3d
    
    # Translate
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z
    
    # Check if behind camera
    if np.any(corners_3d[2, :] <= 0.1):
        return None
    
    # Project to 2D
    corners_2d = K @ corners_3d
    corners_2d = corners_2d[:2, :] / corners_3d[2, :]
    
    return corners_2d.T


def draw_3d_box(image, box_3d, K, color=(0, 255, 0), thickness=2, 
                score=None, class_name=None, show_info=True):
    """Draw 3D bounding box on image"""
    corners_2d = project_3d_box(box_3d, K)
    
    if corners_2d is None:
        return image
    
    image = image.copy()
    corners_2d = corners_2d.astype(np.int32)
    
    # Draw bottom face
    for i in range(4):
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[(i+1)%4]), color, thickness)
    
    # Draw top face
    for i in range(4, 8):
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[4+(i+1)%4]), color, thickness)
    
    # Draw vertical edges
    for i in range(4):
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[i+4]), color, thickness)
    
    # Draw front indicator
    front_center = ((corners_2d[0] + corners_2d[3] + corners_2d[4] + corners_2d[7]) / 4).astype(np.int32)
    cv2.circle(image, tuple(front_center), 5, (255, 0, 0), -1)
    
    if show_info:
        # Label
        if class_name and score is not None:
            label = f'{class_name} {score:.2f}'
        elif class_name:
            label = class_name
        else:
            label = None
        
        if label:
            label_y = int(corners_2d[:, 1].min()) - 5
            label_pos = (int(corners_2d[:, 0].min()), max(20, label_y))
            cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, color, 2, cv2.LINE_AA)
        
        # Depth
        depth_text = f'{box_3d[2]:.1f}m'
        depth_pos = (int(corners_2d[:, 0].min()), int(corners_2d[:, 1].max()) + 20)
        cv2.putText(image, depth_text, depth_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 2, cv2.LINE_AA)
        
        # Dimensions (optional)
        dim_text = f'H:{box_3d[3]:.1f} W:{box_3d[4]:.1f} L:{box_3d[5]:.1f}'
        dim_pos = (int(corners_2d[:, 0].min()), int(corners_2d[:, 1].max()) + 40)
        cv2.putText(image, dim_text, dim_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, color, 1, cv2.LINE_AA)
    
    return image


def create_side_by_side(image_original, image_with_boxes, detection_count):
    """Create side-by-side comparison"""
    h, w = image_original.shape[:2]
    
    # Create canvas
    canvas = np.zeros((h, w*2, 3), dtype=np.uint8)
    
    # Place images
    canvas[:, :w] = image_original
    canvas[:, w:] = image_with_boxes
    
    # Add labels
    cv2.putText(canvas, 'Original', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
               1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f'3D Detection ({detection_count} objects)', (w + 20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Add divider line
    cv2.line(canvas, (w, 0), (w, h), (255, 255, 255), 2)
    
    return canvas


def process_images(engine, input_paths, output_dir, conf_threshold=0.5, show_info=True):
    """Process multiple images"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing {len(input_paths)} images")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"{'='*60}\n")
    
    K = engine.intrinsics
    
    for img_path in tqdm(input_paths, desc="Processing images"):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        # Run inference
        boxes_3d, scores, class_names, image_processed = engine.predict_image(image)
        
        # Filter by confidence
        keep_mask = scores >= conf_threshold
        boxes_3d_filtered = boxes_3d[keep_mask]
        scores_filtered = scores[keep_mask]
        class_names_filtered = [class_names[i] for i in range(len(class_names)) if keep_mask[i]]
        
        # Draw 3D boxes
        image_with_boxes = image_processed.copy()
        for box, score, cls in zip(boxes_3d_filtered, scores_filtered, class_names_filtered):
            color = CLASS_COLORS.get(cls, (0, 255, 0))
            image_with_boxes = draw_3d_box(image_with_boxes, box, K, color=color, 
                                          thickness=2, score=score, class_name=cls,
                                          show_info=show_info)
        
        # Create side-by-side
        result = create_side_by_side(image_processed, image_with_boxes, len(boxes_3d_filtered))
        
        # Save
        output_file = output_path / f"{img_path.stem}_detection.jpg"
        cv2.imwrite(str(output_file), result)
    
    print(f"\n✓ Processed {len(input_paths)} images")
    print(f"  Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Image inference with side-by-side visualization'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory of images')
    parser.add_argument('--output-dir', type=str, default='outputs/image_inference',
                       help='Output directory for results')
    parser.add_argument('--fisheye-config', type=str, default=None,
                       help='Path to fisheye config JSON (optional)')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--no-intrinsics-norm', action='store_true',
                       help='Disable intrinsics normalization')
    parser.add_argument('--dim-scale', type=float, default=None,
                       help='Dimension scaling factor (e.g., 0.3)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')
    parser.add_argument('--no-info', action='store_true',
                       help='Hide labels, depth, dimensions on boxes')
    
    args = parser.parse_args()
    
    # Load fisheye config if provided
    fisheye_config = None
    if args.fisheye_config:
        with open(args.fisheye_config, 'r') as f:
            fisheye_config = json.load(f)
        print(f"Loaded fisheye config from {args.fisheye_config}")
    
    # Create engine
    engine = ImageInferenceEngine(
        checkpoint_path=args.checkpoint,
        fisheye_config=fisheye_config,
        device=args.device,
        use_intrinsics_normalization=not args.no_intrinsics_norm,
        dim_scale=args.dim_scale
    )
    
    # Get input paths
    input_path = Path(args.input)
    if input_path.is_file():
        input_paths = [input_path]
    elif input_path.is_dir():
        input_paths = list(input_path.glob('*.png')) + \
                     list(input_path.glob('*.jpg')) + \
                     list(input_path.glob('*.jpeg'))
        input_paths.sort()
    else:
        raise ValueError(f"Input not found: {args.input}")
    
    # Process images
    process_images(
        engine=engine,
        input_paths=input_paths,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        show_info=not args.no_info
    )


if __name__ == "__main__":
    main()