"""Image Inference Script for Two-Stage 3D Detection - FOR RECTIFIED IMAGES

This script is designed to work with PRE-RECTIFIED fisheye images.
It loads rectified camera intrinsics from the calibration pipeline.

WORKFLOW:
  1. Rectify video: python scripts/04_rectify_for_detection.py
  2. Extract frames: ffmpeg -i rectified.mp4 -vf "select='not(mod(n\,10))'" -vsync 0 -qscale:v 2 frames/frame_%06d.png
  3. Run inference: python inference_rectified_images.py --images frames/ --intrinsics-json rectified_camera_params.json --checkpoint model.pth
  4. Review detections and correct labels
  5. Retrain model with corrected labels
"""

import os
import sys
import argparse
from pathlib import Path
import time
import json

import torch
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models.two_stage_detector import build_model


CLASS_COLORS = {
    'Car': (0, 255, 0),
    'Pedestrian': (255, 165, 0),
    'Cyclist': (0, 191, 255)
}


# Default KITTI P2 calibration matrix (fallback only)
KITTI_DEFAULT_INTRINSICS = np.array([
    [721.5377, 0.0, 609.5593],
    [0.0, 721.5377, 172.854],
    [0.0, 0.0, 1.0]
], dtype=np.float32)


def load_intrinsics_from_json(json_path):
    """
    Load rectified camera intrinsics from JSON file.
    
    Args:
        json_path: Path to rectified_camera_params.json
        
    Returns:
        K: 3x3 intrinsic matrix as numpy array
    """
    with open(json_path, 'r') as f:
        params = json.load(f)
    
    if 'rectified_intrinsics' in params:
        # New format from calibration pipeline
        intrinsics = params['rectified_intrinsics']
        K = np.array([
            [intrinsics['fx'], 0.0, intrinsics['cx']],
            [0.0, intrinsics['fy'], intrinsics['cy']],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        print(f"Loaded rectified intrinsics from {json_path}")
        print(f"  Camera model: {params.get('camera_model', 'unknown')}")
        if 'image_size' in params:
            print(f"  Image size: {params['image_size']['width']}x{params['image_size']['height']}")
    
    elif 'camera_matrix' in params:
        # Alternative format
        intrinsics = params['camera_matrix']
        K = np.array([
            [intrinsics['fx'], 0.0, intrinsics['cx']],
            [0.0, intrinsics['fy'], intrinsics['cy']],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        print(f"Loaded intrinsics from {json_path}")
    
    else:
        raise ValueError(f"Invalid JSON format in {json_path}")
    
    return K


def load_intrinsics_from_npy(npy_path):
    """
    Load intrinsics from .npy file (legacy support).
    
    Args:
        npy_path: Path to .npy file containing 3x3 matrix
        
    Returns:
        K: 3x3 intrinsic matrix
    """
    K = np.load(npy_path).astype(np.float32)
    print(f"Loaded intrinsics from {npy_path}")
    return K


class ImageInferenceEngine:
    def __init__(self, checkpoint_path, intrinsics=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'classes' in checkpoint:
            self.classes = checkpoint['classes']
            print(f"Loaded trained classes: {self.classes}")
        else:
            self.classes = ['Car']
            print(f"Warning: Using default classes: {self.classes}")
        
        # Build model
        self.model = build_model(active_classes=self.classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Set intrinsics
        if intrinsics is None:
            self.intrinsics = torch.from_numpy(KITTI_DEFAULT_INTRINSICS).float().to(self.device)
            print("WARNING: Using default KITTI intrinsics (not recommended for production)")
        else:
            self.intrinsics = torch.from_numpy(intrinsics).float().to(self.device)
            print("✓ Using custom intrinsics")
        
        print(f"\nCamera intrinsics:")
        print(f"  fx={self.intrinsics[0,0]:.2f}, fy={self.intrinsics[1,1]:.2f}")
        print(f"  cx={self.intrinsics[0,2]:.2f}, cy={self.intrinsics[1,2]:.2f}")
        
        print(f"\n✓ Model loaded from epoch {checkpoint['epoch']}")
        print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
        
        # Normalization params (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Resize to standard size (e.g., 1280x384 for KITTI)
        h, w = image.shape[:2]
        target_h, target_w = 384, 1280
        
        image_resized = cv2.resize(image, (target_w, target_h))
        
        # Normalize
        image_norm = image_resized.astype(np.float32) / 255.0
        image_norm = (image_norm - self.mean) / self.std
        
        # To tensor [1, 3, H, W]
        image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).unsqueeze(0).float()
        
        # Compute scale factors for box rescaling
        scale_x = w / target_w
        scale_y = h / target_h
        
        # CRITICAL FIX: Scale intrinsics to match resized image
        # Model expects K for the INPUT image size (target_w x target_h)
        K_scaled = self.intrinsics.clone()
        K_scaled[0, 0] /= scale_x  # fx for resized image
        K_scaled[1, 1] /= scale_y  # fy for resized image
        K_scaled[0, 2] /= scale_x  # cx for resized image
        K_scaled[1, 2] /= scale_y  # cy for resized image
        
        return image_tensor, K_scaled, (scale_x, scale_y)
    
    @torch.no_grad()
    def predict_image(self, image):
        """Run inference on single image"""
        # Preprocess - returns scaled intrinsics
        image_tensor, K_scaled, scales = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # Inference with SCALED intrinsics (matching resized image)
        start_time = time.time()
        predictions = self.model(image_tensor, K_scaled.unsqueeze(0), gt_boxes_2d=None)
        inference_time = time.time() - start_time
        
        # Decode predictions
        boxes_3d, scores, class_names = self.decode_predictions(predictions, scales)
        
        return boxes_3d, scores, class_names, inference_time
    
    def decode_predictions(self, predictions, scales):
        """Decode predictions with scale adjustment"""
        boxes_2d = predictions['boxes_2d'][0]
        depth_pred = predictions['depth'][0]
        dims_pred = predictions['dimensions'][0]
        rot_bins, rot_res = predictions['rotation'][0]
        scores = predictions['scores_2d'][0]
        class_names = predictions['classes'][0]
        
        if len(boxes_2d) == 0:
            return np.zeros((0, 7)), np.zeros(0), []
        
        # Decode depth with offset
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
        
        dims = dims_pred.cpu().numpy()
        
        # CRITICAL FIX: Use full-resolution intrinsics directly
        # self.intrinsics is already at full resolution (3848×2168)
        scale_x, scale_y = scales
        K = self.intrinsics.cpu().numpy().copy()
        # K is already at full resolution - DO NOT scale it again!
        
        # Scale boxes back to original image coordinates
        boxes_2d_np = boxes_2d.cpu().numpy()
        boxes_2d_np[:, [0, 2]] *= scale_x
        boxes_2d_np[:, [1, 3]] *= scale_y
        
        # Unproject to 3D (using bottom-center) with FULL RESOLUTION K
        x_2d = (boxes_2d_np[:, 0] + boxes_2d_np[:, 2]) / 2
        y_2d = boxes_2d_np[:, 3]  # Bottom edge
        
        x_cam = (x_2d - K[0, 2]) * depth / K[0, 0]
        y_cam_bottom = (y_2d - K[1, 2]) * depth / K[1, 1]
        z_cam = depth
        
        # Shift to geometric center
        y_cam = y_cam_bottom - dims[:, 0] / 2.0
        
        boxes_3d = np.stack([x_cam, y_cam, z_cam, dims[:, 0], dims[:, 1], dims[:, 2], rotation], axis=1)
        scores_np = scores.cpu().numpy()
        
        return boxes_3d, scores_np, class_names


def project_3d_box(box_3d, K):
    """Project 3D box to 2D with CORRECTED rotation matrix"""
    x, y, z, h, w, l, ry = box_3d
    
    # Define 8 corners in object coordinate system
    x_corners = [l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2]
    y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    z_corners = [w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2]
    
    corners_3d = np.array([x_corners, y_corners, z_corners])
    
    # CORRECTED rotation matrix for KITTI
    R = np.array([
        [np.cos(ry), 0, -np.sin(ry)],
        [0, 1, 0],
        [np.sin(ry), 0, np.cos(ry)]
    ])
    corners_3d = R @ corners_3d
    
    # Translate to camera coordinates
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


def draw_3d_box(image, box_3d, K, color=(0, 255, 0), thickness=2, score=None, class_name=None):
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
    
    # Draw front indicator (blue dot)
    front_center = ((corners_2d[0] + corners_2d[3] + corners_2d[4] + corners_2d[7]) / 4).astype(np.int32)
    cv2.circle(image, tuple(front_center), 5, (255, 0, 0), -1)
    
    # Draw label
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
    
    # Draw depth
    depth_text = f'{box_3d[2]:.1f}m'
    depth_pos = (int(corners_2d[:, 0].min()), int(corners_2d[:, 1].max()) + 20)
    cv2.putText(image, depth_text, depth_pos, cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, color, 2, cv2.LINE_AA)
    
    return image


def save_kitti_label(output_path, boxes_3d, scores, class_names, image_shape):
    """
    Save detections in KITTI label format for correction/retraining
    
    Format: type truncated occluded alpha bbox_2d dimensions location rotation_y score
    """
    h, w = image_shape[:2]
    
    with open(output_path, 'w') as f:
        for box_3d, score, cls in zip(boxes_3d, scores, class_names):
            x, y, z, box_h, box_w, box_l, ry = box_3d
            
            # Project to get 2D bbox (approximate)
            # For now, use placeholder values - you can refine this
            bbox_2d = [0, 0, w-1, h-1]  # Full image bbox as placeholder
            
            # KITTI format
            line = f"{cls} "
            line += f"{0.0:.2f} "  # truncated
            line += f"{0} "  # occluded
            line += f"{0.0:.2f} "  # alpha (approximate from ry)
            line += f"{bbox_2d[0]:.2f} {bbox_2d[1]:.2f} {bbox_2d[2]:.2f} {bbox_2d[3]:.2f} "
            line += f"{box_h:.2f} {box_w:.2f} {box_l:.2f} "
            line += f"{x:.2f} {y:.2f} {z:.2f} "
            line += f"{ry:.2f} "
            line += f"{score:.2f}\n"
            
            f.write(line)


def process_images(engine, image_paths, output_dir, conf_threshold=0.5, 
                  save_labels=False, save_visualizations=True):
    """Process multiple images and save outputs"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if save_visualizations:
        viz_dir = output_path / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
    
    if save_labels:
        labels_dir = output_path / 'labels'
        labels_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Processing {len(image_paths)} images")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Save labels: {save_labels}")
    print(f"Save visualizations: {save_visualizations}")
    print(f"{'='*70}\n")
    
    K = engine.intrinsics.cpu().numpy()
    
    inference_times = []
    detection_counts = []
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        # Run inference
        boxes_3d, scores, class_names, inf_time = engine.predict_image(image)
        inference_times.append(inf_time)
        
        # Filter by confidence
        keep_mask = scores >= conf_threshold
        boxes_3d_filtered = boxes_3d[keep_mask]
        scores_filtered = scores[keep_mask]
        class_names_filtered = [class_names[i] for i in range(len(class_names)) if keep_mask[i]]
        
        detection_counts.append(len(boxes_3d_filtered))
        
        # Save KITTI labels if requested
        if save_labels:
            label_file = labels_dir / f"{img_path.stem}.txt"
            save_kitti_label(label_file, boxes_3d_filtered, scores_filtered, 
                           class_names_filtered, image.shape)
        
        # Save visualization if requested
        if save_visualizations:
            image_vis = image.copy()
            for box, score, cls in zip(boxes_3d_filtered, scores_filtered, class_names_filtered):
                color = CLASS_COLORS.get(cls, (0, 255, 0))
                image_vis = draw_3d_box(image_vis, box, K, color=color, thickness=2, 
                                       score=score, class_name=cls)
            
            # Add detection count
            info_text = f"Detections: {len(boxes_3d_filtered)}"
            cv2.putText(image_vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (0, 255, 0), 2, cv2.LINE_AA)
            
            viz_file = viz_dir / f"{img_path.stem}_detection.jpg"
            cv2.imwrite(str(viz_file), image_vis)
    
    # Print statistics
    avg_inf_time = np.mean(inference_times)
    avg_fps = 1.0 / avg_inf_time
    avg_detections = np.mean(detection_counts)
    total_detections = sum(detection_counts)
    
    print(f"\n{'='*70}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Images processed: {len(inference_times)}")
    print(f"Average inference time: {avg_inf_time*1000:.2f} ms")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average detections per image: {avg_detections:.1f}")
    print(f"Total detections: {total_detections}")
    
    if save_labels:
        print(f"\n✓ KITTI labels saved to: {labels_dir}")
        print(f"  You can now correct these labels and retrain")
    
    if save_visualizations:
        print(f"\n✓ Visualizations saved to: {viz_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Image inference for two-stage 3D detection (RECTIFIED IMAGES)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT: This script expects PRE-RECTIFIED images!

Workflow:
  1. Rectify your fisheye video:
     python scripts/04_rectify_for_detection.py --input raw.mp4 --output rectified.mp4
  
  2. Extract frames:
     ffmpeg -i rectified.mp4 -vf "select='not(mod(n\,10))'" -vsync 0 -qscale:v 2 frames/frame_%%06d.png
  
  3. Run inference on rectified frames:
     python inference_rectified_images.py \\
       --checkpoint model.pth \\
       --images frames/ \\
       --output-dir detections/ \\
       --intrinsics-json data/caliberation_results/rectified_camera_params.json \\
       --save-labels

  4. Correct labels in detections/labels/

  5. Retrain model with corrected labels

Examples:
  # Basic inference with visualizations
  python inference_rectified_images.py \\
    --checkpoint best_model.pth \\
    --images domain_images/ \\
    --output-dir outputs/inference \\
    --intrinsics-json rectified_camera_params.json
  
  # Save KITTI labels for correction and retraining
  python inference_rectified_images.py \\
    --checkpoint best_model.pth \\
    --images domain_images/ \\
    --output-dir outputs/inference \\
    --intrinsics-json rectified_camera_params.json \\
    --save-labels \\
    --conf-threshold 0.3
        """
    )
    
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--images', type=str, required=True, 
                       help='Path to directory containing rectified images')
    parser.add_argument('--output-dir', type=str, required=True, 
                       help='Output directory for results')
    parser.add_argument('--intrinsics-json', type=str, default=None,
                       help='Path to rectified_camera_params.json (RECOMMENDED)')
    parser.add_argument('--intrinsics', type=str, default=None,
                       help='Path to intrinsics .npy file (legacy support)')
    parser.add_argument('--conf-threshold', type=float, default=0.5, 
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device (cuda/cpu)')
    parser.add_argument('--save-labels', action='store_true',
                       help='Save KITTI format labels for correction/retraining')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Do not save visualization images')
    
    args = parser.parse_args()
    
    # Validate images directory exists
    images_dir = Path(args.images)
    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {args.images}")
        sys.exit(1)
    
    # Get image paths
    image_paths = sorted(list(images_dir.glob('*.png')) + 
                        list(images_dir.glob('*.jpg')) + 
                        list(images_dir.glob('*.jpeg')))
    
    if len(image_paths) == 0:
        print(f"ERROR: No images found in {args.images}")
        sys.exit(1)
    
    print(f"Found {len(image_paths)} images")
    
    # Load intrinsics
    intrinsics = None
    if args.intrinsics_json:
        # Preferred: Load from JSON
        try:
            intrinsics = load_intrinsics_from_json(args.intrinsics_json)
        except Exception as e:
            print(f"ERROR loading intrinsics from JSON: {e}")
            sys.exit(1)
    
    elif args.intrinsics:
        # Legacy: Load from .npy
        try:
            intrinsics = load_intrinsics_from_npy(args.intrinsics)
        except Exception as e:
            print(f"ERROR loading intrinsics from .npy: {e}")
            sys.exit(1)
    
    else:
        print("WARNING: No intrinsics provided. Using KITTI defaults.")
        print("         This is NOT recommended for fisheye cameras!")
        print("         Use --intrinsics-json to provide rectified camera parameters.")
    
    # Create engine
    engine = ImageInferenceEngine(
        checkpoint_path=args.checkpoint,
        intrinsics=intrinsics,
        device=args.device
    )
    
    # Process images
    process_images(
        engine=engine,
        image_paths=image_paths,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        save_labels=args.save_labels,
        save_visualizations=not args.no_visualizations
    )


if __name__ == "__main__":
    main()
