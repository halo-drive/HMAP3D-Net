"""Video Inference Script V4 - Aligned with V4 Training Architecture

This script strictly follows V4 training methodology:
1. Normalized intrinsics (resolution-invariant)
2. 24-bin rotation head (full 360°)
3. Stable depth prediction (no log_variance)
4. YOLOv8 as 2D detector (model.detector_2d)
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

from models.two_stage_detector_v4 import build_model_v4


CLASS_COLORS = {
    'Car': (0, 255, 0),
    'Pedestrian': (255, 165, 0),
    'Cyclist': (0, 191, 255)
}


# COCO class IDs to our classes mapping
COCO_TO_OUR_CLASSES = {
    2: 'Car',           # COCO car
    0: 'Pedestrian',    # COCO person
    1: 'Cyclist',       # COCO bicycle (rough mapping)
    3: 'Car',           # COCO motorcycle → Car
    5: 'Car',           # COCO bus → Car
    7: 'Car',           # COCO truck → Car
}


def load_intrinsics_from_json(json_path):
    """Load rectified camera intrinsics from JSON"""
    with open(json_path, 'r') as f:
        params = json.load(f)
    
    if 'rectified_intrinsics' in params:
        intrinsics = params['rectified_intrinsics']
        K = np.array([
            [intrinsics['fx'], 0.0, intrinsics['cx']],
            [0.0, intrinsics['fy'], intrinsics['cy']],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        print(f"Loaded rectified intrinsics:")
        print(f"  fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
        print(f"  cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
        if 'image_size' in params:
            w, h = params['image_size']['width'], params['image_size']['height']
            print(f"  Image size: {w}x{h}")
            print(f"  Normalized fx={K[0,0]/w:.3f}, fy={K[1,1]/h:.3f}")
    elif 'camera_matrix' in params:
        intrinsics = params['camera_matrix']
        K = np.array([
            [intrinsics['fx'], 0.0, intrinsics['cx']],
            [0.0, intrinsics['fy'], intrinsics['cy']],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        print(f"Loaded intrinsics: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    else:
        raise ValueError(f"Invalid JSON format in {json_path}")
    
    return K


class VideoInferenceEngineV4:
    def __init__(self, checkpoint_path, intrinsics, device='cuda', target_size=None):
        """
        V4 Inference Engine - Strict adherence to training architecture
        
        Args:
            checkpoint_path: Path to V4 checkpoint (.pth)
            intrinsics: Camera intrinsics matrix [3, 3]
            device: 'cuda' or 'cpu'
            target_size: Optional (width, height) for resizing input frames
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"\n{'='*70}")
        print(f"V4 INFERENCE ENGINE INITIALIZATION")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        
        # Load V4 checkpoint
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.classes = checkpoint.get('classes', ['Car'])
        print(f"Classes: {self.classes}")
        
        # Build V4 model (same as training)
        self.model = build_model_v4(active_classes=self.classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        
        # Load YOLOv8 as 2D detector (model.detector_2d)
        print(f"\nLoading YOLOv8 2D detector...")
        try:
            from ultralytics import YOLO
            self.model.detector_2d = YOLO('yolov8m.pt')  # Medium model
            print(f"✓ YOLOv8m loaded (COCO-pretrained)")
        except ImportError:
            print(f"ERROR: ultralytics not installed")
            print(f"Install with: pip install ultralytics")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR loading YOLOv8: {e}")
            sys.exit(1)
        
        self.model.eval()
        
        # Store intrinsics
        self.intrinsics_original = torch.from_numpy(intrinsics).float()
        self.target_size = target_size
        
        print(f"\n✓ V4 Model loaded successfully")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Val loss: {checkpoint.get('best_val_loss', 'N/A')}")
        
        if target_size:
            print(f"\nPreprocessing: Resize to {target_size[0]}x{target_size[1]}")
        else:
            print(f"\nPreprocessing: Original resolution (no resize)")
        
        # ImageNet normalization (same as training)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        print(f"{'='*70}\n")
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame - SAME as V4 training data pipeline
        
        Returns:
            frame_tensor: [1, 3, H, W] normalized tensor
            K_scaled: [3, 3] scaled intrinsics
            scale_factors: (scale_x, scale_y) for reverse mapping
        """
        h_orig, w_orig = frame.shape[:2]
        
        # Resize if target_size specified
        if self.target_size:
            target_w, target_h = self.target_size
            frame_resized = cv2.resize(frame, (target_w, target_h))
            
            # Scale intrinsics proportionally
            scale_x = target_w / w_orig
            scale_y = target_h / h_orig
            
            K_scaled = self.intrinsics_original.clone()
            K_scaled[0, 0] *= scale_x  # fx
            K_scaled[1, 1] *= scale_y  # fy
            K_scaled[0, 2] *= scale_x  # cx
            K_scaled[1, 2] *= scale_y  # cy
            
            scale_factors = (w_orig / target_w, h_orig / target_h)
        else:
            frame_resized = frame
            K_scaled = self.intrinsics_original.clone()
            scale_factors = (1.0, 1.0)
        
        # Normalize (ImageNet stats)
        frame_norm = frame_resized.astype(np.float32) / 255.0
        frame_norm = (frame_norm - self.mean) / self.std
        
        # To tensor [1, 3, H, W]
        frame_tensor = torch.from_numpy(frame_norm.transpose(2, 0, 1)).unsqueeze(0).float()
        
        return frame_tensor, K_scaled, scale_factors
    
    @torch.no_grad()
    def predict_frame(self, frame):
        """
        Run inference on single frame
        
        Pipeline (same as V4 training forward pass):
        1. Preprocess frame
        2. Run model(images, intrinsics, gt_boxes_2d=None)
           - model uses detector_2d to get 2D boxes
           - Encodes normalized intrinsics
           - Predicts depth, dimensions, rotation, foreground
        3. Decode predictions to 3D boxes
        
        Returns:
            boxes_3d: [N, 7] array (x, y, z, h, w, l, ry)
            scores: [N] confidence scores
            class_names: [N] list of class names
            inference_time: float (seconds)
        """
        frame_tensor, K_scaled, scale_factors = self.preprocess_frame(frame)
        frame_tensor = frame_tensor.to(self.device)
        K_scaled = K_scaled.to(self.device)
        
        # Inference (V4 model forward pass)
        start_time = time.time()
        predictions = self.model(
            frame_tensor, 
            K_scaled.unsqueeze(0),
            gt_boxes_2d=None  # Use model.detector_2d for 2D detection
        )
        inference_time = time.time() - start_time
        
        # Decode predictions
        boxes_3d, scores, class_names = self.decode_predictions_v4(
            predictions, scale_factors, K_scaled
        )
        
        return boxes_3d, scores, class_names, inference_time
    
    def decode_predictions_v4(self, predictions, scale_factors, K_scaled):
        """
        Decode V4 predictions to 3D boxes
        
        V4 prediction structure (from training):
        - boxes_2d: list of [N, 4] 2D boxes from detector_2d
        - depth: list of [N, 2] (depth, offset)
        - dimensions: list of [N, 3] (h, w, l)
        - rotation: list of (bins [N, 24], residuals [N, 24])
        - foreground: list of [N, 2] (foreground logits)
        """
        # Extract predictions for batch index 0
        boxes_2d_batch = predictions['boxes_2d']
        
        if len(boxes_2d_batch) == 0:
            return np.zeros((0, 7)), np.zeros(0), []
        
        boxes_2d_result = boxes_2d_batch[0]
        
        # Handle YOLOv8 Results object
        if hasattr(boxes_2d_result, 'boxes'):
            boxes_obj = boxes_2d_result.boxes
            boxes_2d = boxes_obj.xyxy.cpu()      # [N, 4]
            scores_2d = boxes_obj.conf.cpu()     # [N]
            labels_coco = boxes_obj.cls.long().cpu()  # [N] COCO class IDs
            
            # Filter for relevant classes
            valid_mask = torch.tensor([
                label.item() in COCO_TO_OUR_CLASSES 
                for label in labels_coco
            ])
            
            if not valid_mask.any():
                return np.zeros((0, 7)), np.zeros(0), []
            
            boxes_2d = boxes_2d[valid_mask]
            scores_2d = scores_2d[valid_mask]
            labels_coco = labels_coco[valid_mask]
            
            # Map COCO classes to our classes
            class_names = [
                COCO_TO_OUR_CLASSES.get(label.item(), 'Car') 
                for label in labels_coco
            ]
        else:
            # Shouldn't reach here with proper V4 setup
            return np.zeros((0, 7)), np.zeros(0), []
        
        if len(boxes_2d) == 0:
            return np.zeros((0, 7)), np.zeros(0), []
        
        # Get 3D predictions (aligned with filtered 2D boxes)
        depth_pred = predictions['depth'][0]      # [N, 2]
        dims_pred = predictions['dimensions'][0]  # [N, 3]
        rot_bins, rot_res = predictions['rotation'][0]  # [N, 24], [N, 24]
        
        # Decode depth (V4: depth + offset)
        depth = depth_pred[:, 0].cpu().numpy()
        depth_offset = depth_pred[:, 1].cpu().numpy()
        depth_final = depth + depth_offset
        
        # Decode rotation (V4: 24 bins, no direction head)
        rot_bin_idx = torch.argmax(rot_bins, dim=1).cpu().numpy()
        bin_size = 2 * np.pi / 24  # 24 bins for full 360°
        
        rotation = (rot_bin_idx + 0.5) * bin_size
        rot_res_selected = rot_res[torch.arange(len(rot_bin_idx)), torch.from_numpy(rot_bin_idx)]
        rotation = rotation + rot_res_selected.cpu().numpy()
        
        # Map to [-π, π]
        rotation = (rotation + np.pi) % (2 * np.pi) - np.pi
        
        # Decode dimensions (V4: h, w, l)
        dims = dims_pred.cpu().numpy()  # [N, 3]
        
        # Get intrinsics
        K = K_scaled.cpu().numpy()
        scale_x, scale_y = scale_factors
        
        # Scale 2D boxes back to original resolution
        boxes_2d_np = boxes_2d.numpy()
        boxes_2d_np[:, [0, 2]] *= scale_x  # x1, x2
        boxes_2d_np[:, [1, 3]] *= scale_y  # y1, y2
        
        # Unproject to 3D camera coordinates (KITTI convention)
        x_2d = (boxes_2d_np[:, 0] + boxes_2d_np[:, 2]) / 2  # Center x
        y_2d = boxes_2d_np[:, 3]  # Bottom edge y
        
        x_cam = (x_2d - K[0, 2]) * depth_final / K[0, 0]
        y_cam_bottom = (y_2d - K[1, 2]) * depth_final / K[1, 1]
        z_cam = depth_final
        
        # Shift y to geometric center
        y_cam = y_cam_bottom - dims[:, 0] / 2.0  # Bottom - height/2
        
        # Assemble 3D boxes: [x, y, z, h, w, l, rotation_y]
        boxes_3d = np.stack([
            x_cam,       # x in camera frame
            y_cam,       # y in camera frame (center)
            z_cam,       # z (depth)
            dims[:, 0],  # height
            dims[:, 1],  # width
            dims[:, 2],  # length
            rotation     # rotation_y
        ], axis=1)
        
        scores_np = scores_2d.numpy()
        
        return boxes_3d, scores_np, class_names


def project_3d_box(box_3d, K):
    """Project 3D box to 2D image plane"""
    x, y, z, h, w, l, ry = box_3d
    
    # 8 corners of 3D bounding box
    x_corners = [l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2]
    y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    z_corners = [w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2]
    
    corners_3d = np.array([x_corners, y_corners, z_corners])
    
    # Rotation matrix (around y-axis)
    R = np.array([
        [np.cos(ry), 0, -np.sin(ry)],
        [0, 1, 0],
        [np.sin(ry), 0, np.cos(ry)]
    ])
    corners_3d = R @ corners_3d
    
    # Translation
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


def draw_3d_box(frame, box_3d, K, color=(0, 255, 0), thickness=2, 
                score=None, class_name=None):
    """Draw 3D bounding box on frame"""
    corners_2d = project_3d_box(box_3d, K)
    
    if corners_2d is None:
        return frame
    
    corners_2d = corners_2d.astype(np.int32)
    
    # Draw bottom face (0-1-2-3)
    for i in range(4):
        cv2.line(frame, tuple(corners_2d[i]), tuple(corners_2d[(i+1)%4]), color, thickness)
    
    # Draw top face (4-5-6-7)
    for i in range(4, 8):
        cv2.line(frame, tuple(corners_2d[i]), tuple(corners_2d[4+(i+1)%4]), color, thickness)
    
    # Draw vertical edges
    for i in range(4):
        cv2.line(frame, tuple(corners_2d[i]), tuple(corners_2d[i+4]), color, thickness)
    
    # Draw front indicator (red circle)
    front_center = ((corners_2d[0] + corners_2d[3] + corners_2d[4] + corners_2d[7]) / 4).astype(np.int32)
    cv2.circle(frame, tuple(front_center), 5, (0, 0, 255), -1)
    
    # Draw label with score
    if class_name and score is not None:
        label = f'{class_name} {score:.2f}'
        label_y = int(corners_2d[:, 1].min()) - 5
        label_pos = (int(corners_2d[:, 0].min()), max(20, label_y))
        cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, color, 2, cv2.LINE_AA)
    
    # Draw depth
    depth_text = f'{box_3d[2]:.1f}m'
    depth_pos = (int(corners_2d[:, 0].min()), int(corners_2d[:, 1].max()) + 20)
    cv2.putText(frame, depth_text, depth_pos, cv2.FONT_HERSHEY_SIMPLEX,
               0.5, color, 2, cv2.LINE_AA)
    
    return frame


def process_video(engine, video_path, output_path, conf_threshold=0.3, show_fps=True):
    """Process rectified video with V4 inference"""
    print(f"\n{'='*70}")
    print(f"VIDEO PROCESSING")
    print(f"{'='*70}")
    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"{'='*70}\n")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print()
    
    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Get original intrinsics for visualization
    K_vis = engine.intrinsics_original.cpu().numpy()
    
    inference_times = []
    detection_counts = []
    
    pbar = tqdm(total=total_frames, desc="Processing")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run V4 inference
        boxes_3d, scores, class_names, inf_time = engine.predict_frame(frame)
        inference_times.append(inf_time)
        
        # Filter by confidence threshold
        keep_mask = scores >= conf_threshold
        boxes_3d_filtered = boxes_3d[keep_mask]
        scores_filtered = scores[keep_mask]
        class_names_filtered = [class_names[i] for i in range(len(class_names)) if keep_mask[i]]
        
        detection_counts.append(len(boxes_3d_filtered))
        
        # Visualize
        frame_vis = frame.copy()
        for box, score, cls in zip(boxes_3d_filtered, scores_filtered, class_names_filtered):
            color = CLASS_COLORS.get(cls, (0, 255, 0))
            frame_vis = draw_3d_box(frame_vis, box, K_vis, color=color,
                                   thickness=2, score=score, class_name=cls)
        
        # Show FPS and detection count
        if show_fps:
            avg_inf_time = np.mean(inference_times[-30:])
            current_fps = 1.0 / avg_inf_time if avg_inf_time > 0 else 0
            
            info_text = f"FPS: {current_fps:.1f} | Detections: {len(boxes_3d_filtered)}"
            cv2.putText(frame_vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        out.write(frame_vis)
        pbar.update(1)
        frame_idx += 1
    
    pbar.close()
    cap.release()
    out.release()
    
    # Print statistics
    avg_inf_time = np.mean(inference_times)
    avg_fps = 1.0 / avg_inf_time if avg_inf_time > 0 else 0
    avg_detections = np.mean(detection_counts) if detection_counts else 0
    
    print(f"\n{'='*70}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Output saved: {output_path}")
    print(f"Frames processed: {frame_idx}")
    print(f"Average inference time: {avg_inf_time*1000:.2f} ms")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average detections per frame: {avg_detections:.1f}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='V4 Inference - Strict Training Architecture Adherence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full resolution (original 3848x2168)
  python inference_rectified_videosv4.py \\
    --checkpoint outputs/two_stage_v4/checkpoints/checkpoint_best.pth \\
    --video rectified_video.mp4 \\
    --output output_v4.mp4 \\
    --intrinsics-json rectified_camera_params.json
  
  # Resized to 1280x720 (faster)
  python inference_rectified_videosv4.py \\
    --checkpoint outputs/two_stage_v4/checkpoints/checkpoint_best.pth \\
    --video rectified_video.mp4 \\
    --output output_v4_720p.mp4 \\
    --intrinsics-json rectified_camera_params.json \\
    --target-size 1280 720
        """
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to V4 checkpoint (.pth file)')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to rectified input video')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output video with 3D boxes')
    parser.add_argument('--intrinsics-json', type=str, required=True,
                       help='Path to rectified_camera_params.json')
    parser.add_argument('--target-size', type=int, nargs=2, default=None,
                       help='Optional resize: width height (e.g., 1280 720)')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                       help='Confidence threshold for detections (default: 0.3)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--no-fps', action='store_true',
                       help='Hide FPS overlay on output video')
    
    args = parser.parse_args()
    
    # Validate inputs
    for path, name in [(args.video, 'Video'), (args.intrinsics_json, 'Intrinsics'), 
                       (args.checkpoint, 'Checkpoint')]:
        if not Path(path).exists():
            print(f"ERROR: {name} file not found: {path}")
            sys.exit(1)
    
    # Load intrinsics
    try:
        intrinsics = load_intrinsics_from_json(args.intrinsics_json)
    except Exception as e:
        print(f"ERROR loading intrinsics: {e}")
        sys.exit(1)
    
    # Parse target size
    target_size = tuple(args.target_size) if args.target_size else None
    
    # Create V4 inference engine
    engine = VideoInferenceEngineV4(
        checkpoint_path=args.checkpoint,
        intrinsics=intrinsics,
        device=args.device,
        target_size=target_size
    )
    
    # Process video
    process_video(
        engine=engine,
        video_path=args.video,
        output_path=args.output,
        conf_threshold=args.conf_threshold,
        show_fps=not args.no_fps
    )


if __name__ == "__main__":
    main()