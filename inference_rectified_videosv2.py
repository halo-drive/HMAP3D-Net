"""Video Inference Script V2 - PATCHED for Quick Testing

PATCHES APPLIED:
1. Ignore collapsed direction head - use rotation head only (360° coverage)
2. Dimension correction - adjust for UK/European vehicle dimensions
3. Heuristic orientation estimation from 2D box geometry

This is a QUICK FIX for testing while retraining happens.
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

from models.two_stage_detector_v2 import build_model_v2


CLASS_COLORS = {
    'Car': (0, 255, 0),
    'Pedestrian': (255, 165, 0),
    'Cyclist': (0, 191, 255)
}


# KITTI dimension statistics (from training data)
KITTI_MEAN_DIMS = np.array([1.526, 1.629, 3.883])  # h, w, l
KITTI_STD_DIMS = np.array([0.192, 0.227, 0.845])

# Your domain dimensions (estimated from visual inspection + UK vehicle mix)
# Adjusted for more vans, SUVs, and European vehicles
YOUR_MEAN_DIMS = np.array([1.68, 1.82, 4.35])  # h, w, l (taller, wider, longer)
YOUR_STD_DIMS = np.array([0.25, 0.28, 1.05])  # More variation


def correct_dimensions(pred_dims):
    """
    Correct dimensions from KITTI priors to your domain priors
    
    Args:
        pred_dims: [N, 3] predicted dimensions from network
    Returns:
        corrected_dims: [N, 3] corrected dimensions
    """
    # Denormalize from KITTI distribution
    normalized = (pred_dims - KITTI_MEAN_DIMS) / KITTI_STD_DIMS
    
    # Renormalize to your domain distribution
    corrected = normalized * YOUR_STD_DIMS + YOUR_MEAN_DIMS
    
    # Clamp to reasonable ranges
    corrected[:, 0] = np.clip(corrected[:, 0], 1.2, 2.5)  # height: 1.2-2.5m
    corrected[:, 1] = np.clip(corrected[:, 1], 1.4, 2.2)  # width: 1.4-2.2m
    corrected[:, 2] = np.clip(corrected[:, 2], 3.0, 6.0)  # length: 3.0-6.0m
    
    return corrected


def estimate_orientation_heuristic(box_2d, depth, x_cam, K):
    """
    Heuristic orientation estimation from 2D box geometry
    
    This is a backup when direction head fails.
    Uses box aspect ratio and lateral position.
    """
    x_center = (box_2d[0] + box_2d[2]) / 2
    box_width = box_2d[2] - box_2d[0]
    box_height = box_2d[3] - box_2d[1]
    
    aspect_ratio = box_width / box_height
    image_center_x = K[0, 2]
    
    # Lateral position in camera frame
    lateral_offset = x_cam
    
    # Heuristic rules:
    # 1. Wide boxes (aspect > 1.5) are usually perpendicular
    # 2. Tall boxes are facing camera or away
    # 3. Use lateral position to disambiguate
    
    if aspect_ratio > 1.5:  # Wide box → perpendicular
        if lateral_offset > 0:  # Right side
            return -np.pi / 2  # Facing left
        else:  # Left side
            return np.pi / 2  # Facing right
    else:  # Tall box → longitudinal
        if x_center < image_center_x * 0.8:  # Far left
            return np.pi / 4  # Angled
        elif x_center > image_center_x * 1.2:  # Far right
            return -np.pi / 4  # Angled
        else:  # Center
            if depth < 20:  # Close
                return np.pi  # Facing camera
            else:  # Far
                return 0  # Facing away
    
    return 0  # Default


def load_intrinsics_from_json(json_path):
    """Load rectified camera intrinsics from JSON file"""
    with open(json_path, 'r') as f:
        params = json.load(f)
    
    if 'rectified_intrinsics' in params:
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


class VideoInferenceEngineV2Patched:
    def __init__(self, checkpoint_path, intrinsics, device='cuda'):
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
        
        # Build V2 model
        self.model = build_model_v2(active_classes=self.classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Set intrinsics
        self.intrinsics = torch.from_numpy(intrinsics).float().to(self.device)
        
        print(f"\nCamera intrinsics:")
        print(f"  fx={self.intrinsics[0,0]:.2f}, fy={self.intrinsics[1,1]:.2f}")
        print(f"  cx={self.intrinsics[0,2]:.2f}, cy={self.intrinsics[1,2]:.2f}")
        print(f"\n✓ Model V2 loaded from epoch {checkpoint['epoch']}")
        print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
        
        print("\n🔧 PATCHES ACTIVE:")
        print("  - Direction head DISABLED (using rotation head only)")
        print("  - Dimension correction ENABLED")
        print("  - Heuristic orientation estimation ENABLED")
        
        # Normalization params (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        h, w = frame.shape[:2]
        target_h, target_w = 384, 1280
        
        frame_resized = cv2.resize(frame, (target_w, target_h))
        
        # Normalize
        frame_norm = frame_resized.astype(np.float32) / 255.0
        frame_norm = (frame_norm - self.mean) / self.std
        
        # To tensor [1, 3, H, W]
        frame_tensor = torch.from_numpy(frame_norm.transpose(2, 0, 1)).unsqueeze(0).float()
        
        # Scale factors
        scale_x = w / target_w
        scale_y = h / target_h
        
        # Scale intrinsics to match resized image
        K_scaled = self.intrinsics.clone()
        K_scaled[0, 0] /= scale_x
        K_scaled[1, 1] /= scale_y
        K_scaled[0, 2] /= scale_x
        K_scaled[1, 2] /= scale_y
        
        return frame_tensor, K_scaled, (scale_x, scale_y)
    
    @torch.no_grad()
    def predict_frame(self, frame):
        """Run inference on single frame"""
        frame_tensor, K_scaled, scales = self.preprocess_frame(frame)
        frame_tensor = frame_tensor.to(self.device)
        
        # Inference
        start_time = time.time()
        predictions = self.model(frame_tensor, K_scaled.unsqueeze(0), gt_boxes_2d=None)
        inference_time = time.time() - start_time
        
        # Decode with patches
        boxes_3d, scores, class_names = self.decode_predictions_patched(predictions, scales)
        
        return boxes_3d, scores, class_names, inference_time
    
    def decode_predictions_patched(self, predictions, scales):
        """Decode predictions with PATCHES applied"""
        boxes_2d = predictions['boxes_2d'][0]
        depth_pred = predictions['depth'][0]
        dims_pred = predictions['dimensions'][0]
        direction_logits = predictions['direction'][0]  # Read but don't use
        rot_bins, rot_res = predictions['rotation'][0]
        scores = predictions['scores_2d'][0]
        class_names = predictions['classes'][0]
        
        # IoU-based confidence (if available)
        if 'iou' in predictions and len(predictions['iou'][0]) > 0:
            iou_scores = predictions['iou'][0].cpu().numpy()
            scores_combined = scores.cpu().numpy() * iou_scores
            scores = torch.from_numpy(scores_combined).to(scores.device)
        
        if len(boxes_2d) == 0:
            return np.zeros((0, 7)), np.zeros(0), []
        
        # Decode depth
        depth = depth_pred[:, 0].cpu().numpy()
        depth_offset = depth_pred[:, 2].cpu().numpy()
        depth = depth + depth_offset
        
        # PATCH 1: Ignore direction head, use rotation head for full 360°
        rot_bin_idx = torch.argmax(rot_bins, dim=1).cpu().numpy()
        bin_size = 2 * np.pi / 12
        rotation = (rot_bin_idx + 0.5) * bin_size
        rot_res_selected = rot_res[torch.arange(len(rot_bin_idx)), torch.from_numpy(rot_bin_idx)]
        rotation = rotation + rot_res_selected.cpu().numpy()
        
        # Map to [-π, π] without direction head
        rotation = (rotation + np.pi) % (2 * np.pi) - np.pi
        
        # Decode dimensions
        dims = dims_pred.cpu().numpy()
        
        # PATCH 2: Dimension correction
        dims_corrected = correct_dimensions(dims)
        
        # Get intrinsics
        scale_x, scale_y = scales
        K = self.intrinsics.cpu().numpy().copy()
        
        # Scale 2D boxes back
        boxes_2d_np = boxes_2d.cpu().numpy()
        boxes_2d_np[:, [0, 2]] *= scale_x
        boxes_2d_np[:, [1, 3]] *= scale_y
        
        # Unproject to 3D
        x_2d = (boxes_2d_np[:, 0] + boxes_2d_np[:, 2]) / 2
        y_2d = boxes_2d_np[:, 3]  # Bottom edge
        
        x_cam = (x_2d - K[0, 2]) * depth / K[0, 0]
        y_cam_bottom = (y_2d - K[1, 2]) * depth / K[1, 1]
        z_cam = depth
        
        # Shift to geometric center
        y_cam = y_cam_bottom - dims_corrected[:, 0] / 2.0
        
        # PATCH 3: Heuristic orientation refinement
        rotation_refined = rotation.copy()
        for i in range(len(rotation)):
            # Use heuristic if rotation seems uncertain
            heuristic_rot = estimate_orientation_heuristic(
                boxes_2d_np[i], 
                depth[i], 
                x_cam[i], 
                K
            )
            
            # Blend: trust rotation head more if loss was low
            # Since rotation loss was 0.005-0.008, we trust it 70%, heuristic 30%
            rotation_refined[i] = 0.7 * rotation[i] + 0.3 * heuristic_rot
        
        boxes_3d = np.stack([
            x_cam, 
            y_cam, 
            z_cam, 
            dims_corrected[:, 0],  # Corrected dimensions
            dims_corrected[:, 1], 
            dims_corrected[:, 2], 
            rotation_refined
        ], axis=1)
        
        scores_np = scores.cpu().numpy()
        
        return boxes_3d, scores_np, class_names


def project_3d_box(box_3d, K):
    """Project 3D box to 2D"""
    x, y, z, h, w, l, ry = box_3d
    
    # 8 corners
    x_corners = [l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2]
    y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    z_corners = [w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2]
    
    corners_3d = np.array([x_corners, y_corners, z_corners])
    
    # Rotation
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
    
    if np.any(corners_3d[2, :] <= 0.1):
        return None
    
    # Project
    corners_2d = K @ corners_3d
    corners_2d = corners_2d[:2, :] / corners_3d[2, :]
    
    return corners_2d.T


def draw_3d_box(frame, box_3d, K, color=(0, 255, 0), thickness=2, score=None, class_name=None):
    """Draw 3D box"""
    corners_2d = project_3d_box(box_3d, K)
    
    if corners_2d is None:
        return frame
    
    corners_2d = corners_2d.astype(np.int32)
    
    # Bottom face
    for i in range(4):
        cv2.line(frame, tuple(corners_2d[i]), tuple(corners_2d[(i+1)%4]), color, thickness)
    
    # Top face
    for i in range(4, 8):
        cv2.line(frame, tuple(corners_2d[i]), tuple(corners_2d[4+(i+1)%4]), color, thickness)
    
    # Vertical edges
    for i in range(4):
        cv2.line(frame, tuple(corners_2d[i]), tuple(corners_2d[i+4]), color, thickness)
    
    # Front indicator
    front_center = ((corners_2d[0] + corners_2d[3] + corners_2d[4] + corners_2d[7]) / 4).astype(np.int32)
    cv2.circle(frame, tuple(front_center), 5, (255, 0, 0), -1)
    
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
        cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, color, 2, cv2.LINE_AA)
    
    # Depth
    depth_text = f'{box_3d[2]:.1f}m'
    depth_pos = (int(corners_2d[:, 0].min()), int(corners_2d[:, 1].max()) + 20)
    cv2.putText(frame, depth_text, depth_pos, cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, color, 2, cv2.LINE_AA)
    
    return frame


def process_video(engine, video_path, output_path, conf_threshold=0.5, show_fps=True):
    """Process video"""
    print(f"\n{'='*70}")
    print(f"Processing video: {video_path}")
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
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    K = engine.intrinsics.cpu().numpy().copy()
    
    inference_times = []
    detection_counts = []
    
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        boxes_3d, scores, class_names, inf_time = engine.predict_frame(frame)
        inference_times.append(inf_time)
        
        keep_mask = scores >= conf_threshold
        boxes_3d_filtered = boxes_3d[keep_mask]
        scores_filtered = scores[keep_mask]
        class_names_filtered = [class_names[i] for i in range(len(class_names)) if keep_mask[i]]
        
        detection_counts.append(len(boxes_3d_filtered))
        
        frame_vis = frame.copy()
        for box, score, cls in zip(boxes_3d_filtered, scores_filtered, class_names_filtered):
            color = CLASS_COLORS.get(cls, (0, 255, 0))
            frame_vis = draw_3d_box(frame_vis, box, K, color=color, thickness=2, 
                                   score=score, class_name=cls)
        
        if show_fps:
            avg_inf_time = np.mean(inference_times[-30:])
            current_fps = 1.0 / avg_inf_time
            
            info_text = f"FPS: {current_fps:.1f} | Detections: {len(boxes_3d_filtered)}"
            cv2.putText(frame_vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        out.write(frame_vis)
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    avg_inf_time = np.mean(inference_times)
    avg_fps = 1.0 / avg_inf_time
    avg_detections = np.mean(detection_counts)
    
    print(f"\n{'='*70}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Output saved to: {output_path}")
    print(f"Average inference time: {avg_inf_time*1000:.2f} ms")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average detections per frame: {avg_detections:.1f}")


def main():
    parser = argparse.ArgumentParser(description='V2 Inference - PATCHED VERSION')
    
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--intrinsics-json', type=str, required=True)
    parser.add_argument('--conf-threshold', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-fps', action='store_true')
    
    args = parser.parse_args()
    
    if not Path(args.video).exists():
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)
    
    if not Path(args.intrinsics_json).exists():
        print(f"ERROR: Intrinsics not found: {args.intrinsics_json}")
        sys.exit(1)
    
    try:
        intrinsics = load_intrinsics_from_json(args.intrinsics_json)
    except Exception as e:
        print(f"ERROR loading intrinsics: {e}")
        sys.exit(1)
    
    engine = VideoInferenceEngineV2Patched(
        checkpoint_path=args.checkpoint,
        intrinsics=intrinsics,
        device=args.device
    )
    
    process_video(
        engine=engine,
        video_path=args.video,
        output_path=args.output,
        conf_threshold=args.conf_threshold,
        show_fps=not args.no_fps
    )


if __name__ == "__main__":
    main()