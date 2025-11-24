"""
Video Inference Script for Monocular 3D Object Detection

Run trained model on video files and generate annotated output.

Usage:
    # Basic usage with default intrinsics
    python inference_video.py \
        --checkpoint outputs/experiments/netash_thirdline_*/checkpoints/checkpoint_epoch_049.pth \
        --video input_video.mp4 \
        --output output_video.mp4
    
    # With custom intrinsics
    python inference_video.py \
        --checkpoint path/to/checkpoint.pth \
        --video input.mp4 \
        --output output.mp4 \
        --intrinsics intrinsics.json \
        --score-threshold 0.6
    
    # Save individual frames
    python inference_video.py \
        --checkpoint path/to/checkpoint.pth \
        --video input.mp4 \
        --output output.mp4 \
        --save-frames \
        --frame-dir output_frames
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.monocular_3d_detector import build_model


def decode_heatmap_peaks(heatmap, threshold=0.5, max_objects=50):
    """Extract ONE peak per object with spatial grouping"""
    heatmap_torch = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    
    kernel = 15
    pad = (kernel - 1) // 2
    
    hmax = F.max_pool2d(heatmap_torch, kernel_size=kernel, stride=1, padding=pad)
    keep = (hmax == heatmap_torch).float()
    
    heatmap_nms = (heatmap_torch * keep).squeeze().numpy()
    
    peak_coords = np.where(heatmap_nms > threshold)
    peak_scores = heatmap_nms[peak_coords]
    
    peaks = []
    sorted_idx = np.argsort(peak_scores)[::-1]
    
    for idx in sorted_idx:
        y, x = peak_coords[0][idx], peak_coords[1][idx]
        score = peak_scores[idx]
        
        too_close = False
        for existing_peak in peaks:
            dist = np.sqrt((y - existing_peak[0])**2 + (x - existing_peak[1])**2)
            if dist < 10:
                too_close = True
                break
        
        if not too_close:
            peaks.append([y, x, score])
            
        if len(peaks) >= max_objects:
            break
    
    return np.array(peaks) if peaks else np.zeros((0, 3))


def decode_predictions(predictions, intrinsics, threshold=0.5, stride=4):
    """Decode model predictions into 3D bounding boxes"""
    batch_size = predictions['heatmap'].shape[0]
    
    depth_bins = [0, 10, 20, 35, 50, 70, 200]
    depth_bin_centers = [(depth_bins[i] + depth_bins[i+1])/2 for i in range(len(depth_bins)-1)]
    
    num_rot_bins = 8
    rot_bin_size = 2 * np.pi / num_rot_bins
    
    all_detections = []
    
    for b in range(batch_size):
        heatmap = predictions['heatmap'][b, 0].cpu().numpy()
        
        # Adaptive threshold
        peaks = None
        for thresh in [0.6, 0.7, 0.8]:
            candidate_peaks = decode_heatmap_peaks(heatmap, threshold=thresh)
            if 2 <= len(candidate_peaks) <= 10:
                peaks = candidate_peaks
                break
        
        if peaks is None:
            peaks = decode_heatmap_peaks(heatmap, threshold=threshold)
        
        if len(peaks) == 0:
            all_detections.append({
                'boxes_3d': torch.zeros(0, 7),
                'scores': torch.zeros(0)
            })
            continue
        
        K = intrinsics[b].cpu().numpy()
        
        boxes_3d = []
        scores = []
        
        for peak in peaks:
            y_hm, x_hm, score = peak
            y_hm, x_hm = int(y_hm), int(x_hm)
            
            # Decode depth
            depth_logits = predictions['depth_bin'][b, :, y_hm, x_hm].cpu().numpy()
            depth_bin_idx = np.argmax(depth_logits)
            depth_residual = predictions['depth_residual'][b, depth_bin_idx, y_hm, x_hm].item()
            depth = depth_bin_centers[depth_bin_idx] + depth_residual
            depth = max(0.1, depth)
            
            # Decode dimensions
            dims = predictions['dimensions'][b, :, y_hm, x_hm].cpu().numpy()
            h, w, l = np.abs(dims)
            
            # Decode rotation
            rot_logits = predictions['rotation_bin'][b, :, y_hm, x_hm].cpu().numpy()
            rot_bin_idx = np.argmax(rot_logits)
            rot_residual = predictions['rotation_residual'][b, rot_bin_idx, y_hm, x_hm].item()
            yaw_ego = (rot_bin_idx + 0.5) * rot_bin_size + rot_residual
            
            # Decode 2D offset
            offset = predictions['offset_2d'][b, :, y_hm, x_hm].cpu().numpy()
            
            # Unproject to camera frame
            u = (x_hm + offset[0]) * stride
            v = (y_hm + offset[1]) * stride
            
            x_cam = (u - K[0, 2]) * depth / K[0, 0]
            y_cam = (v - K[1, 2]) * depth / K[1, 1]
            z_cam = depth
            
            # Convert rotation ego→camera
            yaw_cam = -yaw_ego
            
            boxes_3d.append([x_cam, y_cam, z_cam, h, w, l, yaw_cam])
            scores.append(score)
        
        all_detections.append({
            'boxes_3d': torch.tensor(boxes_3d, dtype=torch.float32),
            'scores': torch.tensor(scores, dtype=torch.float32)
        })
    
    return all_detections


def project_3d_box_to_2d(box_3d, K):
    """Project 3D box to 2D image (camera frame)"""
    x, y, z, h, w, l, yaw = box_3d
    
    x_corners = [w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2]
    y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    z_corners = [l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2]
    
    corners_3d = np.array([x_corners, y_corners, z_corners])
    
    R = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    corners_3d = R @ corners_3d
    
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z
    
    if np.any(corners_3d[2, :] <= 0.1):
        return np.zeros((8, 2))
    
    corners_2d = K @ corners_3d
    corners_2d = corners_2d[:2, :] / corners_3d[2, :]
    
    return corners_2d.T


def draw_3d_box(image, box_3d, K, color=(0, 255, 0), thickness=2, score=None):
    """Draw 3D bounding box on image"""
    corners_2d = project_3d_box_to_2d(box_3d, K)
    
    if np.all(corners_2d == 0):
        return image
    
    corners_2d = corners_2d.astype(np.int32)
    
    # Draw bottom face
    for i in range(4):
        pt1 = tuple(corners_2d[i])
        pt2 = tuple(corners_2d[(i+1) % 4])
        cv2.line(image, pt1, pt2, color, thickness)
    
    # Draw top face
    for i in range(4, 8):
        pt1 = tuple(corners_2d[i])
        pt2 = tuple(corners_2d[4 + (i+1) % 4])
        cv2.line(image, pt1, pt2, color, thickness)
    
    # Draw vertical lines
    for i in range(4):
        pt1 = tuple(corners_2d[i])
        pt2 = tuple(corners_2d[i + 4])
        cv2.line(image, pt1, pt2, color, thickness)
    
    # Front face marker
    front_center = ((corners_2d[0] + corners_2d[1]) / 2).astype(np.int32)
    cv2.circle(image, tuple(front_center), 5, color, -1)
    
    # Draw score if provided
    if score is not None:
        text_pos = tuple(corners_2d[0])
        cv2.putText(image, f'{score:.2f}', text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, color, 2)
    
    return image


def load_intrinsics(intrinsics_path):
    """Load camera intrinsics from JSON file"""
    with open(intrinsics_path, 'r') as f:
        data = json.load(f)
    
    if 'matrix' in data:
        K = np.array(data['matrix'], dtype=np.float32)
    elif 'fx' in data and 'fy' in data and 'cx' in data and 'cy' in data:
        K = np.array([
            [data['fx'], 0, data['cx']],
            [0, data['fy'], data['cy']],
            [0, 0, 1]
        ], dtype=np.float32)
    else:
        raise ValueError("Invalid intrinsics format")
    
    return K


def estimate_intrinsics(width, height):
    """Estimate default camera intrinsics based on image size"""
    # Typical automotive camera FOV: ~50-60 degrees
    focal_length = width * 0.8  # Approximate
    
    K = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return K


def process_video(
    checkpoint_path,
    video_path,
    output_path,
    intrinsics=None,
    score_threshold=0.5,
    save_frames=False,
    frame_dir='output_frames',
    show_fps=True
):
    """
    Process video with trained model
    
    Args:
        checkpoint_path: Path to model checkpoint
        video_path: Input video path
        output_path: Output video path
        intrinsics: Camera intrinsics (3x3 matrix or path to JSON)
        score_threshold: Detection confidence threshold
        save_frames: Save individual annotated frames
        frame_dir: Directory for saved frames
        show_fps: Display FPS on output
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = build_model(pretrained=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Loaded checkpoint from epoch: {epoch}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo properties:")
    print(f"  Resolution: {orig_width}x{orig_height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    # Model input size
    model_width, model_height = 1280, 720
    
    # Setup camera intrinsics
    if intrinsics is None:
        print(f"\nNo intrinsics provided, estimating from image size...")
        K_orig = estimate_intrinsics(orig_width, orig_height)
    elif isinstance(intrinsics, str):
        print(f"\nLoading intrinsics from: {intrinsics}")
        K_orig = load_intrinsics(intrinsics)
    else:
        K_orig = np.array(intrinsics, dtype=np.float32)
    
    # Scale intrinsics to model input size
    scale_x = model_width / orig_width
    scale_y = model_height / orig_height
    K_model = K_orig.copy()
    K_model[0, :] *= scale_x
    K_model[1, :] *= scale_y
    
    print(f"\nCamera intrinsics (scaled to {model_width}x{model_height}):")
    print(K_model)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))
    
    # Setup frame saving
    if save_frames:
        frame_dir = Path(frame_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving frames to: {frame_dir}")
    
    # Process video
    print(f"\nProcessing video...")
    print(f"Score threshold: {score_threshold}")
    print("=" * 60)
    
    frame_count = 0
    total_detections = 0
    processing_times = []
    
    pbar = tqdm(total=total_frames, desc="Processing")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Resize frame for model
        frame_resized = cv2.resize(frame, (model_width, model_height))
        
        # Prepare input
        img_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        K_tensor = torch.from_numpy(K_model).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            predictions = model(img_tensor)
            detections = decode_predictions(predictions, K_tensor, threshold=score_threshold)
        
        # Get predictions
        pred_boxes = detections[0]['boxes_3d'].numpy()
        pred_scores = detections[0]['scores'].numpy()
        
        total_detections += len(pred_boxes)
        
        # Scale intrinsics back for visualization
        K_vis = K_orig.copy()
        
        # Draw detections on original frame
        annotated_frame = frame.copy()
        
        for box, score in zip(pred_boxes, pred_scores):
            # Scale box coordinates back to original resolution
            box_scaled = box.copy()
            box_scaled[0] *= (orig_width / model_width)   # x
            box_scaled[1] *= (orig_height / model_height) # y
            box_scaled[2] *= (orig_width / model_width)   # z (depth scaled by width for consistency)
            
            color = (0, 255, 0) if score > 0.7 else (0, 165, 255)
            annotated_frame = draw_3d_box(annotated_frame, box_scaled, K_vis, 
                                         color=color, thickness=2, score=score)
        
        # Add info overlay
        info_y = 30
        cv2.putText(annotated_frame, f'Detections: {len(pred_boxes)}', (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if show_fps:
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            current_fps = 1.0 / processing_time if processing_time > 0 else 0
            avg_fps = len(processing_times) / sum(processing_times) if processing_times else 0
            
            cv2.putText(annotated_frame, f'FPS: {current_fps:.1f} (avg: {avg_fps:.1f})', 
                       (10, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write frame
        out.write(annotated_frame)
        
        # Save individual frame if requested
        if save_frames:
            frame_path = frame_dir / f'frame_{frame_count:06d}.jpg'
            cv2.imwrite(str(frame_path), annotated_frame)
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    
    # Cleanup
    cap.release()
    out.release()
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Output saved to: {output_path}")
    print(f"\nStatistics:")
    print(f"  Total frames: {frame_count}")
    print(f"  Total detections: {total_detections}")
    print(f"  Avg detections/frame: {total_detections/frame_count:.1f}")
    
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        avg_fps = 1.0 / avg_time
        print(f"  Avg processing time: {avg_time*1000:.1f}ms/frame")
        print(f"  Avg FPS: {avg_fps:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Video Inference for Monocular 3D Detection')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--video', type=str, required=True,
                       help='Input video path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output video path')
    parser.add_argument('--intrinsics', type=str, default=None,
                       help='Camera intrinsics JSON file (optional, will estimate if not provided)')
    parser.add_argument('--score-threshold', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--save-frames', action='store_true',
                       help='Save individual annotated frames')
    parser.add_argument('--frame-dir', type=str, default='output_frames',
                       help='Directory for saved frames')
    parser.add_argument('--no-fps', action='store_true',
                       help='Hide FPS display')
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    if not Path(args.video).exists():
        print(f"Error: Video not found: {args.video}")
        return
    
    # Run processing
    process_video(
        checkpoint_path=args.checkpoint,
        video_path=args.video,
        output_path=args.output,
        intrinsics=args.intrinsics,
        score_threshold=args.score_threshold,
        save_frames=args.save_frames,
        frame_dir=args.frame_dir,
        show_fps=not args.no_fps
    )


if __name__ == "__main__":
    main()
