"""
Inference Script for Monocular 3D Object Detection

Test trained checkpoints on validation images and visualize predictions.

Usage:
    # Test latest checkpoint from running experiment
    python inference.py --checkpoint outputs/experiments/mono3d_production_mse_TIMESTAMP/checkpoints/checkpoint_latest.pth --num-samples 10
    
    # Test best checkpoint
    python inference.py --checkpoint outputs/experiments/mono3d_production_mse_TIMESTAMP/checkpoints/checkpoint_best.pth --num-samples 20
    
    # Save to custom directory
    python inference.py --checkpoint path/to/checkpoint.pth --output inference_results --score-threshold 0.5
"""

import os
import sys
import argparse
from pathlib import Path
import json
import random

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.nuscenes_dataset import NuScenesMonocular3D
from models.monocular_3d_detector import build_model
from torch.utils.data import DataLoader


def decode_heatmap_peaks(heatmap, threshold=0.5, max_objects=50):
    """Extract ONE peak per object with spatial grouping"""
    heatmap_torch = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    
    # Very aggressive NMS: 15x15 kernel
    kernel = 15  # 60-pixel suppression radius
    pad = (kernel - 1) // 2
    
    hmax = F.max_pool2d(heatmap_torch, kernel_size=kernel, stride=1, padding=pad)
    keep = (hmax == heatmap_torch).float()
    
    heatmap_nms = (heatmap_torch * keep).squeeze().numpy()
    
    # Lower threshold since we have strong NMS
    peak_coords = np.where(heatmap_nms > threshold)
    peak_scores = heatmap_nms[peak_coords]
    
    # Additional filtering: minimum distance between peaks
    peaks = []
    sorted_idx = np.argsort(peak_scores)[::-1]
    
    for idx in sorted_idx:
        y, x = peak_coords[0][idx], peak_coords[1][idx]
        score = peak_scores[idx]
        
        # Check distance to existing peaks
        too_close = False
        for existing_peak in peaks:
            dist = np.sqrt((y - existing_peak[0])**2 + (x - existing_peak[1])**2)
            if dist < 10:  # Minimum 10-pixel separation in heatmap (40px in image)
                too_close = True
                break
        
        if not too_close:
            peaks.append([y, x, score])
            
        if len(peaks) >= max_objects:
            break
    
    return np.array(peaks) if peaks else np.zeros((0, 3))

def decode_predictions(predictions, intrinsics, threshold=0.3, stride=4):
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
        for thresh in [0.5, 0.6, 0.7]:
            candidate_peaks = decode_heatmap_peaks(heatmap, threshold=thresh)
            if 2 <= len(candidate_peaks) <= 10:
                peaks = candidate_peaks
                break
        
        if peaks is None:
            peaks = decode_heatmap_peaks(heatmap, threshold=threshold)
        
        if len(peaks) == 0:
            all_detections.append({
                'boxes_3d': torch.zeros(0, 7),
                'scores': torch.zeros(0),
                'boxes_2d': torch.zeros(0, 4)
            })
            continue
        
        K = intrinsics[b].cpu().numpy()
        
        boxes_3d = []
        scores = []
        boxes_2d = []
        
        for peak in peaks:
            y_hm, x_hm, score = peak
            y_hm, x_hm = int(y_hm), int(x_hm)
            
            # Decode depth (camera frame)
            depth_logits = predictions['depth_bin'][b, :, y_hm, x_hm].cpu().numpy()
            depth_bin_idx = np.argmax(depth_logits)
            depth_residual = predictions['depth_residual'][b, depth_bin_idx, y_hm, x_hm].item()
            depth = depth_bin_centers[depth_bin_idx] + depth_residual
            depth = max(0.1, depth)
            
            # Decode dimensions (ego frame - rotation invariant)
            dims = predictions['dimensions'][b, :, y_hm, x_hm].cpu().numpy()
            h, w, l = np.abs(dims)
            
            # Decode rotation (ego frame)
            rot_logits = predictions['rotation_bin'][b, :, y_hm, x_hm].cpu().numpy()
            rot_bin_idx = np.argmax(rot_logits)
            rot_residual = predictions['rotation_residual'][b, rot_bin_idx, y_hm, x_hm].item()
            yaw_ego = (rot_bin_idx + 0.5) * rot_bin_size + rot_residual
            
            # Decode 2D offset
            offset = predictions['offset_2d'][b, :, y_hm, x_hm].cpu().numpy()
            
            # Unproject center to camera frame
            u = (x_hm + offset[0]) * stride
            v = (y_hm + offset[1]) * stride
            
            x_cam = (u - K[0, 2]) * depth / K[0, 0]
            y_cam = (v - K[1, 2]) * depth / K[1, 1]
            z_cam = depth
            
            # Convert rotation from ego to camera frame
            # Ego: yaw around Z-axis (up)
            # Camera: yaw around Y-axis (down)
            # Ego yaw=0 means vehicle facing forward (+X in ego = +Z in camera)
            yaw_cam = -yaw_ego  # Simple negation for Z→Y axis rotation
            
            # Store in camera frame for visualization
            boxes_3d.append([x_cam, y_cam, z_cam, h, w, l, yaw_cam])
            scores.append(score)
            
            # Compute 2D box
            u_min, u_max = u - 50, u + 50
            v_min, v_max = v - 50, v + 50
            boxes_2d.append([u_min, v_min, u_max, v_max])
        
        all_detections.append({
            'boxes_3d': torch.tensor(boxes_3d, dtype=torch.float32),
            'scores': torch.tensor(scores, dtype=torch.float32),
            'boxes_2d': torch.tensor(boxes_2d, dtype=torch.float32)
        })
    
    return all_detections


def project_3d_box_to_2d(box_3d, K):
    """
    Project 3D box to 2D image
    Input box is in CAMERA FRAME (x=right, y=down, z=forward)
    """
    x, y, z, h, w, l, yaw = box_3d
    
    # 3D box corners in object-centric frame (before rotation)
    x_corners = [w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2]
    y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    z_corners = [l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2]
    
    corners_3d = np.array([x_corners, y_corners, z_corners])
    
    # Rotate around Y-axis (vertical in camera frame)
    R = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    corners_3d = R @ corners_3d
    
    # Translate to box center
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z
    
    # Check if behind camera
    if np.any(corners_3d[2, :] <= 0.1):
        # Return dummy corners if behind camera
        return np.zeros((8, 2))
    
    # Project to image
    corners_2d = K @ corners_3d
    corners_2d = corners_2d[:2, :] / corners_2d[2, :]
    
    return corners_2d.T


def draw_3d_box(image, box_3d, K, color=(0, 255, 0), thickness=2):
    """Draw 3D bounding box on image"""
    corners_2d = project_3d_box_to_2d(box_3d, K)
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
    
    return image


def visualize_sample(image, gt_boxes, pred_boxes, pred_scores, K, output_path):
    """
    Create visualization comparing GT and predictions
    
    Args:
        image: (H, W, 3) numpy array
        gt_boxes: List of GT boxes [x, y, z, h, w, l, yaw]
        pred_boxes: Predicted boxes
        pred_scores: Prediction confidence scores
        K: Camera intrinsics
        output_path: Save path
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Input Image', fontsize=14)
    axes[0].axis('off')
    
    # Ground truth
    img_gt = image.copy()
    for box in gt_boxes:
        img_gt = draw_3d_box(img_gt, box, K, color=(0, 255, 0), thickness=2)
    
    axes[1].imshow(img_gt)
    axes[1].set_title(f'Ground Truth ({len(gt_boxes)} objects)', fontsize=14)
    axes[1].axis('off')
    
    # Predictions
    img_pred = image.copy()
    for box, score in zip(pred_boxes, pred_scores):
        color = (255, 0, 0) if score > 0.5 else (255, 165, 0)  # Red if confident, orange if uncertain
        img_pred = draw_3d_box(img_pred, box, K, color=color, thickness=2)
        
        # Add score text
        corners_2d = project_3d_box_to_2d(box, K)
        text_pos = tuple(corners_2d[0].astype(np.int32))
        cv2.putText(img_pred, f'{score:.2f}', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 2)
    
    axes[2].imshow(img_pred)
    axes[2].set_title(f'Predictions ({len(pred_boxes)} detections)', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_inference(checkpoint_path, num_samples=10, output_dir='inference_results', 
                 score_threshold=0.3, seed=42):
    """
    Run inference on validation samples
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Number of validation samples to test
        output_dir: Output directory for results
        score_threshold: Detection confidence threshold
        seed: Random seed for sample selection
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    config = checkpoint.get('config', None)
    if config is None:
        print("Warning: No config found in checkpoint, using defaults")
        config = {
            'dataset': {'root_dir': '/media/ashwin-benchdev/eb2a8889-3ec9-411a-826e-816cf9759b02/NuScenes-Full-Dataset'},
            'model': {'input_size': (720, 1280)}
        }
    
    # Build model
    print("Building model...")
    model = build_model(pretrained=False)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Loaded checkpoint from epoch: {epoch}")
    
    # Load validation dataset
    print("\nLoading validation dataset...")
    val_dataset = NuScenesMonocular3D(
        root_dir=config['dataset']['root_dir'],
        split='val',
        input_size=config['model']['input_size'],
        augment=False
    )
    
    # Select random samples
    total_samples = len(val_dataset)
    sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    print(f"\nRunning inference on {len(sample_indices)} samples...")
    print(f"Score threshold: {score_threshold}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Process samples
    for idx, sample_idx in enumerate(tqdm(sample_indices, desc="Processing samples")):
        sample = val_dataset[sample_idx]
        
        # Prepare input
        image = sample['image'].unsqueeze(0).to(device)
        intrinsics = sample['intrinsics'].unsqueeze(0)
        
        # Ground truth boxes
        mask = sample['mask'].numpy()
        valid_indices = np.where(mask > 0)[0]
        
        gt_boxes = []
        for i in valid_indices:
            indices_yx = sample['indices'][i].numpy()
            y_hm, x_hm = indices_yx
            
            # Decode GT (approximate)
            depth_bin = sample['depth_bin'][i].item()
            depth_res = sample['depth_residual'][i].item()
            dims = sample['dimensions'][i].numpy()
            rot_bin = sample['rotation_bin'][i].item()
            rot_res = sample['rotation_residual'][i].item()
            
            depth_bins = [0, 10, 20, 35, 50, 70, 200]
            depth_bin_centers = [(depth_bins[i] + depth_bins[i+1])/2 for i in range(len(depth_bins)-1)]
            depth = depth_bin_centers[depth_bin] + depth_res
            
            rot_bin_size = 2 * np.pi / 8
            rotation = (rot_bin + 0.5) * rot_bin_size + rot_res
            
            K = intrinsics[0].numpy()
            u = x_hm * 4
            v = y_hm * 4
            x_cam = (u - K[0, 2]) * depth / K[0, 0]
            y_cam = (v - K[1, 2]) * depth / K[1, 1]
            
            gt_boxes.append([x_cam, y_cam, depth, dims[0], dims[1], dims[2], rotation])
        
        # Run inference
        with torch.no_grad():
            predictions = model(image)
            detections = decode_predictions(predictions, intrinsics, threshold=score_threshold)
        
        # Get predictions
        pred_boxes = detections[0]['boxes_3d'].numpy()
        pred_scores = detections[0]['scores'].numpy()
        
        # Convert image to numpy for visualization
        img_np = image[0].cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        # Visualize
        output_path = output_dir / f'sample_{idx:03d}_idx_{sample_idx}.png'
        visualize_sample(
            img_np, 
            gt_boxes, 
            pred_boxes, 
            pred_scores,
            intrinsics[0].numpy(),
            output_path
        )
    
    print("\n" + "=" * 60)
    print(f"Inference complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Generated {len(sample_indices)} visualizations")


def main():
    parser = argparse.ArgumentParser(description='Inference for Monocular 3D Detection')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of validation samples to test')
    parser.add_argument('--output', type=str, default='inference_results',
                       help='Output directory for results')
    parser.add_argument('--score-threshold', type=float, default=0.3,
                       help='Detection confidence threshold')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sample selection')
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    # Run inference
    run_inference(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        output_dir=args.output,
        score_threshold=args.score_threshold,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
