"""
Visualization utilities for monitoring training progress

Generates comparison images showing:
- Ground truth 3D boxes (green)
- Predicted 3D boxes (red)
- Heatmap predictions
- Depth predictions
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def project_3d_box_to_2d(box_3d, K):
    """
    Project 3D bounding box to 2D image plane
    
    Args:
        box_3d: (7,) array [x, y, z, h, w, l, yaw] in camera frame
        K: (3, 3) camera intrinsic matrix
    
    Returns:
        corners_2d: (8, 2) array of 2D corner coordinates
    """
    x, y, z, h, w, l, yaw = box_3d
    
    # 3D box corners in camera frame (before yaw rotation)
    # Camera frame: X=right, Y=down, Z=forward
    # Bottom face (closer to ground, +Y) = first 4 corners
    # Top face (roof, -Y from center) = last 4 corners
    
    x_corners = [w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2]      # Width (lateral)
    y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]      # Height (vertical)
    z_corners = [l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2]      # Length (depth)
    
    # Rotate around Y axis (vertical in camera frame)
    corners_3d = np.array([x_corners, y_corners, z_corners])
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
    
    # Project to image
    corners_2d = K @ corners_3d
    corners_2d = corners_2d[:2, :] / (corners_2d[2, :] + 1e-8)  # Add epsilon for safety
    
    return corners_2d.T

def draw_3d_box(image, box_3d, K, color=(0, 255, 0), thickness=2):
    """
    Draw 3D bounding box on image
    
    Args:
        image: (H, W, 3) numpy array
        box_3d: (7,) array [x, y, z, h, w, l, yaw]
        K: (3, 3) camera intrinsic matrix
        color: RGB tuple
        thickness: Line thickness
    """
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
    
    # Draw front face marker (distinguish orientation)
    front_center = ((corners_2d[0] + corners_2d[1]) / 2).astype(np.int32)
    cv2.circle(image, tuple(front_center), 5, color, -1)
    
    return image


def visualize_predictions(
    model,
    val_loader,
    device,
    output_dir,
    epoch,
    num_samples=4,
    score_threshold=0.3
):
    """
    Generate visualization comparing GT and predictions
    
    Args:
        model: Trained model
        val_loader: Validation dataloader
        device: torch device
        output_dir: Directory to save visualizations
        epoch: Current epoch number
        num_samples: Number of validation samples to visualize
        score_threshold: Detection confidence threshold
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get a batch of validation samples
    batch = next(iter(val_loader))
    images = batch['image'].to(device)
    targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
              for k, v in batch.items() if k != 'image'}
    
    # Limit to num_samples
    images = images[:num_samples]
    targets = {k: v[:num_samples] if isinstance(v, torch.Tensor) else v 
              for k, v in targets.items()}
    intrinsics = batch['intrinsics'][:num_samples]
    
    # Forward pass
    with torch.no_grad():
        predictions = model(images)
        
        # Decode predictions
        detections = model.module.decode_detections(
            predictions, intrinsics, threshold=score_threshold
        ) if hasattr(model, 'module') else model.decode_detections(
            predictions, intrinsics, threshold=score_threshold
        )
    
    # Create visualization for each image
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 6*num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    
    for idx in range(num_samples):
        # Convert image to numpy
        img_np = images[idx].cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8).copy()
        
        K = intrinsics[idx].cpu().numpy()
        
        # Get ground truth boxes
        mask = targets['mask'][idx].cpu().numpy()
        valid_indices = np.where(mask > 0)[0]
        
        gt_boxes = []
        for i in valid_indices:
            indices_yx = targets['indices'][idx, i].cpu().numpy()
            y_hm, x_hm = indices_yx
            
            # Get GT from targets
            depth_bin = targets['depth_bin'][idx, i].item()
            depth_res = targets['depth_residual'][idx, i].item()
            dims = targets['dimensions'][idx, i].cpu().numpy()
            rot_bin = targets['rotation_bin'][idx, i].item()
            rot_res = targets['rotation_residual'][idx, i].item()
            
            # Approximate depth (use bin center + residual)
            depth_bins = [0, 10, 20, 35, 50, 70, 200]
            depth_bin_centers = [(depth_bins[i] + depth_bins[i+1])/2 for i in range(len(depth_bins)-1)]
            depth = depth_bin_centers[depth_bin] + depth_res
            
            # Approximate rotation
            rot_bin_size = 2 * np.pi / 8
            rotation = (rot_bin + 0.5) * rot_bin_size + rot_res
            
            # Unproject center to 3D
            u = x_hm * 4  # stride = 4
            v = y_hm * 4
            x_cam = (u - K[0, 2]) * depth / K[0, 0]
            y_cam = (v - K[1, 2]) * depth / K[1, 1]
            
            gt_boxes.append([x_cam, y_cam, depth, dims[0], dims[1], dims[2], rotation])
        
        # Get predicted boxes
        pred_boxes = detections[idx]['boxes_3d'].cpu().numpy()
        pred_scores = detections[idx]['scores'].cpu().numpy()
        
        # Image 1: Ground truth
        img_gt = img_np.copy()
        for box in gt_boxes:
            img_gt = draw_3d_box(img_gt, box, K, color=(0, 255, 0), thickness=2)
        
        axes[idx, 0].imshow(img_gt)
        axes[idx, 0].set_title(f'Ground Truth ({len(gt_boxes)} objects)', fontsize=12)
        axes[idx, 0].axis('off')
        
        # Image 2: Predictions
        img_pred = img_np.copy()
        for box, score in zip(pred_boxes, pred_scores):
            if score > score_threshold:
                img_pred = draw_3d_box(img_pred, box, K, color=(255, 0, 0), thickness=2)
        
        axes[idx, 1].imshow(img_pred)
        axes[idx, 1].set_title(f'Predictions ({len(pred_boxes)} detections)', fontsize=12)
        axes[idx, 1].axis('off')
        
        # Image 3: Heatmap overlay
        heatmap = predictions['heatmap'][idx, 0].cpu().numpy()
        heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
        
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title(f'Heatmap (max: {heatmap.max():.3f})', fontsize=12)
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    save_path = output_dir / f'detections_epoch_{epoch:03d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  → Saved visualization: {save_path}")
    
    return str(save_path)


def visualize_heatmap_grid(predictions, targets, output_dir, epoch, num_samples=8):
    """
    Visualize predicted vs ground truth heatmaps in a grid
    
    Args:
        predictions: Model predictions dict
        targets: Ground truth dict
        output_dir: Save directory
        epoch: Current epoch
        num_samples: Number of samples to show
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pred_heatmaps = predictions['heatmap'][:num_samples, 0].cpu().numpy()
    gt_heatmaps = targets['heatmap'][:num_samples, 0].cpu().numpy()
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3*num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    
    for idx in range(num_samples):
        # Ground truth
        axes[idx, 0].imshow(gt_heatmaps[idx], cmap='hot', vmin=0, vmax=1)
        axes[idx, 0].set_title(f'GT Heatmap (sample {idx})')
        axes[idx, 0].axis('off')
        
        # Prediction
        axes[idx, 1].imshow(pred_heatmaps[idx], cmap='hot', vmin=0, vmax=1)
        axes[idx, 1].set_title(f'Pred Heatmap (max: {pred_heatmaps[idx].max():.3f})')
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    
    save_path = output_dir / f'heatmaps_epoch_{epoch:03d}.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"  → Saved heatmap grid: {save_path}")
