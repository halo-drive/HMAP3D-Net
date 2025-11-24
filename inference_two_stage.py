"""
Inference Script for Two-Stage 3D Detection
Supports: KITTI validation, nuScenes, arbitrary images
"""

import os
import sys
import argparse
from pathlib import Path
import json
import time

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data.kitti_dataset import KITTI3DDataset
from models.two_stage_detector import build_model


class InferenceEngine:
    """Inference engine for monocular 3D detection"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading checkpoint: {checkpoint_path}")
        self.model = build_model()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
        print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    def preprocess_image(self, image_path, target_size=(384, 1280)):
        """Load and preprocess image"""
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Resize
        image_resized = cv2.resize(image, (target_size[1], target_size[0]))
        
        # To tensor
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        return image_tensor, image, (orig_h, orig_w)
    
    def get_default_intrinsics(self, image_shape, target_size=(384, 1280)):
        """
        Estimate default camera intrinsics
        Assumes typical automotive camera FOV ~50-60 degrees
        """
        h, w = image_shape
        target_h, target_w = target_size
        
        # Scale factors
        scale_x = target_w / w
        scale_y = target_h / h
        
        # Default focal length (typical for automotive cameras)
        # Assuming 50-degree horizontal FOV
        fx = target_w / (2 * np.tan(np.radians(50) / 2))
        fy = fx
        
        # Principal point at image center
        cx = target_w / 2
        cy = target_h / 2
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return torch.from_numpy(K).float()
    
    @torch.no_grad()
    def predict(self, image_tensor, intrinsics):
        """Run inference"""
        image_tensor = image_tensor.to(self.device)
        intrinsics = intrinsics.to(self.device)
        
        start_time = time.time()
        predictions = self.model(image_tensor, intrinsics.unsqueeze(0), gt_boxes_2d=None)
        inference_time = time.time() - start_time
        
        return predictions, inference_time
    
    def decode_predictions(self, predictions, intrinsics):
        """Decode predictions to 3D boxes"""
        boxes_2d = predictions['boxes_2d'][0]
        depth_pred = predictions['depth'][0]
        dims_pred = predictions['dimensions'][0]
        rot_bins, rot_res = predictions['rotation'][0]
        scores = predictions['scores_2d'][0]
        
        if len(boxes_2d) == 0:
            return np.zeros((0, 7)), np.zeros(0)
        
        # Decode depth
        depth = depth_pred[:, 0].cpu().numpy()
        
        # Decode rotation
        rot_bin_idx = torch.argmax(rot_bins, dim=1).cpu().numpy()
        bin_size = 2 * np.pi / 12
        rotation = (rot_bin_idx + 0.5) * bin_size
        rot_res_selected = rot_res[torch.arange(len(rot_bin_idx)), torch.from_numpy(rot_bin_idx)]
        rotation = rotation + rot_res_selected.cpu().numpy()
        rotation = (rotation + np.pi) % (2 * np.pi) - np.pi
        
        # Decode dimensions
        dims = dims_pred.cpu().numpy()
        
        # Unproject to 3D
        K = intrinsics.cpu().numpy()
        boxes_2d_np = boxes_2d.cpu().numpy()
        centers_2d = (boxes_2d_np[:, :2] + boxes_2d_np[:, 2:]) / 2
        
        x_cam = (centers_2d[:, 0] - K[0, 2]) * depth / K[0, 0]
        y_cam = (centers_2d[:, 1] - K[1, 2]) * depth / K[1, 1]
        z_cam = depth
        
        # [x, y, z, h, w, l, ry]
        boxes_3d = np.stack([x_cam, y_cam, z_cam, dims[:, 0], dims[:, 1], dims[:, 2], rotation], axis=1)
        scores_np = scores.cpu().numpy()
        
        return boxes_3d, scores_np


def project_3d_box(box_3d, K):
    """
    Project 3D box to 2D - CORRECTED KITTI CONVENTION
    LENGTH along X, WIDTH along Z
    """
    x, y, z, h, w, l, ry = box_3d
    
    # Correct dimension mapping
    x_corners = [l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2]  # LENGTH along X
    y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]  # HEIGHT along Y
    z_corners = [w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2]  # WIDTH along Z
    
    corners_3d = np.array([x_corners, y_corners, z_corners])
    
    # Rotation around Y-axis
    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    corners_3d = R @ corners_3d
    
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z
    
    if np.any(corners_3d[2, :] <= 0.1):
        return None
    
    corners_2d = K @ corners_3d
    corners_2d = corners_2d[:2, :] / corners_3d[2, :]
    
    return corners_2d.T


def draw_3d_box(image, box_3d, K, color=(0, 255, 0), thickness=2, score=None):
    """Draw 3D box on image"""
    corners_2d = project_3d_box(box_3d, K)
    
    if corners_2d is None:
        return image
    
    corners_2d = corners_2d.astype(np.int32)
    
    # Bottom face
    for i in range(4):
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[(i+1)%4]), color, thickness)
    
    # Top face
    for i in range(4, 8):
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[4+(i+1)%4]), color, thickness)
    
    # Vertical edges
    for i in range(4):
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[i+4]), color, thickness)
    
    # Front face marker
    front_center = ((corners_2d[0] + corners_2d[3] + corners_2d[4] + corners_2d[7]) / 4).astype(np.int32)
    cv2.circle(image, tuple(front_center), 5, (255, 0, 0), -1)
    
    # Draw score if provided
    if score is not None:
        text_pos = (int(corners_2d[:, 0].min()), int(corners_2d[:, 1].min()) - 5)
        cv2.putText(image, f'{score:.2f}', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 2)
    
    # Draw depth text
    depth_text = f'{box_3d[2]:.1f}m'
    depth_pos = (int(corners_2d[:, 0].min()), int(corners_2d[:, 1].max()) + 15)
    cv2.putText(image, depth_text, depth_pos, cv2.FONT_HERSHEY_SIMPLEX, 
               0.4, color, 1)
    
    return image


def visualize_predictions(image, boxes_3d, scores, K, output_path):
    """Visualize predictions"""
    img_viz = image.copy()
    
    for box_3d, score in zip(boxes_3d, scores):
        img_viz = draw_3d_box(img_viz, box_3d, K, color=(0, 255, 0), thickness=2, score=score)
    
    # Add info text
    info_text = f'Detections: {len(boxes_3d)}'
    cv2.putText(img_viz, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               1.0, (0, 255, 0), 2)
    
    # Save
    cv2.imwrite(str(output_path), cv2.cvtColor(img_viz, cv2.COLOR_RGB2BGR))


def visualize_comparison(image, boxes_3d_pred, boxes_3d_gt, scores, K, output_path):
    """Visualize predictions vs ground truth"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Ground truth
    img_gt = image.copy()
    for box in boxes_3d_gt:
        img_gt = draw_3d_box(img_gt, box, K, color=(0, 255, 0), thickness=2)
    
    axes[1].imshow(img_gt)
    axes[1].set_title(f'Ground Truth ({len(boxes_3d_gt)} objects)')
    axes[1].axis('off')
    
    # Predictions
    img_pred = image.copy()
    for box, score in zip(boxes_3d_pred, scores):
        img_pred = draw_3d_box(img_pred, box, K, color=(255, 0, 0), thickness=2, score=score)
    
    axes[2].imshow(img_pred)
    axes[2].set_title(f'Predictions ({len(boxes_3d_pred)} detections)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_iou_3d(box1, box2):
    """Compute 3D IoU (simplified - bird's eye view)"""
    # Extract bird's eye view coordinates
    x1, y1, z1, h1, w1, l1, ry1 = box1
    x2, y2, z2, h2, w2, l2, ry2 = box2
    
    # Simplified: use bounding rectangles in BEV
    # This is approximate - full 3D IoU requires rotated box intersection
    
    # For now, compute depth difference as a proxy
    depth_diff = abs(z1 - z2)
    if depth_diff > 2.0:  # More than 2m depth difference
        return 0.0
    
    # Lateral distance
    lateral_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    if lateral_dist > 2.0:  # More than 2m lateral distance
        return 0.0
    
    # Simple overlap based on distance
    max_dist = 3.0
    overlap = max(0, 1.0 - (lateral_dist + depth_diff) / max_dist)
    
    return overlap


def evaluate_kitti_val(engine, data_root, output_dir, num_samples=None):
    """Evaluate on KITTI validation set"""
    print("\n" + "="*60)
    print("EVALUATING ON KITTI VALIDATION SET")
    print("="*60)
    
    val_dataset = KITTI3DDataset(
        root_dir=data_root,
        split='val',
        filter_classes=['Car']
    )
    
    if num_samples:
        val_dataset.indices = val_dataset.indices[:num_samples]
    
    print(f"Validation samples: {len(val_dataset)}")
    
    output_dir = Path(output_dir)
    viz_dir = output_dir / 'kitti_val_results'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    inference_times = []
    
    for idx in tqdm(range(len(val_dataset)), desc="Running inference"):
        sample = val_dataset[idx]
        
        image_tensor = sample['image'].unsqueeze(0)
        intrinsics = sample['intrinsics']
        boxes_3d_gt = sample['boxes_3d'].numpy()
        img_idx = sample['img_idx']
        
        # Inference
        predictions, inf_time = engine.predict(image_tensor, intrinsics)
        boxes_3d_pred, scores = engine.decode_predictions(predictions, intrinsics)
        
        inference_times.append(inf_time)
        
        # Save visualization every 10 samples
        if idx % 10 == 0:
            image = (sample['image'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            K = intrinsics.numpy()
            
            viz_path = viz_dir / f'sample_{idx:04d}.png'
            visualize_comparison(image, boxes_3d_pred, boxes_3d_gt, scores, K, viz_path)
        
        # Store results
        results.append({
            'img_idx': img_idx,
            'num_gt': len(boxes_3d_gt),
            'num_pred': len(boxes_3d_pred),
            'inference_time': inf_time
        })
    
    # Compute statistics
    avg_time = np.mean(inference_times)
    fps = 1.0 / avg_time
    
    num_gt_total = sum(r['num_gt'] for r in results)
    num_pred_total = sum(r['num_pred'] for r in results)
    
    print("\n" + "="*60)
    print("KITTI VALIDATION RESULTS")
    print("="*60)
    print(f"Samples evaluated: {len(results)}")
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"Total GT objects: {num_gt_total}")
    print(f"Total predictions: {num_pred_total}")
    print(f"Avg detections per image: {num_pred_total/len(results):.2f}")
    print(f"\nVisualizations saved to: {viz_dir}")
    
    # Save metrics
    metrics = {
        'num_samples': len(results),
        'avg_inference_time_ms': avg_time * 1000,
        'fps': fps,
        'total_gt_objects': num_gt_total,
        'total_predictions': num_pred_total,
        'avg_detections_per_image': num_pred_total / len(results)
    }
    
    with open(output_dir / 'kitti_val_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def run_single_image(engine, image_path, output_path, intrinsics_path=None):
    """Run inference on single image"""
    print(f"\nProcessing: {image_path}")
    
    # Load image
    image_tensor, image_orig, orig_size = engine.preprocess_image(image_path)
    
    # Load or estimate intrinsics
    if intrinsics_path and Path(intrinsics_path).exists():
        K = np.loadtxt(intrinsics_path)
        intrinsics = torch.from_numpy(K).float()
        print("  Using provided intrinsics")
    else:
        intrinsics = engine.get_default_intrinsics(orig_size)
        print("  Using estimated intrinsics (default automotive camera)")
    
    # Inference
    predictions, inf_time = engine.predict(image_tensor, intrinsics)
    boxes_3d, scores = engine.decode_predictions(predictions, intrinsics)
    
    print(f"  Inference time: {inf_time*1000:.2f} ms")
    print(f"  Detections: {len(boxes_3d)}")
    
    # Visualize
    K = intrinsics.numpy()
    
    # Resize original image to match processed size
    image_resized = cv2.resize(image_orig, (1280, 384))
    
    visualize_predictions(image_resized, boxes_3d, scores, K, output_path)
    print(f"  Saved: {output_path}")


def run_image_directory(engine, image_dir, output_dir, intrinsics_path=None):
    """Run inference on directory of images"""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))
    print(f"\nFound {len(image_files)} images in {image_dir}")
    
    inference_times = []
    
    for img_path in tqdm(image_files, desc="Processing images"):
        output_path = output_dir / f'{img_path.stem}_3d.png'
        
        image_tensor, image_orig, orig_size = engine.preprocess_image(img_path)
        
        if intrinsics_path and Path(intrinsics_path).exists():
            K = np.loadtxt(intrinsics_path)
            intrinsics = torch.from_numpy(K).float()
        else:
            intrinsics = engine.get_default_intrinsics(orig_size)
        
        predictions, inf_time = engine.predict(image_tensor, intrinsics)
        boxes_3d, scores = engine.decode_predictions(predictions, intrinsics)
        
        inference_times.append(inf_time)
        
        K = intrinsics.numpy()
        image_resized = cv2.resize(image_orig, (1280, 384))
        
        visualize_predictions(image_resized, boxes_3d, scores, K, output_path)
    
    avg_time = np.mean(inference_times)
    print(f"\nAverage inference time: {avg_time*1000:.2f} ms ({1.0/avg_time:.2f} FPS)")
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Two-Stage 3D Detection Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['kitti-val', 'single', 'directory'],
                       help='Inference mode')
    
    # Mode-specific arguments
    parser.add_argument('--data-root', type=str, help='KITTI dataset root (for kitti-val mode)')
    parser.add_argument('--image', type=str, help='Single image path (for single mode)')
    parser.add_argument('--image-dir', type=str, help='Image directory (for directory mode)')
    parser.add_argument('--intrinsics', type=str, help='Camera intrinsics file (3x3 matrix)')
    
    parser.add_argument('--output-dir', type=str, default='outputs/inference',
                       help='Output directory for results')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to evaluate (for kitti-val)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = InferenceEngine(args.checkpoint, device=args.device)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference based on mode
    if args.mode == 'kitti-val':
        if not args.data_root:
            raise ValueError("--data-root required for kitti-val mode")
        evaluate_kitti_val(engine, args.data_root, output_dir, args.num_samples)
    
    elif args.mode == 'single':
        if not args.image:
            raise ValueError("--image required for single mode")
        output_path = output_dir / f'{Path(args.image).stem}_3d.png'
        run_single_image(engine, args.image, output_path, args.intrinsics)
    
    elif args.mode == 'directory':
        if not args.image_dir:
            raise ValueError("--image-dir required for directory mode")
        run_image_directory(engine, args.image_dir, output_dir, args.intrinsics)


if __name__ == "__main__":
    main()
