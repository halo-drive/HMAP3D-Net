"""Inference Script - Fixed 3D Unprojection"""

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


CLASS_COLORS = {
    'Car': (0, 255, 0),
    'Pedestrian': (255, 165, 0),
    'Cyclist': (0, 191, 255)
}


class InferenceEngine:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        print(f"Loading checkpoint: {checkpoint_path}")
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
        print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    @torch.no_grad()
    def predict(self, image_tensor, intrinsics):
        image_tensor = image_tensor.to(self.device)
        intrinsics = intrinsics.to(self.device)
        
        start_time = time.time()
        predictions = self.model(image_tensor, intrinsics.unsqueeze(0), gt_boxes_2d=None)
        inference_time = time.time() - start_time
        
        return predictions, inference_time
    
    def decode_predictions(self, predictions, intrinsics):
        """Decode predictions - FIXED UNPROJECTION"""
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
        depth = depth + depth_offset  # Apply learned offset
        
        # Decode rotation
        rot_bin_idx = torch.argmax(rot_bins, dim=1).cpu().numpy()
        bin_size = 2 * np.pi / 12
        rotation = (rot_bin_idx + 0.5) * bin_size
        rot_res_selected = rot_res[torch.arange(len(rot_bin_idx)), torch.from_numpy(rot_bin_idx)]
        rotation = rotation + rot_res_selected.cpu().numpy()
        rotation = (rotation + np.pi) % (2 * np.pi) - np.pi
        
        dims = dims_pred.cpu().numpy()
        
        # CRITICAL FIX: Use bottom-center of 2D box
        K = intrinsics.cpu().numpy()
        boxes_2d_np = boxes_2d.cpu().numpy()
        
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
    x, y, z, h, w, l, ry = box_3d
    
    x_corners = [l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2]
    y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    z_corners = [w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2]
    
    corners_3d = np.array([x_corners, y_corners, z_corners])
    
    R = np.array([
        [np.cos(ry), 0, -np.sin(ry)],
        [0, 1, 0],
        [np.sin(ry), 0, np.cos(ry)]
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


def draw_3d_box(image, box_3d, K, color=(0, 255, 0), thickness=2, score=None, class_name=None):
    corners_2d = project_3d_box(box_3d, K)
    
    if corners_2d is None:
        return image
    
    corners_2d = corners_2d.astype(np.int32)
    
    for i in range(4):
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[(i+1)%4]), color, thickness)
    
    for i in range(4, 8):
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[4+(i+1)%4]), color, thickness)
    
    for i in range(4):
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[i+4]), color, thickness)
    
    front_center = ((corners_2d[0] + corners_2d[3] + corners_2d[4] + corners_2d[7]) / 4).astype(np.int32)
    cv2.circle(image, tuple(front_center), 5, (255, 0, 0), -1)
    
    if class_name and score is not None:
        label = f'{class_name} {score:.2f}'
    elif class_name:
        label = class_name
    else:
        label = None
    
    if label:
        label_y = int(corners_2d[:, 1].min()) - 5
        label_pos = (int(corners_2d[:, 0].min()), max(15, label_y))
        cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, color, 2)
    
    depth_text = f'{box_3d[2]:.1f}m'
    depth_pos = (int(corners_2d[:, 0].min()), int(corners_2d[:, 1].max()) + 15)
    cv2.putText(image, depth_text, depth_pos, cv2.FONT_HERSHEY_SIMPLEX, 
               0.4, color, 1)
    
    return image


def visualize_comparison(image, boxes_3d_pred, boxes_3d_gt, scores, K, output_path, 
                        pred_classes=None, gt_classes=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    img_gt = image.copy()
    for idx, box in enumerate(boxes_3d_gt):
        class_name = gt_classes[idx] if gt_classes and idx < len(gt_classes) else None
        color = CLASS_COLORS.get(class_name, (0, 255, 0)) if class_name else (0, 255, 0)
        img_gt = draw_3d_box(img_gt, box, K, color=color, thickness=2, class_name=class_name)
    
    axes[1].imshow(img_gt)
    axes[1].set_title(f'Ground Truth ({len(boxes_3d_gt)} objects)')
    axes[1].axis('off')
    
    img_pred = image.copy()
    for idx, (box, score) in enumerate(zip(boxes_3d_pred, scores)):
        class_name = pred_classes[idx] if pred_classes and idx < len(pred_classes) else None
        color = CLASS_COLORS.get(class_name, (255, 0, 0)) if class_name else (255, 0, 0)
        img_pred = draw_3d_box(img_pred, box, K, color=color, thickness=2, 
                              score=score, class_name=class_name)
    
    axes[2].imshow(img_pred)
    axes[2].set_title(f'Predictions ({len(boxes_3d_pred)} detections)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_kitti_val(engine, data_root, output_dir, num_samples=None):
    print("\n" + "="*60)
    print("EVALUATING ON KITTI VALIDATION SET")
    print("="*60)
    
    val_dataset = KITTI3DDataset(
        root_dir=data_root,
        split='val',
        filter_classes=engine.classes
    )
    
    if num_samples:
        val_dataset.indices = val_dataset.indices[:num_samples]
    
    print(f"Validation samples: {len(val_dataset)}")
    
    output_dir = Path(output_dir)
    viz_dir = output_dir / 'kitti_val_results'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    inference_times = []
    class_stats = {cls: {'gt': 0, 'pred': 0} for cls in engine.classes}
    
    for idx in tqdm(range(len(val_dataset)), desc="Running inference"):
        sample = val_dataset[idx]
        
        image_tensor = sample['image'].unsqueeze(0)
        intrinsics = sample['intrinsics']
        boxes_3d_gt = sample['boxes_3d'].numpy()
        labels_gt = sample['labels'].numpy()
        
        gt_classes = [val_dataset.CLASSES[label] for label in labels_gt]
        
        for cls in gt_classes:
            if cls in class_stats:
                class_stats[cls]['gt'] += 1
        
        predictions, inf_time = engine.predict(image_tensor, intrinsics)
        boxes_3d_pred, scores, pred_classes = engine.decode_predictions(predictions, intrinsics)
        
        for cls in pred_classes:
            if cls in class_stats:
                class_stats[cls]['pred'] += 1
        
        inference_times.append(inf_time)
        
        if idx % 10 == 0:
            image = (sample['image'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            K = intrinsics.numpy()
            
            viz_path = viz_dir / f'sample_{idx:04d}.png'
            visualize_comparison(image, boxes_3d_pred, boxes_3d_gt, scores, K, viz_path,
                               pred_classes, gt_classes)
        
        results.append({
            'img_idx': sample['img_idx'],
            'num_gt': len(boxes_3d_gt),
            'num_pred': len(boxes_3d_pred),
            'inference_time': inf_time
        })
    
    avg_time = np.mean(inference_times)
    fps = 1.0 / avg_time
    
    print("\n" + "="*60)
    print("KITTI VALIDATION RESULTS")
    print("="*60)
    print(f"Samples evaluated: {len(results)}")
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"\nPer-class statistics:")
    for cls in engine.classes:
        print(f"  {cls:12s}: GT={class_stats[cls]['gt']:5d}  Pred={class_stats[cls]['pred']:5d}")
    print(f"\nVisualizations saved to: {viz_dir}")
    
    metrics = {
        'num_samples': len(results),
        'avg_inference_time_ms': avg_time * 1000,
        'fps': fps,
        'classes': engine.classes,
        'class_stats': class_stats,
    }
    
    with open(output_dir / 'kitti_val_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['kitti-val'])
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='outputs/inference')
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    engine = InferenceEngine(args.checkpoint, device=args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'kitti-val':
        evaluate_kitti_val(engine, args.data_root, output_dir, args.num_samples)


if __name__ == "__main__":
    main()
