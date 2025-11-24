"""Training Script for Two-Stage 3D Detection - Multi-Class Support"""

import os
import sys
import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from data.kitti_dataset import KITTI3DDataset
from models.two_stage_detector import build_model
from models.losses_two_stage import Total3DLoss


def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    intrinsics = torch.stack([item['intrinsics'] for item in batch])
    boxes_2d = [item['boxes_2d'] for item in batch]
    boxes_3d = [item['boxes_3d'] for item in batch]
    labels = [item['labels'] for item in batch]
    img_indices = [item['img_idx'] for item in batch]
    
    return {
        'images': images,
        'boxes_2d': boxes_2d,
        'boxes_3d': boxes_3d,
        'labels': labels,
        'intrinsics': intrinsics,
        'img_indices': img_indices
    }


def prepare_targets(batch_boxes_3d):
    all_depth = []
    all_dims = []
    all_rotation = []
    
    for boxes_3d in batch_boxes_3d:
        if len(boxes_3d) > 0:
            depth = boxes_3d[:, 2]
            dims = boxes_3d[:, 3:6]
            rotation = boxes_3d[:, 6]
            
            all_depth.append(depth)
            all_dims.append(dims)
            all_rotation.append(rotation)
    
    if len(all_depth) > 0:
        targets = {
            'depth': torch.cat(all_depth),
            'dimensions': torch.cat(all_dims),
            'rotation': torch.cat(all_rotation)
        }
    else:
        targets = {
            'depth': torch.zeros(0),
            'dimensions': torch.zeros(0, 3),
            'rotation': torch.zeros(0)
        }
    
    return targets


def decode_predictions(predictions, intrinsics):
    """Decode model predictions to 3D boxes"""
    boxes_2d = predictions['boxes_2d'][0]
    depth_pred = predictions['depth'][0]
    dims_pred = predictions['dimensions'][0]
    rot_bins, rot_res = predictions['rotation'][0]
    
    if len(boxes_2d) == 0:
        return np.zeros((0, 7))
    
    depth = depth_pred[:, 0].cpu().numpy()
    
    rot_bin_idx = torch.argmax(rot_bins, dim=1).cpu().numpy()
    bin_size = 2 * np.pi / 12
    rotation = (rot_bin_idx + 0.5) * bin_size
    rot_res_selected = rot_res[torch.arange(len(rot_bin_idx)), torch.from_numpy(rot_bin_idx)]
    rotation = rotation + rot_res_selected.cpu().numpy()
    rotation = (rotation + np.pi) % (2 * np.pi) - np.pi
    
    dims = dims_pred.cpu().numpy()
    
    K = intrinsics.cpu().numpy()
    boxes_2d_np = boxes_2d.cpu().numpy()
    centers_2d = (boxes_2d_np[:, :2] + boxes_2d_np[:, 2:]) / 2
    
    x_cam = (centers_2d[:, 0] - K[0, 2]) * depth / K[0, 0]
    y_cam = (centers_2d[:, 1] - K[1, 2]) * depth / K[1, 1]
    z_cam = depth
    
    boxes_3d = np.stack([x_cam, y_cam, z_cam, dims[:, 0], dims[:, 1], dims[:, 2], rotation], axis=1)
    
    return boxes_3d


def project_3d_box(box_3d, K):
    """Project 3D box to 2D - CORRECTED KITTI CONVENTION"""
    x, y, z, h, w, l, ry = box_3d
    
    x_corners = [l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2]  # LENGTH along X
    y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]  # HEIGHT along Y
    z_corners = [w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2]  # WIDTH along Z
    
    corners_3d = np.array([x_corners, y_corners, z_corners])
    
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


def draw_3d_box(image, box_3d, K, color=(0, 255, 0), thickness=2):
    """Draw 3D box on image"""
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
    
    return image


@torch.no_grad()
def visualize_samples(model, dataloader, device, output_dir, epoch, num_samples=4):
    """Visualize validation samples"""
    model.eval()
    
    viz_dir = output_dir / 'visualizations' / f'epoch_{epoch:03d}'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    batch = next(iter(dataloader))
    images = batch['images'][:num_samples].to(device)
    boxes_2d = batch['boxes_2d'][:num_samples]
    boxes_3d = batch['boxes_3d'][:num_samples]
    intrinsics = batch['intrinsics'][:num_samples].to(device)
    
    predictions = model(images, intrinsics, gt_boxes_2d=boxes_2d)
    
    for i in range(num_samples):
        pred_boxes_3d = decode_predictions({
            'boxes_2d': [predictions['boxes_2d'][i]],
            'depth': [predictions['depth'][i]],
            'dimensions': [predictions['dimensions'][i]],
            'rotation': [(predictions['rotation'][i][0], predictions['rotation'][i][1])]
        }, intrinsics[i])
        
        gt_boxes_3d = boxes_3d[i].cpu().numpy()
        
        img_np = (images[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        K = intrinsics[i].cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(img_np)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        img_gt = img_np.copy()
        for box in gt_boxes_3d:
            img_gt = draw_3d_box(img_gt, box, K, color=(0, 255, 0), thickness=2)
        
        axes[1].imshow(img_gt)
        axes[1].set_title(f'Ground Truth ({len(gt_boxes_3d)} objects)')
        axes[1].axis('off')
        
        img_pred = img_np.copy()
        for box in pred_boxes_3d:
            img_pred = draw_3d_box(img_pred, box, K, color=(255, 0, 0), thickness=2)
        
        axes[2].imshow(img_pred)
        axes[2].set_title(f'Predictions ({len(pred_boxes_3d)} detections)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(viz_dir / f'sample_{i:02d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  → Saved visualizations to {viz_dir}")


def train_epoch(model, dataloader, optimizer, loss_fn, scaler, device, epoch):
    """Train for one epoch with mixed precision"""
    model.train()
    
    total_loss = 0
    loss_depth_sum = 0
    loss_dim_sum = 0
    loss_rot_sum = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        images = batch['images'].to(device)
        boxes_2d = batch['boxes_2d']
        boxes_3d = batch['boxes_3d']
        intrinsics = batch['intrinsics'].to(device)
        
        with autocast():
            predictions = model(images, intrinsics, gt_boxes_2d=boxes_2d)
            targets = prepare_targets(boxes_3d)
            
            if len(targets['depth']) == 0:
                continue
            
            targets = {k: v.to(device) for k, v in targets.items()}
            
            pred_depth = torch.cat(predictions['depth'])
            pred_dims = torch.cat(predictions['dimensions'])
            pred_rot_bins = torch.cat([r[0] for r in predictions['rotation']])
            pred_rot_res = torch.cat([r[1] for r in predictions['rotation']])
            
            pred_dict = {
                'depth': pred_depth,
                'dimensions': pred_dims,
                'rotation': (pred_rot_bins, pred_rot_res)
            }
            
            loss, loss_dict = loss_fn(pred_dict, targets)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        loss_depth_sum += loss_dict['loss_depth']
        loss_dim_sum += loss_dict['loss_dimension']
        loss_rot_sum += loss_dict['loss_rotation']
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{loss.item():.3f}",
            'depth': f"{loss_dict['loss_depth']:.3f}",
            'dim': f"{loss_dict['loss_dimension']:.3f}",
            'rot': f"{loss_dict['loss_rotation']:.3f}"
        })
    
    avg_losses = {
        'total': total_loss / num_batches,
        'depth': loss_depth_sum / num_batches,
        'dimension': loss_dim_sum / num_batches,
        'rotation': loss_rot_sum / num_batches
    }
    
    return avg_losses


@torch.no_grad()
def validate(model, dataloader, loss_fn, device):
    model.eval()
    
    total_loss = 0
    loss_depth_sum = 0
    loss_dim_sum = 0
    loss_rot_sum = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Validation"):
        images = batch['images'].to(device)
        boxes_2d = batch['boxes_2d']
        boxes_3d = batch['boxes_3d']
        intrinsics = batch['intrinsics'].to(device)
        
        with autocast():
            predictions = model(images, intrinsics, gt_boxes_2d=boxes_2d)
            targets = prepare_targets(boxes_3d)
            
            if len(targets['depth']) == 0:
                continue
            
            targets = {k: v.to(device) for k, v in targets.items()}
            
            pred_depth = torch.cat(predictions['depth'])
            pred_dims = torch.cat(predictions['dimensions'])
            pred_rot_bins = torch.cat([r[0] for r in predictions['rotation']])
            pred_rot_res = torch.cat([r[1] for r in predictions['rotation']])
            
            pred_dict = {
                'depth': pred_depth,
                'dimensions': pred_dims,
                'rotation': (pred_rot_bins, pred_rot_res)
            }
            
            loss, loss_dict = loss_fn(pred_dict, targets)
        
        total_loss += loss.item()
        loss_depth_sum += loss_dict['loss_depth']
        loss_dim_sum += loss_dict['loss_dimension']
        loss_rot_sum += loss_dict['loss_rotation']
        num_batches += 1
    
    avg_losses = {
        'total': total_loss / num_batches,
        'depth': loss_depth_sum / num_batches,
        'dimension': loss_dim_sum / num_batches,
        'rotation': loss_rot_sum / num_batches
    }
    
    return avg_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--classes', type=str, nargs='+', 
                       default=['Car', 'Pedestrian', 'Cyclist'],
                       help='KITTI classes to train on')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='outputs/two_stage')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--viz-interval', type=int, default=5)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    writer = SummaryWriter(output_dir / 'logs')
    
    # Save config
    config = vars(args).copy()
    config['classes'] = args.classes
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nLoading datasets...")
    print(f"Training classes: {args.classes}")
    
    train_dataset = KITTI3DDataset(
        root_dir=args.data_root,
        split='train',
        filter_classes=args.classes
    )
    
    val_dataset = KITTI3DDataset(
        root_dir=args.data_root,
        split='val',
        filter_classes=args.classes
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    print(f"Train: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images")
    
    print("\nBuilding model...")
    model = build_model(active_classes=args.classes)
    model = model.to(device)
    
    loss_fn = Total3DLoss(loss_weights={'depth': 1.0, 'dimension': 1.0, 'rotation': 0.5})
    
    trainable_params = []
    for name, param in model.named_parameters():
        if 'detector_2d' not in name and '_yolo' not in name:
            trainable_params.append(param)
    
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
    
    print("\nStarting training...")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.epochs):
        train_losses = train_epoch(model, train_loader, optimizer, loss_fn, scaler, device, epoch)
        val_losses = validate(model, val_loader, loss_fn, device)
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_losses['total']:.4f} (D:{train_losses['depth']:.3f} "
              f"Dim:{train_losses['dimension']:.3f} Rot:{train_losses['rotation']:.3f})")
        print(f"  Val Loss:   {val_losses['total']:.4f} (D:{val_losses['depth']:.3f} "
              f"Dim:{val_losses['dimension']:.3f} Rot:{val_losses['rotation']:.3f})")
        
        if epoch % args.viz_interval == 0:
            visualize_samples(model, val_loader, device, output_dir, epoch, num_samples=4)
        
        for key in train_losses:
            writer.add_scalar(f'train/{key}', train_losses[key], epoch)
            writer.add_scalar(f'val/{key}', val_losses[key], epoch)
        
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'classes': args.classes
        }
        
        torch.save(checkpoint, checkpoint_dir / 'checkpoint_latest.pth')
        
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save(checkpoint, checkpoint_dir / 'checkpoint_best.pth')
            print(f"  → New best model saved!")
        
        scheduler.step()
        print("=" * 60)
    
    writer.close()
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
