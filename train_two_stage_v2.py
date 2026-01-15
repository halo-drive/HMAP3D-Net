"""Training Script for Two-Stage 3D Detection V2 - CLEAN VERSION

Fixed issues:
- Direction weight increased to 2.0 (fights mode collapse)
- IoU loss disabled temporarily (was causing gradient explosion)
- Visualization disabled (missing decode_predictions function)
- Clean code with no duplicates
"""

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

sys.path.insert(0, str(Path(__file__).parent))

from data.kitti_dataset import KITTI3DDataset
from models.two_stage_detector_v2 import build_model_v2
from models.losses_two_stage_v2 import Total3DLossV2


def augment_intrinsics(K, augment_range=0.3):
    """Randomly perturb camera intrinsics for training augmentation"""
    K_aug = K.clone()
    batch_size = K.shape[0]
    device = K.device
    
    for i in range(batch_size):
        # Randomly scale focal lengths
        fx_scale = torch.empty(1, device=device).uniform_(1 - augment_range, 1 + augment_range).item()
        fy_scale = torch.empty(1, device=device).uniform_(1 - augment_range, 1 + augment_range).item()
        
        K_aug[i, 0, 0] *= fx_scale  # fx
        K_aug[i, 1, 1] *= fy_scale  # fy
        
        # Slightly perturb principal point
        cx_scale = torch.empty(1, device=device).uniform_(1 - augment_range*0.5, 1 + augment_range*0.5).item()
        cy_scale = torch.empty(1, device=device).uniform_(1 - augment_range*0.5, 1 + augment_range*0.5).item()
        
        K_aug[i, 0, 2] *= cx_scale  # cx
        K_aug[i, 1, 2] *= cy_scale  # cy
    
    return K_aug


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
    """Prepare targets for all prediction heads"""
    all_depth = []
    all_dims = []
    all_rotation = []
    all_boxes_3d = []
    
    for boxes_3d in batch_boxes_3d:
        if len(boxes_3d) > 0:
            depth = boxes_3d[:, 2]
            dims = boxes_3d[:, 3:6]
            rotation = boxes_3d[:, 6]
            
            all_depth.append(depth)
            all_dims.append(dims)
            all_rotation.append(rotation)
            all_boxes_3d.append(boxes_3d)
    
    if len(all_depth) > 0:
        targets = {
            'depth': torch.cat(all_depth),
            'dimensions': torch.cat(all_dims),
            'rotation': torch.cat(all_rotation),
            'boxes_3d': torch.cat(all_boxes_3d),
            'foreground': torch.ones(sum(len(d) for d in all_depth))
        }
    else:
        targets = {
            'depth': torch.zeros(0),
            'dimensions': torch.zeros(0, 3),
            'rotation': torch.zeros(0),
            'boxes_3d': torch.zeros(0, 7),
            'foreground': torch.zeros(0)
        }
    
    return targets


def train_epoch(model, dataloader, optimizer, loss_fn, scaler, device, epoch, augment_intrinsics_flag=True):
    """Train for one epoch"""
    model.train(mode=True)
    
    total_loss = 0
    loss_depth_sum = 0
    loss_dim_sum = 0
    loss_dir_sum = 0
    loss_rot_sum = 0
    loss_iou_sum = 0
    loss_fg_sum = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        images = batch['images'].to(device)
        boxes_2d = batch['boxes_2d']
        boxes_3d = batch['boxes_3d']
        intrinsics = batch['intrinsics'].to(device)
        
        # Augment intrinsics
        if augment_intrinsics_flag:
            intrinsics = augment_intrinsics(intrinsics, augment_range=0.3)
        
        with autocast():
            predictions = model(images, intrinsics, gt_boxes_2d=boxes_2d)
            targets = prepare_targets(boxes_3d)
            
            if len(targets['depth']) == 0:
                continue
            
            targets = {k: v.to(device) for k, v in targets.items()}
            
            pred_depth = torch.cat(predictions['depth'])
            pred_dims = torch.cat(predictions['dimensions'])
            pred_direction = torch.cat(predictions['direction'])
            pred_rot_bins = torch.cat([r[0] for r in predictions['rotation']])
            pred_rot_res = torch.cat([r[1] for r in predictions['rotation']])
            pred_iou = torch.cat(predictions['iou'])
            pred_fg = torch.cat(predictions['foreground'])
            
            pred_dict = {
                'depth': pred_depth,
                'dimensions': pred_dims,
                'direction': pred_direction,
                'rotation': (pred_rot_bins, pred_rot_res),
                'iou': pred_iou,
                'foreground': pred_fg,
                # NO pred_boxes_3d - IoU loss disabled
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
        loss_dir_sum += loss_dict['loss_direction']
        loss_rot_sum += loss_dict['loss_rotation']
        loss_iou_sum += loss_dict['loss_iou']
        loss_fg_sum += loss_dict['loss_foreground']
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{loss.item():.3f}",
            'depth': f"{loss_dict['loss_depth']:.3f}",
            'dim': f"{loss_dict['loss_dimension']:.3f}",
            'dir': f"{loss_dict['loss_direction']:.3f}",
            'iou': f"{loss_dict['loss_iou']:.3f}"
        })
    
    avg_losses = {
        'total': total_loss / num_batches,
        'depth': loss_depth_sum / num_batches,
        'dimension': loss_dim_sum / num_batches,
        'direction': loss_dir_sum / num_batches,
        'rotation': loss_rot_sum / num_batches,
        'iou': loss_iou_sum / num_batches,
        'foreground': loss_fg_sum / num_batches
    }
    
    return avg_losses


@torch.no_grad()
def validate(model, dataloader, loss_fn, device):
    model.eval()
    
    total_loss = 0
    loss_depth_sum = 0
    loss_dim_sum = 0
    loss_dir_sum = 0
    loss_rot_sum = 0
    loss_iou_sum = 0
    loss_fg_sum = 0
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
            pred_direction = torch.cat(predictions['direction'])
            pred_rot_bins = torch.cat([r[0] for r in predictions['rotation']])
            pred_rot_res = torch.cat([r[1] for r in predictions['rotation']])
            pred_iou = torch.cat(predictions['iou'])
            pred_fg = torch.cat(predictions['foreground'])
            
            pred_dict = {
                'depth': pred_depth,
                'dimensions': pred_dims,
                'direction': pred_direction,
                'rotation': (pred_rot_bins, pred_rot_res),
                'iou': pred_iou,
                'foreground': pred_fg
                # NO pred_boxes_3d
            }
            
            loss, loss_dict = loss_fn(pred_dict, targets)
        
        total_loss += loss.item()
        loss_depth_sum += loss_dict['loss_depth']
        loss_dim_sum += loss_dict['loss_dimension']
        loss_dir_sum += loss_dict['loss_direction']
        loss_rot_sum += loss_dict['loss_rotation']
        loss_iou_sum += loss_dict['loss_iou']
        loss_fg_sum += loss_dict['loss_foreground']
        num_batches += 1
    
    avg_losses = {
        'total': total_loss / num_batches,
        'depth': loss_depth_sum / num_batches,
        'dimension': loss_dim_sum / num_batches,
        'direction': loss_dir_sum / num_batches,
        'rotation': loss_rot_sum / num_batches,
        'iou': loss_iou_sum / num_batches,
        'foreground': loss_fg_sum / num_batches
    }
    
    return avg_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--classes', type=str, nargs='+', 
                       default=['Car', 'Pedestrian', 'Cyclist'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='outputs/two_stage_v2')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--viz-interval', type=int, default=5)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    writer = SummaryWriter(output_dir / 'logs')
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("Loading datasets...")
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
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images\n")
    
    print("Building model V2...")
    model = build_model_v2(active_classes=args.classes)
    model = model.to(device)
    
    # CRITICAL: Direction weight = 2.0 (fights mode collapse)
    loss_fn = Total3DLossV2(loss_weights={
        'depth': 1.0,
        'dimension': 1.0,
        'direction': 5.0,      # INCREASED from 0.5
        'rotation': 1.0,
        'iou': 0.0,            # DISABLED (was causing explosion)
        'foreground': 0.3
    })
    
    trainable_params = []
    for name, param in model.named_parameters():
        if 'detector_2d' not in name and '_yolo' not in name:
            trainable_params.append(param)
    
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}\n")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"Resuming from {args.resume}\n")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
    
    print("Starting training...")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.epochs):
        train_losses = train_epoch(model, train_loader, optimizer, loss_fn, scaler, device, epoch)
        val_losses = validate(model, val_loader, loss_fn, device)
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_losses['total']:.4f} "
              f"(D:{train_losses['depth']:.2f} Dim:{train_losses['dimension']:.2f} "
              f"Dir:{train_losses['direction']:.3f} Rot:{train_losses['rotation']:.2f})")
        print(f"  Val Loss:   {val_losses['total']:.4f} "
              f"(D:{val_losses['depth']:.2f} Dim:{val_losses['dimension']:.2f} "
              f"Dir:{val_losses['direction']:.3f} Rot:{val_losses['rotation']:.2f})")
        
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
            print("  → New best model saved!")
        
        scheduler.step()
        print("=" * 60)
    
    writer.close()
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()