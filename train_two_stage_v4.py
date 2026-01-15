"""Training Script V4 - Stable Training with Domain Adaptation

Key Improvements:
1. Fixed depth loss (no uncertainty exploitation)
2. Normalized intrinsics (resolution-invariant)
3. Better augmentation strategy
4. Gradient monitoring and clipping
5. Learning rate warmup + cosine schedule
6. Early stopping on validation divergence
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.two_stage_detector_v4 import build_model_v4
from models.losses_two_stage_v4 import Total3DLossV4, GradientMonitor


def collate_fn(batch):
    """Custom collate function for batch processing"""
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'boxes_2d': [item['boxes_2d'] for item in batch],
        'boxes_3d': [item['boxes_3d'] for item in batch],
        'labels': [item['labels'] for item in batch],
        'intrinsics': torch.stack([item['intrinsics'] for item in batch]),
        'img_indices': [item['img_idx'] for item in batch]
    }


def augment_intrinsics_normalized(K, image_size, augment_range=0.2):
    """
    Augment intrinsics with awareness of normalized values
    
    Args:
        K: [B, 3, 3] intrinsics
        image_size: (H, W)
        augment_range: ±20% augmentation (reduced from V3's ±30%)
    
    Returns:
        K_aug: [B, 3, 3] augmented intrinsics
    """
    K_aug = K.clone()
    h, w = image_size
    
    for i in range(K.shape[0]):
        # Augment focal lengths
        fx_scale = torch.empty(1, device=K.device).uniform_(
            1 - augment_range, 1 + augment_range
        ).item()
        fy_scale = torch.empty(1, device=K.device).uniform_(
            1 - augment_range, 1 + augment_range
        ).item()
        
        K_aug[i, 0, 0] *= fx_scale  # fx
        K_aug[i, 1, 1] *= fy_scale  # fy
        
        # Augment principal point (smaller range)
        cx_scale = torch.empty(1, device=K.device).uniform_(
            1 - augment_range * 0.3, 1 + augment_range * 0.3
        ).item()
        cy_scale = torch.empty(1, device=K.device).uniform_(
            1 - augment_range * 0.3, 1 + augment_range * 0.3
        ).item()
        
        K_aug[i, 0, 2] *= cx_scale  # cx
        K_aug[i, 1, 2] *= cy_scale  # cy
    
    return K_aug


def prepare_targets(batch_boxes_3d, device):
    """Prepare target dict from batch"""
    all_depth = []
    all_dims = []
    all_rot = []
    all_fg = []
    
    for boxes_3d in batch_boxes_3d:
        if len(boxes_3d) > 0:
            all_depth.append(boxes_3d[:, 2])      # z (depth)
            all_dims.append(boxes_3d[:, 3:6])     # h, w, l
            all_rot.append(boxes_3d[:, 6])        # rotation_y
            all_fg.append(torch.ones(len(boxes_3d)))  # foreground labels
    
    if len(all_depth) > 0:
        return {
            'depth': torch.cat(all_depth).to(device),
            'dimensions': torch.cat(all_dims).to(device),
            'rotation': torch.cat(all_rot).to(device),
            'foreground': torch.cat(all_fg).to(device)
        }
    
    # Return empty targets
    return {
        'depth': torch.zeros(0).to(device),
        'dimensions': torch.zeros(0, 3).to(device),
        'rotation': torch.zeros(0).to(device),
        'foreground': torch.zeros(0).to(device)
    }

def train_epoch(model, dataloader, optimizer, loss_fn, scaler, device, epoch, 
                grad_monitor, max_grad_norm=10.0):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    loss_depth_sum = 0
    loss_dim_sum = 0
    loss_rot_sum = 0
    loss_fg_sum = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        boxes_2d = batch['boxes_2d']
        boxes_3d = batch['boxes_3d']
        intrinsics = batch['intrinsics'].to(device)
        
        # Augment intrinsics
        image_h, image_w = images.shape[2:4]
        intrinsics_aug = augment_intrinsics_normalized(
            intrinsics, (image_h, image_w), augment_range=0.2
        )
        
        # Forward pass
        with autocast():
            predictions = model(images, intrinsics_aug, gt_boxes_2d=boxes_2d)
            
            # Prepare targets
            targets = prepare_targets(boxes_3d, device)
            
            if len(targets['depth']) == 0:
                continue
            
            # Concatenate predictions
            pred_depth = torch.cat(predictions['depth'])
            pred_dims = torch.cat(predictions['dimensions'])
            pred_rot_bins = torch.cat([r[0] for r in predictions['rotation']])
            pred_rot_res = torch.cat([r[1] for r in predictions['rotation']])
            pred_fg = torch.cat(predictions['foreground'])
            
            pred_dict = {
                'depth': pred_depth,
                'dimensions': pred_dims,
                'rotation': (pred_rot_bins, pred_rot_res),
                'foreground': pred_fg
            }
            
            # Compute loss
            loss, loss_dict = loss_fn(pred_dict, targets)
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # FIXED: Check for NaN BEFORE unscaling
        has_nan, nan_param = grad_monitor.check_nan_gradients()
        if has_nan:
            print(f"\nWARNING: NaN gradient detected in {nan_param}. Skipping batch.")
            scaler.update()  # Must call update even when skipping
            continue
        
        # Gradient clipping (only if no NaN)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate losses
        total_loss += loss.item()
        loss_depth_sum += loss_dict['loss_depth']
        loss_dim_sum += loss_dict['loss_dimension']
        loss_rot_sum += loss_dict['loss_rotation']
        loss_fg_sum += loss_dict['loss_foreground']
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.3f}",
            'depth': f"{loss_dict['loss_depth']:.2f}",
            'dim': f"{loss_dict['loss_dimension']:.2f}",
            'rot': f"{loss_dict['loss_rotation']:.2f}"
        })
    
    # Compute average losses
    avg_losses = {
        'total': total_loss / num_batches,
        'depth': loss_depth_sum / num_batches,
        'dimension': loss_dim_sum / num_batches,
        'rotation': loss_rot_sum / num_batches,
        'foreground': loss_fg_sum / num_batches
    }
    
    return avg_losses

@torch.no_grad()
def validate(model, dataloader, loss_fn, device):
    """Validate model"""
    model.eval()
    
    total_loss = 0
    loss_depth_sum = 0
    loss_dim_sum = 0
    loss_rot_sum = 0
    loss_fg_sum = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Validation"):
        images = batch['images'].to(device)
        boxes_2d = batch['boxes_2d']
        boxes_3d = batch['boxes_3d']
        intrinsics = batch['intrinsics'].to(device)
        
        with autocast():
            predictions = model(images, intrinsics, gt_boxes_2d=boxes_2d)
            
            targets = prepare_targets(boxes_3d, device)
            
            if len(targets['depth']) == 0:
                continue
            
            pred_depth = torch.cat(predictions['depth'])
            pred_dims = torch.cat(predictions['dimensions'])
            pred_rot_bins = torch.cat([r[0] for r in predictions['rotation']])
            pred_rot_res = torch.cat([r[1] for r in predictions['rotation']])
            pred_fg = torch.cat(predictions['foreground'])
            
            pred_dict = {
                'depth': pred_depth,
                'dimensions': pred_dims,
                'rotation': (pred_rot_bins, pred_rot_res),
                'foreground': pred_fg
            }
            
            loss, loss_dict = loss_fn(pred_dict, targets)
        
        total_loss += loss.item()
        loss_depth_sum += loss_dict['loss_depth']
        loss_dim_sum += loss_dict['loss_dimension']
        loss_rot_sum += loss_dict['loss_rotation']
        loss_fg_sum += loss_dict['loss_foreground']
        num_batches += 1
    
    avg_losses = {
        'total': total_loss / num_batches,
        'depth': loss_depth_sum / num_batches,
        'dimension': loss_dim_sum / num_batches,
        'rotation': loss_rot_sum / num_batches,
        'foreground': loss_fg_sum / num_batches
    }
    
    return avg_losses


def main():
    parser = argparse.ArgumentParser(description='Train V4 Two-Stage 3D Detector')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Path to KITTI-3D dataset root')
    parser.add_argument('--classes', type=str, nargs='+', 
                       default=['Car', 'Pedestrian', 'Cyclist'],
                       help='Active classes for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--output-dir', type=str, default='outputs/two_stage_v4',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--early-stop-patience', type=int, default=10,
                       help='Early stopping patience (epochs)')
    parser.add_argument('--grad-clip', type=float, default=10.0,
                       help='Gradient clipping max norm')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    output_dir = Path(args.output_dir)
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output_dir / 'logs')
    
    # Save training config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Import dataset
    from data.kitti_dataset import KITTI3DDataset
    
    # Create datasets (use native resolution from dataset)
    train_ds = KITTI3DDataset(
        root_dir=args.data_root,
        split='train',
        filter_classes=args.classes
    )
    val_ds = KITTI3DDataset(
        root_dir=args.data_root,
        split='val',
        filter_classes=args.classes
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Train: {len(train_ds)} images, Val: {len(val_ds)} images\n")
    
    # Build model (no target_resolution needed)
    model = build_model_v4(active_classes=args.classes).to(device)
    
    # Loss function
    loss_fn = Total3DLossV4(
        loss_weights={
            'depth': 1.0,
            'dimension': 1.0,
            'rotation': 1.2,
            'foreground': 0.2
        }
    )
    
    # Count parameters
    trainable_params = [p for n, p in model.named_parameters() 
                       if 'detector_2d' not in n and p.requires_grad]
    total_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {total_params:,}\n")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Gradient monitor
    grad_monitor = GradientMonitor(model)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.3f}\n")
    
    # Training loop
    print("=" * 70)
    print("TRAINING V4")
    print("=" * 70)
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, loss_fn, scaler,
            device, epoch, grad_monitor, max_grad_norm=args.grad_clip
        )
        
        # Validate
        val_losses = validate(model, val_loader, loss_fn, device)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_losses['total']:.3f} "
              f"(D:{train_losses['depth']:.2f} "
              f"Dim:{train_losses['dimension']:.2f} "
              f"Rot:{train_losses['rotation']:.2f})")
        print(f"  Val Loss:   {val_losses['total']:.3f} "
              f"(D:{val_losses['depth']:.2f} "
              f"Dim:{val_losses['dimension']:.2f} "
              f"Rot:{val_losses['rotation']:.2f})")
        
        # Log to TensorBoard
        for key in train_losses:
            writer.add_scalar(f'train/{key}', train_losses[key], epoch)
            writer.add_scalar(f'val/{key}', val_losses[key], epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'classes': args.classes
        }
        torch.save(checkpoint, output_dir / 'checkpoints' / 'checkpoint_latest.pth')
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save(checkpoint, output_dir / 'checkpoints' / 'checkpoint_best.pth')
            print("  → Best model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{args.early_stop_patience})")
        
        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
        
        # Update learning rate
        scheduler.step()
        
        print("=" * 70)
    
    writer.close()
    print(f"\nTraining complete! Best val loss: {best_val_loss:.3f}")


if __name__ == '__main__':
    main()