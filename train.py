"""
Training Script for Monocular 3D Object Detection

Implements complete training pipeline:
- Multi-GPU support (DataParallel)
- Mixed precision training (AMP)
- Gradient accumulation
- Learning rate scheduling (OneCycleLR)
- Validation with early stopping
- Checkpoint management (save/resume)
- TensorBoard logging

Usage:
    python train.py --config configs/train_config.py
"""


import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
import json
from visualization import visualize_predictions, visualize_heatmap_grid
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.nuscenes_dataset import NuScenesMonocular3D
from models.monocular_3d_detector import build_model
from models.losses import DetectionLoss


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer:
    """Training pipeline orchestrator"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.start_epoch = 0
        self.best_loss = float('inf')
        
        # Setup directories
        self.setup_directories()
        
        # Initialize model
        print("Initializing model...")
        self.model = build_model(pretrained=False)
        self.model = self.model.to(self.device)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        # Initialize datasets
        print("Loading datasets...")
        self.train_dataset = NuScenesMonocular3D(
            root_dir=config['dataset']['root_dir'],
            split='train',
            input_size=config['model']['input_size'],
            augment=True
        )
        
        self.val_dataset = NuScenesMonocular3D(
            root_dir=config['dataset']['root_dir'],
            split='val',
            input_size=config['model']['input_size'],
            augment=False
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=True
        )
        
        # Loss function
        self.criterion = DetectionLoss(
            heatmap_weight=config['loss_weights']['heatmap'],
            depth_weight=config['loss_weights']['depth'],
            dimension_weight=config['loss_weights']['dimension'],
            rotation_weight=config['loss_weights']['rotation'],
            offset_weight=config['loss_weights']['offset']
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        

        # Learning rate scheduler with warmup
        total_steps = len(self.train_loader) * config['training']['epochs']
        warmup_steps = len(self.train_loader) * 3  # 3 epochs warmup

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['training']['learning_rate'],
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,  # Start LR = max_lr / 25
            final_div_factor=10000.0
        )
        
        # Mixed precision training
        self.scaler = GradScaler(enabled=config['training']['use_amp'])
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Resume from checkpoint if specified
        if config['training']['resume_checkpoint']:
            self.load_checkpoint(config['training']['resume_checkpoint'])
        
        print(f"\nTraining Configuration:")
        print(f"  Device: {self.device}")
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Val samples: {len(self.val_dataset)}")
        print(f"  Batch size: {config['training']['batch_size']}")
        print(f"  Epochs: {config['training']['epochs']}")
        print(f"  Learning rate: {config['training']['learning_rate']}")
        print(f"  Mixed precision: {config['training']['use_amp']}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}\n")
    
    def setup_directories(self):
        """Create output directories"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = self.config['experiment']['name']
        
        self.exp_dir = Path(self.config['experiment']['output_dir']) / f"{exp_name}_{timestamp}"
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.log_dir = self.exp_dir / 'logs'
        self.vis_dir = self.exp_dir / 'visualizations'
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = self.exp_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save periodic checkpoint
        if (epoch + 1) % self.config['training']['save_interval'] == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1:03d}.pth'
            torch.save(checkpoint, epoch_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            print(f"  → Saved best checkpoint (loss: {loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint to resume training"""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        
        print(f"  Resumed from epoch {self.start_epoch}, best_loss: {self.best_loss:.4f}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        # Loss meters
        loss_meters = {
            'total': AverageMeter(),
            'heatmap': AverageMeter(),
            'depth_bin': AverageMeter(),
            'depth_residual': AverageMeter(),
            'dimension': AverageMeter(),
            'rotation_bin': AverageMeter(),
            'rotation_residual': AverageMeter(),
            'offset': AverageMeter()
        }
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in batch.items() if k != 'image'}
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config['training']['use_amp']):
                predictions = self.model(images)
                losses = self.criterion(predictions, targets)

            # Check for invalid loss before backward
            if torch.isnan(losses['total_loss']) or torch.isinf(losses['total_loss']):
                print(f"\nWARNING: Invalid loss detected at iteration {batch_idx}, skipping batch")
                continue

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(losses['total_loss']).backward()

            # Gradient clipping
            if self.config['training']['grad_clip'] > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['grad_clip']
                )

            # Optimizer step with NaN check
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            # Update meters
            batch_size = images.size(0)
            loss_meters['total'].update(losses['total_loss'].item(), batch_size)
            loss_meters['heatmap'].update(losses['loss_heatmap'].item(), batch_size)
            loss_meters['depth_bin'].update(losses['loss_depth_bin'].item(), batch_size)
            loss_meters['depth_residual'].update(losses['loss_depth_residual'].item(), batch_size)
            loss_meters['dimension'].update(losses['loss_dimension'].item(), batch_size)
            loss_meters['rotation_bin'].update(losses['loss_rotation_bin'].item(), batch_size)
            loss_meters['rotation_residual'].update(losses['loss_rotation_residual'].item(), batch_size)
            loss_meters['offset'].update(losses['loss_offset'].item(), batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_meters['total'].avg:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Log to TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            if batch_idx % self.config['training']['log_interval'] == 0:
                self.writer.add_scalar('train/total_loss', loss_meters['total'].val, global_step)
                self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], global_step)
        
        # Epoch summary
        print(f"\n  Train Loss: {loss_meters['total'].avg:.4f}")
        print(f"    └─ Heatmap:    {loss_meters['heatmap'].avg:.4f}")
        print(f"    └─ Depth:      {loss_meters['depth_bin'].avg:.4f} + {loss_meters['depth_residual'].avg:.4f}")
        print(f"    └─ Dimension:  {loss_meters['dimension'].avg:.4f}")
        print(f"    └─ Rotation:   {loss_meters['rotation_bin'].avg:.4f} + {loss_meters['rotation_residual'].avg:.4f}")
        print(f"    └─ Offset:     {loss_meters['offset'].avg:.4f}")
        
        # Log epoch metrics
        for key, meter in loss_meters.items():
            self.writer.add_scalar(f'train_epoch/{key}_loss', meter.avg, epoch)
        
        return loss_meters['total'].avg
    
    def validate(self, epoch):
        """Validate on validation set"""
        self.model.eval()
        
        loss_meters = {
            'total': AverageMeter(),
            'heatmap': AverageMeter(),
            'depth_bin': AverageMeter(),
            'depth_residual': AverageMeter(),
            'dimension': AverageMeter(),
            'rotation_bin': AverageMeter(),
            'rotation_residual': AverageMeter(),
            'offset': AverageMeter()
        }
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validation")
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items() if k != 'image'}
                
                # Forward pass
                with autocast(enabled=self.config['training']['use_amp']):
                    predictions = self.model(images)
                    losses = self.criterion(predictions, targets)
                
                # Update meters
                batch_size = images.size(0)
                loss_meters['total'].update(losses['total_loss'].item(), batch_size)
                loss_meters['heatmap'].update(losses['loss_heatmap'].item(), batch_size)
                loss_meters['depth_bin'].update(losses['loss_depth_bin'].item(), batch_size)
                loss_meters['depth_residual'].update(losses['loss_depth_residual'].item(), batch_size)
                loss_meters['dimension'].update(losses['loss_dimension'].item(), batch_size)
                loss_meters['rotation_bin'].update(losses['loss_rotation_bin'].item(), batch_size)
                loss_meters['rotation_residual'].update(losses['loss_rotation_residual'].item(), batch_size)
                loss_meters['offset'].update(losses['loss_offset'].item(), batch_size)
                
                pbar.set_postfix({'val_loss': f"{loss_meters['total'].avg:.4f}"})
        
        # Validation summary
        print(f"\n  Val Loss: {loss_meters['total'].avg:.4f}")
        print(f"    └─ Heatmap:    {loss_meters['heatmap'].avg:.4f}")
        print(f"    └─ Depth:      {loss_meters['depth_bin'].avg:.4f} + {loss_meters['depth_residual'].avg:.4f}")
        print(f"    └─ Dimension:  {loss_meters['dimension'].avg:.4f}")
        print(f"    └─ Rotation:   {loss_meters['rotation_bin'].avg:.4f} + {loss_meters['rotation_residual'].avg:.4f}")
        print(f"    └─ Offset:     {loss_meters['offset'].avg:.4f}")
        
        # Log validation metrics
        for key, meter in loss_meters.items():
            self.writer.add_scalar(f'val/{key}_loss', meter.avg, epoch)
        
        return loss_meters['total'].avg
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60 + "\n")
        
        try:
            for epoch in range(self.start_epoch, self.config['training']['epochs']):
                epoch_start = time.time()
                
                # Train
                train_loss = self.train_epoch(epoch)
                
                # Validate
                val_loss = self.validate(epoch)
                if (epoch + 1) % self.config['training']['visualize_interval'] == 0:
                    print(f"\n  Generating visualizations...")
                    try:
                        vis_path = visualize_predictions(
                            self.model,
                            self.val_loader,
                            self.device,
                            self.vis_dir,
                            epoch,
                            num_samples=4,
                            score_threshold=0.3
                        )
                        # Log to TensorBoard
                        from PIL import Image
                        vis_img = Image.open(vis_path)
                        vis_tensor = torch.from_numpy(np.array(vis_img)).permute(2, 0, 1)
                        self.writer.add_image(f'detections/epoch_{epoch}', vis_tensor, epoch)
                    except Exception as e:
                        print(f"  Warning: Visualization failed: {e}")
                
                # Check if best model
                is_best = val_loss < self.best_loss
                if is_best:
                    self.best_loss = val_loss
                
                # Save checkpoint
                self.save_checkpoint(epoch, val_loss, is_best)
                
                epoch_time = time.time() - epoch_start
                print(f"\n  Epoch time: {epoch_time:.2f}s")
                print(f"  Best val loss: {self.best_loss:.4f}\n")
                print("="*60 + "\n")
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            print("Saving checkpoint...")
            val_loss = float('inf')  # Use inf if validation not completed
            self.save_checkpoint(epoch, val_loss, is_best=False)
        
        except Exception as e:
            print(f"\n\nError during training: {e}")
            import traceback
            traceback.print_exc()
            print("Saving checkpoint...")
            val_loss = float('inf')  # Use inf if validation not completed
            self.save_checkpoint(epoch, val_loss, is_best=False)
            raise
        
        finally:
            self.writer.close()
            print("\nTraining completed!")
            print(f"Best model saved at: {self.checkpoint_dir / 'checkpoint_best.pth'}")
            print(f"Final validation loss: {self.best_loss:.4f}")


def get_default_config():
    """Default training configuration"""
    return {
        'experiment': {
            'name': 'mono3d_nuscenes',
            'output_dir': './outputs/experiments'
        },
        'dataset': {
            'root_dir': '/media/ashwin-benchdev/*/NuScenes-Full-Dataset'
        },
        'model': {
            'input_size': (720, 1280)
        },
        'training': {
            'batch_size': 8,
            'epochs': 50,
            'learning_rate': 2e-5,
            'weight_decay': 1e-4,
            'num_workers': 4,
            'use_amp': True,
            'grad_clip': 5.0,
            'save_interval': 5,
            'log_interval': 50,
            'visualize_interval': 5,
            'resume_checkpoint': None
        },
        'loss_weights': {
            'heatmap': 1.0,
            'depth': 1.0,
            'dimension': 0.1,
            'rotation': 0.5,
            'offset': 0.1
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Train Monocular 3D Detection')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--name', type=str, default='mono3d_nuscenes', help='Experiment name')
    
    args = parser.parse_args()
    
    # Load config
    config = get_default_config()
    
    # Override with command line arguments
    config['experiment']['name'] = args.name
    config['training']['batch_size'] = args.batch_size
    config['training']['epochs'] = args.epochs
    config['training']['learning_rate'] = args.lr
    config['training']['resume_checkpoint'] = args.resume
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
