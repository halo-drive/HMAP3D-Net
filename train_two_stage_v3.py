"""Training Script V3 - Fixed with progress updates"""
import os, sys, argparse, json
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from data.kitti_dataset import KITTI3DDataset
from models.two_stage_detector_v3 import build_model_v3
from models.losses_two_stage_v3 import Total3DLossV3

def augment_intrinsics(K, augment_range=0.3):
    K_aug = K.clone()
    for i in range(K.shape[0]):
        fx_s = torch.empty(1, device=K.device).uniform_(1-augment_range, 1+augment_range).item()
        fy_s = torch.empty(1, device=K.device).uniform_(1-augment_range, 1+augment_range).item()
        K_aug[i,0,0] *= fx_s; K_aug[i,1,1] *= fy_s
        cx_s = torch.empty(1, device=K.device).uniform_(1-augment_range*0.5, 1+augment_range*0.5).item()
        cy_s = torch.empty(1, device=K.device).uniform_(1-augment_range*0.5, 1+augment_range*0.5).item()
        K_aug[i,0,2] *= cx_s; K_aug[i,1,2] *= cy_s
    return K_aug

def collate_fn(batch):
    return {'images': torch.stack([item['image'] for item in batch]),
            'boxes_2d': [item['boxes_2d'] for item in batch],
            'boxes_3d': [item['boxes_3d'] for item in batch],
            'labels': [item['labels'] for item in batch],
            'intrinsics': torch.stack([item['intrinsics'] for item in batch]),
            'img_indices': [item['img_idx'] for item in batch]}

def prepare_targets(batch_boxes_3d):
    all_d, all_dim, all_rot, all_b3d = [], [], [], []
    for b in batch_boxes_3d:
        if len(b) > 0:
            all_d.append(b[:,2]); all_dim.append(b[:,3:6]); all_rot.append(b[:,6]); all_b3d.append(b)
    if len(all_d) > 0:
        return {'depth': torch.cat(all_d), 'dimensions': torch.cat(all_dim),
                'rotation': torch.cat(all_rot), 'boxes_3d': torch.cat(all_b3d),
                'foreground': torch.ones(sum(len(d) for d in all_d))}
    return {'depth': torch.zeros(0), 'dimensions': torch.zeros(0,3),
            'rotation': torch.zeros(0), 'boxes_3d': torch.zeros(0,7), 'foreground': torch.zeros(0)}

def train_epoch(model, dataloader, optimizer, loss_fn, scaler, device, epoch):
    model.train()
    tl, ld, ldim, lr, li, lf, nb = 0,0,0,0,0,0,0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        imgs, b2d, b3d, K = batch['images'].to(device), batch['boxes_2d'], batch['boxes_3d'], batch['intrinsics'].to(device)
        K = augment_intrinsics(K, 0.3)
        with autocast():
            preds = model(imgs, K, gt_boxes_2d=b2d)
            tgts = prepare_targets(b3d)
            if len(tgts['depth']) == 0: continue
            tgts = {k: v.to(device) for k,v in tgts.items()}
            pd = torch.cat(preds['depth'])
            pdim = torch.cat(preds['dimensions'])
            prb = torch.cat([r[0] for r in preds['rotation']])
            prr = torch.cat([r[1] for r in preds['rotation']])
            piou = torch.cat(preds['iou'])
            pfg = torch.cat(preds['foreground'])
            pred_dict = {'depth': pd, 'dimensions': pdim, 'rotation': (prb, prr), 'iou': piou, 'foreground': pfg}
            loss, ldict = loss_fn(pred_dict, tgts)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer); scaler.update()
        tl += loss.item(); ld += ldict['loss_depth']; ldim += ldict['loss_dimension']
        lr += ldict['loss_rotation']; li += ldict['loss_iou']; lf += ldict['loss_foreground']; nb += 1
        # ADDED: Update progress bar with current losses
        pbar.set_postfix({'loss': f"{loss.item():.3f}", 'depth': f"{ldict['loss_depth']:.2f}", 
                         'dim': f"{ldict['loss_dimension']:.2f}", 'rot': f"{ldict['loss_rotation']:.2f}"})
    return {'total': tl/nb, 'depth': ld/nb, 'dimension': ldim/nb, 'rotation': lr/nb, 'iou': li/nb, 'foreground': lf/nb}

@torch.no_grad()
def validate(model, dataloader, loss_fn, device):
    model.eval()
    tl, ld, ldim, lr, li, lf, nb = 0,0,0,0,0,0,0
    for batch in tqdm(dataloader, desc="Validation"):
        imgs, b2d, b3d, K = batch['images'].to(device), batch['boxes_2d'], batch['boxes_3d'], batch['intrinsics'].to(device)
        with autocast():
            preds = model(imgs, K, gt_boxes_2d=b2d)
            tgts = prepare_targets(b3d)
            if len(tgts['depth']) == 0: continue
            tgts = {k: v.to(device) for k,v in tgts.items()}
            pd = torch.cat(preds['depth'])
            pdim = torch.cat(preds['dimensions'])
            prb = torch.cat([r[0] for r in preds['rotation']])
            prr = torch.cat([r[1] for r in preds['rotation']])
            piou = torch.cat(preds['iou'])
            pfg = torch.cat(preds['foreground'])
            pred_dict = {'depth': pd, 'dimensions': pdim, 'rotation': (prb, prr), 'iou': piou, 'foreground': pfg}
            loss, ldict = loss_fn(pred_dict, tgts)
        tl += loss.item(); ld += ldict['loss_depth']; ldim += ldict['loss_dimension']
        lr += ldict['loss_rotation']; li += ldict['loss_iou']; lf += ldict['loss_foreground']; nb += 1
    return {'total': tl/nb, 'depth': ld/nb, 'dimension': ldim/nb, 'rotation': lr/nb, 'iou': li/nb, 'foreground': lf/nb}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--classes', type=str, nargs='+', default=['Car','Pedestrian','Cyclist'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='outputs/two_stage_v3')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    output_dir = Path(args.output_dir)
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output_dir / 'logs')
    
    train_ds = KITTI3DDataset(root_dir=args.data_root, split='train', filter_classes=args.classes)
    val_ds = KITTI3DDataset(root_dir=args.data_root, split='val', filter_classes=args.classes)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}\n")
    
    model = build_model_v3(active_classes=args.classes).to(device)
    loss_fn = Total3DLossV3(loss_weights={'depth': 1.0, 'dimension': 1.0, 'rotation': 1.5, 'iou': 0.0, 'foreground': 0.3})
    
    trainable = [p for n,p in model.named_parameters() if 'detector_2d' not in n and '_yolo' not in n]
    print(f"Params: {sum(p.numel() for p in trainable):,}\n")
    
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = GradScaler()
    
    start_ep, best_val = 0, float('inf')
    if args.resume:
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        start_ep, best_val = ckpt['epoch']+1, ckpt['best_val_loss']
    
    print("Training V3...\n" + "="*60)
    for ep in range(start_ep, args.epochs):
        tr = train_epoch(model, train_ld, opt, loss_fn, scaler, device, ep)
        va = validate(model, val_ld, loss_fn, device)
        print(f"\nEp {ep}: Train {tr['total']:.3f} (D:{tr['depth']:.2f} Dim:{tr['dimension']:.2f} Rot:{tr['rotation']:.2f})")
        print(f"        Val   {va['total']:.3f} (D:{va['depth']:.2f} Dim:{va['dimension']:.2f} Rot:{va['rotation']:.2f})")
        
        # FIXED: TensorBoard logging
        for k in tr: 
            writer.add_scalar(f'train/{k}', tr[k], ep)
            writer.add_scalar(f'val/{k}', va[k], ep)
        writer.add_scalar('learning_rate', opt.param_groups[0]['lr'], ep)
        
        ckpt = {'epoch': ep, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(),
                'train_losses': tr, 'val_losses': va, 'best_val_loss': best_val, 'classes': args.classes}
        torch.save(ckpt, output_dir / 'checkpoints' / 'checkpoint_latest.pth')
        
        if va['total'] < best_val:
            best_val = va['total']
            torch.save(ckpt, output_dir / 'checkpoints' / 'checkpoint_best.pth')
            print("  → Best!")
        sched.step()
        print("="*60)
    writer.close()
    print(f"\nDone! Best: {best_val:.3f}")

if __name__ == "__main__": main()