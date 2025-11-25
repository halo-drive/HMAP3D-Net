"""Diagnostic Script - Analyze Two-Stage 3D Detection Pipeline"""

import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models.two_stage_detector import build_model


def diagnose_video(checkpoint_path, video_path, num_frames=50, conf_threshold=0.5):
    """Diagnose issues in 3D detection pipeline"""
    
    print("\n" + "="*70)
    print("TWO-STAGE 3D DETECTION - DIAGNOSTIC ANALYSIS")
    print("="*70 + "\n")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    classes = checkpoint.get('classes', ['Car'])
    print(f"Model trained on classes: {classes}")
    print(f"Training checkpoint: Epoch {checkpoint['epoch']}, Loss {checkpoint['best_val_loss']:.4f}\n")
    
    model = build_model(active_classes=classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Analyzing first {num_frames} frames\n")
    
    # Normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # Statistics
    stats = {
        'yolo_detections': [],
        'yolo_filtered': [],
        'final_3d': [],
        'depth_values': [],
        'dimension_values': [],
        'confidence_scores': [],
        'class_distribution': {cls: 0 for cls in classes}
    }
    
    print("="*70)
    print("FRAME-BY-FRAME ANALYSIS")
    print("="*70)
    
    for frame_idx in tqdm(range(num_frames), desc="Analyzing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        target_h, target_w = 384, 1280
        
        frame_resized = cv2.resize(frame_rgb, (target_w, target_h))
        frame_norm = frame_resized.astype(np.float32) / 255.0
        frame_norm = (frame_norm - mean) / std
        frame_tensor = torch.from_numpy(frame_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        
        # Create dummy intrinsics (KITTI default)
        K = torch.tensor([
            [721.5377, 0.0, 609.5593],
            [0.0, 721.5377, 172.854],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=device)
        
        # Inference
        with torch.no_grad():
            predictions = model(frame_tensor, K.unsqueeze(0), gt_boxes_2d=None)
        
        # Extract stats
        boxes_2d = predictions['boxes_2d'][0]
        scores_2d = predictions['scores_2d'][0]
        depth_pred = predictions['depth'][0]
        dims_pred = predictions['dimensions'][0]
        classes_pred = predictions['classes'][0]
        
        # Count YOLO raw detections (before class filtering)
        # Note: We can't directly see YOLO raw output after class filtering
        # But we can infer from the final output
        
        num_yolo_raw = len(boxes_2d)  # After class filtering
        num_yolo_filtered = num_yolo_raw  # Same as above in current impl
        
        # Filter by confidence
        conf_mask = scores_2d >= conf_threshold
        num_final = conf_mask.sum().item()
        
        stats['yolo_detections'].append(num_yolo_raw)
        stats['yolo_filtered'].append(num_yolo_filtered)
        stats['final_3d'].append(num_final)
        
        if num_final > 0:
            # Get depth values
            depth_vals = (depth_pred[:, 0] + depth_pred[:, 2]).cpu().numpy()  # depth + offset
            stats['depth_values'].extend(depth_vals[conf_mask.cpu().numpy()].tolist())
            
            # Get dimensions
            dims_vals = dims_pred.cpu().numpy()
            stats['dimension_values'].extend(dims_vals[conf_mask.cpu().numpy()].tolist())
            
            # Get confidence scores
            stats['confidence_scores'].extend(scores_2d[conf_mask].cpu().numpy().tolist())
            
            # Class distribution
            for cls in classes_pred:
                if cls in stats['class_distribution']:
                    stats['class_distribution'][cls] += 1
        
        # Print sample frames
        if frame_idx in [0, 10, 20, 30, 40]:
            print(f"\nFrame {frame_idx}:")
            print(f"  YOLO 2D detections: {num_yolo_raw}")
            print(f"  After confidence filter (>{conf_threshold}): {num_final}")
            if num_final > 0:
                print(f"  Classes: {classes_pred}")
                print(f"  Confidence range: {scores_2d[conf_mask].min():.2f} - {scores_2d[conf_mask].max():.2f}")
                print(f"  Depth range: {depth_vals[conf_mask.cpu().numpy()].min():.1f}m - {depth_vals[conf_mask.cpu().numpy()].max():.1f}m")
    
    cap.release()
    
    # Print summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    avg_yolo = np.mean(stats['yolo_detections'])
    avg_filtered = np.mean(stats['yolo_filtered'])
    avg_final = np.mean(stats['final_3d'])
    
    print(f"\n Detection Statistics (per frame):")
    print(f"  Average YOLO 2D detections: {avg_yolo:.1f}")
    print(f"  Average after class filter: {avg_filtered:.1f}")
    print(f"  Average final 3D (conf>{conf_threshold}): {avg_final:.1f}")
    
    if len(stats['depth_values']) > 0:
        print(f"\n 3D Attribute Statistics:")
        print(f"  Depth: {np.mean(stats['depth_values']):.1f}m ± {np.std(stats['depth_values']):.1f}m")
        print(f"    Range: {np.min(stats['depth_values']):.1f}m - {np.max(stats['depth_values']):.1f}m")
        
        dims = np.array(stats['dimension_values'])
        print(f"  Dimensions (H,W,L):")
        print(f"    Height: {dims[:, 0].mean():.2f}m ± {dims[:, 0].std():.2f}m")
        print(f"    Width:  {dims[:, 1].mean():.2f}m ± {dims[:, 1].std():.2f}m")
        print(f"    Length: {dims[:, 2].mean():.2f}m ± {dims[:, 2].std():.2f}m")
    
    print(f"\n Class Distribution:")
    for cls, count in stats['class_distribution'].items():
        print(f"  {cls}: {count} detections")
    
    if len(stats['confidence_scores']) > 0:
        print(f"\n Confidence Scores:")
        print(f"  Mean: {np.mean(stats['confidence_scores']):.3f}")
        print(f"  Median: {np.median(stats['confidence_scores']):.3f}")
        print(f"  Range: {np.min(stats['confidence_scores']):.3f} - {np.max(stats['confidence_scores']):.3f}")
    
    # Diagnosis
    print("\n" + "="*70)
    print("🔍 DIAGNOSIS & RECOMMENDATIONS")
    print("="*70)
    
    if avg_yolo < 1.0:
        print("\n CRITICAL: Very few YOLO 2D detections!")
        print("   Possible causes:")
        print("   1. Video content doesn't match YOLO's COCO classes")
        print("   2. Video quality/lighting issues")
        print("   3. Objects too small/far/occluded")
        print("   Recommendation:")
        print("   - Check what YOLO detects on a few frames manually")
        print("   - Try lowering YOLO confidence threshold in model code")
        print("   - Consider fine-tuning YOLO on your video domain")
    
    elif avg_final < 0.5:
        print("\n  WARNING: YOLO finds objects but confidence is low")
        print("   Possible causes:")
        print("   1. 3D regression heads not confident in predictions")
        print("   2. Model overfitted to KITTI, poor generalization")
        print("   3. Confidence threshold too high")
        print("   Recommendation:")
        print("   - Lower confidence threshold to 0.3 or 0.2")
        print("   - Check if predictions are reasonable (even if low confidence)")
        print("   - May need more diverse training data")
    
    else:
        print("\n✓ Detection pipeline working normally")
        print("  If boxes still look wrong, check:")
        print("  - Camera intrinsics calibration")
        print("  - Depth scale calibration")
        print("  - Rotation convention")
    
    if len(stats['depth_values']) > 0:
        avg_depth = np.mean(stats['depth_values'])
        if avg_depth > 50:
            print("\n  WARNING: Very large depth values (>50m)")
            print("   This suggests depth estimation is off")
            print("   Check:")
            print("   - Camera intrinsics (fx, fy might be wrong)")
            print("   - Depth head may need recalibration")
        elif avg_depth < 5:
            print("\n  WARNING: Very small depth values (<5m)")
            print("   Objects appear too close")
            print("   Check camera intrinsics and training data scale")
    
    print("\n" + "="*70)
    print(" NEXT STEPS")
    print("="*70)
    print("\n1. Run this diagnostic on your training video too:")
    print("   python3 diagnose_pipeline.py --checkpoint <CKPT> --video <TRAIN_VIDEO>")
    print("\n2. Compare results: Do you get similar detection rates?")
    print("\n3. Try lowering confidence threshold:")
    print("   python3 inference_video.py ... --conf-threshold 0.3")
    print("\n4. Visualize a few frames to see what YOLO detects")
    print("\n5. Check if camera intrinsics match your video")


def main():
    parser = argparse.ArgumentParser(description='Diagnose 3D detection pipeline issues')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--num-frames', type=int, default=50)
    parser.add_argument('--conf-threshold', type=float, default=0.5)
    
    args = parser.parse_args()
    
    diagnose_video(
        args.checkpoint,
        args.video,
        args.num_frames,
        args.conf_threshold
    )


if __name__ == "__main__":
    main()