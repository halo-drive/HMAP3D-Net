"""Visual Debugging - Compare YOLO 2D vs Final 3D Detections"""

import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from models.two_stage_detector import build_model
from ultralytics import YOLO


def visualize_comparison(frame_path, checkpoint_path, output_path, conf_threshold=0.5):
    """
    Create side-by-side comparison:
    1. Raw YOLO 2D detections
    2. YOLO after class filtering
    3. Final 3D projections
    """
    
    # Load frame
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Error: Cannot load image {frame_path}")
        return
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    
    print(f"Loaded frame: {w}x{h}")
    
    # 1. Test YOLO directly (without any filtering)
    print("\n" + "="*60)
    print("STEP 1: Raw YOLO Detection (No Filtering)")
    print("="*60)
    
    yolo = YOLO('yolov8x.pt')
    yolo_results = yolo.predict(frame_rgb, verbose=False, conf=0.25)
    
    yolo_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    yolo_scores = yolo_results[0].boxes.conf.cpu().numpy()
    yolo_classes = yolo_results[0].boxes.cls.cpu().numpy().astype(int)
    
    # COCO class names
    coco_names = yolo_results[0].names
    
    print(f"YOLO found {len(yolo_boxes)} objects:")
    for i in range(len(yolo_boxes)):
        cls_id = yolo_classes[i]
        cls_name = coco_names[cls_id]
        score = yolo_scores[i]
        print(f"  {i+1}. {cls_name} (conf: {score:.3f})")
    
    # Draw YOLO detections
    img1 = frame_rgb.copy()
    for i in range(len(yolo_boxes)):
        box = yolo_boxes[i].astype(int)
        score = yolo_scores[i]
        cls_name = coco_names[yolo_classes[i]]
        
        cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
        label = f'{cls_name} {score:.2f}'
        cv2.putText(img1, label, (box[0], box[1]-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # 2. Load 3D model and see what it detects
    print("\n" + "="*60)
    print("STEP 2: 3D Model Detection (With Class Filtering)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint.get('classes', ['Car'])
    
    print(f"Model trained on: {classes}")
    
    model = build_model(active_classes=classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Preprocess
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    target_h, target_w = 384, 1280
    frame_resized = cv2.resize(frame_rgb, (target_w, target_h))
    frame_norm = frame_resized.astype(np.float32) / 255.0
    frame_norm = (frame_norm - mean) / std
    frame_tensor = torch.from_numpy(frame_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    
    # KITTI default intrinsics
    K = torch.tensor([
        [721.5377, 0.0, 609.5593],
        [0.0, 721.5377, 172.854],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device=device)
    
    # Inference
    with torch.no_grad():
        predictions = model(frame_tensor, K.unsqueeze(0), gt_boxes_2d=None)
    
    boxes_2d = predictions['boxes_2d'][0]
    scores_2d = predictions['scores_2d'][0]
    classes_pred = predictions['classes'][0]
    
    print(f"After class filtering: {len(boxes_2d)} objects")
    for i in range(len(boxes_2d)):
        print(f"  {i+1}. {classes_pred[i]} (conf: {scores_2d[i]:.3f})")
    
    # Scale boxes back to original size
    scale_x = w / target_w
    scale_y = h / target_h
    
    img2 = frame_rgb.copy()
    for i in range(len(boxes_2d)):
        box = boxes_2d[i].cpu().numpy()
        box[[0, 2]] *= scale_x
        box[[1, 3]] *= scale_y
        box = box.astype(int)
        
        score = scores_2d[i].cpu().item()
        cls = classes_pred[i]
        
        color = (0, 255, 0) if cls == 'Car' else (255, 165, 0)
        cv2.rectangle(img2, (box[0], box[1]), (box[2], box[3]), color, 2)
        label = f'{cls} {score:.2f}'
        cv2.putText(img2, label, (box[0], box[1]-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 3. Final detections after confidence filtering
    print("\n" + "="*60)
    print(f"STEP 3: Final Detections (conf > {conf_threshold})")
    print("="*60)
    
    conf_mask = scores_2d >= conf_threshold
    num_final = conf_mask.sum().item()
    
    print(f"After confidence filter: {num_final} objects")
    
    img3 = frame_rgb.copy()
    for i in range(len(boxes_2d)):
        if not conf_mask[i]:
            continue
        
        box = boxes_2d[i].cpu().numpy()
        box[[0, 2]] *= scale_x
        box[[1, 3]] *= scale_y
        box = box.astype(int)
        
        score = scores_2d[i].cpu().item()
        cls = classes_pred[i]
        
        color = (0, 255, 0) if cls == 'Car' else (255, 165, 0)
        cv2.rectangle(img3, (box[0], box[1]), (box[2], box[3]), color, 3)
        label = f'{cls} {score:.2f}'
        cv2.putText(img3, label, (box[0], box[1]-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    axes[0].imshow(img1)
    axes[0].set_title(f'1. Raw YOLO\n({len(yolo_boxes)} detections)', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(img2)
    axes[1].set_title(f'2. After Class Filter\n({len(boxes_2d)} detections)', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(img3)
    axes[2].set_title(f'3. After Conf Filter (>{conf_threshold})\n({num_final} detections)', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison to: {output_path}")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    if len(yolo_boxes) == 0:
        print("\n CRITICAL: YOLO found NO objects!")
        print("   Your video may not contain recognizable objects")
        print("   Or YOLO confidence is too high (default 0.25)")
    elif len(boxes_2d) == 0:
        print("\n  WARNING: All YOLO detections filtered out!")
        print("   YOLO found objects, but none match trained classes:")
        print(f"   Trained classes: {classes}")
        print("   Try training on more classes or check if video has these objects")
    elif num_final == 0:
        print("\n  WARNING: All detections have low confidence!")
        print(f"   {len(boxes_2d)} objects detected but all below {conf_threshold}")
        print("   Solutions:")
        print("   1. Lower confidence threshold: --conf-threshold 0.3")
        print("   2. Model may be overfitting to KITTI")
        print("   3. Check camera intrinsics")
    else:
        print(f"\n✓ Pipeline working: {num_final} final detections")
        print("   If boxes still look wrong, check:")
        print("   - Camera intrinsics")
        print("   - Depth/dimension calibration")
    
    return {
        'yolo_raw': len(yolo_boxes),
        'after_class_filter': len(boxes_2d),
        'after_conf_filter': num_final
    }


def main():
    parser = argparse.ArgumentParser(description='Visual debugging of 2D to 3D pipeline')
    parser.add_argument('--frame', type=str, required=True, help='Path to test frame')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--output', type=str, default='pipeline_debug.png',
                       help='Output comparison image')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Final confidence threshold')
    
    args = parser.parse_args()
    
    visualize_comparison(args.frame, args.checkpoint, args.output, args.conf_threshold)


if __name__ == "__main__":
    main()