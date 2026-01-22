#!/usr/bin/env python3
"""Export Stage 2 (3D heads) to ONNX with proper preprocessing"""

import torch
import torch.nn as nn
from pathlib import Path
import argparse


class Stage2HeadsOnly(nn.Module):
    """Stage 2 heads without YOLO - accepts preprocessed images + 2D boxes"""
    def __init__(self, checkpoint_path):
        super().__init__()
        
        # Load full model
        from models.two_stage_detector import build_model
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'classes' in checkpoint:
            classes = checkpoint['classes']
        else:
            classes = ['Car']
        
        full_model = build_model(active_classes=classes)
        full_model.load_state_dict(checkpoint['model_state_dict'])
        full_model.eval()
        
        # Extract only the 3D prediction components
        self.feature_backbone = full_model.feature_backbone
        self.roi_align = full_model.roi_align
        self.avgpool = full_model.avgpool
        self.depth_head = full_model.depth_head
        self.dimension_head = full_model.dimension_head
        self.rotation_head = full_model.rotation_head
        
        print("✓ Extracted Stage 2 heads from checkpoint")
        print(f"  Classes: {classes}")
    
    def forward(self, images, boxes_2d):
        """
        Args:
            images: [1, 3, H, W] - Preprocessed RGB image (ImageNet normalized)
            boxes_2d: [100, 5] - Format: [batch_idx, x1, y1, x2, y2]
                                 Padded with zeros if < 100 boxes
        
        Returns:
            depth: [100, 3] - [depth, log_var, offset]
            dimensions: [100, 3] - [h, w, l]
            rotation_bins: [100, 12] - Bin classification logits
            rotation_res: [100, 12] - Residual for each bin
        """
        batch_size = images.shape[0]
        
        # Extract features
        features = self.feature_backbone(images)  # [1, 2048, H/32, W/32]
        
        # RoI Align
        roi_features = self.roi_align(features, boxes_2d)  # [100, 2048, 7, 7]
        roi_features = self.avgpool(roi_features)  # [100, 2048, 1, 1]
        roi_features = roi_features.squeeze(-1).squeeze(-1)  # [100, 2048]
        
        # 3D prediction heads
        depth, depth_log_var, depth_offset = self.depth_head(roi_features)
        depth_output = torch.stack([depth, depth_log_var, depth_offset], dim=1)  # [100, 3]
        
        dims = self.dimension_head(roi_features)  # [100, 3]
        
        rot_bins, rot_res = self.rotation_head(roi_features)  # [100, 12], [100, 12]
        
        return depth_output, dims, rot_bins, rot_res


def export_to_onnx(checkpoint_path, output_path, img_height=640, img_width=640, max_boxes=100):
    """Export Stage 2 heads to ONNX"""
    
    print(f"\n{'='*80}")
    print("EXPORTING STAGE 2 (3D HEADS) TO ONNX")
    print(f"{'='*80}\n")
    
    # Create model
    model = Stage2HeadsOnly(checkpoint_path)
    model.eval()
    
    # Dummy inputs
    dummy_image = torch.randn(1, 3, img_height, img_width)
    dummy_boxes = torch.zeros(max_boxes, 5)  # [batch_idx, x1, y1, x2, y2]
    
    # Add some realistic boxes for trace
    dummy_boxes[0] = torch.tensor([0, 100, 100, 200, 200])  # batch=0, box coordinates
    dummy_boxes[1] = torch.tensor([0, 300, 150, 450, 300])
    
    print(f"Input shapes:")
    print(f"  images: {tuple(dummy_image.shape)} - [batch, channels, height, width]")
    print(f"  boxes_2d: {tuple(dummy_boxes.shape)} - [max_boxes, 5]")
    
    # Test forward pass
    with torch.no_grad():
        depth, dims, rot_bins, rot_res = model(dummy_image, dummy_boxes)
    
    print(f"\nOutput shapes:")
    print(f"  depth: {tuple(depth.shape)} - [max_boxes, 3] (depth, log_var, offset)")
    print(f"  dimensions: {tuple(dims.shape)} - [max_boxes, 3] (h, w, l)")
    print(f"  rotation_bins: {tuple(rot_bins.shape)} - [max_boxes, 12]")
    print(f"  rotation_res: {tuple(rot_res.shape)} - [max_boxes, 12]")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_image, dummy_boxes),
        output_path,
        input_names=['images', 'boxes_2d'],
        output_names=['depth', 'dimensions', 'rotation_bins', 'rotation_res'],
        dynamic_axes={
            # Static shapes - no dynamic axes for TensorRT optimization
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"\n✓ ONNX model exported to: {output_path}")
    
    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verification passed")
    
    # Print model info
    print(f"\nONNX Model Info:")
    print(f"  IR version: {onnx_model.ir_version}")
    print(f"  Producer: {onnx_model.producer_name}")
    print(f"  Opset: {onnx_model.opset_import[0].version}")
    
    print(f"\n{'='*80}")
    print("PREPROCESSING REQUIREMENTS FOR DRIVEWORKS:")
    print(f"{'='*80}")
    print("""
This ONNX model expects PREPROCESSED inputs:

1. Images [1, 3, 640, 640]:
   - Scale: pixel / 255.0 → [0, 1]
   - Normalize with ImageNet statistics:
     * mean = [0.485, 0.456, 0.406]
     * std  = [0.229, 0.224, 0.225]
   - Formula: (pixel/255 - mean) / std

2. Boxes [100, 5]:
   - Format: [batch_idx, x1, y1, x2, y2]
   - Coordinates in image space [0, 640]
   - Pad with zeros if < 100 boxes

DataConditioner JSON (stage2_3d_heads.bin.json):
{
  "dataConditionerParams" : {
    "meanValue" : [123.675, 116.28, 103.53],
    "stdev" : [58.395, 57.12, 57.375],
    "splitPlanes" : true,
    "pixelScaleCoefficient": 0.003921568627,
    "ignoreAspectRatio" : true,
    "doPerPlaneMeanNormalization" : false
  }
}
""")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export Stage 2 heads to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to .pth checkpoint')
    parser.add_argument('--output', type=str, default='stage2_3d_heads.onnx',
                       help='Output ONNX path')
    parser.add_argument('--img-height', type=int, default=640)
    parser.add_argument('--img-width', type=int, default=640)
    parser.add_argument('--max-boxes', type=int, default=100)
    
    args = parser.parse_args()
    
    export_to_onnx(
        args.checkpoint,
        args.output,
        args.img_height,
        args.img_width,
        args.max_boxes
    )


if __name__ == '__main__':
    main()
