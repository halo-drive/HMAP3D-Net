"""
Monocular 3D Object Detection Network
Architecture: DLA-34 backbone + Multi-head detection

Target platform: NVIDIA Drive AGX Orin (DLA-compatible operations only)
Input: 1280x720x3 RGB images (scaled from nuScenes 1600x900)
Output: 3D bounding boxes in camera coordinate frame

Network outputs (stride 4, 180x320 feature maps):
- Heatmap: 1 channel (vehicle center detection)
- Depth bins: 6 channels (depth classification: 0-10, 10-20, 20-35, 35-50, 50-70, 70+m)
- Depth residual: 6 channels (continuous offset per bin)
- Dimensions: 3 channels (height, width, length in meters)
- Rotation bins: 8 channels (yaw angle in 45° sectors)
- Rotation residual: 8 channels (fine angle adjustment per bin)
- 2D offset: 2 channels (sub-pixel center localization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math


def fill_up_weights(up):
    """Initialize upsampling weights for bilinear interpolation"""
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class BasicBlock(nn.Module):
    """Basic residual block for DLA"""
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                              stride=stride, padding=dilation,
                              bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                              stride=1, padding=dilation,
                              bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    """Root node for aggregating features in DLA"""
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    """Hierarchical tree structure for DLA"""
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                              dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                              dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                            stride, root_dim=0,
                            root_kernel_size=root_kernel_size,
                            dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                            root_dim=root_dim + out_channels,
                            root_kernel_size=root_kernel_size,
                            dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                           root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA34(nn.Module):
    """
    Deep Layer Aggregation - 34 layers
    Original paper: https://arxiv.org/abs/1707.06484
    
    Architecture designed for dense prediction tasks with hierarchical
    feature aggregation. Suitable for NVIDIA DLA acceleration (standard ops).
    """
    def __init__(self, levels=[1, 1, 1, 2, 2, 1],
                 channels=[16, 32, 64, 128, 256, 512],
                 block=BasicBlock):
        super(DLA34, self).__init__()
        self.channels = channels
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                     padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True))
        
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                          level_root=False, root_residual=True)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                          level_root=True, root_residual=True)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                          level_root=True, root_residual=True)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                          level_root=True, root_residual=True)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                         stride=stride if i == 0 else 1,
                         padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        """
        Forward pass through DLA-34 backbone
        
        Args:
            x: Input tensor (B, 3, 720, 1280)
        
        Returns:
            features: Output feature map (B, 512, 180, 320) at stride 4
        """
        y = []
        x = self.base_layer(x)  # (B, 16, 720, 1280)
        for i in range(3):
            x = getattr(self, f'level{i}')(x)
            y.append(x)
        
        # Output from level2 has stride 4 relative to input
        # level2 output channels = 64 (from channels[2])
        return x  # (B, 64, 180, 320)


class DeformableConvUpsampler(nn.Module):
    """
    Standard upsampling module (NO deformable convolutions for DLA compatibility)
    Uses transposed convolution for learnable upsampling.
    """
    def __init__(self, in_channels, out_channels):
        super(DeformableConvUpsampler, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=4, stride=2, padding=1,
            output_padding=0, bias=False)
        fill_up_weights(self.up)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MonocularDetectionHead(nn.Module):
    """
    Multi-task detection head for monocular 3D object detection
    
    All heads use standard convolutions (DLA-compatible):
    - 3x3 conv (feature extraction)
    - 1x1 conv (projection to output channels)
    """
    def __init__(self, in_channels=64):
        super(MonocularDetectionHead, self).__init__()
        
        # Shared feature extraction (optional, can be per-head)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Heatmap head: Object center detection (1 class: vehicle)
        self.heatmap = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, bias=True)
        )
        # Initialize with bias for sigmoid output
        self.heatmap[-1].bias.data.fill_(-2.19)  # Prior for sigmoid (low initial response)
        
        # Depth prediction head (bins + residuals)
        self.depth_bin = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 6, kernel_size=1, bias=True)  # 6 depth bins
        )
        
        self.depth_residual = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 6, kernel_size=1, bias=True)  # Residual per bin
        )
        
        # Dimension head: 3D box dimensions (h, w, l)
        self.dimensions = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1, bias=True)
        )
        
        # Rotation head (bins + residuals)
        self.rotation_bin = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 8, kernel_size=1, bias=True)  # 8 bins (45° sectors)
        )
        
        self.rotation_residual = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 8, kernel_size=1, bias=True)
        )
        
        # 2D offset head: Sub-pixel center localization
        self.offset_2d = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, bias=True)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None and m != self.heatmap[-1]:
                    nn.init.constant_(m.bias, 0)
                # Kaiming/He initialization for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through all detection heads
        
        Args:
            x: Feature map from backbone (B, 512, H, W)
        
        Returns:
            Dictionary containing all predictions:
            {
                'heatmap': (B, 1, H, W) - sigmoid probabilities
                'depth_bin': (B, 6, H, W) - depth bin logits
                'depth_residual': (B, 6, H, W) - residual offsets
                'dimensions': (B, 3, H, W) - (h, w, l)
                'rotation_bin': (B, 8, H, W) - rotation bin logits
                'rotation_residual': (B, 8, H, W) - angle residuals
                'offset_2d': (B, 2, H, W) - (Δu, Δv)
            }
        """
        # Shared feature extraction
        shared_feat = self.shared_conv(x)
        
        # Compute all predictions
        heatmap = torch.sigmoid(self.heatmap(shared_feat))
        depth_bin = self.depth_bin(shared_feat)
        depth_residual = self.depth_residual(shared_feat)
        dimensions = self.dimensions(shared_feat)
        rotation_bin = self.rotation_bin(shared_feat)
        rotation_residual = self.rotation_residual(shared_feat)
        offset_2d = self.offset_2d(shared_feat)
        
        return {
            'heatmap': heatmap,
            'depth_bin': depth_bin,
            'depth_residual': depth_residual,
            'dimensions': dimensions,
            'rotation_bin': rotation_bin,
            'rotation_residual': rotation_residual,
            'offset_2d': offset_2d
        }


class MonocularDetector3D(nn.Module):
    """
    Complete monocular 3D object detection network
    
    Architecture:
    - Backbone: DLA-34 (stride 4, output 512 channels)
    - Head: Multi-task detection head
    
    Target: NVIDIA Drive AGX Orin DLA acceleration
    - All operations are DLA-compatible (standard convolutions)
    - No deformable convolutions or dynamic operations
    - Fixed input size for TensorRT optimization
    """
    def __init__(self, 
                 input_size=(720, 1280),
                 depth_bins=[0, 10, 20, 35, 50, 70, 200],
                 num_rotation_bins=8):
        super(MonocularDetector3D, self).__init__()
        
        self.input_size = input_size
        self.depth_bins = depth_bins
        self.num_depth_bins = len(depth_bins) - 1  # 6 bins
        self.num_rotation_bins = num_rotation_bins
        
        # Backbone
        self.backbone = DLA34()
        
        # Detection head
        self.head = MonocularDetectionHead(in_channels=64)
        
        # Metadata for inference
        self.register_buffer('depth_bin_boundaries', 
                           torch.tensor(depth_bins, dtype=torch.float32))
        self.rotation_bin_size = 2 * math.pi / num_rotation_bins
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input RGB image (B, 3, 720, 1280)
        
        Returns:
            Dictionary of predictions at stride 4 (B, C, 180, 320)
        """
        # Extract features
        features = self.backbone(x)
        
        # Generate predictions
        predictions = self.head(features)
        
        return predictions
    
    def decode_detections(self, predictions, K, threshold=0.3, max_detections=100):
        """
        Decode network outputs to 3D bounding boxes in camera frame
        
        Args:
            predictions: Dictionary from forward pass
            K: Camera intrinsic matrix (3x3) or batch (B, 3, 3)
            threshold: Heatmap confidence threshold
            max_detections: Maximum number of detections per image
        
        Returns:
            List of detection dictionaries per batch element:
            {
                'boxes_3d': (N, 7) - [x, y, z, h, w, l, yaw] in camera frame
                'scores': (N,) - confidence scores
                'centers_2d': (N, 2) - [u, v] image coordinates
            }
        """
        batch_size = predictions['heatmap'].shape[0]
        device = predictions['heatmap'].device
        
        # Extract heatmap peaks (NMS)
        heatmap = predictions['heatmap']
        batch_detections = []
        
        for b in range(batch_size):
            # Get peaks using max pooling (local NMS)
            hmax = F.max_pool2d(heatmap[b:b+1], kernel_size=3, stride=1, padding=1)
            keep = (hmax == heatmap[b:b+1]).float()
            heatmap_b = heatmap[b, 0] * keep[0, 0]
            
            # Threshold and top-k
            scores_flat = heatmap_b.view(-1)
            topk_scores, topk_inds = torch.topk(scores_flat, min(max_detections, scores_flat.numel()))
            
            # Filter by threshold
            keep_inds = topk_scores > threshold
            topk_scores = topk_scores[keep_inds]
            topk_inds = topk_inds[keep_inds]
            
            if topk_inds.numel() == 0:
                batch_detections.append({
                    'boxes_3d': torch.zeros((0, 7), device=device),
                    'scores': torch.zeros((0,), device=device),
                    'centers_2d': torch.zeros((0, 2), device=device)
                })
                continue
            
            # Convert flat indices to 2D coordinates
            H, W = heatmap_b.shape
            topk_ys = (topk_inds // W).float()
            topk_xs = (topk_inds % W).float()
            
            # Decode predictions at peak locations
            # Apply 2D offset for sub-pixel localization
            offset_2d = predictions['offset_2d'][b]
            offset_xs = offset_2d[0, topk_ys.long(), topk_xs.long()]
            offset_ys = offset_2d[1, topk_ys.long(), topk_xs.long()]
            topk_xs = topk_xs + offset_xs
            topk_ys = topk_ys + offset_ys
            
            # Map to input image coordinates (stride 4)
            centers_u = topk_xs * 4
            centers_v = topk_ys * 4
            
            # Decode depth
            depth_bin_logits = predictions['depth_bin'][b, :, topk_ys.long(), topk_xs.long()].t()
            depth_residuals = predictions['depth_residual'][b, :, topk_ys.long(), topk_xs.long()].t()
            depth_bin_probs = F.softmax(depth_bin_logits, dim=1)
            depth_bin_indices = torch.argmax(depth_bin_probs, dim=1)
            
            # Compute depth from bins + residuals
            bin_centers = (self.depth_bin_boundaries[:-1] + self.depth_bin_boundaries[1:]) / 2
            depth_from_bins = bin_centers[depth_bin_indices]
            residual_values = depth_residuals[torch.arange(len(depth_bin_indices)), depth_bin_indices]
            depths = depth_from_bins + residual_values
            
            # Decode dimensions
            dims = predictions['dimensions'][b, :, topk_ys.long(), topk_xs.long()].t()
            heights, widths, lengths = dims[:, 0], dims[:, 1], dims[:, 2]
            
            # Decode rotation
            rotation_bin_logits = predictions['rotation_bin'][b, :, topk_ys.long(), topk_xs.long()].t()
            rotation_residuals = predictions['rotation_residual'][b, :, topk_ys.long(), topk_xs.long()].t()
            rotation_bin_probs = F.softmax(rotation_bin_logits, dim=1)
            rotation_bin_indices = torch.argmax(rotation_bin_probs, dim=1)
            
            # Compute yaw angle
            bin_angle_centers = torch.arange(self.num_rotation_bins, device=device).float() * self.rotation_bin_size
            yaw_from_bins = bin_angle_centers[rotation_bin_indices]
            residual_angles = rotation_residuals[torch.arange(len(rotation_bin_indices)), rotation_bin_indices]
            yaws = yaw_from_bins + residual_angles
            
            # Unproject 2D centers to 3D using depth
            K_b = K if K.dim() == 2 else K[b]
            fx, fy = K_b[0, 0], K_b[1, 1]
            cx, cy = K_b[0, 2], K_b[1, 2]
            
            X_cam = (centers_u - cx) * depths / fx
            Y_cam = (centers_v - cy) * depths / fy
            Z_cam = depths
            
            # Assemble 3D boxes: [x, y, z, h, w, l, yaw]
            boxes_3d = torch.stack([X_cam, Y_cam, Z_cam, heights, widths, lengths, yaws], dim=1)
            
            batch_detections.append({
                'boxes_3d': boxes_3d,
                'scores': topk_scores,
                'centers_2d': torch.stack([centers_u, centers_v], dim=1)
            })
        
        return batch_detections


def build_model(pretrained=True):
    """
    Build MonocularDetector3D model
    
    Args:
        pretrained: Load ImageNet pretrained weights for DLA-34 backbone
    
    Returns:
        model: MonocularDetector3D instance
    """
    model = MonocularDetector3D(
        input_size=(720, 1280),
        depth_bins=[0, 10, 20, 35, 50, 70, 200],
        num_rotation_bins=8
    )
    
    if pretrained:
        # Note: Implement DLA-34 ImageNet pretrained weight loading here
        # For now, training from scratch or using custom pretraining
        print("Note: ImageNet pretrained weights not loaded. Training from scratch.")
    
    return model


if __name__ == "__main__":
    # Test model instantiation and forward pass
    model = build_model(pretrained=False)
    model.eval()
    
    # Dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, 720, 1280)
    
    # Forward pass
    with torch.no_grad():
        predictions = model(x)
    
    print("Model Architecture Test:")
    print("=" * 60)
    print(f"Input shape: {x.shape}")
    print("\nOutput shapes:")
    for key, value in predictions.items():
        print(f"  {key:20s}: {tuple(value.shape)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test decoding
    K = torch.tensor([[1013.14, 0, 653.02],
                      [0, 1013.14, 393.21],
                      [0, 0, 1]], dtype=torch.float32)
    K = K.unsqueeze(0).repeat(batch_size, 1, 1)
    
    detections = model.decode_detections(predictions, K, threshold=0.1)
    print(f"\nDetections per batch element:")
    for i, det in enumerate(detections):
        print(f"  Batch {i}: {det['boxes_3d'].shape[0]} detections")
