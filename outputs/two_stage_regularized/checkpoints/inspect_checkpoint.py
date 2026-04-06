#!/usr/bin/env python3
import torch
import sys

checkpoint_path = sys.argv[1]
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Get the actual model weights
state_dict = ckpt['model_state_dict']

print(f"\n{'='*80}")
print(f"MODEL STATE DICT ANALYSIS")
print(f"{'='*80}\n")

print(f"Total parameters: {len(state_dict)}\n")

# Group by component
backbone = [k for k in state_dict.keys() if 'feature_backbone' in k]
depth_head = [k for k in state_dict.keys() if 'depth_head' in k]
dim_head = [k for k in state_dict.keys() if 'dimension_head' in k]
rot_head = [k for k in state_dict.keys() if 'rotation_head' in k]

print(f"Feature Backbone: {len(backbone)} layers")
if backbone:
    print(f"  First: {backbone[0]}")
    print(f"  Last:  {backbone[-1]}")

print(f"\nDepth Head: {len(depth_head)} layers")
for k in depth_head[:5]:
    print(f"  {k}: {state_dict[k].shape}")

print(f"\nDimension Head: {len(dim_head)} layers")
for k in dim_head[:5]:
    print(f"  {k}: {state_dict[k].shape}")

print(f"\nRotation Head: {len(rot_head)} layers")
for k in rot_head[:5]:
    print(f"  {k}: {state_dict[k].shape}")

# Check first conv to determine input preprocessing
first_conv = [k for k in state_dict.keys() if 'conv1.weight' in k or 'feature_backbone.0.weight' in k]
if first_conv:
    weight_shape = state_dict[first_conv[0]].shape
    print(f"\n{'='*80}")
    print(f"INPUT LAYER ANALYSIS:")
    print(f"{'='*80}")
    print(f"  Layer: {first_conv[0]}")
    print(f"  Shape: {weight_shape}")
    print(f"  Input channels: {weight_shape[1]}")
    print(f"  → Expects RGB (3 channels)")
