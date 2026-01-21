"""Inspect checkpoint to see actual layer shapes"""

import torch
import sys

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoint.pth"

print(f"Loading: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\nCheckpoint keys:", checkpoint.keys())

state_dict = checkpoint['model_state_dict']

print("\n" + "="*70)
print("Layer shapes in checkpoint:")
print("="*70)

# Find intrinsics and head layers
for key in sorted(state_dict.keys()):
    if 'intrinsics' in key.lower() or 'depth_head' in key or 'dimension_head' in key:
        shape = state_dict[key].shape
        print(f"{key}: {shape}")

print("\n" + "="*70)
print("Key findings:")
print("="*70)

# Check if intrinsics encoder exists
has_intrinsics = any('intrinsics_encoder' in k for k in state_dict.keys())
print(f"Has intrinsics_encoder: {has_intrinsics}")

# Check depth head first layer size
depth_head_key = 'depth_head.fc.0.weight'
if depth_head_key in state_dict:
    shape = state_dict[depth_head_key].shape
    print(f"depth_head first layer: {shape} -> expects input size {shape[1]}")
    if shape[1] == 2048:
        print("  → NOT intrinsics-conditioned (2048 = ResNet only)")
    elif shape[1] == 2176:
        print("  → IS intrinsics-conditioned (2176 = 2048 + 128)")
