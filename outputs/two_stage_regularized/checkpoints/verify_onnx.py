#!/usr/bin/env python3
"""Verify ONNX model inputs/outputs match expectations"""

import onnx
import sys

model_path = sys.argv[1]
model = onnx.load(model_path)

print("\n" + "="*80)
print("ONNX MODEL VERIFICATION")
print("="*80)

print("\nINPUTS:")
for inp in model.graph.input:
    print(f"  {inp.name}:")
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"    Shape: {dims}")
    print(f"    Type: {inp.type.tensor_type.elem_type}")

print("\nOUTPUTS:")
for out in model.graph.output:
    print(f"  {out.name}:")
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"    Shape: {dims}")
    print(f"    Type: {out.type.tensor_type.elem_type}")

print("\n" + "="*80)
