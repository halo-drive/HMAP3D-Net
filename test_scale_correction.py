# test_scale_correction.py
import numpy as np

# From KITTI → Your camera
fx_kitti = 721
fx_yours = 2700
scale_factor = fx_kitti / fx_yours  # 0.267

print(f"Scale factor: {scale_factor}")
print(f"If predicted depth = 10m → actual = {10 * scale_factor:.1f}m")
print(f"If predicted height = 5m → actual = {5 * scale_factor:.1f}m")

# Expected car dimensions
print("\nExpected corrections:")
print(f"Car height should be ~1.5m")
print(f"If you see 5m boxes → scale by {1.5/5:.2f}")
print(f"If you see 0.5m boxes → scale by {1.5/0.5:.2f}")
