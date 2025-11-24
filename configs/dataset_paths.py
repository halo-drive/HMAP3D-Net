"""Dataset path configuration"""

NUSCENES_ROOT = "/media/ashwin-benchdev/eb2a8889-3ec9-411a-826e-816cf9759b02/NuScenes-Full-Dataset"
NUSCENES_VERSION = "v1.0-trainval"
NUSCENES_DATAROOT = f"{NUSCENES_ROOT}/NuScenes/trainval"
NUSCENES_ANNOTATIONS = f"{NUSCENES_ROOT}/nuscenes_prepared/{NUSCENES_VERSION}"

# Verify paths exist
import os
assert os.path.exists(NUSCENES_ROOT), f"Dataset root not found: {NUSCENES_ROOT}"
assert os.path.exists(NUSCENES_DATAROOT), f"Data blobs not found: {NUSCENES_DATAROOT}"
assert os.path.exists(NUSCENES_ANNOTATIONS), f"Annotations not found: {NUSCENES_ANNOTATIONS}"
print("✓ Dataset paths verified")
