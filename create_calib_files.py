import json
import numpy as np
from pathlib import Path

# Load your rectified intrinsics
with open('/home/ashwin-benchdev/fisheye-caliberation/data/caliberation_results/rectified_camera_params.json') as f:
    params = json.load(f)

K = np.array(params['rectified_intrinsics']['matrix'])

# Create P2 matrix (K with no translation: [K | 0])
P2 = np.hstack([K, np.zeros((3, 1))])

# Create calib files for each frame (000000.txt to 000009.txt)
calib_dir = Path('/home/ashwin-benchdev/NetAsh3D/domain_dataset/training/calib')
calib_dir.mkdir(parents=True, exist_ok=True)

for i in range(10):
    calib_file = calib_dir / f'{i:06d}.txt'
    with open(calib_file, 'w') as f:
        # KITTI format requires P0, P1, P2, P3, R0_rect, Tr_velo_to_cam
        f.write(f'P0: {" ".join(map(str, P2.flatten()))}\n')
        f.write(f'P1: {" ".join(map(str, P2.flatten()))}\n')
        f.write(f'P2: {" ".join(map(str, P2.flatten()))}\n')
        f.write(f'P3: {" ".join(map(str, P2.flatten()))}\n')
        f.write('R0_rect: 1 0 0 0 1 0 0 0 1\n')
        f.write('Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n')

print(f"Created 10 calibration files in {calib_dir}")
