"""Test KITTI 3D Box Visualization - DIMENSION MAPPING FIXED"""

import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from data.kitti_dataset import KITTI3DDataset


def project_3d_box(box_3d, K):
    """
    Project 3D box to 2D - CORRECT KITTI CONVENTION
    
    KITTI convention:
    - rotation_y = 0 means facing along +X (right)
    - Therefore: LENGTH is along LOCAL X-axis
    """
    x, y, z, h, w, l, ry = box_3d
    
    # Object-centric frame (CORRECTED):
    # Local X = forward (length) - when ry=0, points along global +X
    # Local Y = up (height)
    # Local Z = right side (width)
    
    x_corners = [l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2]  # LENGTH along X (forward)
    y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]  # HEIGHT along Y (up)
    z_corners = [w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2]  # WIDTH along Z (right)
    
    corners_3d = np.array([x_corners, y_corners, z_corners])
    
    # Rotation around Y-axis
    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    corners_3d = R @ corners_3d
    
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z
    
    if np.any(corners_3d[2, :] <= 0.1):
        return None
    
    corners_2d = K @ corners_3d
    corners_2d = corners_2d[:2, :] / corners_3d[2, :]
    
    return corners_2d.T


def draw_3d_box(image, box_3d, K, color=(0, 255, 0), thickness=2):
    """Draw 3D box on image"""
    corners_2d = project_3d_box(box_3d, K)
    
    if corners_2d is None:
        return image
    
    corners_2d = corners_2d.astype(np.int32)
    
    # Bottom face (0,1,2,3)
    for i in range(4):
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[(i+1)%4]), color, thickness)
    
    # Top face (4,5,6,7)
    for i in range(4, 8):
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[4+(i+1)%4]), color, thickness)
    
    # Vertical edges
    for i in range(4):
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[i+4]), color, thickness)
    
    # Front face marker: corners 0,3,4,7 (front face in local frame, +X face)
    # After corner reordering with length along X
    front_center = ((corners_2d[0] + corners_2d[3] + corners_2d[4] + corners_2d[7]) / 4).astype(np.int32)
    cv2.circle(image, tuple(front_center), 6, (255, 0, 0), -1)
    
    return image


def test_3d_visualization():
    dataset = KITTI3DDataset(
        root_dir='/media/ashwin-benchdev/eb2a8889-3ec9-411a-826e-816cf9759b02/KITTI-3D',
        split='val',
        filter_classes=['Car']
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx in range(4):
        sample = dataset[idx]
        
        image = (sample['image'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        boxes_3d = sample['boxes_3d'].numpy()
        K = sample['intrinsics'].numpy()
        
        print(f"\nSample {idx}: {len(boxes_3d)} objects")
        if len(boxes_3d) > 0:
            box = boxes_3d[0]
            print(f"  First box:")
            print(f"    Location (x,y,z): ({box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f})")
            print(f"    Dimensions (h,w,l): ({box[3]:.2f}, {box[4]:.2f}, {box[5]:.2f})")
            print(f"    Rotation: {box[6]:.2f} rad = {np.degrees(box[6]):.1f}°")
        
        img_viz = image.copy()
        for box_3d in boxes_3d:
            img_viz = draw_3d_box(img_viz, box_3d, K, color=(0, 255, 0), thickness=2)
        
        axes[idx].imshow(img_viz)
        axes[idx].set_title(f'Sample {idx} - {len(boxes_3d)} objects', fontsize=12)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('kitti_3d_CORRECTED.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: kitti_3d_CORRECTED.png")
    print("\nVerify:")
    print("  1. Boxes should tightly fit vehicles (not elongated)")
    print("  2. Red dots point to vehicle FRONTS")
    print("  3. Box orientations match vehicle headings")


if __name__ == "__main__":
    test_3d_visualization()
