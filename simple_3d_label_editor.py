#!/usr/bin/env python3
"""
Simple 3D Label Editor for KITTI Format

Edit predicted 3D bounding boxes on rectified images:
- Adjust position (x, y, z)
- Adjust dimensions (height, width, length)
- Adjust rotation
- Delete boxes
- Save corrections

Usage:
    python simple_3d_label_editor.py \\
        --images /path/to/rectified/images \\
        --labels /path/to/predicted/labels \\
        --calib /path/to/rectified_camera_params.json \\
        --output /path/to/save/corrected/labels

Keyboard Controls:
    Navigation:
        n - Next image
        p - Previous image
        Tab - Select next box
        
    Position (selected box):
        w/s - Move depth (forward/backward)
        a/d - Move left/right (x-axis)
        z/x - Move up/down (y-axis)
        
    Dimensions (selected box):
        t/g - Increase/decrease height
        y/h - Increase/decrease width
        u/j - Increase/decrease length
        
    Rotation (selected box):
        r/f - Rotate clockwise/counter-clockwise
        
    Actions:
        Delete - Delete selected box
        v - Save corrections for current image
        Shift+S - Save all and quit
        q - Quit without saving
        
    Adjustment speed:
        1 - Fine (0.05m)
        2 - Normal (0.1m)
        3 - Coarse (0.5m)
"""

import cv2
import numpy as np
from pathlib import Path
import json
import sys


class Simple3DLabelEditor:
    def __init__(self, image_dir, label_dir, calib_file, output_dir=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.output_dir = Path(output_dir) if output_dir else self.label_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load calibration
        print(f"Loading calibration from: {calib_file}")
        with open(calib_file, 'r') as f:
            params = json.load(f)
        
        if 'rectified_intrinsics' in params:
            intrinsics = params['rectified_intrinsics']
        elif 'camera_matrix' in params:
            intrinsics = params['camera_matrix']
        else:
            raise ValueError("Invalid calibration file format")
        
        self.K = np.array([
            [intrinsics['fx'], 0, intrinsics['cx']],
            [0, intrinsics['fy'], intrinsics['cy']],
            [0, 0, 1]
        ], dtype=np.float32)
        
        print(f"Camera intrinsics loaded:")
        print(f"  fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}")
        print(f"  cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}")
        
        # Get image files
        self.image_files = sorted(list(self.image_dir.glob('*.png')) + 
                                  list(self.image_dir.glob('*.jpg')))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"\nFound {len(self.image_files)} images")
        print(f"Labels directory: {self.label_dir}")
        print(f"Output directory: {self.output_dir}")
        
        self.current_idx = 0
        self.selected_box = 0
        self.step_size = 0.1  # meters
        self.modified = False
        self.unsaved_changes = {}
        
        # Colors
        self.color_selected = (0, 255, 255)  # Yellow for selected
        self.color_unselected = (0, 255, 0)  # Green for others
        self.color_text = (255, 255, 255)
        
    def load_labels(self, frame_id):
        """Load KITTI labels for a frame"""
        label_file = self.label_dir / f'{frame_id}.txt'
        if not label_file.exists():
            return []
        
        boxes = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                
                box = {
                    'type': parts[0],
                    'truncated': float(parts[1]),
                    'occluded': int(parts[2]),
                    'alpha': float(parts[3]),
                    'bbox_2d': [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
                    'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],  # h,w,l
                    'location': [float(parts[11]), float(parts[12]), float(parts[13])],  # x,y,z
                    'rotation_y': float(parts[14]),
                    'score': float(parts[15]) if len(parts) > 15 else 1.0
                }
                boxes.append(box)
        
        return boxes
    
    def save_labels(self, frame_id, boxes):
        """Save corrected labels in KITTI format"""
        label_file = self.output_dir / f'{frame_id}.txt'
        
        with open(label_file, 'w') as f:
            for box in boxes:
                line = f"{box['type']} "
                line += f"{box['truncated']:.2f} "
                line += f"{box['occluded']} "
                line += f"{box['alpha']:.2f} "
                line += f"{box['bbox_2d'][0]:.2f} {box['bbox_2d'][1]:.2f} {box['bbox_2d'][2]:.2f} {box['bbox_2d'][3]:.2f} "
                line += f"{box['dimensions'][0]:.2f} {box['dimensions'][1]:.2f} {box['dimensions'][2]:.2f} "
                line += f"{box['location'][0]:.2f} {box['location'][1]:.2f} {box['location'][2]:.2f} "
                line += f"{box['rotation_y']:.2f} "
                line += f"{box['score']:.2f}\n"
                f.write(line)
        
        print(f"✓ Saved: {label_file.name}")
        return True
    
    def project_3d_box(self, box):
        """Project 3D box to 2D image coordinates"""
        h, w, l = box['dimensions']
        x, y, z = box['location']
        ry = box['rotation_y']
        
        # Define 8 corners in object coordinate system
        x_corners = [l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2]
        y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
        z_corners = [w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2]
        
        corners_3d = np.array([x_corners, y_corners, z_corners])
        
        # Rotation matrix
        R = np.array([
            [np.cos(ry), 0, -np.sin(ry)],
            [0, 1, 0],
            [np.sin(ry), 0, np.cos(ry)]
        ])
        corners_3d = R @ corners_3d
        
        # Translation
        corners_3d[0, :] += x
        corners_3d[1, :] += y
        corners_3d[2, :] += z
        
        # Check if behind camera
        if np.any(corners_3d[2, :] <= 0.1):
            return None
        
        # Project to 2D
        corners_2d = self.K @ corners_3d
        corners_2d = corners_2d[:2, :] / corners_3d[2, :]
        
        return corners_2d.T.astype(np.int32)
    
    def draw_box(self, image, box, color, selected=False):
        """Draw 3D bounding box on image"""
        corners = self.project_3d_box(box)
        if corners is None:
            return image
        
        thickness = 3 if selected else 2
        
        # Draw bottom face (indices 0-3)
        for i in range(4):
            cv2.line(image, tuple(corners[i]), tuple(corners[(i+1)%4]), color, thickness)
        
        # Draw top face (indices 4-7)
        for i in range(4, 8):
            cv2.line(image, tuple(corners[i]), tuple(corners[4+(i+1)%4]), color, thickness)
        
        # Draw vertical edges
        for i in range(4):
            cv2.line(image, tuple(corners[i]), tuple(corners[i+4]), color, thickness)
        
        # Draw front indicator (blue dot)
        front_center = ((corners[0] + corners[3] + corners[4] + corners[7]) / 4).astype(np.int32)
        cv2.circle(image, tuple(front_center), 6, (255, 0, 0), -1)
        
        # Draw label with box info
        h, w, l = box['dimensions']
        x, y, z = box['location']
        
        label_lines = [
            f"{box['type']} {box['score']:.2f}",
            f"D:{z:.1f}m",
            f"H:{h:.1f} W:{w:.1f} L:{l:.1f}"
        ]
        
        label_y = int(corners[:, 1].min()) - 10
        for i, line in enumerate(label_lines):
            y_pos = max(20, label_y - i*20)
            cv2.putText(image, line, (int(corners[:, 0].min()), y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        
        return image
    
    def draw_ui(self, image, frame_id, boxes):
        """Draw UI overlay with instructions and status"""
        vis = image.copy()
        
        # Draw all boxes
        for i, box in enumerate(boxes):
            color = self.color_selected if i == self.selected_box else self.color_unselected
            vis = self.draw_box(vis, box, color, selected=(i == self.selected_box))
        
        # Status bar background
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (vis.shape[1], 180), (0, 0, 0), -1)
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        # Status information
        status_lines = [
            f"Frame: {self.current_idx+1}/{len(self.image_files)} | {frame_id} | Boxes: {len(boxes)}",
            f"Selected: Box {self.selected_box+1}/{len(boxes)} | Step: {self.step_size:.2f}m | {'[MODIFIED]' if self.modified else ''}",
            "",
            "NAV: [n]ext [p]rev [Tab]select | POS: [w/s]depth [a/d]x [z/x]y | DIM: [t/g]H [y/h]W [u/j]L",
            "ROT: [r/f]rotate | ACTION: [Del]delete [v]save [Shift+S]save all [q]quit | STEP: [1]fine [2]norm [3]coarse"
        ]
        
        y_offset = 25
        for i, line in enumerate(status_lines):
            font_scale = 0.6 if i < 2 else 0.5
            cv2.putText(vis, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, self.color_text, 2 if i < 2 else 1, cv2.LINE_AA)
            y_offset += 30
        
        # Show selected box info
        if boxes and self.selected_box < len(boxes):
            box = boxes[self.selected_box]
            h, w, l = box['dimensions']
            x, y, z = box['location']
            
            info_lines = [
                f"Location: X={x:.2f} Y={y:.2f} Z={z:.2f}",
                f"Dimensions: H={h:.2f} W={w:.2f} L={l:.2f}",
                f"Rotation: {np.degrees(box['rotation_y']):.1f}°"
            ]
            
            y_offset = vis.shape[0] - 70
            for line in info_lines:
                cv2.putText(vis, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, self.color_selected, 2, cv2.LINE_AA)
                y_offset += 25
        
        return vis
    
    def run(self):
        """Main editing loop"""
        print("\n" + "="*70)
        print("3D LABEL EDITOR - CONTROLS")
        print("="*70)
        print("Navigation: [n]ext [p]rev [Tab]select")
        print("Position: [w/s]depth [a/d]left/right [z/x]up/down")
        print("Dimensions: [t/g]height [y/h]width [u/j]length")
        print("Rotation: [r/f]rotate")
        print("Actions: [Delete]remove [v]save [Shift+S]save all [q]quit")
        print("Step size: [1]fine(0.05) [2]normal(0.1) [3]coarse(0.5)")
        print("="*70 + "\n")
        
        cv2.namedWindow('3D Label Editor', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('3D Label Editor', 1920, 1080)
        
        while True:
            img_path = self.image_files[self.current_idx]
            frame_id = img_path.stem
            
            # Load image and labels
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Error loading image: {img_path}")
                self.current_idx = (self.current_idx + 1) % len(self.image_files)
                continue
            
            # Load labels (from cache if modified, else from file)
            if frame_id in self.unsaved_changes:
                boxes = self.unsaved_changes[frame_id]
            else:
                boxes = self.load_labels(frame_id)
                self.unsaved_changes[frame_id] = boxes
            
            # Draw UI
            vis = self.draw_ui(image, frame_id, boxes)
            cv2.imshow('3D Label Editor', vis)
            
            # Handle keyboard input
            key = cv2.waitKey(0) & 0xFF
            self.modified = False
            
            # Navigation
            if key == ord('n'):  # Next frame
                self.current_idx = (self.current_idx + 1) % len(self.image_files)
                self.selected_box = 0
                
            elif key == ord('p'):  # Previous frame
                self.current_idx = (self.current_idx - 1) % len(self.image_files)
                self.selected_box = 0
                
            elif key == 9:  # Tab - select next box
                if boxes:
                    self.selected_box = (self.selected_box + 1) % len(boxes)
            
            # Step size adjustment
            elif key == ord('1'):
                self.step_size = 0.05
                print(f"Step size: {self.step_size}m (fine)")
            elif key == ord('2'):
                self.step_size = 0.1
                print(f"Step size: {self.step_size}m (normal)")
            elif key == ord('3'):
                self.step_size = 0.5
                print(f"Step size: {self.step_size}m (coarse)")
            
            # Edit selected box
            elif boxes and self.selected_box < len(boxes):
                box = boxes[self.selected_box]
                
                # Position adjustments
                if key == ord('w'):  # Depth forward
                    box['location'][2] += self.step_size
                    self.modified = True
                elif key == ord('s'):  # Depth backward
                    box['location'][2] -= self.step_size
                    self.modified = True
                elif key == ord('a'):  # Left
                    box['location'][0] -= self.step_size
                    self.modified = True
                elif key == ord('d'):  # Right
                    box['location'][0] += self.step_size
                    self.modified = True
                elif key == ord('z'):  # Up
                    box['location'][1] -= self.step_size
                    self.modified = True
                elif key == ord('x'):  # Down
                    box['location'][1] += self.step_size
                    self.modified = True
                
                # Dimension adjustments
                elif key == ord('t'):  # Height +
                    box['dimensions'][0] += self.step_size
                    self.modified = True
                elif key == ord('g'):  # Height -
                    box['dimensions'][0] = max(0.1, box['dimensions'][0] - self.step_size)
                    self.modified = True
                elif key == ord('y'):  # Width +
                    box['dimensions'][1] += self.step_size
                    self.modified = True
                elif key == ord('h'):  # Width -
                    box['dimensions'][1] = max(0.1, box['dimensions'][1] - self.step_size)
                    self.modified = True
                elif key == ord('u'):  # Length +
                    box['dimensions'][2] += self.step_size
                    self.modified = True
                elif key == ord('j'):  # Length -
                    box['dimensions'][2] = max(0.1, box['dimensions'][2] - self.step_size)
                    self.modified = True
                
                # Rotation adjustment
                elif key == ord('r'):  # Rotate clockwise
                    box['rotation_y'] += 0.1
                    self.modified = True
                elif key == ord('f'):  # Rotate counter-clockwise
                    box['rotation_y'] -= 0.1
                    self.modified = True
                
                # Delete box
                elif key == 255:  # Delete key
                    boxes.pop(self.selected_box)
                    self.selected_box = max(0, min(self.selected_box, len(boxes) - 1))
                    self.modified = True
                    print(f"Deleted box {self.selected_box + 1}")
            
            # Save current frame
            if key == ord('v'):
                if self.save_labels(frame_id, boxes):
                    if frame_id in self.unsaved_changes:
                        del self.unsaved_changes[frame_id]
            
            # Save all and quit (Shift+S)
            elif key == ord('S'):
                print("\nSaving all modified frames...")
                for fid, boxes_data in self.unsaved_changes.items():
                    self.save_labels(fid, boxes_data)
                print("All changes saved. Exiting...")
                break
            
            # Quit without saving
            elif key == ord('q'):
                if self.unsaved_changes:
                    print(f"\nWarning: {len(self.unsaved_changes)} frames have unsaved changes!")
                    print("Press 'S' (Shift+S) to save all, or 'q' again to quit without saving")
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 == ord('q'):
                        print("Exiting without saving...")
                        break
                    elif key2 == ord('S'):
                        print("\nSaving all modified frames...")
                        for fid, boxes_data in self.unsaved_changes.items():
                            self.save_labels(fid, boxes_data)
                        print("All changes saved. Exiting...")
                        break
                else:
                    print("Exiting...")
                    break
        
        cv2.destroyAllWindows()
        print("\n" + "="*70)
        print(f"Corrected labels saved to: {self.output_dir}")
        print("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Simple 3D Label Editor for KITTI format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--images', type=str, required=True,
                       help='Directory containing rectified images')
    parser.add_argument('--labels', type=str, required=True,
                       help='Directory containing predicted KITTI labels')
    parser.add_argument('--calib', type=str, required=True,
                       help='Path to rectified_camera_params.json')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for corrected labels (default: same as --labels)')
    
    args = parser.parse_args()
    
    try:
        editor = Simple3DLabelEditor(
            image_dir=args.images,
            label_dir=args.labels,
            calib_file=args.calib,
            output_dir=args.output
        )
        editor.run()
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
