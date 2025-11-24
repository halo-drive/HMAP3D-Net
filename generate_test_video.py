"""Helper Script - Generate Test Video for 3D Detection"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def create_synthetic_test_video(output_path, duration_sec=10, fps=30):
    """Create a simple synthetic test video with moving rectangles"""
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration_sec * fps
    
    print(f"Creating synthetic test video...")
    print(f"  Duration: {duration_sec} seconds")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Output: {output_path}")
    
    for frame_idx in range(total_frames):
        # Create blank frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Draw road
        road_y = int(height * 0.6)
        cv2.rectangle(frame, (0, road_y), (width, height), (100, 100, 100), -1)
        
        # Draw lane markings
        for x in range(0, width, 100):
            cv2.line(frame, (x, road_y + 50), (x + 40, road_y + 50), (255, 255, 255), 3)
        
        # Simulate moving "car"
        t = frame_idx / total_frames
        car_x = int(width * 0.2 + width * 0.6 * t)
        car_y = road_y + 80
        car_w, car_h = 120, 80
        
        cv2.rectangle(frame, 
                     (car_x - car_w//2, car_y - car_h//2),
                     (car_x + car_w//2, car_y + car_h//2),
                     (0, 0, 180), -1)
        
        # Add text
        cv2.putText(frame, f"Frame {frame_idx+1}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✓ Test video created: {output_path}")


def download_kitti_sample():
    """Instructions for downloading KITTI sample videos"""
    print("\n" + "="*60)
    print("KITTI Sample Video Download Instructions")
    print("="*60)
    print("\n1. Visit KITTI Raw Data:")
    print("   http://www.cvlibs.net/datasets/kitti/raw_data.php")
    print("\n2. Download a driving sequence, e.g.:")
    print("   - 2011_09_26_drive_0001 (City)")
    print("   - 2011_09_26_drive_0005 (Highway)")
    print("\n3. Extract images and create video:")
    print("   cd 2011_09_26/2011_09_26_drive_0001_sync/image_02/data")
    print("   ffmpeg -framerate 10 -pattern_type glob -i '*.png' \\")
    print("          -c:v libx264 -pix_fmt yuv420p kitti_sample.mp4")
    print("\n4. Use the video for inference:")
    print("   python3 inference_video.py \\")
    print("       --checkpoint outputs/two_stage_regularized/checkpoints/checkpoint_best.pth \\")
    print("       --video kitti_sample.mp4 \\")
    print("       --output kitti_output_3d.mp4")
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Generate test video for 3D detection')
    parser.add_argument('--mode', type=str, choices=['synthetic', 'kitti-info'], 
                       default='synthetic',
                       help='synthetic: create test video, kitti-info: show download instructions')
    parser.add_argument('--output', type=str, default='test_video.mp4',
                       help='Output video path (for synthetic mode)')
    parser.add_argument('--duration', type=int, default=10,
                       help='Duration in seconds (for synthetic mode)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (for synthetic mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'synthetic':
        create_synthetic_test_video(args.output, args.duration, args.fps)
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print(f"\n1. Run inference on the test video:")
        print(f"   python3 inference_video.py \\")
        print(f"       --checkpoint outputs/two_stage_regularized/checkpoints/checkpoint_best.pth \\")
        print(f"       --video {args.output} \\")
        print(f"       --output test_video_3d.mp4\n")
        
        print("2. Note: Synthetic video won't have realistic 3D detections")
        print("   For real testing, use actual driving footage or KITTI data")
        
    elif args.mode == 'kitti-info':
        download_kitti_sample()


if __name__ == "__main__":
    main()