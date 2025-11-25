#!/usr/bin/env python3
"""Apply Critical Fix: Proper Denormalization for YOLO"""

import sys
import shutil
from pathlib import Path


def apply_fix():
    """Apply the denormalization fix to two_stage_detector.py"""
    
    model_path = Path('models/two_stage_detector.py')
    
    if not model_path.exists():
        print(f"ERROR: {model_path} not found!")
        print("Make sure you're running this from the NetAsh3D directory")
        return False
    
    print("="*70)
    print("APPLYING CRITICAL FIX: YOLO Image Denormalization")
    print("="*70)
    
    # Read current content
    with open(model_path, 'r') as f:
        lines = f.readlines()
    
    # Backup
    backup_path = str(model_path) + '.backup_denorm_fix'
    shutil.copy(model_path, backup_path)
    print(f"\n✓ Backup created: {backup_path}")
    
    # Find the buggy lines
    found = False
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Look for the buggy code
        if 'img = images[i].cpu().numpy().transpose(1, 2, 0) * 255' in line:
            found = True
            indent = len(line) - len(line.lstrip())
            spaces = ' ' * indent
            
            print(f"\n✓ Found buggy code at line {i+1}")
            print(f"  OLD: {line.strip()}")
            
            # Replace with fixed code
            fixed_lines.append(f"{spaces}# FIXED: Properly denormalize before YOLO\n")
            fixed_lines.append(f"{spaces}IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)\n")
            fixed_lines.append(f"{spaces}IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)\n")
            fixed_lines.append(f"{spaces}\n")
            fixed_lines.append(f"{spaces}img = images[i].cpu().numpy().transpose(1, 2, 0)\n")
            fixed_lines.append(f"{spaces}img = img * IMAGENET_STD + IMAGENET_MEAN  # Denormalize\n")
            fixed_lines.append(f"{spaces}img = (img * 255).clip(0, 255).astype('uint8')  # Scale\n")
            
            print(f"  NEW: Multi-line fix with proper denormalization")
            
            # Skip next line if it's the img.astype line
            if i+1 < len(lines) and 'img = img.astype' in lines[i+1]:
                i += 1
                print(f"  (Also removed line {i+1}: {lines[i].strip()})")
            
        else:
            fixed_lines.append(line)
        
        i += 1
    
    if not found:
        print("\n⚠️  WARNING: Could not find the exact buggy code!")
        print("   The file may have already been fixed or has different formatting.")
        print("   Please check manually.")
        return False
    
    # Check if numpy is imported
    has_numpy = any('import numpy as np' in line for line in fixed_lines)
    
    if not has_numpy:
        print("\n✓ Adding numpy import")
        # Find where to insert (after other imports)
        insert_pos = 0
        for i, line in enumerate(fixed_lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_pos = i + 1
        fixed_lines.insert(insert_pos, 'import numpy as np\n')
    
    # Write fixed version
    with open(model_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"\n✓ Fix applied successfully!")
    print(f"  Fixed file: {model_path}")
    
    print("\n" + "="*70)
    print("WHAT WAS FIXED")
    print("="*70)
    print("\nOLD CODE (BUGGY):")
    print("  img = images[i].cpu().numpy().transpose(1, 2, 0) * 255")
    print("  img = img.astype('uint8')")
    print("\n  Problem: Doesn't undo ImageNet normalization")
    print("  Result: YOLO sees garbage → detects nothing")
    
    print("\nNEW CODE (FIXED):")
    print("  img = images[i].cpu().numpy().transpose(1, 2, 0)")
    print("  img = img * IMAGENET_STD + IMAGENET_MEAN  # Denormalize")
    print("  img = (img * 255).clip(0, 255).astype('uint8')  # Scale")
    print("\n  Fix: Properly undoes normalization")
    print("  Result: YOLO sees correct images → detects objects!")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Test the fix:")
    print("   python3 diagnose_pipeline.py \\")
    print("       --checkpoint outputs/two_stage_regularized/checkpoints/checkpoint_best.pth \\")
    print("       --video test_video.mp4 \\")
    print("       --num-frames 50")
    print("\n2. Run visual debug:")
    print("   python3 debug_pipeline_visual.py \\")
    print("       --frame test_frame.png \\")
    print("       --checkpoint outputs/two_stage_regularized/checkpoints/checkpoint_best.pth \\")
    print("       --output pipeline_debug_fixed.png")
    print("\n3. Test on full video:")
    print("   python3 inference_fisheye_video.py \\")
    print("       --checkpoint outputs/two_stage_regularized/checkpoints/checkpoint_best.pth \\")
    print("       --video your_video.mp4 \\")
    print("       --output output_fixed.mp4 \\")
    print("       --fisheye-config intrinsics.json")
    
    print("\n" + "="*70)
    
    return True


def main():
    print("\n")
    success = apply_fix()
    
    if success:
        print("\n✅ FIX APPLIED SUCCESSFULLY!\n")
        sys.exit(0)
    else:
        print("\n❌ FIX FAILED - Please apply manually\n")
        print("See QUESTIONS_ANSWERED.md for manual instructions")
        sys.exit(1)


if __name__ == "__main__":
    main()