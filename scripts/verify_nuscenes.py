#!/usr/bin/env python3
"""
nuScenes Dataset Verification Script
Validates data accessibility and extracts key statistics for monocular 3D detection.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import sys
import numpy as np  


# Dataset paths
NUSCENES_ROOT = "/media/ashwin-benchdev/eb2a8889-3ec9-411a-826e-816cf9759b02/NuScenes-Full-Dataset"
NUSCENES_VERSION = "v1.0-trainval"
NUSCENES_DATAROOT = f"{NUSCENES_ROOT}/NuScenes/trainval"
NUSCENES_ANNOTATIONS = f"{NUSCENES_ROOT}/nuscenes_prepared/{NUSCENES_VERSION}"

class NuScenesVerifier:
    def __init__(self):
        self.annotation_path = Path(NUSCENES_ANNOTATIONS)
        self.data_root = Path(NUSCENES_DATAROOT)

        # Will hold loaded tables
        self.sample = None
        self.sample_data = None
        self.sample_annotation = None
        self.calibrated_sensor = None
        self.ego_pose = None
        self.instance = None
        self.category = None
        self.sensor = None

    def verify_paths(self):
        """Verify all required paths exist"""
        print("=" * 60)
        print("STEP 1: Path Verification")
        print("=" * 60)

        checks = [
            ("Dataset Root", Path(NUSCENES_ROOT)),
            ("Data Blobs", self.data_root),
            ("Annotations", self.annotation_path),
        ]

        all_good = True
        for name, path in checks:
            exists = path.exists()
            status = "✓" if exists else "✗"
            print(f"{status} {name}: {path}")
            if not exists:
                all_good = False

        if not all_good:
            print("\n✗ ERROR: Some paths do not exist!")
            sys.exit(1)

        print("\n✓ All paths verified\n")
        return True

    def load_annotations(self):
        """Load all annotation JSON files"""
        print("=" * 60)
        print("STEP 2: Loading Annotation Tables")
        print("=" * 60)
        
        json_files = {
            'sample': 'sample.json',
            'sample_data': 'sample_data.json',
            'sample_annotation': 'sample_annotation.json',
            'calibrated_sensor': 'calibrated_sensor.json',
            'ego_pose': 'ego_pose.json',
            'instance': 'instance.json',  # ADD THIS LINE
            'category': 'category.json',
            'sensor': 'sensor.json',
        }
        
        for attr, filename in json_files.items():
            filepath = self.annotation_path / filename
            print(f"Loading {filename}... ", end='', flush=True)
            
            if not filepath.exists():
                print(f"✗ NOT FOUND")
                sys.exit(1)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            setattr(self, attr, data)
            print(f"✓ {len(data)} records")
        
        print("\n✓ All annotations loaded\n")

    def analyze_cam_front(self):
        """Analyze CAM_FRONT specific data"""
        print("=" * 60)
        print("STEP 3: CAM_FRONT Analysis")
        print("=" * 60)
        
        # Find CAM_FRONT sensor token
        cam_front_sensor = None
        for sensor in self.sensor:
            if sensor['channel'] == 'CAM_FRONT':
                cam_front_sensor = sensor
                break
        
        if not cam_front_sensor:
            print("✗ CAM_FRONT sensor not found in sensor.json")
            sys.exit(1)
        
        print(f"✓ CAM_FRONT sensor token: {cam_front_sensor['token']}")
        print(f"  Modality: {cam_front_sensor['modality']}")
        
        # BUILD INDEX FIRST - O(n) instead of O(n*m)
        print("\nBuilding calibrated_sensor lookup index...")
        calib_lookup = {c['token']: c for c in self.calibrated_sensor}
        print(f"✓ Indexed {len(calib_lookup)} calibrations")
        
        # Now count CAM_FRONT sample_data entries - FAST
        print("Filtering CAM_FRONT samples...")
        cam_front_samples = []
        for sd in self.sample_data:
            calib = calib_lookup.get(sd['calibrated_sensor_token'])
            if calib and calib['sensor_token'] == cam_front_sensor['token']:
                cam_front_samples.append(sd)
        
        print(f"✓ CAM_FRONT sample_data entries: {len(cam_front_samples)}")
        
        # Verify image files exist (search across blob directories)
        print("\nVerifying image file accessibility...")
        test_samples = cam_front_samples[:10]
        accessible = 0
        
        # List all blob directories
        blob_dirs = sorted([d for d in self.data_root.iterdir() 
                        if d.is_dir() and 'trainval' in d.name and 'blobs' in d.name])
        print(f"  Found {len(blob_dirs)} blob directories")
        
        for sd in test_samples:
            filename = sd['filename']  # e.g., "sweeps/CAM_FRONT/xxx.jpg"
            
            # Try each blob directory
            found = False
            for blob_dir in blob_dirs:
                img_path = blob_dir / filename
                if img_path.exists():
                    accessible += 1
                    found = True
                    break
            
            if not found:
                print(f"  Warning: Could not find {filename}")
        
        print(f"✓ Sample check: {accessible}/{len(test_samples)} images accessible")
        
        if accessible == 0:
            print("✗ ERROR: No images accessible. Check blob directory structure.")
            print(f"  Searched in: {self.data_root}")
            print(f"  Blob dirs: {[d.name for d in blob_dirs]}")
            sys.exit(1)
        
        # Get camera intrinsics
        print("\nExtracting camera intrinsics...")
        cam_front_calibs = [c for c in self.calibrated_sensor 
                        if c['sensor_token'] == cam_front_sensor['token']]
        
        if cam_front_calibs:
            calib = cam_front_calibs[0]
            intrinsic = calib['camera_intrinsic']
            print(f"✓ Camera intrinsic matrix:")
            print(f"  fx={intrinsic[0][0]:.2f}, fy={intrinsic[1][1]:.2f}")
            print(f"  cx={intrinsic[0][2]:.2f}, cy={intrinsic[1][2]:.2f}")
            print(f"  Image resolution: {sd['width']}×{sd['height']}")
        
        self.cam_front_samples = cam_front_samples
        self.cam_front_sensor_token = cam_front_sensor['token']
        
        print("\n✓ CAM_FRONT analysis complete\n")

    def analyze_annotations(self):
        """Analyze 3D annotation statistics"""
        print("=" * 60)
        print("STEP 4: 3D Annotation Analysis")
        print("=" * 60)
        
        # Build instance lookup for category resolution
        print("Building instance→category lookup...")
        instance_lookup = {inst['token']: inst for inst in self.instance}
        print(f"✓ Indexed {len(instance_lookup)} instances")
        
        # Category distribution
        category_counts = defaultdict(int)
        for ann in self.sample_annotation:
            inst_token = ann['instance_token']
            inst = instance_lookup.get(inst_token)
            if inst:
                cat_token = inst['category_token']
                cat = next((c for c in self.category if c['token'] == cat_token), None)
                if cat:
                    category_counts[cat['name']] += 1
        
        print("Object category distribution:")
        # Sort by count
        for cat_name, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat_name:30s}: {count:6d}")
        
        # Filter vehicle annotations
        vehicle_categories = ['vehicle.car', 'vehicle.truck', 'vehicle.bus', 
                            'vehicle.construction', 'vehicle.trailer']
        vehicle_count = sum(count for cat, count in category_counts.items() 
                        if any(vc in cat for vc in vehicle_categories))
        
        print(f"\n✓ Total vehicle annotations: {vehicle_count}")
        print(f"✓ Total annotations: {len(self.sample_annotation)}")
        
        # Analyze 3D box parameters
        print("\nAnalyzing 3D bounding box dimensions (first 1000 vehicle samples)...")
        vehicle_sizes = []
        checked = 0
        for ann in self.sample_annotation:
            if checked >= 1000:
                break
            inst = instance_lookup.get(ann['instance_token'])
            if inst:
                cat = next((c for c in self.category if c['token'] == inst['category_token']), None)
                if cat and 'vehicle' in cat['name']:
                    vehicle_sizes.append(ann['size'])  # [width, length, height] in nuScenes
                    checked += 1
        
        if vehicle_sizes:
            import numpy as np
            sizes_array = np.array(vehicle_sizes)
            print(f"  Width  (mean±std): {sizes_array[:, 0].mean():.2f}±{sizes_array[:, 0].std():.2f}m")
            print(f"  Length (mean±std): {sizes_array[:, 1].mean():.2f}±{sizes_array[:, 1].std():.2f}m")
            print(f"  Height (mean±std): {sizes_array[:, 2].mean():.2f}±{sizes_array[:, 2].std():.2f}m")
        
        print("\n✓ Annotation analysis complete\n")

    def summarize(self):
        """Print final summary"""
        print("=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)

        print(f"Dataset Version:       {NUSCENES_VERSION}")
        print(f"Total Scenes:          {len(self.sample)}")
        print(f"CAM_FRONT Samples:     {len(self.cam_front_samples)}")
        print(f"Total 3D Annotations:  {len(self.sample_annotation)}")

        # Estimate training samples (keyframe samples with annotations)
        keyframe_samples = [s for s in self.sample_data if s['is_key_frame']]
        print(f"Keyframe Samples:      {len(keyframe_samples)}")

        print("\n" + "=" * 60)
        print("✓ nuScenes dataset verified and ready for use")
        print("=" * 60)

        return True

def main():
    verifier = NuScenesVerifier()

    try:
        verifier.verify_paths()
        verifier.load_annotations()
        verifier.analyze_cam_front()
        verifier.analyze_annotations()
        verifier.summarize()

        print("\n✓ SUCCESS: Dataset verification complete")
        return 0

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
