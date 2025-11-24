#!/usr/bin/env python3
"""
Depth Distribution Analysis for nuScenes CAM_FRONT Vehicle Annotations
Analyzes 3D vehicle positions to determine optimal depth bin boundaries.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Dataset paths
NUSCENES_ROOT = "/media/ashwin-benchdev/eb2a8889-3ec9-411a-826e-816cf9759b02/NuScenes-Full-Dataset"
NUSCENES_VERSION = "v1.0-trainval"
NUSCENES_DATAROOT = f"{NUSCENES_ROOT}/NuScenes/trainval"
NUSCENES_ANNOTATIONS = f"{NUSCENES_ROOT}/nuscenes_prepared/{NUSCENES_VERSION}"

class DepthAnalyzer:
    def __init__(self):
        self.annotation_path = Path(NUSCENES_ANNOTATIONS)
        self.output_dir = Path.home() / "NetAsh3D" / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data tables
        self.sample = None
        self.sample_data = None
        self.sample_annotation = None
        self.calibrated_sensor = None
        self.ego_pose = None
        self.instance = None
        self.category = None
        self.sensor = None
        
        # Analysis results
        self.vehicle_depths = []
        
    def load_data(self):
        """Load required annotation tables"""
        print("=" * 60)
        print("Loading nuScenes Annotations")
        print("=" * 60)
        
        json_files = {
            'sample': 'sample.json',
            'sample_data': 'sample_data.json',
            'sample_annotation': 'sample_annotation.json',
            'calibrated_sensor': 'calibrated_sensor.json',
            'ego_pose': 'ego_pose.json',
            'instance': 'instance.json',
            'category': 'category.json',
            'sensor': 'sensor.json',
        }
        
        for attr, filename in json_files.items():
            filepath = self.annotation_path / filename
            print(f"Loading {filename}... ", end='', flush=True)
            with open(filepath, 'r') as f:
                data = json.load(f)
            setattr(self, attr, data)
            print(f"✓ {len(data)} records")
        
        print()
    
    def build_lookups(self):
        """Build lookup dictionaries for fast access"""
        print("Building lookup indices...")
        
        self.instance_lookup = {inst['token']: inst for inst in self.instance}
        self.category_lookup = {cat['token']: cat for cat in self.category}
        self.calib_lookup = {c['token']: c for c in self.calibrated_sensor}
        self.ego_lookup = {e['token']: e for e in self.ego_pose}
        self.sample_data_lookup = {sd['token']: sd for sd in self.sample_data}
        
        # Build sample_token → annotations mapping
        self.sample_annotations = defaultdict(list)
        for ann in self.sample_annotation:
            self.sample_annotations[ann['sample_token']].append(ann)
        
        # Find CAM_FRONT sensor
        self.cam_front_sensor = next(
            (s for s in self.sensor if s['channel'] == 'CAM_FRONT'), None
        )
        
        print(f"✓ Indexed {len(self.instance_lookup)} instances")
        print(f"✓ Indexed {len(self.sample_annotations)} samples with annotations")
        print(f"✓ CAM_FRONT sensor: {self.cam_front_sensor['token']}\n")
    
    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion [w, x, y, z] to 3x3 rotation matrix"""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
    
    def transform_to_camera_frame(self, point_global, ego_pose, calib_sensor):
        """
        Transform point from global frame to camera frame.
        
        Args:
            point_global: [x, y, z] in global/ego coordinates
            ego_pose: ego_pose record (vehicle position in global frame)
            calib_sensor: calibrated_sensor record (sensor position in ego frame)
        
        Returns:
            [x, y, z] in camera frame
        """
        point_global = np.array(point_global)
        
        # Global → Ego transformation
        ego_translation = np.array(ego_pose['translation'])
        ego_rotation = self.quaternion_to_rotation_matrix(ego_pose['rotation'])
        
        # Ego → Sensor transformation
        sensor_translation = np.array(calib_sensor['translation'])
        sensor_rotation = self.quaternion_to_rotation_matrix(calib_sensor['rotation'])
        
        # Transform: Global → Ego
        point_ego = ego_rotation.T @ (point_global - ego_translation)
        
        # Transform: Ego → Camera
        point_camera = sensor_rotation.T @ (point_ego - sensor_translation)
        
        return point_camera
    
    def is_vehicle_category(self, category_name):
        """Check if category is a vehicle"""
        vehicle_types = ['vehicle.car', 'vehicle.truck', 'vehicle.bus', 
                        'vehicle.construction', 'vehicle.trailer', 'vehicle.motorcycle']
        return any(vtype in category_name for vtype in vehicle_types)
    
    def analyze_depths(self):
        """Extract depth values for all vehicle annotations in CAM_FRONT view"""
        print("=" * 60)
        print("Analyzing Vehicle Depths")
        print("=" * 60)
        
        # Build sample_data index by sample_token for CAM_FRONT only
        print("Building CAM_FRONT sample_data index...")
        cam_front_sample_data = {}
        for sd in self.sample_data:
            calib = self.calib_lookup.get(sd['calibrated_sensor_token'])
            if calib and calib['sensor_token'] == self.cam_front_sensor['token']:
                if sd.get('is_key_frame', False):  # Only keyframes
                    cam_front_sample_data[sd['sample_token']] = sd
        
        print(f"✓ Found {len(cam_front_sample_data)} CAM_FRONT keyframe samples")
        
        keyframe_samples = [s for s in self.sample]
        print(f"Processing {len(keyframe_samples)} samples...")
        
        processed = 0
        vehicles_found = 0
        
        for i, sample in enumerate(keyframe_samples):
            if i % 1000 == 0:
                print(f"  Progress: {i}/{len(keyframe_samples)} samples, "
                    f"{vehicles_found} vehicles found", end='\r')
            
            # Get CAM_FRONT sample_data for this sample
            sample_data = cam_front_sample_data.get(sample['token'])
            if not sample_data:
                continue
            
            # Get ego pose and calibration
            ego_pose = self.ego_lookup[sample_data['ego_pose_token']]
            calib_sensor = self.calib_lookup[sample_data['calibrated_sensor_token']]
            
            # Get annotations for this sample
            annotations = self.sample_annotations.get(sample['token'], [])
            
            for ann in annotations:
                # Filter for vehicles only
                instance = self.instance_lookup.get(ann['instance_token'])
                if not instance:
                    continue
                
                category = self.category_lookup.get(instance['category_token'])
                if not category or not self.is_vehicle_category(category['name']):
                    continue
                
                # Transform annotation position to camera frame
                point_global = ann['translation']
                point_camera = self.transform_to_camera_frame(
                    point_global, ego_pose, calib_sensor
                )
                
                # Depth is Z coordinate in camera frame
                depth = point_camera[2]
                
                # Filter reasonable depths (positive, within sensor range)
                if 0 < depth < 200:  # Max 200m
                    self.vehicle_depths.append(depth)
                    vehicles_found += 1
            
            processed += 1
        
        print(f"\n✓ Processed {processed} samples")
        print(f"✓ Found {len(self.vehicle_depths)} vehicle depth measurements\n")

    def compute_statistics(self):
        """Compute depth distribution statistics"""
        print("=" * 60)
        print("Depth Distribution Statistics")
        print("=" * 60)
        
        depths = np.array(self.vehicle_depths)
        
        print(f"Total samples:        {len(depths)}")
        print(f"Minimum depth:        {depths.min():.2f}m")
        print(f"Maximum depth:        {depths.max():.2f}m")
        print(f"Mean depth:           {depths.mean():.2f}m")
        print(f"Median depth:         {np.median(depths):.2f}m")
        print(f"Std deviation:        {depths.std():.2f}m")
        print()
        
        # Percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print("Percentile analysis:")
        for p in percentiles:
            val = np.percentile(depths, p)
            print(f"  {p:2d}th percentile:    {val:6.2f}m")
        print()
        
        # Distance ranges
        ranges = [
            (0, 10),
            (10, 20),
            (20, 30),
            (30, 40),
            (40, 50),
            (50, 60),
            (60, 80),
            (80, 100),
            (100, 200)
        ]
        
        print("Distribution by distance range:")
        for start, end in ranges:
            count = np.sum((depths >= start) & (depths < end))
            pct = 100 * count / len(depths)
            print(f"  {start:3d}-{end:3d}m: {count:7d} ({pct:5.2f}%)")
        print()
    
    def recommend_bins(self):
        """Recommend depth bin boundaries based on distribution"""
        print("=" * 60)
        print("Recommended Depth Bin Configurations")
        print("=" * 60)
        
        depths = np.array(self.vehicle_depths)
        
        # Strategy 1: Equal percentile bins
        print("\n1. Equal Percentile Bins (4 bins):")
        bins_4 = [
            depths.min(),
            np.percentile(depths, 25),
            np.percentile(depths, 50),
            np.percentile(depths, 75),
            depths.max()
        ]
        print(f"   Bins: [0-{bins_4[1]:.1f}m] [{bins_4[1]:.1f}-{bins_4[2]:.1f}m] "
              f"[{bins_4[2]:.1f}-{bins_4[3]:.1f}m] [{bins_4[3]:.1f}m+]")
        
        # Strategy 2: Practical distance bins (6 bins)
        print("\n2. Practical Distance Bins (6 bins) - RECOMMENDED:")
        bins_6 = [0, 10, 20, 35, 50, 70, 200]
        for i in range(len(bins_6)-1):
            count = np.sum((depths >= bins_6[i]) & (depths < bins_6[i+1]))
            pct = 100 * count / len(depths)
            print(f"   Bin {i}: [{bins_6[i]:3d}-{bins_6[i+1]:3d}m): {count:7d} samples ({pct:5.2f}%)")
        
        # Strategy 3: Fine-grained bins (8 bins)
        print("\n3. Fine-grained Bins (8 bins):")
        bins_8 = [0, 8, 16, 25, 35, 45, 60, 80, 200]
        for i in range(len(bins_8)-1):
            count = np.sum((depths >= bins_8[i]) & (depths < bins_8[i+1]))
            pct = 100 * count / len(depths)
            print(f"   Bin {i}: [{bins_8[i]:3d}-{bins_8[i+1]:3d}m): {count:7d} samples ({pct:5.2f}%)")
        
        print("\n" + "=" * 60)
        print("RECOMMENDATION:")
        print("=" * 60)
        print("Use 6-bin configuration: [0-10, 10-20, 20-35, 35-50, 50-70, 70+]m")
        print("Rationale:")
        print("  - Covers 95%+ of vehicle detections")
        print("  - Finer resolution in critical near range (0-35m)")
        print("  - Coarser bins for distant vehicles (less depth precision needed)")
        print("  - Manageable network output size (6 bins)")
        print("=" * 60)
        print()
        
        return bins_6
    
    def plot_histogram(self, bins_recommended):
        """Generate depth distribution histogram"""
        print("Generating depth histogram...")
        
        depths = np.array(self.vehicle_depths)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Full histogram
        ax1 = axes[0]
        ax1.hist(depths, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(np.median(depths), color='red', linestyle='--', 
                   label=f'Median: {np.median(depths):.1f}m')
        ax1.axvline(np.percentile(depths, 95), color='orange', linestyle='--',
                   label=f'95th %ile: {np.percentile(depths, 95):.1f}m')
        ax1.set_xlabel('Depth (meters)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('nuScenes CAM_FRONT Vehicle Depth Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Bin-based histogram with recommended bins
        ax2 = axes[1]
        ax2.hist(depths, bins=bins_recommended, alpha=0.7, color='forestgreen', edgecolor='black')
        for bin_edge in bins_recommended[1:-1]:
            ax2.axvline(bin_edge, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
        ax2.set_xlabel('Depth (meters)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Recommended 6-Bin Configuration', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add bin labels
        for i in range(len(bins_recommended)-1):
            mid = (bins_recommended[i] + bins_recommended[i+1]) / 2
            count = np.sum((depths >= bins_recommended[i]) & (depths < bins_recommended[i+1]))
            ax2.text(mid, ax2.get_ylim()[1]*0.9, f'Bin {i}\n{count} samples',
                    ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'depth_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Histogram saved: {output_path}\n")
        
        plt.close()
    
    def save_results(self, bins_recommended):
        """Save analysis results to file"""
        output_path = self.output_dir / 'depth_analysis_summary.txt'
        
        depths = np.array(self.vehicle_depths)
        
        with open(output_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("nuScenes CAM_FRONT Vehicle Depth Distribution Analysis\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Total vehicle samples: {len(depths)}\n")
            f.write(f"Depth range: {depths.min():.2f}m - {depths.max():.2f}m\n")
            f.write(f"Mean: {depths.mean():.2f}m, Median: {np.median(depths):.2f}m\n")
            f.write(f"Std: {depths.std():.2f}m\n\n")
            
            f.write("RECOMMENDED CONFIGURATION FOR NETWORK:\n")
            f.write("-" * 70 + "\n")
            f.write("Number of depth bins: 6\n")
            f.write(f"Bin boundaries: {bins_recommended}\n\n")
            
            f.write("Bin configuration for network head:\n")
            for i in range(len(bins_recommended)-1):
                count = np.sum((depths >= bins_recommended[i]) & (depths < bins_recommended[i+1]))
                pct = 100 * count / len(depths)
                f.write(f"  Bin {i}: [{bins_recommended[i]:3d}-{bins_recommended[i+1]:3d}m) "
                       f"- {count:7d} samples ({pct:5.2f}%)\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"✓ Summary saved: {output_path}\n")

def main():
    analyzer = DepthAnalyzer()
    
    try:
        analyzer.load_data()
        analyzer.build_lookups()
        analyzer.analyze_depths()
        analyzer.compute_statistics()
        bins_recommended = analyzer.recommend_bins()
        analyzer.plot_histogram(bins_recommended)
        analyzer.save_results(bins_recommended)
        
        print("=" * 60)
        print("✓ DEPTH ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {analyzer.output_dir}/")
        print("  - depth_distribution.png")
        print("  - depth_analysis_summary.txt")
        print("\nNext step: Use recommended 6-bin configuration in network architecture")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
