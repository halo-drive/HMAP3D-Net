"""Check KITTI class distribution"""

from pathlib import Path
from collections import Counter

data_root = Path('/media/ashwin-benchdev/eb2a8889-3ec9-411a-826e-816cf9759b02/KITTI-3D')
label_dir = data_root / 'training' / 'label_2'

print("Analyzing KITTI dataset classes...\n")

all_classes = []
class_counts = Counter()

for label_file in label_dir.glob('*.txt'):
    with open(label_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(' ')
            if len(parts) < 15:
                continue
            
            cls = parts[0]
            all_classes.append(cls)
            class_counts[cls] += 1

print("KITTI 3D Object Classes:")
print("=" * 50)
for cls, count in class_counts.most_common():
    print(f"  {cls:20s}: {count:6d} instances")

print("\n" + "=" * 50)
print(f"Total instances: {len(all_classes)}")
print(f"Unique classes: {len(class_counts)}")

# Check which have sufficient data for training
print("\nRecommended classes (>1000 instances):")
for cls, count in class_counts.most_common():
    if count > 1000:
        print(f"  ✓ {cls} ({count} instances)")
    else:
        print(f"  ✗ {cls} ({count} instances - too few)")
