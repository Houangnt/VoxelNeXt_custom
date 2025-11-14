import numpy as np
import glob
import os

# Paths
points_dir = '/home/ducanh/HoangNT/VoxelNeXt/PCDet/points'
labels_dir = '/home/ducanh/HoangNT/VoxelNeXt/PCDet/labels'

print("=" * 80)
print("ANALYZING ALL POINT CLOUDS")
print("=" * 80)

# Get all bin files
bin_files = sorted(glob.glob(os.path.join(points_dir, '*.bin')))
print(f"\nFound {len(bin_files)} point cloud files\n")

# Initialize min/max
x_min, x_max = float('inf'), float('-inf')
y_min, y_max = float('inf'), float('-inf')
z_min, z_max = float('inf'), float('-inf')

total_points = 0

# Process all point clouds
for i, bin_file in enumerate(bin_files):
    try:
        points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
        
        x_min = min(x_min, points[:, 0].min())
        x_max = max(x_max, points[:, 0].max())
        y_min = min(y_min, points[:, 1].min())
        y_max = max(y_max, points[:, 1].max())
        z_min = min(z_min, points[:, 2].min())
        z_max = max(z_max, points[:, 2].max())
        
        total_points += len(points)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(bin_files)} files...")
    except Exception as e:
        print(f"Error processing {bin_file}: {e}")

print("\n" + "=" * 80)
print("POINT CLOUD STATISTICS")
print("=" * 80)
print(f"Total points across all files: {total_points:,}")
print(f"\nActual data range:")
print(f"  X: [{x_min:.2f}, {x_max:.2f}] (width: {x_max - x_min:.2f}m)")
print(f"  Y: [{y_min:.2f}, {y_max:.2f}] (width: {y_max - y_min:.2f}m)")
print(f"  Z: [{z_min:.2f}, {z_max:.2f}] (height: {z_max - z_min:.2f}m)")

# Analyze GT boxes
print("\n" + "=" * 80)
print("ANALYZING ALL GT BOXES")
print("=" * 80)

label_files = sorted(glob.glob(os.path.join(labels_dir, '*.txt')))
print(f"\nFound {len(label_files)} label files\n")

gt_x_min, gt_x_max = float('inf'), float('-inf')
gt_y_min, gt_y_max = float('inf'), float('-inf')
gt_z_min, gt_z_max = float('inf'), float('-inf')

total_boxes = 0

for i, label_file in enumerate(label_files):
    try:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 8:
                    x, y, z, l, w, h, ry, cls = parts
                    x, y, z = float(x), float(y), float(z)
                    l, w, h = float(l), float(w), float(h)
                    
                    # Consider box extent (center +/- size/2)
                    gt_x_min = min(gt_x_min, x - l/2)
                    gt_x_max = max(gt_x_max, x + l/2)
                    gt_y_min = min(gt_y_min, y - w/2)
                    gt_y_max = max(gt_y_max, y + w/2)
                    gt_z_min = min(gt_z_min, z - h/2)
                    gt_z_max = max(gt_z_max, z + h/2)
                    
                    total_boxes += 1
    except Exception as e:
        print(f"Error processing {label_file}: {e}")

print(f"Total GT boxes: {total_boxes}")
print(f"\nGT boxes range (including box extent):")
print(f"  X: [{gt_x_min:.2f}, {gt_x_max:.2f}] (width: {gt_x_max - gt_x_min:.2f}m)")
print(f"  Y: [{gt_y_min:.2f}, {gt_y_max:.2f}] (width: {gt_y_max - gt_y_min:.2f}m)")
print(f"  Z: [{gt_z_min:.2f}, {gt_z_max:.2f}] (height: {gt_z_max - gt_z_min:.2f}m)")

# Calculate recommended range with buffer
print("\n" + "=" * 80)
print("RECOMMENDED POINT_CLOUD_RANGE")
print("=" * 80)

# Use point cloud range as base, add small buffer
buffer = 1.0  # 1 meter buffer

# For X and Y, use symmetric range if possible, otherwise use actual range
x_range_min = max(0, np.floor(x_min - buffer))
x_range_max = np.ceil(x_max + buffer)
y_range_min = np.floor(y_min - buffer)
y_range_max = np.ceil(y_max + buffer)
z_range_min = np.floor(z_min - buffer)
z_range_max = np.ceil(z_max + buffer)

# Make Y symmetric around 0 for better performance
y_abs_max = max(abs(y_range_min), abs(y_range_max))
y_range_min = -y_abs_max
y_range_max = y_abs_max

print(f"\nOption 1 - Tight fit (covers all data with 1m buffer):")
pc_range_tight = [x_range_min, y_range_min, z_range_min, x_range_max, y_range_max, z_range_max]
print(f"  POINT_CLOUD_RANGE: {pc_range_tight}")

# Option 2 - Round to nice numbers
x_range_min_round = 0
x_range_max_round = int(np.ceil(x_range_max / 10) * 10)
y_range_abs_round = int(np.ceil(y_abs_max / 10) * 10)
z_range_min_round = int(np.floor(z_range_min / 5) * 5)
z_range_max_round = int(np.ceil(z_range_max / 5) * 5)

pc_range_round = [x_range_min_round, -y_range_abs_round, z_range_min_round, 
                  x_range_max_round, y_range_abs_round, z_range_max_round]
print(f"\nOption 2 - Rounded (nice numbers, easier to remember):")
print(f"  POINT_CLOUD_RANGE: {pc_range_round}")

# Verify coverage
print("\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)

def check_coverage(pc_range, x_min, x_max, y_min, y_max, z_min, z_max, 
                   gt_x_min, gt_x_max, gt_y_min, gt_y_max, gt_z_min, gt_z_max):
    points_covered = (
        x_min >= pc_range[0] and x_max <= pc_range[3] and
        y_min >= pc_range[1] and y_max <= pc_range[4] and
        z_min >= pc_range[2] and z_max <= pc_range[5]
    )
    
    boxes_covered = (
        gt_x_min >= pc_range[0] and gt_x_max <= pc_range[3] and
        gt_y_min >= pc_range[1] and gt_y_max <= pc_range[4] and
        gt_z_min >= pc_range[2] and gt_z_max <= pc_range[5]
    )
    
    return points_covered, boxes_covered

for i, (name, pc_range) in enumerate([("Option 1", pc_range_tight), 
                                       ("Option 2", pc_range_round)], 1):
    points_ok, boxes_ok = check_coverage(
        pc_range, x_min, x_max, y_min, y_max, z_min, z_max,
        gt_x_min, gt_x_max, gt_y_min, gt_y_max, gt_z_min, gt_z_max
    )
    
    print(f"\n{name}: {pc_range}")
    print(f"  ✓ All points covered: {points_ok}")
    print(f"  ✓ All GT boxes covered: {boxes_ok}")
    
    if points_ok and boxes_ok:
        print(f"  RECOMMENDED - This range covers all data!")

print("\n" + "=" * 80)
print("Copy one of the recommended ranges to your custom_dataset.yaml:")
print("=" * 80)
print(f"\nPOINT_CLOUD_RANGE: {pc_range_round}")
print("\n" + "=" * 80)
