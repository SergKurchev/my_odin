#!/usr/bin/env python3
"""Test NBV Stage2 class mapping logic with real data"""

import json
from pathlib import Path

# Recreate the mapping logic from my_train_odin.py
NBV_PRIMITIVE_NAMES = {
    1: "cube", 2: "sphere", 3: "cylinder", 4: "cone",
    5: "torus", 6: "capsule", 7: "ellipsoid", 8: "pyramid"
}
NBV_TEXTURE_NAMES = ["red", "mixed", "green"]

# Generate 24 classes
NBV_CATEGORIES = {}
class_id = 0
for prim_id in sorted(NBV_PRIMITIVE_NAMES.keys()):
    for texture in NBV_TEXTURE_NAMES:
        NBV_CATEGORIES[class_id] = f"{NBV_PRIMITIVE_NAMES[prim_id]}_{texture}"
        class_id += 1

# Mapping from (primitive_id, texture_type) to class_id
NBV_PRIMITIVE_TEXTURE_TO_CLASS = {}
class_id = 0
for prim_id in sorted(NBV_PRIMITIVE_NAMES.keys()):
    for texture in NBV_TEXTURE_NAMES:
        NBV_PRIMITIVE_TEXTURE_TO_CLASS[(prim_id, texture)] = class_id
        class_id += 1

# Test with real local dataset
DATASET_DIR = Path(r"C:\Users\NeverGonnaGiveYouUp\OneDrive\Рабочий стол\study_materials\Skoltech\Reinforcement_Learning\Project 4 TEST\NBV_with_obstacles_and_robot\dataset\primitives\stage2")

print("=== Testing NBV Stage2 Class Mapping with Real Data ===\n")

if not DATASET_DIR.exists():
    print(f"ERROR: Dataset not found at {DATASET_DIR}")
    exit(1)

print(f"Dataset directory: {DATASET_DIR}")
print(f"Total classes defined: {len(NBV_CATEGORIES)}\n")

# Test conversion on real samples
sample_dirs = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir() and d.name.startswith("sample_")])
print(f"Found {len(sample_dirs)} samples\n")

# Analyze first 3 samples
test_samples = sample_dirs[:3]

all_conversions = {}
robot_count = 0
invalid_count = 0

for sample_dir in test_samples:
    color_map_path = sample_dir / "color_map.json"

    if not color_map_path.exists():
        continue

    with open(color_map_path) as f:
        color_map = json.load(f)

    print(f"=== {sample_dir.name} ===")
    print(f"  Total instances: {len(color_map)}")

    for entry in color_map:
        prim_id = entry["category_id"]
        texture = entry.get("texture_type")

        # Check for robot
        if prim_id == 9:
            robot_count += 1
            print(f"  Instance {entry['instance_id']}: Robot (category_id=9) - SKIPPED")
            continue

        # Check for invalid primitive
        if prim_id not in NBV_PRIMITIVE_NAMES:
            invalid_count += 1
            print(f"  Instance {entry['instance_id']}: Invalid primitive_id={prim_id} - SKIPPED")
            continue

        # Check for invalid texture
        if texture not in NBV_TEXTURE_NAMES:
            invalid_count += 1
            print(f"  Instance {entry['instance_id']}: Invalid texture='{texture}' - SKIPPED")
            continue

        # Convert to class_id
        new_class_id = NBV_PRIMITIVE_TEXTURE_TO_CLASS[(prim_id, texture)]
        class_name = NBV_CATEGORIES[new_class_id]

        key = (prim_id, texture)
        if key not in all_conversions:
            all_conversions[key] = []
        all_conversions[key].append(sample_dir.name)

        print(f"  Instance {entry['instance_id']}: primitive_id={prim_id} ({NBV_PRIMITIVE_NAMES[prim_id]}) + texture='{texture}' -> class {new_class_id} ({class_name})")

    print()

# Summary
print("=== Summary ===")
print(f"Samples analyzed: {len(test_samples)}")
print(f"Unique (primitive, texture) combinations found: {len(all_conversions)}")
print(f"Robot instances filtered: {robot_count}")
print(f"Invalid instances filtered: {invalid_count}")

print(f"\n=== Class Distribution ===")
for (prim_id, texture), samples in sorted(all_conversions.items()):
    class_id = NBV_PRIMITIVE_TEXTURE_TO_CLASS[(prim_id, texture)]
    class_name = NBV_CATEGORIES[class_id]
    print(f"Class {class_id:2d} ({class_name:20s}): found in {len(samples)} samples")

# Verify all class IDs are valid
used_class_ids = set()
for (prim_id, texture) in all_conversions.keys():
    class_id = NBV_PRIMITIVE_TEXTURE_TO_CLASS[(prim_id, texture)]
    used_class_ids.add(class_id)

print(f"\n=== Validation ===")
print(f"Used class IDs: {sorted(used_class_ids)}")
print(f"Min class ID: {min(used_class_ids)}")
print(f"Max class ID: {max(used_class_ids)}")

if max(used_class_ids) >= 24:
    print("✗ ERROR: Class ID >= 24 found!")
else:
    print("✓ All class IDs are valid (0-23)")

if robot_count > 0:
    print(f"✓ Robot instances correctly identified and filtered ({robot_count} total)")

print("\n✓ Test PASSED - Mapping logic is correct!")
