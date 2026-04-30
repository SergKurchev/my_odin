#!/usr/bin/env python3
"""Test NBV Stage2 data loader with real data"""

import json
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

# Import from my_train_odin.py
from my_train_odin import (
    NBV_CATEGORIES,
    NBV_PRIMITIVE_NAMES,
    NBV_TEXTURE_NAMES,
    NBV_PRIMITIVE_TEXTURE_TO_CLASS,
    get_nbv_stage2_dataset_dicts
)

# Test with real local dataset
DATASET_DIR = r"C:\Users\NeverGonnaGiveYouUp\OneDrive\Рабочий стол\study_materials\Skoltech\Reinforcement_Learning\Project 4 TEST\NBV_with_obstacles_and_robot\dataset\primitives\stage2"

print("=== Testing NBV Stage2 Data Loader ===\n")

# Check if dataset exists
if not Path(DATASET_DIR).exists():
    print(f"ERROR: Dataset not found at {DATASET_DIR}")
    sys.exit(1)

print(f"Dataset directory: {DATASET_DIR}")
print(f"Total classes: {len(NBV_CATEGORIES)}\n")

# Create a minimal splits file for testing
splits_file = "test_splits.json"
sample_dirs = sorted([d.name for d in Path(DATASET_DIR).iterdir() if d.is_dir() and d.name.startswith("sample_")])
print(f"Found {len(sample_dirs)} samples")

# Use first 3 samples for testing
test_samples = sample_dirs[:3]
splits = {"train": test_samples, "val": [], "test": []}

with open(splits_file, 'w') as f:
    json.dump(splits, f)

print(f"Test samples: {test_samples}\n")

# Test the data loader
print("=== Testing get_nbv_stage2_dataset_dicts() ===\n")

try:
    dataset_dicts = get_nbv_stage2_dataset_dicts(DATASET_DIR, splits_file, "train")
    print(f"✓ Loaded {len(dataset_dicts)} chunks\n")

    if len(dataset_dicts) > 0:
        # Analyze first chunk
        first_chunk = dataset_dicts[0]
        print(f"First chunk info:")
        print(f"  Sample ID: {first_chunk['sample_id']}")
        print(f"  Part ID: {first_chunk['part_id']}")
        print(f"  Num frames: {len(first_chunk['file_names'])}")
        print(f"  Image size: {first_chunk['width']}x{first_chunk['height']}")

        # Check color_map mapping
        color_map = first_chunk['color_map']
        print(f"\n  Color map entries: {len(color_map)}")

        # Analyze class distribution
        class_ids = set()
        for color, info in color_map.items():
            class_ids.add(info['category_id'])

        print(f"  Unique class IDs in this chunk: {sorted(class_ids)}")
        print(f"\n  Class mapping examples:")

        for color, info in list(color_map.items())[:5]:
            class_id = info['category_id']
            class_name = NBV_CATEGORIES.get(class_id, "UNKNOWN")
            print(f"    Color {color} -> class {class_id} ({class_name})")

        # Check all chunks for class distribution
        all_class_ids = set()
        for chunk in dataset_dicts:
            for color, info in chunk['color_map'].items():
                all_class_ids.add(info['category_id'])

        print(f"\n=== Overall Statistics ===")
        print(f"Total chunks: {len(dataset_dicts)}")
        print(f"Unique classes found: {len(all_class_ids)}")
        print(f"Class IDs: {sorted(all_class_ids)}")
        print(f"\nClass names:")
        for class_id in sorted(all_class_ids):
            print(f"  {class_id:2d}: {NBV_CATEGORIES[class_id]}")

        # Verify no robot class
        if 9 in all_class_ids:
            print("\n⚠ WARNING: Robot class (9) found in data! Should be filtered out.")
        else:
            print("\n✓ Robot class correctly filtered out")

        # Check for any class IDs >= 24
        invalid_classes = [c for c in all_class_ids if c >= 24]
        if invalid_classes:
            print(f"\n⚠ WARNING: Invalid class IDs found: {invalid_classes}")
        else:
            print("✓ All class IDs are valid (0-23)")

        print("\n=== Test PASSED ===")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    # Cleanup
    if Path(splits_file).exists():
        Path(splits_file).unlink()
