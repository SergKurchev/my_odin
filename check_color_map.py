#!/usr/bin/env python3
"""
Quick script to check category_id values in color_map.json files.
This will tell us if categories are 0-indexed or 1-indexed in the dataset.
"""

import json
from pathlib import Path
import sys

def check_color_map(dataset_dir):
    """Check category_id values in color_map.json files."""
    dataset_path = Path(dataset_dir)

    # Find all color_map.json files
    color_maps = list(dataset_path.glob("**/color_map.json"))

    if not color_maps:
        print(f"No color_map.json files found in {dataset_dir}")
        return

    print(f"Found {len(color_maps)} color_map.json files\n")

    all_category_ids = set()

    for cm_path in color_maps[:5]:  # Check first 5 samples
        print(f"Checking: {cm_path.parent.name}/color_map.json")

        with open(cm_path, 'r') as f:
            color_map = json.load(f)

        # Handle both dict and list formats
        if isinstance(color_map, dict):
            items = color_map.values()
        elif isinstance(color_map, list):
            items = color_map
        else:
            print(f"  Unknown format: {type(color_map)}")
            continue

        for item in items:
            cat_id = item.get("category_id")
            cat_name = item.get("category_name", "unknown")
            if cat_id is not None:
                all_category_ids.add(cat_id)
                print(f"  category_id={cat_id}, category_name={cat_name}")
        print()

    print(f"\nAll unique category_id values found: {sorted(all_category_ids)}")

    if all_category_ids:
        min_id = min(all_category_ids)
        max_id = max(all_category_ids)

        if min_id == 0:
            print("✓ Categories are 0-indexed (Ripe=0, Unripe=1, Half-ripe=2)")
            print("  GT categories should NOT be decremented in visualization")
        elif min_id == 1:
            print("✓ Categories are 1-indexed (Ripe=1, Unripe=2, Half-ripe=3)")
            print("  GT categories SHOULD be decremented by 1 in visualization")
        else:
            print(f"⚠ Unexpected indexing: min={min_id}, max={max_id}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_color_map.py <dataset_dir>")
        print("Example: python check_color_map.py /path/to/multiview_dataset")
        sys.exit(1)

    check_color_map(sys.argv[1])
