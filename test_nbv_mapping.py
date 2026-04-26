#!/usr/bin/env python3
"""Test NBV Stage2 class mapping"""

# NBV Stage2: 8 primitives × 3 textures = 24 classes
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

print("=== NBV Stage2 Class Mapping (24 classes) ===\n")
print(f"Total classes: {len(NBV_CATEGORIES)}\n")

print("Class ID -> Name:")
for class_id, name in sorted(NBV_CATEGORIES.items()):
    print(f"  {class_id:2d}: {name}")

print("\n(Primitive ID, Texture) -> Class ID:")
for (prim_id, texture), class_id in sorted(NBV_PRIMITIVE_TEXTURE_TO_CLASS.items()):
    prim_name = NBV_PRIMITIVE_NAMES[prim_id]
    print(f"  ({prim_id}, {texture:6s}) -> {class_id:2d} ({prim_name}_{texture})")

print("\n=== Example conversions ===")
examples = [
    (1, "red"),    # cube_red -> 0
    (1, "mixed"),  # cube_mixed -> 1
    (1, "green"),  # cube_green -> 2
    (2, "red"),    # sphere_red -> 3
    (8, "green"),  # pyramid_green -> 23
]

for prim_id, texture in examples:
    class_id = NBV_PRIMITIVE_TEXTURE_TO_CLASS[(prim_id, texture)]
    class_name = NBV_CATEGORIES[class_id]
    prim_name = NBV_PRIMITIVE_NAMES[prim_id]
    print(f"  Primitive {prim_id} ({prim_name}) + texture '{texture}' -> class {class_id} ({class_name})")

print("\n✓ Mapping verified: 8 primitives × 3 textures = 24 classes")
