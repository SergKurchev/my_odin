import os
import sys
import json
import numpy as np
import torch
import cv2
from pathlib import Path
from imageio import imread
from unittest.mock import MagicMock

# -----------------------------------------------------------------------------
# --- Extract plot_3d_strawberry from vis_utils.py ---
def get_function_source(file_path, func_name):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    start_line = -1
    for i, line in enumerate(lines):
        if line.startswith(f"def {func_name}("):
            start_line = i
            break
    
    if start_line == -1:
        return None
        
    func_lines = []
    # Find end by looking for next non-indented line after the function starts
    for i in range(start_line, len(lines)):
        if i > start_line and lines[i].strip() and not lines[i].startswith(" ") and not lines[i].startswith("\t"):
            break
        func_lines.append(lines[i])
    
    return "".join(func_lines)

vis_utils_path = Path("odin/utils/vis_utils.py")
func_source = get_function_source(vis_utils_path, "plot_3d_strawberry")
if func_source:
    print("Found plot_3d_strawberry in vis_utils.py. Executing source...")
    # Provide necessary globals for the function
    namespace = {"np": np, "os": os, "json": json}
    exec(func_source, namespace)
    plot_3d_strawberry = namespace["plot_3d_strawberry"]
else:
    print("Error: Could not find plot_3d_strawberry in vis_utils.py")
    sys.exit(1)

# -----------------------------------------------------------------------------
# 2. NBV Stage 2 Constants (from my_train_odin.py)
# -----------------------------------------------------------------------------
NBV_PRIMITIVE_NAMES = {
    1: "cube", 2: "sphere", 3: "cylinder", 4: "cone",
    5: "torus", 6: "capsule", 7: "ellipsoid", 8: "pyramid"
}
NBV_TEXTURE_NAMES = ["red", "mixed", "green"]

NBV_PRIMITIVE_TEXTURE_TO_CLASS = {}
class_id = 0
for prim_id in sorted(NBV_PRIMITIVE_NAMES.keys()):
    for texture in NBV_TEXTURE_NAMES:
        NBV_PRIMITIVE_TEXTURE_TO_CLASS[(prim_id, texture)] = class_id
        class_id += 1

def quat_to_rotmat(x, y, z, w):
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
    q = np.array([x, y, z, w], dtype=np.float32)
    q /= (np.linalg.norm(q) + 1e-12)
    x, y, z, w = q

    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    return R

# -----------------------------------------------------------------------------
# 3. Projection Logic (from backproject.py)
# -----------------------------------------------------------------------------
def unproject(intrinsics, poses, depths):
    """
    Exact replication of Strawberry3DEvaluator projection logic using NumPy.
    intrinsics: [V, 3, 3] (numpy)
    poses: [V, 4, 4] (numpy)
    depths: [V, H, W] (numpy)
    """
    V, H, W = depths.shape
    
    # Create coordinate grid
    u = np.arange(0, W, dtype=np.float32)
    v = np.arange(0, H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v) # Default indexing is 'xy'
    
    all_pts_world = []
    
    for i in range(V):
        fx, fy, cx, cy = intrinsics[i, 0, 0], intrinsics[i, 1, 1], intrinsics[i, 0, 2], intrinsics[i, 1, 2]
        Z = depths[i]
        
        # Back-project to camera coordinates
        X_cam = (uu - cx) * Z / fx
        Y_cam = -(vv - cy) * Z / fy
        Z_cam = Z
        
        pts_cam = np.stack([X_cam, Y_cam, Z_cam], axis=-1) # [H, W, 3]
        
        R = poses[i, :3, :3]
        t = poses[i, :3, 3]
        
        # pts_world = pts_cam @ R.T + t
        pts_world = np.dot(pts_cam, R.T) + t
        all_pts_world.append(pts_world)
        
    return np.stack(all_pts_world)

# -----------------------------------------------------------------------------
# 4. Main Verification Logic
# -----------------------------------------------------------------------------
def run_real_verification():
    dataset_dir = Path(r"C:\Users\NeverGonnaGiveYouUp\OneDrive\Рабочий стол\study_materials\Skoltech\Reinforcement_Learning\Project 4 TEST\NBV_with_obstacles_and_robot\dataset\primitives\stage2")
    sample_id = "sample_00000"
    s_dir = dataset_dir / sample_id
    
    print(f"Loading data from {s_dir}...")
    
    with open(s_dir / "cameras.json", "r") as f:
        cameras_dict = json.load(f)
    with open(s_dir / "color_map.json", "r") as f:
        color_map = json.load(f)
        
    # Pick first 5 frames like ODIN chunk
    frame_keys = sorted(cameras_dict.keys())[:5]
    
    depths = []
    rgbs = []
    masks = []
    intrinsics = []
    poses = []
    
    target_size = 224 # Match NBV Stage 2 native resolution
    
    for fk in frame_keys:
        cam_data = cameras_dict[fk]
        fi = int(fk)
        name = f"{fi:05d}"
        
        # Load RGB
        rgb = imread(s_dir / "rgb" / f"{name}.png")[..., :3]
        old_h, old_w = rgb.shape[:2]
        if (old_h, old_w) != (target_size, target_size):
            rgb = cv2.resize(rgb, (target_size, target_size))
        rgbs.append(rgb / 255.0)
        
        # Load Depth
        depth = np.load(s_dir / "depth" / f"{name}.npy")
        if depth.shape != (target_size, target_size):
            depth = cv2.resize(depth, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        depths.append(depth)
        
        # Intrinsics
        intr = cam_data["intrinsics"]
        K = np.array([
            [intr["fx"], 0, intr["cx"]],
            [0, intr["fy"], intr["cy"]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Scaling if needed
        if (old_h, old_w) != (target_size, target_size):
            K[0, 0] *= (target_size / old_w)
            K[0, 2] *= (target_size / old_w)
            K[1, 1] *= (target_size / old_h)
            K[1, 2] *= (target_size / old_h)
        intrinsics.append(K)
        
        # Pose
        R = quat_to_rotmat(*cam_data["rotation"])
        # Convert OpenGL (+Z backward) to ODIN (+Z forward)
        R[:, 2] = -R[:, 2]
        t = np.array(cam_data["position"], dtype=np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R
        pose[:3, 3] = t
        poses.append(pose)
        
        # Load Mask
        mask_img = imread(s_dir / "masks" / f"{name}.png").astype(np.uint8)
        mask_img = cv2.resize(mask_img, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        masks.append(mask_img)

    # Convert to Tensors
    depths_t = torch.from_numpy(np.stack(depths)).float()
    poses_t = torch.from_numpy(np.stack(poses)).float()
    intrinsics_t = torch.from_numpy(np.stack(intrinsics)).float()
    
    print("Performing back-projection...")
    our_pc = unproject(np.stack(intrinsics), np.stack(poses), np.stack(depths)) # [V, H, W, 3]
    
    # Reshape for visualizer
    pc_flat = our_pc.reshape(-1, 3)
    color_flat = np.stack(rgbs).reshape(-1, 3)
    
    # Prepare GT Masks and Labels
    # We need to map each instance from color_map to a binary mask over the whole point cloud
    gt_masks_list = []
    gt_labels_list = []
    
    print("Preparing GT masks for 24 classes...")
    for item in color_map:
        prim_id = item["category_id"]
        if prim_id == 9: continue # Robot
        if prim_id not in NBV_PRIMITIVE_NAMES: continue
        
        texture = item.get("texture_type")
        class_id = NBV_PRIMITIVE_TEXTURE_TO_CLASS.get((prim_id, texture))
        if class_id is None: continue
        
        # Create mask for this object across all frames
        obj_color = tuple(item["color"])
        full_obj_mask = []
        for m_img in masks:
            mr, mg, mb = m_img[:, :, 0], m_img[:, :, 1], m_img[:, :, 2]
            px = (mr == obj_color[0]) & (mg == obj_color[1]) & (mb == obj_color[2])
            full_obj_mask.append(px)
            
        full_obj_mask = np.stack(full_obj_mask).flatten() # Match flat PC
        if np.any(full_obj_mask):
            gt_masks_list.append(full_obj_mask.astype(np.float32))
            gt_labels_list.append(class_id)
            
    if not gt_masks_list:
        print("Warning: No objects found in sample!")
        gt_masks_np = None
        gt_labels_np = None
    else:
        gt_masks_np = np.stack(gt_masks_list) # [Objects, Points]
        gt_labels_np = np.array(gt_labels_list)
        
    print(f"Total points: {len(pc_flat)}")
    print(f"Total objects found: {len(gt_labels_list)}")
    
    output_dir = "real_verification_results"
    scene_name = f"real_logic_{sample_id}"
    
    print(f"Generating HTML in {output_dir}...")
    plot_3d_strawberry(
        pc_flat, color_flat, 
        masks=None, labels=None, # Predictions
        gt_masks=gt_masks_np, gt_labels=gt_labels_np,
        scene_name=scene_name,
        data_dir=output_dir,
        num_frames=len(frame_keys),
        image_size=(target_size, target_size)
    )
    
    html_path = os.path.abspath(os.path.join(output_dir, f"{scene_name}.html"))
    print(f"\nSUCCESS! Visualization generated at:\n{html_path}")

if __name__ == "__main__":
    run_real_verification()
