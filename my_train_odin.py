import gc
import re
import warnings
warnings.filterwarnings('ignore')

import copy
import json
import logging
import os
import weakref
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from imageio import imread
from PIL import Image
from torch.cuda.amp import autocast

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from torch.nn.parallel import DistributedDataParallel
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
    AMPTrainer,
    SimpleTrainer,
    hooks
)
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import DatasetEvaluator, inference_on_dataset
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.structures import Instances, BitMasks

from odin import (
    add_maskformer2_video_config,
    add_maskformer2_config,
    build_detection_train_loader,
    build_detection_test_loader,
)
from odin.data_video.segmentation_benchmark.evaluate_semantic_instance import Scannet_Evaluator
from odin.utils.util_video_to_3d import convert_video_instances_to_3d
from odin.utils.util_3d import convert_3d_to_2d_dict_format
from odin.modeling.backproject.backproject import backprojector_dataloader, multiscsale_voxelize
from odin.modeling.bayesian.swag import SWAG
import pandas as pd

torch.multiprocessing.set_sharing_strategy('file_system')


# -------------------------------------------------------------------------
# Custom Checkpointer for SWAG compatibility
# -------------------------------------------------------------------------
class SWAGCompatibleCheckpointer(DetectionCheckpointer):
    """
    Custom checkpointer that handles SWAG state with int values.
    Converts int values to tensors before loading to avoid Detectron2 errors.
    """
    def _load_model(self, checkpoint):
        """
        Override to convert int values (like n_models) to tensors before loading.
        """
        if "model" in checkpoint:
            # Convert any int values in the checkpoint to tensors
            self._convert_ints_to_tensors(checkpoint["model"])
        return super()._load_model(checkpoint)

    def _convert_ints_to_tensors(self, state_dict):
        """
        Recursively convert int/float values to tensors in state_dict.
        """
        for key, value in list(state_dict.items()):
            if isinstance(value, int):
                state_dict[key] = torch.tensor(value, dtype=torch.int64)
            elif isinstance(value, float):
                state_dict[key] = torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, dict):
                self._convert_ints_to_tensors(value)


# -------------------------------------------------------------------------
# 1. Dataset Registration
# -------------------------------------------------------------------------
def quat_to_rotmat(x, y, z, w):
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
    # Ensure normalization for safety - use float32 consistently
    q = np.array([x, y, z, w], dtype=np.float32)
    q /= (np.linalg.norm(q) + 1e-12)
    x, y, z, w = q

    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    return R

CATEGORIES = {0: "Ripe", 1: "Unripe", 2: "Half-ripe"} # 3 is Peduncle but ignored
NUM_CLASSES = len(CATEGORIES)

# NBV Stage2: 8 primitives × 3 textures = 24 classes
# Primitive IDs: 1-8 (cube, sphere, cylinder, cone, torus, capsule, ellipsoid, pyramid)
# Texture types: red, mixed, green
# Robot (category_id=9) is excluded (background)
NBV_PRIMITIVE_NAMES = {
    1: "cube", 2: "sphere", 3: "cylinder", 4: "cone",
    5: "torus", 6: "capsule", 7: "ellipsoid", 8: "pyramid"
}
NBV_TEXTURE_NAMES = ["red", "mixed", "green"]

# Generate 24 classes: primitive_texture (e.g., "cube_red", "sphere_mixed", etc.)
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

def get_nbv_stage2_dataset_dicts(dataset_dir: str, splits_file: str, split: str):
    """
    NBV Stage2 specific loader.
    Differences from Strawberry:
    - cameras.json is a dict {"00000": {...}, "00001": {...}}
    - color_map.json is a list [{color, instance_id, category_id, category_name, texture_type}, ...]
    - No 'ripeness' field, has 'texture_type' instead
    - category_id ranges 0-9 (8 primitives + robot)
    """
    with open(splits_file, "r") as f:
        splits = json.load(f)

    sample_ids = splits.get(split, [])
    dataset_dicts = []

    for sid in sample_ids:
        s_dir = Path(dataset_dir) / sid
        if not s_dir.exists():
            s_dir = Path(dataset_dir) / f"sample_{str(sid).zfill(5)}"
        if not s_dir.exists():
            continue

        cameras_path = s_dir / "cameras.json"
        color_map_path = s_dir / "color_map.json"

        if not cameras_path.exists() or not color_map_path.exists():
            continue

        with open(cameras_path, "r") as f:
            cameras_dict = json.load(f)  # Dict format!
        with open(color_map_path, "r") as f:
            color_map = json.load(f)  # List format!

        # Convert cameras dict to sorted list
        cameras = []
        for frame_key in sorted(cameras_dict.keys()):
            cam_data = cameras_dict[frame_key]
            # Convert to Strawberry-like format
            cameras.append({
                "frame_index": int(frame_key),
                "intrinsics": cam_data["intrinsics"],
                "position": cam_data["position"],
                "rotation": cam_data["rotation"]
            })

        # Build color_to_info from list and convert to 24-class format
        # Skip robot (category_id=9), convert (primitive_id, texture_type) to class_id
        color_to_info = {}
        for v in color_map:
            prim_id = v["category_id"]

            # Skip robot (background)
            if prim_id == 9:
                continue

            # Skip if not a valid primitive
            if prim_id not in NBV_PRIMITIVE_NAMES:
                continue

            texture = v.get("texture_type")
            if texture not in NBV_TEXTURE_NAMES:
                continue

            # Convert (primitive_id, texture_type) to unified class_id (0-23)
            new_class_id = NBV_PRIMITIVE_TEXTURE_TO_CLASS[(prim_id, texture)]

            # Store with new class_id
            new_info = v.copy()
            new_info["category_id"] = new_class_id
            color_to_info[tuple(v["color"])] = new_info

        # Process in chunks
        CHUNK_SIZE = 5
        for start_idx in range(0, len(cameras), CHUNK_SIZE):
            chunk_cams = cameras[start_idx:start_idx+CHUNK_SIZE]
            if len(chunk_cams) < CHUNK_SIZE:
                continue

            file_names = []
            depth_file_names = []
            masks_file_names = []
            intrinsics = []
            poses = []

            for cam in chunk_cams:
                fi = cam["frame_index"]
                name = f"{fi:05d}"

                rgb_p = s_dir / "rgb" / f"{name}.png"
                depth_p = s_dir / "depth" / f"{name}.npy"
                mask_p = s_dir / "masks" / f"{name}.png"

                if rgb_p.exists() and depth_p.exists():
                    file_names.append(str(rgb_p))
                    depth_file_names.append(str(depth_p))
                    masks_file_names.append(str(mask_p) if mask_p.exists() else None)

                    intr = cam["intrinsics"]
                    K = np.array([
                        [intr["fx"], 0, intr["cx"]],
                        [0, intr["fy"], intr["cy"]],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    intrinsics.append(K)

                    R = quat_to_rotmat(*cam["rotation"])
                    t = np.array(cam["position"], dtype=np.float32)

                    pose = np.eye(4, dtype=np.float32)
                    pose[:3, :3] = R
                    pose[:3, 3] = t
                    poses.append(pose)

            if len(file_names) > 0:
                part_id = start_idx // CHUNK_SIZE
                first_intr = chunk_cams[0]["intrinsics"]
                img_w = int(round(first_intr["cx"] * 2))
                img_h = int(round(first_intr["cy"] * 2))

                record = {
                    "file_name": file_names[0],
                    "image_id": f"{sid}_part{part_id}",
                    "file_names": file_names,
                    "depth_file_names": depth_file_names,
                    "masks_file_names": masks_file_names,
                    "height": img_h,
                    "width": img_w,
                    "intrinsics": intrinsics,
                    "poses": poses,
                    "length": len(file_names),
                    "color_map": color_to_info,
                    "sample_id": sid,
                    "part_id": part_id,
                }
                dataset_dicts.append(record)

    return dataset_dicts

def get_strawberry_dataset_dicts(dataset_dir: str, splits_file: str, split: str):
    """
    Reads splits.json and builds detectron2 formatted dataset dicts.
    """
    with open(splits_file, "r") as f:
        splits = json.load(f)
    
    sample_ids = splits.get(split, [])
    dataset_dicts = []
    
    for sid in sample_ids:
        # Assuming sample_id format is like 'sample_00000' or just '00000'
        # The path corresponds to dataset_dir / sample_NNNNN
        # Try both formats
        s_dir = Path(dataset_dir) / sid
        if not s_dir.exists():
            s_dir = Path(dataset_dir) / f"sample_{str(sid).zfill(5)}"
            
        if not s_dir.exists():
            continue
            
        # load cameras.json and color_map.json
        cameras_path = s_dir / "cameras.json"
        color_map_path = s_dir / "color_map.json"
        
        if not cameras_path.exists() or not color_map_path.exists():
            continue
            
        with open(cameras_path, "r") as f:
            cameras = json.load(f)
        with open(color_map_path, "r") as f:
            color_map = json.load(f)

        # Convert color map to a more usable format mapping specific rgb to instance_id and cat_id
        # Support both dict format (Strawberry) and list format (NBV Stage2)
        if isinstance(color_map, dict):
            color_to_info = {tuple(v["color"]): v for v in color_map.values()}
        elif isinstance(color_map, list):
            color_to_info = {tuple(v["color"]): v for v in color_map}
        else:
            raise ValueError(f"Unsupported color_map format: {type(color_map)}")
            
        # Модификация: бьём видео на куски по 5 кадров для увеличения числа сэмплов и экономии VRAM
        CHUNK_SIZE = 5
        for start_idx in range(0, len(cameras), CHUNK_SIZE):
            chunk_cams = cameras[start_idx:start_idx+CHUNK_SIZE]
            if len(chunk_cams) < CHUNK_SIZE:
                continue
                
            file_names = []
            depth_file_names = []
            masks_file_names = []
            intrinsics = []
            poses = []
            
            for cam in chunk_cams:
                fi = cam["frame_index"]
                name = f"{fi:05d}"
                
                rgb_p = s_dir / "rgb" / f"{name}.png"
                depth_p = s_dir / "depth" / f"{name}.npy"
                mask_p = s_dir / "masks" / f"{name}.png"
                
                if rgb_p.exists() and depth_p.exists():
                    file_names.append(str(rgb_p))
                    depth_file_names.append(str(depth_p))
                    masks_file_names.append(str(mask_p) if mask_p.exists() else None)
                    
                    intr = cam["intrinsics"]
                    K = np.array([
                        [intr["fx"], 0, intr["cx"]],
                        [0, intr["fy"], intr["cy"]],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    intrinsics.append(K)
                    
                    R = quat_to_rotmat(*cam["rotation"])
                    t = np.array(cam["position"], dtype=np.float32)
                    
                    # Стандартная матрица [R | t]
                    pose = np.eye(4, dtype=np.float32)
                    pose[:3, :3] = R
                    pose[:3, 3] = t
                    
                    poses.append(pose)
                    
            if len(file_names) > 0:
                part_id = start_idx // CHUNK_SIZE
                # Получаем размеры из интринзики первого кадра (cx, cy — центр изображения)
                first_intr = chunk_cams[0]["intrinsics"]
                img_w = int(round(first_intr["cx"] * 2))
                img_h = int(round(first_intr["cy"] * 2))
                record = {
                    "file_name": file_names[0], # Primary file (первый кадр в чанке)
                    "image_id": f"{sid}_part{part_id}",
                    "width": img_w,   # обязательно для detectron2
                    "height": img_h,  # обязательно для detectron2
                    "file_names": file_names,
                    "depth_file_names": depth_file_names,
                    "masks_file_names": masks_file_names,
                    "intrinsics": intrinsics,
                    "poses": poses,
                    "length": len(file_names),
                    "color_map": color_to_info
                }
                dataset_dicts.append(record)
            
    return dataset_dicts

def register_nbv_stage2_datasets(dataset_dir: str, splits_file: str):
    """Register NBV Stage2 datasets with proper categories."""
    for split in ["train", "val", "test"]:
        dataset_name = f"nbv_stage2_{split}"
        DatasetCatalog.register(dataset_name, lambda s=split: get_nbv_stage2_dataset_dicts(dataset_dir, splits_file, s))
        MetadataCatalog.get(dataset_name).set(
            thing_classes=list(NBV_CATEGORIES.values()),
            evaluator_type="nbv_stage2"
        )

def register_strawberry_datasets(dataset_dir: str, splits_file: str):
    for split in ["train", "val", "test"]:
        dataset_name = f"strawberry_{split}"
        DatasetCatalog.register(dataset_name, lambda s=split: get_strawberry_dataset_dicts(dataset_dir, splits_file, s))
        MetadataCatalog.get(dataset_name).set(
            thing_classes=list(CATEGORIES.values()),
            evaluator_type="strawberry"
        )

def detect_dataset_type(dataset_dir: str) -> str:
    """
    Auto-detect dataset type by checking format of cameras.json and color_map.json.
    Returns: "strawberry" or "nbv_stage2"
    """
    dataset_path = Path(dataset_dir)

    # Find first sample directory
    sample_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("sample_")])
    if not sample_dirs:
        raise ValueError(f"No sample directories found in {dataset_dir}")

    sample_dir = sample_dirs[0]
    cameras_path = sample_dir / "cameras.json"
    color_map_path = sample_dir / "color_map.json"

    if not cameras_path.exists() or not color_map_path.exists():
        raise ValueError(f"Missing cameras.json or color_map.json in {sample_dir}")

    with open(cameras_path, "r") as f:
        cameras = json.load(f)
    with open(color_map_path, "r") as f:
        color_map = json.load(f)

    # NBV Stage2: cameras is dict, color_map is list
    # Strawberry: cameras is list, color_map is dict
    if isinstance(cameras, dict) and isinstance(color_map, list):
        return "nbv_stage2"
    elif isinstance(cameras, list) and isinstance(color_map, dict):
        return "strawberry"
    else:
        raise ValueError(f"Unknown dataset format: cameras={type(cameras)}, color_map={type(color_map)}")


# -------------------------------------------------------------------------
# 2. Dataset Mapper
# -------------------------------------------------------------------------
class StrawberryDatasetMapper:
    def __init__(self, cfg, is_train: bool, dataset_type: str = "strawberry"):
        self.cfg = cfg
        self.is_train = is_train
        self.dataset_type = dataset_type

        # Select appropriate categories based on dataset type
        if dataset_type == "nbv_stage2":
            self.categories = NBV_CATEGORIES
            self.num_classes = len(NBV_CATEGORIES)
        else:
            self.categories = CATEGORIES
            self.num_classes = NUM_CLASSES

        self.size_divisibility = cfg.INPUT.SIZE_DIVISIBILITY
        self.num_frames = cfg.INPUT.SAMPLING_FRAME_NUM
        
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        
        # Subsample frames if needed
        num_sample_frames = min(self.num_frames, dataset_dict["length"])
        if self.is_train:
            sample_ids = np.random.choice(dataset_dict["length"], num_sample_frames, replace=False)
            sample_ids.sort()
        else:
            # Usually uniformly sample during testing, but ODIN evaluates full sequence or fixed
            sample_factor = max(1, dataset_dict["length"] // num_sample_frames)
            sample_ids = [(i * sample_factor) % dataset_dict["length"] for i in range(num_sample_frames)]
            
        file_names = [dataset_dict["file_names"][i] for i in sample_ids]
        depth_file_names = [dataset_dict["depth_file_names"][i] for i in sample_ids]
        masks_file_names = [dataset_dict["masks_file_names"][i] for i in sample_ids]
        intrinsics_list = [dataset_dict["intrinsics"][i] for i in sample_ids]
        poses_list = [dataset_dict["poses"][i] for i in sample_ids]
        
        color_map = dataset_dict["color_map"]
        
        images = []
        depths = []
        poses = []
        intrinsics = []
        instances_all = []
        
        target_size = self.cfg.INPUT.IMAGE_SIZE # usually 320
        
        for idx in range(num_sample_frames):
            # RGB
            img = imread(file_names[idx])[..., :3]
            old_h, old_w = img.shape[:2]
            
            # Resize
            img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            img_tensor = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            images.append(img_tensor)
            
            # Depth
            depth = np.load(depth_file_names[idx])
            depth = depth[::-1, :].copy() 
            depth = cv2.resize(depth, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            
            # Фильтр "мусора" (белые стены и черный пол)
            rr, gg, bb = img[:,:,0], img[:,:,1], img[:,:,2]
            is_white = (rr > 220) & (gg > 220) & (bb > 220)
            is_black = (rr < 20) & (gg < 20) & (bb < 20)
            depth[is_white | is_black] = 0

            depths.append(torch.as_tensor(depth, dtype=torch.float32))
            
            # Intrinsics Scaling
            K = intrinsics_list[idx].copy()
            K[0, 0] *= (target_size / old_w)
            K[0, 2] *= (target_size / old_w)
            K[1, 1] *= (target_size / old_h)
            K[1, 2] *= (target_size / old_h)
            intrinsics.append(torch.from_numpy(K))
            
            # Poses
            poses.append(torch.from_numpy(poses_list[idx]))
            
            # Masks processing
            image_shape = (target_size, target_size)
            instances = Instances(image_shape)
            has_masks = False
            
            if masks_file_names[idx] is not None and os.path.exists(masks_file_names[idx]):
                mask_img = imread(masks_file_names[idx]).astype(np.uint8)
                mask_img = cv2.resize(mask_img, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
                mr, mg, mb = mask_img[:, :, 0], mask_img[:, :, 1], mask_img[:, :, 2]
                
                gt_classes = []
                gt_masks = []
                instance_ids = []

                # Support both dict and list formats for color_map
                color_map_items = color_map.items() if isinstance(color_map, dict) else [(tuple(v["color"]), v) for v in color_map]

                for color, info in color_map_items:
                    # Check if category_id is valid for current dataset
                    if info["category_id"] not in self.categories:
                        continue
                    px = (mr == color[0]) & (mg == color[1]) & (mb == color[2])
                    if np.any(px):
                        gt_classes.append(info["category_id"])
                        gt_masks.append(px)
                        instance_ids.append(info["instance_id"])
                
                if len(gt_classes) > 0:
                    instances.gt_classes = torch.tensor(gt_classes, dtype=torch.int64)
                    instances.instance_ids = torch.tensor(instance_ids, dtype=torch.int64)
                    instances.gt_masks = BitMasks(torch.stack([torch.from_numpy(m) for m in gt_masks])).tensor
                    has_masks = True
            
            if not has_masks:
                instances.gt_classes = torch.zeros(0, dtype=torch.int64)
                instances.instance_ids = torch.zeros(0, dtype=torch.int64)
                instances.gt_masks = torch.zeros((0, target_size, target_size))
                
            instances_all.append(instances)
        
        h, w = images[0].shape[1:]
        dataset_dict["new_image_shape"] = (h, w)
        dataset_dict["decoder_3d"] = self.cfg.MODEL.DECODER_3D
        dataset_dict["do_camera_drop"] = getattr(self.cfg.INPUT, "CAMERA_DROP", False)
        dataset_dict["max_frames"] = len(images)  # the actual number of frames loaded
        dataset_dict["use_ghost"] = getattr(self.cfg, "USE_GHOST_POINTS", False)
        dataset_dict["multiplier"] = 1.0  # Loss multiplier for dataset balancing
        dataset_dict["images"] = images
        dataset_dict["padding_masks"] = torch.zeros((h, w), dtype=torch.bool)
        dataset_dict["depths"] = depths
        dataset_dict["poses"] = poses
        dataset_dict["intrinsics"] = intrinsics
        dataset_dict["instances_all"] = instances_all
        dataset_dict["image"] = images[0] # primary image
        dataset_dict["instances"] = instances_all[0]
        dataset_dict["valids"] = [d > 0.001 for d in depths] 

        if self.cfg.MODEL.DECODER_3D:
            # Backproject multi-scale features placeholders to XYZ coordinates
            scales = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
            pad_h = int(np.ceil(h / self.size_divisibility) * self.size_divisibility - h)
            pad_w = int(np.ceil(w / self.size_divisibility) * self.size_divisibility - w)
            H_padded = h + pad_h
            W_padded = w + pad_w
            num_v = len(images)
            
            features = {k: torch.zeros(num_v, 1, H_padded//s, W_padded//s) for k, s in scales.items()}
            depths_tensor = torch.stack(depths)
            poses_tensor = torch.stack(poses)
            intrinsics_tensor = torch.stack(intrinsics)
            
            # Подготовка original_xyz для всех кадров
            # Обрати внимание: ODIN ожидает [V, H, W, 3]
            multi_scale_xyz, _, original_xyz_list = backprojector_dataloader(
                list(features.values()), depths_tensor, poses_tensor, intrinsics_tensor,
                augment=False, method=self.cfg.MODEL.INTERPOLATION_METHOD, scannet_pc=None,
                padding=(pad_h, pad_w), do_rot_scale=getattr(self.cfg, "DO_ROT_SCALE", False)
            )
            
            if getattr(self.cfg.INPUT, "VOXELIZE", False):
                multi_scale_p2v = multiscsale_voxelize(multi_scale_xyz, self.cfg.INPUT.VOXEL_SIZE)
            else:
                multi_scale_p2v = [None] * len(multi_scale_xyz)

            dataset_dict['multi_scale_xyz'] = multi_scale_xyz[::-1]
            dataset_dict['multi_scale_p2v'] = multi_scale_p2v[::-1]
            # original_xyz_list[0] имеет форму [V, H_padded, W_padded, 3]
            dataset_dict['original_xyz'] = original_xyz_list[0]

        dataset_dict["all_classes"] = copy.copy(CATEGORIES)
        dataset_dict["num_classes"] = self.num_classes
        dataset_dict["dataset_name"] = "strawberry_train" if self.is_train else "strawberry_val"

        
        return dataset_dict

# -------------------------------------------------------------------------
# 3. Strawberry 3D Evaluator (PQ, SQ, RQ, mAP)
# -------------------------------------------------------------------------
class Strawberry3DEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, output_dir, cfg):
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self.cfg = cfg
        self._cpu_device = torch.device("cpu")
        self.multiplier = 1000 # Standard for instance encoding
        
        # Инциализируем реальный эвалюатор из ODIN (с поддержкой strawberry)
        self.scannet_evaluator = Scannet_Evaluator("strawberry")
        
        self.reset()
        os.makedirs(self._output_dir, exist_ok=True)
        
    def reset(self):
        self.processed_preds = {} # Store parsed results indexed by idx
        self.processed_gts = {}   # Store parsed results indexed by idx
        self.inference_times = []  # Store time per process call
        self.vis_data = {}        # Optional: store heavy visual data for subset of samples
        self.total_frames = 0     # Track total frames for accurate speed metrics
        self.uncertainties = []   # Store uncertainty (entropy) per sample
        self._current_idx = 0
        
    def process(self, inputs, outputs):
        """
        inputs: List of dataset dicts
        outputs: List of model outputs
        """
        target_samples = ["00000", "sample_00000", "00003", "sample_00003", "00005", "sample_00005"]

        for _in, _out in zip(inputs, outputs):
            start_t = time.perf_counter()
            idx = self._current_idx
            sample_id = str(_in.get("image_id", ""))

            # 1. Parse and store ground truth (3D masks/labels)
            # This is much lighter than the full _in (RGB, Depth, Large XYZ tensors)
            self.processed_gts[idx] = self._parse_gt(_in)

            # 2. Parse and store prediction
            self.processed_preds[idx] = self._parse_pred(_out)

            # 3. Calculate uncertainty (entropy) from Bayesian samples or pred_logits
            if 'uncertainty' in _out and 'predictive_entropy' in _out['uncertainty']:
                # Use Bayesian uncertainty (from multiple samples)
                predictive_entropy = _out['uncertainty']['predictive_entropy']  # [B, Q]
                mean_entropy = predictive_entropy.mean().item()
                self.uncertainties.append(mean_entropy)
                if not hasattr(self, '_uncertainty_source_printed'):
                    print(f"[UNCERTAINTY] Using Bayesian uncertainty: {mean_entropy:.6f}")
                    self._uncertainty_source_printed = True
            elif 'pred_logits' in _out:
                # Fallback: use single-pass entropy (deterministic inference)
                logits = _out['pred_logits']  # Shape: (B, num_queries, num_classes)
                probs = torch.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # (B, num_queries)
                mean_entropy = entropy.mean().item()
                self.uncertainties.append(mean_entropy)
                if not hasattr(self, '_uncertainty_source_printed'):
                    print(f"[UNCERTAINTY] Using deterministic entropy: {mean_entropy:.6f}")
                    self._uncertainty_source_printed = True
            else:
                if not hasattr(self, '_uncertainty_missing_printed'):
                    print(f"[UNCERTAINTY] WARNING: No uncertainty or pred_logits in output! Keys: {_out.keys()}")
                    self._uncertainty_missing_printed = True

            # 4. Check if we need visualization for this sample
            is_target = any(ts in sample_id for ts in target_samples)
            if is_target:
                # Store only the minimum data needed for build_html (ON CPU)
                # We copy and move to CPU to ensure no references to GPU-heavy objects
                self.vis_data[idx] = {
                    "image_id": sample_id,
                    "images": [img.cpu() for img in _in.get("images", [])],
                    "depths": [d.cpu() for d in _in.get("depths", [])],
                    "poses": [p.cpu() for p in _in.get("poses", [])],
                    "intrinsics": [i.cpu() for i in _in.get("intrinsics", [])],
                    "color_map": copy.deepcopy(_in.get("color_map", {})),
                    "instances_all": copy.deepcopy(_in.get("instances_all", [])) # 2D masks for visualization
                }

            # 5. Count frames and timing BEFORE deleting tensors
            num_frames = len(_in.get("images", []))
            self.inference_times.append(time.perf_counter() - start_t)
            self.total_frames += num_frames
            self._current_idx += 1

            # 6. CRITICAL: Explicitly release references to large tensors in inputs
            # To help garbage collection between batches
            if "multi_scale_xyz" in _in: del _in["multi_scale_xyz"]
            if "multi_scale_p2v" in _in: del _in["multi_scale_p2v"]
            if "original_xyz" in _in: del _in["original_xyz"]
            if "images" in _in: del _in["images"]
            if "depths" in _in: del _in["depths"]
        
        # Force garbage collection after each batch
        gc.collect()

    def _parse_gt(self, _in):
        h, w = _in['instances_all'][0].image_size
        num_frames = len(_in['instances_all'])
        target_dict = convert_video_instances_to_3d(
            _in['instances_all'],
            num_frames,
            h, w,
            self._cpu_device,
            convert_point_semantic_instance=True,
            multiplier=self.multiplier
        )
        return {
            "masks": target_dict["masks"].cpu(),
            "labels": target_dict["point_semantic_instance_label"].flatten(0).cpu(),
            "class_labels": target_dict["labels"].cpu()
        }

    def _parse_pred(self, _out):
        pred = _out['instances_3d']
        res = {}
        for key in pred:
            if isinstance(pred[key], torch.Tensor):
                res[key] = pred[key].cpu().numpy()
            else:
                res[key] = pred[key]
        return res

    def evaluate(self):
        logging.getLogger(__name__).info("Evaluating 3D Instance metrics (Strawberry) on full validation set...")
        
        # Free up some memory before processing
        gc.collect()
        torch.cuda.empty_cache()
        
        # 1. Точная выгрузка визуализаций по правилам `generate_sample_viewer.py`
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        try:
            from generate_sample_viewer import build_html
        except ImportError:
            logging.getLogger(__name__).warning("Не удалось импортировать build_html из generate_sample_viewer.py!")
            build_html = None

        if build_html is not None and len(self.vis_data) > 0:
            vis_output_dir = os.path.join(self._output_dir, "visualizations")
            os.makedirs(vis_output_dir, exist_ok=True)

            def rotmat_to_quat(R):
                trace = np.trace(R)
                if trace > 0.0:
                    s = np.sqrt(trace + 1.0) * 2.0
                    qw = 0.25 * s
                    qx = (R[2, 1] - R[1, 2]) / s
                    qy = (R[0, 2] - R[2, 0]) / s
                    qz = (R[1, 0] - R[0, 1]) / s
                elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                    s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                    qw = (R[2, 1] - R[1, 2]) / s
                    qx = 0.25 * s
                    qy = (R[0, 1] + R[1, 0]) / s
                    qz = (R[0, 2] + R[2, 0]) / s
                elif R[1, 1] > R[2, 2]:
                    s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                    qw = (R[0, 2] - R[2, 0]) / s
                    qx = (R[0, 1] + R[1, 0]) / s
                    qy = 0.25 * s
                    qz = (R[1, 2] + R[2, 1]) / s
                else:
                    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                    qw = (R[1, 0] - R[0, 1]) / s
                    qx = (R[0, 2] + R[2, 0]) / s
                    qy = (R[1, 2] + R[2, 1]) / s
                    qz = 0.25 * s
                return [float(qx), float(qy), float(qz), float(qw)]

            for idx in self.vis_data:
                v_data = self.vis_data[idx]
                sample_id = v_data["image_id"]
                
                logging.getLogger(__name__).info(f"Генерация HTML визуализации для {sample_id}...")
                
                cameras = []
                chunks = []
                stride = 2
                
                # Подготовка предсказаний для быстрого доступа
                pred_data = self.processed_preds[idx]
                pred_masks = pred_data.get("pred_masks") # [NumInstances, NumPoints]
                pred_classes = pred_data.get("pred_classes") # [NumInstances]
                
                num_pred_instances = 0
                point_pred_inst = None
                point_pred_cat = None
                
                if pred_masks is not None and len(pred_masks) > 0:
                    num_pred_instances = pred_masks.shape[0]
                    num_pts_total = pred_masks.shape[1]
                    # Создаем карту меток для всех точек сразу (фон = -1)
                    point_pred_inst = np.full(num_pts_total, -1, dtype=np.int32)
                    point_pred_cat = np.full(num_pts_total, -1, dtype=np.int32)

                    # DEBUG: Print pred_classes values
                    if not hasattr(self, '_pred_classes_printed'):
                        print(f"\n=== PRED CLASSES DEBUG ===")
                        print(f"pred_classes raw values: {pred_classes[:min(5, len(pred_classes))]}")
                        print(f"pred_classes after -1: {[int(c) - 1 for c in pred_classes[:min(5, len(pred_classes))]]}")
                        print(f"Expected: 0=Ripe(red), 1=Unripe(green), 2=Half-ripe(orange)")
                        print(f"=== END DEBUG ===\n")
                        self._pred_classes_printed = True

                    # Если маски перекрываются, побеждает последняя
                    for inst_idx in range(num_pred_instances):
                        m = pred_masks[inst_idx] > 0
                        point_pred_inst[m] = inst_idx # 0-indexed для виза
                        # ВАЖНО: pred_classes в ODIN уже приходят 1-индексированными (labels + 1), поэтому вычитаем 1
                        point_pred_cat[m] = int(pred_classes[inst_idx]) - 1

                images = v_data.get("images", [])
                depths = v_data.get("depths", [])
                poses = v_data.get("poses", [])
                intrinsics = v_data.get("intrinsics", [])
                color_map = v_data.get("color_map", {})
                
                # Рассчитываем padding как в маппере для правильной глобальной индексации
                H_orig, W_orig = images[0].shape[1:]
                div = self.cfg.INPUT.SIZE_DIVISIBILITY
                H_padded = int(np.ceil(H_orig / div) * div)
                W_padded = int(np.ceil(W_orig / div) * div)

                for camera_idx in range(len(images)):
                    img = images[camera_idx].numpy().transpose(1, 2, 0)
                    Z_full = depths[camera_idx].numpy()
                    pose = poses[camera_idx].numpy()
                    intr = intrinsics[camera_idx].numpy()
                    
                    fx, fy, cx, cy = intr[0,0], intr[1,1], intr[0,2], intr[1,2]
                    
                    H, W = Z_full.shape
                    u = np.arange(0, W, stride, dtype=np.float32)
                    v = np.arange(0, H, stride, dtype=np.float32)
                    uu, vv = np.meshgrid(u, v)
                    
                    Z_s = Z_full[::stride, ::stride]
                    valid = (Z_s > 0.001) & (Z_s < 5.0)
                    
                    # 1. Сбор GT масок для текущего кадра (фон = -1)
                    inst_gt_frame = np.full((H, W), -1, dtype=np.int32)
                    cat_gt_frame = np.full((H, W), -1, dtype=np.int32)
                    
                    if 'instances_all' in v_data and camera_idx < len(v_data['instances_all']):
                        instances = v_data['instances_all'][camera_idx]
                        if len(instances) > 0:
                            gt_m = instances.gt_masks.numpy() # [N, H, W]
                            gt_c = instances.gt_classes.numpy() # [N]
                            gt_ids = instances.instance_ids.numpy() # [N]

                            # DEBUG: Print GT classes values
                            if not hasattr(self, '_gt_classes_printed'):
                                print(f"\n=== GT CLASSES DEBUG ===")
                                print(f"gt_classes raw values: {gt_c[:min(5, len(gt_c))]}")
                                print(f"Expected: 0=Ripe(red), 1=Unripe(green), 2=Half-ripe(orange)")
                                print(f"=== END DEBUG ===\n")
                                self._gt_classes_printed = True

                            for inst_i in range(len(instances)):
                                m_mask = gt_m[inst_i] > 0
                                inst_gt_frame[m_mask] = int(gt_ids[inst_i])
                                cat_gt_frame[m_mask] = int(gt_c[inst_i])

                    # 2. Проекция
                    X_cam =  (uu - cx) * Z_s / fx
                    Y_cam = -(vv - cy) * Z_s / fy
                    Z_cam =   Z_s
                    pts_cam = np.stack([X_cam, Y_cam, Z_cam], axis=-1)
                    
                    R_mat = pose[:3, :3]
                    t_vec = pose[:3, 3]
                    pts_world = pts_cam @ R_mat.T + t_vec
                    
                    fv = valid.ravel()
                    pts_chunk = pts_world.reshape(-1, 3)[fv]
                    
                    rgb_s = img[::stride, ::stride]
                    r = rgb_s[:,:,0].ravel()[fv]
                    g = rgb_s[:,:,1].ravel()[fv]
                    b = rgb_s[:,:,2].ravel()[fv]
                    
                    # Выборка масок GT
                    inst_gt = inst_gt_frame[::stride, ::stride].ravel()[fv]
                    cat_gt = cat_gt_frame[::stride, ::stride].ravel()[fv]

                    # Выборка масок Pred через глобальную индексацию
                    inst_pred = np.full_like(inst_gt, -1)
                    cat_pred = np.full_like(cat_gt, -1)
                    
                    if point_pred_inst is not None:
                        rows = vv[valid].astype(np.int32)
                        cols = uu[valid].astype(np.int32)
                        global_indices = camera_idx * (H_padded * W_padded) + rows * W_padded + cols
                        
                        valid_global = global_indices < len(point_pred_inst)
                        inst_pred[valid_global] = point_pred_inst[global_indices[valid_global]]
                        cat_pred[valid_global] = point_pred_cat[global_indices[valid_global]]
                    
                    chunk = np.column_stack([pts_chunk, r, g, b, inst_gt, cat_gt, inst_pred, cat_pred]).astype(np.float32)
                    chunks.append(chunk)
                    
                    cam_dict = {
                        "position": t_vec.tolist(),
                        "rotation": rotmat_to_quat(R_mat),
                        "intrinsics": {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)},
                        "frame_index": camera_idx
                    }
                    cameras.append(cam_dict)
                
                if len(chunks) > 0:
                    pts = np.concatenate(chunks, axis=0)
                    is_white = (pts[:, 3] > 220) & (pts[:, 4] > 220) & (pts[:, 5] > 220)
                    is_black = (pts[:, 3] < 20) & (pts[:, 4] < 20) & (pts[:, 5] < 20)
                    pts = pts[~is_white & ~is_black]

                    MAX_POINTS = 800000
                    if len(pts) > MAX_POINTS:
                        rng = np.random.default_rng(42)
                        idx = rng.choice(len(pts), MAX_POINTS, replace=False)
                        pts = pts[idx]

                    html = build_html(pts, cameras, color_map, sample_name=sample_id)
                    out_html_path = os.path.join(vis_output_dir, f"{sample_id}_pred.html")
                    with open(out_html_path, "w", encoding="utf-8") as f:
                        f.write(html)
                    logging.getLogger(__name__).info(f"HTML сохранен: {out_html_path}")

        # 1.1 Замер производительности инференса (уже выполнен Detectron2, но мы добавим свои метрики)
        # Суммарное время инференса берется из системных логов, но мы посчитаем среднее по нашему набору
        num_samples = len(self.processed_preds)
        num_frames_per_sample = self.cfg.INPUT.SAMPLING_FRAME_NUM
        # Мы не можем легко достать чистое время из d2.evaluation.evaluator тут, 
        # поэтому используем средние значения из логов если нужно, или просто считаем общую длительность evaluate
        # Однако, для точности в CSV запишем константы из последнего замера если доступно.
        preds_dict = self.processed_preds
        gts_dict = self.processed_gts
            
        # 3. Расчет AP (mAP, mAP@50, mAP@25) через стандартный механизм ScanNet
        matches = {}
        for i, (k, v) in enumerate(preds_dict.items()):
            gt2pred, pred2gt = self.scannet_evaluator.assign_instances_for_scan(v, gts_dict[i])
            matches[i] = {'gt': gt2pred, 'pred': pred2gt}
            
        num_preds = sum(len(v['pred_classes']) for v in preds_dict.values())
        num_gts = sum(len(v['class_labels']) for v in gts_dict.values())
        logging.getLogger(__name__).info(f"Статистика оценки: Найдено предсказаний: {num_preds}, Всего GT инстансов: {num_gts}")

        # evaluate_matches возвращает кортеж из 5 массивов (has_gt, has_pred, y_true, y_score, hard_fn)
        # ВАЖНО: compute_ap принимает их в другом порядке: (has_gt, has_pred, y_score, y_true, hard_fn)
        eval_data = self.scannet_evaluator.evaluate_matches(matches)
        has_gt, has_pred, y_true, y_score, hard_fn = eval_data
        
        aps = self.scannet_evaluator.compute_ap(has_gt, has_pred, y_score, y_true, hard_fn)
        ap_results = self.scannet_evaluator.compute_averages(aps)
        
        # 4. Расчет PQ, SQ, RQ через наш новый метод
        panoptic_results = self.scannet_evaluator.compute_panoptic_metrics(matches)
        
        # 5. Формирование финального словаря метрик
        metrics = {
            "PQ": panoptic_results["all"]["pq"] * 100.0,
            "SQ": panoptic_results["all"]["sq"] * 100.0,
            "RQ": panoptic_results["all"]["rq"] * 100.0,
            "mAP": ap_results["all_ap"] * 100.0,
            "mAP@50": ap_results["all_ap_50%"] * 100.0,
            "mAP@25": ap_results["all_ap_25%"] * 100.0,
            "mean_uncertainty": np.mean(self.uncertainties) if self.uncertainties else 0.0,
        }
        
        # Режим дебага: выводим PrettyTable в консоль
        if logging.getLogger(__name__).isEnabledFor(logging.DEBUG):
            self.scannet_evaluator.print_results(ap_results, logging.getLogger(__name__))

        # 6. Сбор системных метрик и скорости обучения из EventStorage
        # Используем современный подход Detectron2 без предварительной иницализации "заглушек"
        system_metrics = {}
        try:
            storage = get_event_storage()
            system_metrics["iteration"] = storage.iter
            
            # Безопасно извлекаем историю лоссов и скорости обучения
            histories = storage.histories()
            if "total_loss" in histories:
                system_metrics["total_loss"] = histories["total_loss"].latest()
            if "perf/fps" in histories:
                system_metrics["train/fps"] = histories["perf/fps"].median(20)
            if "perf/sec_sample" in histories:
                system_metrics["train/sec_sample"] = histories["perf/sec_sample"].median(20)
        except AssertionError:
            # EventStorage не активен (например, запуск оценки без трейнера)
            pass

        # 7. Расчет метрик производительности инференса
        logger = logging.getLogger(__name__)
        logger.info(f"Inference timing: {len(self.inference_times)} samples, {self.total_frames} frames")

        if self.inference_times and self.total_frames > 0:
            total_time = sum(self.inference_times)
            avg_sec_sample = total_time / len(self.inference_times)
            metrics["eval/sec_sample"] = avg_sec_sample
            metrics["eval/fps"] = self.total_frames / total_time
            # Keep old names for compatibility if needed
            metrics["perf/val_sec_sample"] = metrics["eval/sec_sample"]
            metrics["perf/val_fps"] = metrics["eval/fps"]
            logger.info(f"Eval speed metrics: {metrics['eval/fps']:.2f} fps, {metrics['eval/sec_sample']:.4f} sec/sample")
        else:
            logger.warning(f"Cannot compute eval speed metrics: inference_times={len(self.inference_times)}, total_frames={self.total_frames}")

        # Формируем итоговый словарь для CSV
        metrics_for_csv = {**system_metrics, **metrics}
        
        # 8. Очистка старых визуализаций (храним первые 5 и последние 10)
        self._cleanup_visualizations()
        
        df_new = pd.DataFrame([metrics_for_csv])
        csv_path = os.path.join(self.cfg.OUTPUT_DIR, "metrics_comparison.csv")
        
        # Переставляем важные колонки в начало для удобства (если они есть)
        important_cols = [
            "iteration", "total_loss",
            "train/fps", "train/sec_sample",
            "eval/fps", "eval/sec_sample",
            "PQ", "mAP@50", "mean_uncertainty"
        ]
        
        # Если файл уже есть, объединяем его с новыми данными для поддержки новых колонок
        if os.path.exists(csv_path):
            try:
                df_old = pd.read_csv(csv_path)
                # Объединяем, заполняя недостающие колонки NaN для старых строк
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
            except Exception as e:
                logging.getLogger("odin_strawberry").warning(f"Ошибка при чтении CSV: {e}. Создаем новый.")
                df_combined = df_new
        else:
            df_combined = df_new

        # Сортируем колонки: важные в начале, остальные потом
        cols = [c for c in important_cols if c in df_combined.columns]
        other_cols = [c for c in df_combined.columns if c not in important_cols]
        df_combined = df_combined[cols + other_cols]
        
        # Перезаписываем файл целиком с актуальными заголовками
        df_combined.to_csv(csv_path, index=False)
        
        iter_str = metrics_for_csv.get("iteration", "?")
        pq_val = metrics.get("PQ", 0.0)
        map50_val = metrics.get("mAP@50", 0.0)
        logging.getLogger(__name__).info(f"Метрики (iter {iter_str}, PQ {pq_val:.2f}, mAP@50 {map50_val:.2f}) записаны в {csv_path}")
        
        # Возвращаем в формате D2 для BestCheckpointer
        # Ключ должен соответствовать тому, что мы будем мониторить
        res = {f"strawberry_3d/{k}": v for k, v in metrics.items()}
        return res

    def _cleanup_visualizations(self):
        """
        Удаляет старые папки визуализаций, оставляя первые 5 и последние 10.
        Защищает диск Kaggle от переполнения.
        """
        import shutil
        vis_root = os.path.join(self.cfg.VISUALIZE_LOG_DIR)
        if not os.path.exists(vis_root):
            return
            
        # Папки вида iter_000000
        dirs = [d for d in os.listdir(vis_root) if os.path.isdir(os.path.join(vis_root, d)) and d.startswith("iter_")]
        if len(dirs) <= 15: # 5 + 10
            return
            
        dirs.sort() # Сортировка по итерациям
        
        # Оставляем первые 5 и последние 10
        to_keep = set(dirs[:5]) | set(dirs[-10:])
        to_delete = [d for d in dirs if d not in to_keep]
        
        logger = logging.getLogger("odin_strawberry")
        for d in to_delete:
            path = os.path.join(vis_root, d)
            try:
                shutil.rmtree(path)
                logger.info(f"--- [CLEANUP] Удалена старая визуализация: {d} ---")
            except Exception as e:
                logger.warning(f"--- [CLEANUP] Ошибка при удалении папки {d}: {e} ---")


# -------------------------------------------------------------------------
# 4. Trainer Override with NaN Recovery
# -------------------------------------------------------------------------
class AMPTrainerWithClipping(AMPTrainer):
    def __init__(self, model, data_loader, optimizer, num_frames=1,
                 nan_recovery_enabled=True, nan_lr_scale=0.1, nan_recovery_iters=100,
                 checkpointer=None, output_dir=None):
        super().__init__(model, data_loader, optimizer)
        self.num_frames = num_frames

        # NaN recovery parameters
        self.nan_recovery_enabled = nan_recovery_enabled
        self.nan_lr_scale = nan_lr_scale  # Scale factor when NaN detected (0.1 = reduce by 10x)
        self.nan_recovery_iters = nan_recovery_iters  # Iterations to recover LR

        # Checkpoint management for NaN recovery
        self.checkpointer = checkpointer
        self.output_dir = output_dir
        self.last_good_checkpoint = None  # Path to last checkpoint before NaN

        # State tracking
        self.nan_recovery_active = False
        self.nan_recovery_start_iter = 0
        self.original_lrs = []  # Store original LRs for each param group
        self.target_lrs = []
        self.nan_count = 0  # Track consecutive NaN occurrences

        logger = logging.getLogger(__name__)
        if self.nan_recovery_enabled:
            logger.info(f"[NaN RECOVERY] Enabled: scale={nan_lr_scale}, recovery_iters={nan_recovery_iters}, checkpoint_restore=True")

    def _handle_nan_recovery(self):
        """Handle NaN detection: restore checkpoint, reduce LR and start recovery process."""
        logger = logging.getLogger(__name__)

        if not self.nan_recovery_enabled:
            raise RuntimeError("NaN detected but recovery is disabled!")

        self.nan_count += 1

        # If too many consecutive NaNs, something is seriously wrong
        if self.nan_count > 10:
            logger.error(f"[NaN RECOVERY] Too many consecutive NaNs ({self.nan_count}). Aborting.")
            raise RuntimeError(f"NaN recovery failed after {self.nan_count} attempts!")

        # Try to restore from last good checkpoint
        logger.info(f"[NaN RECOVERY DEBUG] checkpointer={self.checkpointer is not None}, last_good_checkpoint={self.last_good_checkpoint}")
        if self.checkpointer and self.last_good_checkpoint:
            logger.warning(f"[NaN RECOVERY] Restoring model from last good checkpoint: {self.last_good_checkpoint}")
            try:
                checkpoint = self.checkpointer._load_file(self.last_good_checkpoint)
                self.checkpointer._load_model(checkpoint)
                logger.info(f"[NaN RECOVERY] Model weights restored successfully")
            except Exception as e:
                logger.error(f"[NaN RECOVERY] Failed to restore checkpoint: {e}")
                logger.warning(f"[NaN RECOVERY] Continuing with current weights (risky!)")
        else:
            if not self.checkpointer:
                logger.warning(f"[NaN RECOVERY] No checkpointer available - cannot restore weights")
            if not self.last_good_checkpoint:
                logger.warning(f"[NaN RECOVERY] No last good checkpoint available - cannot restore weights")

        # Store original LRs if not already in recovery
        if not self.nan_recovery_active:
            self.original_lrs = [group['lr'] for group in self.optimizer.param_groups]
            self.target_lrs = self.original_lrs.copy()

        # Reduce LR immediately
        for i, group in enumerate(self.optimizer.param_groups):
            new_lr = self.original_lrs[i] * self.nan_lr_scale
            group['lr'] = new_lr
            logger.warning(f"[NaN RECOVERY] Param group {i}: LR {self.original_lrs[i]:.2e} → {new_lr:.2e} (reduced by {1/self.nan_lr_scale:.1f}x)")

        # Activate recovery mode
        self.nan_recovery_active = True
        self.nan_recovery_start_iter = self.iter

        logger.warning(f"[NaN RECOVERY] Started at iter {self.iter}. Will recover LR over {self.nan_recovery_iters} iterations.")

    def _update_recovery_lr(self):
        """Gradually increase LR back to original during recovery."""
        if not self.nan_recovery_active:
            return

        iters_since_recovery = self.iter - self.nan_recovery_start_iter

        if iters_since_recovery >= self.nan_recovery_iters:
            # Recovery complete
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.target_lrs[i]

            logger = logging.getLogger(__name__)
            logger.info(f"[NaN RECOVERY] Complete at iter {self.iter}. LR restored to original values.")
            self.nan_recovery_active = False
            return

        # Linear warmup from reduced LR to original LR
        progress = iters_since_recovery / self.nan_recovery_iters

        for i, group in enumerate(self.optimizer.param_groups):
            reduced_lr = self.original_lrs[i] * self.nan_lr_scale
            current_lr = reduced_lr + (self.target_lrs[i] - reduced_lr) * progress
            group['lr'] = current_lr

        # Log every 10 iterations during recovery
        if iters_since_recovery % 10 == 0:
            logger = logging.getLogger(__name__)
            logger.info(f"[NaN RECOVERY] Progress: {iters_since_recovery}/{self.nan_recovery_iters} iters, "
                       f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

    def run_step(self):
        """
        Кастомный шаг обучения с поддержкой Gradient Clipping для AMP и NaN recovery.
        """
        assert self.model.training, "[AMPTrainerWithClipping] model was not in training mode!"

        # Update LR if in recovery mode
        self._update_recovery_lr()

        # Замеряем время загрузки данных
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        try:
            # 1. Считаем потери внутри autocast
            with torch.cuda.amp.autocast():
                loss_dict = self.model(data)
                if isinstance(loss_dict, torch.Tensor):
                    losses = loss_dict
                    loss_dict = {"total_loss": loss_dict}
                else:
                    losses = sum(loss_dict.values())

            # Check for NaN in losses
            if not torch.isfinite(losses):
                raise RuntimeError("NaN or Inf detected in loss!")

            # 2. Обнуляем градиенты и делаем backward через scaler
            self.optimizer.zero_grad()
            self.grad_scaler.scale(losses).backward()

            # 3. UNSSCALE перед CLIP_GRADIENTS (это ключевой момент для AMP)
            self.grad_scaler.unscale_(self.optimizer)

            # 4. Обрезаем градиенты (Gradient Clipping)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)

            # 5. Шагаем через scaler
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            # Success! Reset NaN counter
            self.nan_count = 0

        except RuntimeError as e:
            if "NaN" in str(e) or "Inf" in str(e):
                logger = logging.getLogger(__name__)
                logger.error(f"[NaN RECOVERY] Caught error: {e}")

                # Handle NaN recovery
                self._handle_nan_recovery()

                # Skip this batch and continue
                logger.warning(f"[NaN RECOVERY] Skipping batch at iter {self.iter}")

                # Create dummy loss_dict for logging
                loss_dict = {"total_loss": torch.tensor(0.0)}
                data_time = 0.0
            else:
                # Re-raise if not NaN-related
                raise

        # Передаем и лоссы, и время загрузки данных
        self._write_metrics(loss_dict, data_time)
        
        # Дополнительные метрики производительности (Sample = 1 Sequence)
        storage = get_event_storage()
        sec_per_sample = time.perf_counter() - start
        storage.put_scalar("perf/sec_sample", sec_per_sample)
        storage.put_scalar("perf/sec_frame", sec_per_sample / self.num_frames)
        storage.put_scalar("perf/fps", 1.0 / sec_per_sample)

class MyTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        model = self.build_model(cfg)

        # SWAG wrapper (if enabled) - only for predictor, not entire model
        self.swag_model = None
        bayesian_type = getattr(cfg.MODEL, "BAYESIAN_TYPE", "none")
        if bayesian_type == "swag":
            logger = logging.getLogger("odin_strawberry")
            logger.info("Initializing SWAG wrapper for predictor (transformer decoder)...")
            no_cov_mat = getattr(cfg.MODEL.SWAG, "NO_COV_MAT", False)
            max_num_models = getattr(cfg.MODEL.SWAG, "MAX_MODELS", 20)

            # Wrap only the predictor (transformer decoder with class_embed)
            if hasattr(model, 'sem_seg_head') and hasattr(model.sem_seg_head, 'predictor'):
                predictor = model.sem_seg_head.predictor
                self.swag_model = SWAG(predictor, no_cov_mat=no_cov_mat, max_num_models=max_num_models)
                # Attach SWAG model to the head for inference
                model.sem_seg_head.swag_model = self.swag_model
                logger.info(f"SWAG initialized for predictor: no_cov_mat={no_cov_mat}, max_models={max_num_models}")
            else:
                logger.warning("Could not find predictor in model, SWAG disabled")
                bayesian_type = "none"

        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # Оборачиваем в DDP только при multi-GPU (world_size > 1)
        # При одиночном запуске DDP требует init_process_group, которого нет
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(model, device_ids=[comm.get_local_rank()])

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = SWAGCompatibleCheckpointer(model, cfg.OUTPUT_DIR, trainer=weakref.proxy(self))

        # Используем наш кастомный трейнер с обрезкой градиентов
        if cfg.SOLVER.AMP.ENABLED:
            self._trainer = AMPTrainerWithClipping(
                model, data_loader, optimizer,
                num_frames=cfg.INPUT.SAMPLING_FRAME_NUM,
                nan_recovery_enabled=cfg.NAN_RECOVERY_ENABLED,
                nan_lr_scale=cfg.NAN_LR_SCALE,
                nan_recovery_iters=cfg.NAN_RECOVERY_ITERS,
                checkpointer=self.checkpointer,
                output_dir=cfg.OUTPUT_DIR
            )
        else:
            # Для полноты добавим SimpleTrainer с теми же метриками, если понадобится,
            # но сейчас используем только AMP
            self._trainer = SimpleTrainer(model, data_loader, optimizer)

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.register_hooks(self.build_hooks())
        
    def build_hooks(self):
        from detectron2.engine import hooks
        all_hooks = super().build_hooks()
        
        # Хук для отображения номера эпохи в логах
        class EpochHook(hooks.HookBase):
            def __init__(self, dataset_len, batch_size):
                self.dataset_len = dataset_len
                self.batch_size = batch_size

            def before_step(self):
                storage = get_event_storage()
                epoch = self.trainer.iter * self.batch_size / self.dataset_len
                storage.put_scalar("epoch", epoch, smoothing_hint=False)
                
                # Периодически выводим в консоль (раз в 20 шагов, как и D2)
                if (self.trainer.iter + 1) % 20 == 0:
                    import logging
                    logger = logging.getLogger("odin_strawberry")
                    logger.info(f">>> EPOCH: {epoch:.2f} | ITER: {self.trainer.iter + 1} <<<")

        dataset_len = len(DatasetCatalog.get(self.cfg.DATASETS.TRAIN[0]))
        all_hooks.append(EpochHook(dataset_len, self.cfg.SOLVER.IMS_PER_BATCH))
        
        # Хук для остановки по времени (защита от таймаута Kaggle)
        class TimeLimitHook(hooks.HookBase):
            def __init__(self, max_time_hours):
                self.max_time_seconds = max_time_hours * 3600
                self.start_time = None

            def before_train(self):
                self.start_time = time.perf_counter()

            def after_step(self):
                elapsed = time.perf_counter() - self.start_time
                if elapsed > self.max_time_seconds:
                    import logging
                    import sys
                    logger = logging.getLogger("odin_strawberry")
                    logger.warning(f"!!! TIME LIMIT REACHED ({elapsed/3600:.2f}h). Stopping training gracefully... !!!")
                    
                    # Принудительно сохраняем чекпоинт перед выходом
                    # Используем имя 'model_final', чтобы Kaggle-скрипт мог его подхватить
                    try:
                        self.trainer.checkpointer.save("model_final")
                        logger.info("--- [TIME LIMIT] Финальный чекпоинт model_final успешно сохранен. ---")
                    except Exception as e:
                        logger.error(f"--- [TIME LIMIT] Ошибка при сохранении финального чекпоинта: {e} ---")
                    
                    # Завершаем процесс. В Kaggle это приведет к успешному завершению ячейки/скрипта.
                    sys.exit(0)

        max_time = getattr(self.cfg, "MAX_TIME_HOURS", 11.5)
        all_hooks.append(TimeLimitHook(max_time))

        class CheckpointCleanupHook(hooks.HookBase):
            def __init__(self, output_dir, keep_last=2):
                self.output_dir = output_dir
                self.keep_last = keep_last

            def before_train(self):
                # Очищаем сразу при старте (на случай если остались файлы от прошлого падения)
                self._cleanup()

            def after_step(self):
                # Проверяем на каждом шаге, так как PeriodicCheckpointer может сработать
                # Но реально удаляем только если появились новые файлы
                if (self.trainer.iter + 1) % 10 == 0: # Проверяем каждые 10 шагов для надежности
                    self._cleanup()

            def _cleanup(self):
                import os
                import re
                logger = logging.getLogger("odin_strawberry")
                if not os.path.exists(self.output_dir):
                    return
                files = [f for f in os.listdir(self.output_dir) if f.endswith(".pth")]
                
                # Ищем файлы вида model_0001234.pth (исключаем model_best.pth и model_final.pth)
                checkpoint_pattern = re.compile(r"^model_(\d+)\.pth$")
                
                checkpoints = []
                for f in files:
                    match = checkpoint_pattern.match(f)
                    if match:
                        iteration = int(match.group(1))
                        checkpoints.append((iteration, f))
                
                if len(checkpoints) <= self.keep_last:
                    return
                    
                # Сортируем по итерации (старые в начале)
                checkpoints.sort(key=lambda x: x[0])
                
                # Оставляем только последние keep_last
                to_delete = checkpoints[:-self.keep_last]
                
                for iteration, filename in to_delete:
                    file_path = os.path.join(self.output_dir, filename)
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            logger.info(f"--- [CLEANUP] Удален старый чекпоинт: {filename} (Disk Space protection) ---")
                    except Exception as e:
                        logger.warning(f"--- [CLEANUP] Ошибка при удалении {filename}: {e} ---")

        all_hooks.append(CheckpointCleanupHook(self.cfg.OUTPUT_DIR, keep_last=2))

        # Hook to track last good checkpoint for NaN recovery
        class LastGoodCheckpointHook(hooks.HookBase):
            def after_step(self):
                # Update last good checkpoint path after each successful checkpoint save
                if self.trainer.iter % self.trainer.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                    checkpoint_path = os.path.join(
                        self.trainer.cfg.OUTPUT_DIR,
                        f"model_{self.trainer.iter:07d}.pth"
                    )
                    if os.path.exists(checkpoint_path):
                        self.trainer._trainer.last_good_checkpoint = checkpoint_path
                        logger = logging.getLogger(__name__)
                        logger.info(f"[NaN RECOVERY] Updated last good checkpoint: {checkpoint_path}")

        all_hooks.append(LastGoodCheckpointHook())

        # Добавляем BestCheckpointer для сохранения лучшей модели по PQ (согласно протоколу)
        from detectron2.engine.hooks import BestCheckpointer
        all_hooks.append(
            BestCheckpointer(
                self.cfg.TEST.EVAL_PERIOD,
                self.checkpointer,
                "strawberry_3d/PQ",
                "max",
                file_prefix="model_best",
            )
        )

        # SWAG Hook: collect weight statistics during training
        if self.swag_model is not None:
            class SWAGHook(hooks.HookBase):
                def __init__(self, swag_model, cfg, dataset_len, batch_size):
                    self.swag_model = swag_model
                    self.start_epoch = getattr(cfg.MODEL.SWAG, "START_EPOCH", 10)
                    self.update_freq = getattr(cfg.MODEL.SWAG, "UPDATE_FREQ", 5)
                    self.dataset_len = dataset_len
                    self.batch_size = batch_size
                    self.collection_started = False

                def after_step(self):
                    # Calculate current epoch
                    current_epoch = self.trainer.iter * self.batch_size / self.dataset_len

                    # Start collecting after START_EPOCH
                    if current_epoch >= self.start_epoch:
                        if not self.collection_started:
                            logger = logging.getLogger("odin_strawberry")
                            logger.info(f">>> SWAG: Starting weight collection at epoch {current_epoch:.2f} <<<")
                            self.collection_started = True

                        # Collect weights every UPDATE_FREQ iterations
                        if (self.trainer.iter + 1) % self.update_freq == 0:
                            # Get base model (unwrap DDP if needed)
                            model = self.trainer.model
                            if isinstance(model, DistributedDataParallel):
                                model = model.module

                            # Collect only predictor weights (not entire model)
                            if hasattr(model, 'sem_seg_head') and hasattr(model.sem_seg_head, 'predictor'):
                                predictor = model.sem_seg_head.predictor
                                self.swag_model.collect_model(predictor)
                            else:
                                logger = logging.getLogger("odin_strawberry")
                                logger.warning(">>> SWAG: Could not find predictor, skipping collection <<<")

                            if (self.trainer.iter + 1) % (self.update_freq * 20) == 0:
                                logger = logging.getLogger("odin_strawberry")
                                logger.info(f">>> SWAG: Collected {self.swag_model.n_models} predictor weight snapshots <<<")

            logger = logging.getLogger("odin_strawberry")
            logger.info("Adding SWAG hook for weight statistics collection...")
            all_hooks.append(SWAGHook(self.swag_model, self.cfg, dataset_len, self.cfg.SOLVER.IMS_PER_BATCH))

        return all_hooks

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        Переопределяем планировщик, чтобы настроить Warmup (разминку) для стабильности.
        Warmup можно настроить через cfg.SOLVER.WARMUP_ITERS или cfg.WARMUP_RATIO.
        """
        from detectron2.solver import build_lr_scheduler

        # Настраиваем Warmup через ratio (процент от MAX_ITER) или абсолютное значение
        cfg.defrost()

        # Если задан WARMUP_RATIO (процент от SOLVER.MAX_ITER), используем его
        if hasattr(cfg, 'WARMUP_RATIO'):
            warmup_iters = int(cfg.SOLVER.MAX_ITER * cfg.WARMUP_RATIO)
            cfg.SOLVER.WARMUP_ITERS = warmup_iters
        # Иначе используем значение из cfg.SOLVER.WARMUP_ITERS (может быть переопределено через аргументы)

        cfg.SOLVER.WARMUP_FACTOR = 0.01  # Начальный LR = BASE_LR * WARMUP_FACTOR
        cfg.freeze()

        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Запускаем инференс (валидацию) с использованием AMP (Mixed Precision).
        Это позволяет экономить память точно так же, как при обучении.

        Если во время eval возникает NaN, пропускаем эту валидацию и продолжаем обучение.
        """
        # Принудительно очищаем кэш перед валидацией
        torch.cuda.empty_cache()

        try:
            # Оборачиваем инференс в autocast, если AMP включен в конфиге
            if cfg.SOLVER.AMP.ENABLED:
                with autocast():
                    return super().test(cfg, model, evaluators)
            else:
                return super().test(cfg, model, evaluators)
        except RuntimeError as e:
            if "NaN" in str(e):
                logger = logging.getLogger("odin_strawberry")
                logger.warning(f"[EVAL SKIP] NaN detected during evaluation: {e}")
                logger.warning("[EVAL SKIP] Skipping this evaluation and continuing training...")
                # Возвращаем пустой результат, чтобы не прерывать обучение
                return {}
            else:
                # Если это не NaN ошибка, пробрасываем дальше
                raise

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return [Strawberry3DEvaluator(dataset_name, output_folder, cfg)]

    @classmethod
    def build_train_loader(cls, cfg):
        dataset_name = cfg.DATASETS.TRAIN[0]
        dataset_dict = DatasetCatalog.get(dataset_name)

        # Detect dataset type from dataset name
        dataset_type = "nbv_stage2" if "nbv_stage2" in dataset_name else "strawberry"

        mapper = StrawberryDatasetMapper(cfg, is_train=True, dataset_type=dataset_type)
        return build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dict)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        dataset_dict = DatasetCatalog.get(dataset_name)

        # Detect dataset type from dataset name
        dataset_type = "nbv_stage2" if "nbv_stage2" in dataset_name else "strawberry"

        mapper = StrawberryDatasetMapper(cfg, is_train=False, dataset_type=dataset_type)
        return build_detection_test_loader(cfg, mapper=mapper, dataset=dataset_dict)

    @classmethod
    def build_optimizer(cls, cfg, model):
        # Implementation mirrors original train_odin.py
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.SOLVER.BASE_LR)
        return optimizer

    def resume_or_load(self, resume=True):
        """
        Override to handle SWAG state loading.
        """
        checkpoint = super().resume_or_load(resume=resume)

        # Load SWAG state if available
        if self.swag_model is not None and resume:
            swag_file = os.path.join(self.cfg.OUTPUT_DIR, "swag_state.pth")
            if os.path.exists(swag_file):
                logger = logging.getLogger("odin_strawberry")
                logger.info(f"Loading SWAG state from {swag_file}")
                swag_state = torch.load(swag_file, map_location="cpu")
                self.swag_model.load_state_dict(swag_state)
                logger.info(f"SWAG state loaded: {self.swag_model.n_models} models collected")

        return checkpoint

    def _write_metrics(self, loss_dict, data_time, prefix=""):
        """
        Override to save SWAG state with checkpoints.
        """
        super()._write_metrics(loss_dict, data_time, prefix)

        # Save SWAG state periodically (same frequency as checkpoints)
        if self.swag_model is not None and self.swag_model.n_models > 0:
            if (self.iter + 1) % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                swag_file = os.path.join(self.cfg.OUTPUT_DIR, "swag_state.pth")
                torch.save(self.swag_model.state_dict(), swag_file)
                logger = logging.getLogger("odin_strawberry")
                logger.info(f">>> SWAG state saved: {self.swag_model.n_models} models <<<")


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    
    # Defensive check: filter out flags that slipped into opts due to incorrect order
    clean_opts = []
    i = 0
    while i < len(args.opts):
        key = args.opts[i]
        if key.startswith("--"):
            # This is a flag, skip it and its potential value
            print(f"Warning: Found flag {key} in config overrides (opts). Removing it to avoid YACS error.")
            i += 1
            if i < len(args.opts) and not args.opts[i].startswith("--") and "." not in args.opts[i]:
                i += 1
            continue
        clean_opts.append(key)
        i += 1
        
    cfg.merge_from_list(clean_opts)

    # Auto-detect dataset type and register accordingly
    dataset_dir = args.dataset_dir
    splits_file = args.splits_file

    dataset_type = detect_dataset_type(dataset_dir)
    print(f"Detected dataset type: {dataset_type}")

    if dataset_type == "nbv_stage2":
        register_nbv_stage2_datasets(dataset_dir, splits_file)
        cfg.DATASETS.TRAIN = ("nbv_stage2_train",)
        cfg.DATASETS.TEST = ("nbv_stage2_val",)
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(NBV_CATEGORIES)
        print(f"Using NBV Stage2 dataset with {len(NBV_CATEGORIES)} classes")
    else:  # strawberry
        register_strawberry_datasets(dataset_dir, splits_file)
        cfg.DATASETS.TRAIN = ("strawberry_train",)
        cfg.DATASETS.TEST = ("strawberry_val",)
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = NUM_CLASSES
        print(f"Using Strawberry dataset with {NUM_CLASSES} classes")
    
    # Конфигурация размера батча и кол-ва кадров
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    
    # Расчет макс. итераций на основе эпох, если задано
    dataset_len = len(DatasetCatalog.get(cfg.DATASETS.TRAIN[0]))
    steps_per_epoch = dataset_len // args.batch_size
    if args.num_epochs > 0:
        cfg.SOLVER.MAX_ITER = int(args.num_epochs * steps_per_epoch)
        # Автоматически ставим шаги затухания LR (на 70% и 90% пути)
        cfg.SOLVER.STEPS = (int(cfg.SOLVER.MAX_ITER * 0.7), int(cfg.SOLVER.MAX_ITER * 0.9))
        print(f"Calculated MAX_ITER: {cfg.SOLVER.MAX_ITER}, STEPS: {cfg.SOLVER.STEPS}")
    else:
        cfg.SOLVER.MAX_ITER = args.max_iter

    # Настройка периодов (Eval и Checkpoint)
    cfg.TEST.EVAL_PERIOD = args.eval_period
    cfg.CHECKPOINT_PERIOD = args.checkpoint_period
    
    # Bayesian Inference Samples
    # Мы можем передавать это через opts, но для удобства добавим и сюда
    if hasattr(args, "bayesian_samples"):
        cfg.MODEL.BAYESIAN_SAMPLES = args.bayesian_samples
    if args.eval_period == 0:
        eval_period = steps_per_epoch * 2 # Каждые 2 эпохи по умолчанию
    else:
        eval_period = args.eval_period
        
    if args.checkpoint_period == 0:
        checkpoint_period = steps_per_epoch # Каждую эпоху по умолчанию
    else:
        checkpoint_period = args.checkpoint_period
        
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period
    cfg.TEST.EVAL_PERIOD = eval_period

    cfg.SOLVER.BASE_LR = args.lr

    # Warmup configuration
    if args.warmup_ratio is not None:
        cfg.WARMUP_RATIO = args.warmup_ratio  # Will be used in build_lr_scheduler

    cfg.DATALOADER.NUM_WORKERS = 4 
    
    cfg.INPUT.SAMPLING_FRAME_NUM = args.num_frames
    cfg.INPUT.IMAGE_SIZE = args.image_size
    cfg.INPUT.MIN_SIZE_TRAIN = (args.image_size,)
    cfg.INPUT.MAX_SIZE_TRAIN = args.image_size
    cfg.INPUT.MIN_SIZE_TEST = args.image_size
    cfg.INPUT.MAX_SIZE_TEST = args.image_size
    
    # Ограничиваем количество кадров на валидации, чтобы избежать OOM
    cfg.MAX_FRAME_NUM = args.num_frames
    cfg.SOLVER.TEST_IMS_PER_BATCH = 1
    
    # Визуализация
    if args.visualize:
        cfg.VISUALIZE = True
        cfg.VISUALIZE_3D = True
        cfg.VISUALIZE_LOG_DIR = os.path.join(cfg.OUTPUT_DIR, "inference", "visualizations")
        os.makedirs(cfg.VISUALIZE_LOG_DIR, exist_ok=True)

    # Gradient Clipping для стабильности (защита от NaN)
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.1 # Жесткая обрезка для стабильности

    cfg.MAX_TIME_HOURS = getattr(args, "max_time", 11.5)

    # NaN recovery parameters
    cfg.NAN_RECOVERY_ENABLED = getattr(args, "nan_recovery", False)
    cfg.NAN_LR_SCALE = getattr(args, "nan_lr_scale", 0.1)
    cfg.NAN_RECOVERY_ITERS = getattr(args, "nan_recovery_iters", 100)

    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="odin_strawberry")
    return cfg

def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = MyTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = MyTrainer.test(cfg, model)
        return res

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to multiview_dataset")
    parser.add_argument("--splits_file", type=str, required=True, help="Path to splits.json")

    # Параметры для управления из ноутбука
    parser.add_argument("--num_epochs", type=float, default=-1, help="Total epochs (overrides max_iter)")
    parser.add_argument("--max_iter", type=int, default=3000, help="Total iterations")
    parser.add_argument("--eval_period", type=int, default=0, help="Eval every N iterations (0 = auto every 2 epochs)")
    parser.add_argument("--checkpoint_period", type=int, default=0, help="Save checkpoint every N iterations (0 = every epoch)")
    parser.add_argument("--batch_size", type=int, default=1, help="Images per batch")
    parser.add_argument("--num_frames", type=int, default=3, help="Frames per sample")
    parser.add_argument("--image_size", type=int, default=224, help="Input frame resolution")
    parser.add_argument("--lr", type=float, default=0.0001, help="Base learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=None, help="Warmup ratio (0.0-1.0) of total iterations. 0.0 = no warmup, 0.1 = 10%% warmup. If not set, uses default warmup iters.")
    parser.add_argument("--visualize", action="store_true", help="Enable 3D visualization dump")
    parser.add_argument("--max_time", type=float, default=11.5, help="Max time in hours")
    parser.add_argument("--bayesian_samples", type=int, default=1, help="Number of MC samples for Bayesian inference")

    # NaN recovery parameters
    parser.add_argument("--nan_recovery", action="store_true", help="Enable automatic NaN recovery (reduce LR and gradually restore)")
    parser.add_argument("--nan_lr_scale", type=float, default=0.1, help="LR scale factor when NaN detected (default: 0.1 = reduce by 10x)")
    parser.add_argument("--nan_recovery_iters", type=int, default=100, help="Iterations to recover LR back to original (default: 100)")

    args = parser.parse_args()
    
    os.environ['TORCH_CUDNN_V8_API_DISABLED'] = '1'
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
