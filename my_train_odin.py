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
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
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
import pandas as pd

torch.multiprocessing.set_sharing_strategy('file_system')

# -------------------------------------------------------------------------
# 1. Dataset Registration
# -------------------------------------------------------------------------
def quat_to_rotmat(x, y, z, w):
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
    # Ensure normalization for safety
    q = np.array([x, y, z, w], dtype=np.float64)
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
        color_to_info = {tuple(v["color"]): v for v in color_map.values()}
            
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

def register_strawberry_datasets(dataset_dir: str, splits_file: str):
    for split in ["train", "val", "test"]:
        dataset_name = f"strawberry_{split}"
        DatasetCatalog.register(dataset_name, lambda s=split: get_strawberry_dataset_dicts(dataset_dir, splits_file, s))
        MetadataCatalog.get(dataset_name).set(
            thing_classes=list(CATEGORIES.values()),
            evaluator_type="strawberry"
        )


# -------------------------------------------------------------------------
# 2. Dataset Mapper
# -------------------------------------------------------------------------
class StrawberryDatasetMapper:
    def __init__(self, cfg, is_train: bool):
        self.cfg = cfg
        self.is_train = is_train
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
            
            depths.append(torch.as_tensor(depth))
            
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
                
                for color, info in color_map.items():
                    if info["category_id"] not in CATEGORIES:
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
        self.predictions = []
        self.ground_truths = []
        
    def process(self, inputs, outputs):
        """
        inputs: List of dataset dicts
        outputs: List of model outputs
        """
        for _in, _out in zip(inputs, outputs):
            # Сохраняем как есть, парсинг сделаем в evaluate для удобства отладки
            self.predictions.append(_out)
            self.ground_truths.append(_in)

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
        
        # 1. Точная выгрузка визуализаций по правилам `generate_sample_viewer.py`
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        try:
            from generate_sample_viewer import build_html
        except ImportError:
            logging.getLogger(__name__).warning("Не удалось импортировать build_html из generate_sample_viewer.py!")
            build_html = None

        if build_html is not None:
            vis_output_dir = os.path.join(self._output_dir, "visualizations")
            os.makedirs(vis_output_dir, exist_ok=True)

            target_samples = ["00000", "sample_00000", "00003", "sample_00003", "00005", "sample_00005"]

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

            for _in, _out in zip(self.ground_truths, self.predictions):
                sample_id = str(_in.get("image_id", ""))
                
                if not any(ts in sample_id for ts in target_samples):
                    continue
                
                logging.getLogger(__name__).info(f"Генерация HTML визуализации для {sample_id}...")
                
                cameras = []
                chunks = []
                stride = 2
                
                images = _in.get("images", [])
                depths = _in.get("depths", [])
                poses = _in.get("poses", [])
                intrinsics = _in.get("intrinsics", [])
                color_map = _in.get("color_map", {})
                
                for i in range(len(images)):
                    img = images[i].numpy().transpose(1, 2, 0)
                    Z_full = depths[i].numpy() # Уже перевёрнуто (flipped) в маппере
                    pose = poses[i].numpy()
                    intr = intrinsics[i].numpy()
                    
                    fx, fy, cx, cy = intr[0,0], intr[1,1], intr[0,2], intr[1,2]
                    
                    H, W = Z_full.shape
                    u = np.arange(0, W, stride, dtype=np.float32)
                    v = np.arange(0, H, stride, dtype=np.float32)
                    uu, vv = np.meshgrid(u, v)
                    
                    Z_s = Z_full[::stride, ::stride]
                    valid = (Z_s > 0.001) & (Z_s < 5.0)
                    
                    # Проекция как в generate_sample_viewer
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
                    
                    # GT Masks (тут мы бы парсили GT из маппера)
                    inst_gt = np.zeros_like(r, dtype=np.float32)
                    cat_gt = np.zeros_like(r, dtype=np.float32)

                    # Pred Masks (тут мы бы парсили pred из _out["instances_3d"])
                    inst_pred = np.zeros_like(r, dtype=np.float32)
                    cat_pred = np.zeros_like(r, dtype=np.float32)
                    
                    chunk = np.column_stack([pts_chunk, r, g, b, inst_gt, cat_gt, inst_pred, cat_pred]).astype(np.float32)
                    chunks.append(chunk)
                    
                    cam_dict = {
                        "position": t_vec.tolist(),
                        "rotation": rotmat_to_quat(R_mat),
                        "intrinsics": {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)},
                        "frame_index": i
                    }
                    cameras.append(cam_dict)
                
                if len(chunks) > 0:
                    pts = np.concatenate(chunks, axis=0)
                    # Фильтрация `plant` как в оригинале (убираем стену и пол)
                    is_white = (pts[:, 3] > 220) & (pts[:, 4] > 220) & (pts[:, 5] > 220)
                    is_black = (pts[:, 3] < 20) & (pts[:, 4] < 20) & (pts[:, 5] < 20)
                    pts = pts[~is_white & ~is_black]

                    # Прореживание для защиты браузера
                    MAX_POINTS = 800000
                    if len(pts) > MAX_POINTS:
                        idx = np.random.choice(len(pts), MAX_POINTS, replace=False)
                        pts = pts[idx]

                    html = build_html(pts, cameras, color_map, sample_name=sample_id)
                    out_html_path = os.path.join(vis_output_dir, f"{sample_id}_pred.html")
                    with open(out_html_path, "w", encoding="utf-8") as f:
                        f.write(html)
                    logging.getLogger(__name__).info(f"HTML сохранен: {out_html_path}")

        # 2. Подготовка данных для реального эвалюатора
        preds_dict = {}
        gts_dict = {}
        
        for idx, (_in, _out) in enumerate(zip(self.ground_truths, self.predictions)):
            preds_dict[idx] = self._parse_pred(_out)
            gts_dict[idx] = self._parse_gt(_in)
            
        # 3. Расчет AP (mAP, mAP@50, mAP@25) через стандартный механизм ScanNet
        # Scannet_Evaluator.evaluate возвращает данные для AP
        matches = {}
        for i, (k, v) in enumerate(preds_dict.items()):
            gt2pred, pred2gt = self.scannet_evaluator.assign_instances_for_scan(v, gts_dict[i])
            matches[i] = {'gt': gt2pred, 'pred': pred2gt}
            
        aps = self.scannet_evaluator.evaluate_matches(matches)
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
        }
        
        # Режим дебага: выводим PrettyTable в консоль
        if logging.getLogger(__name__).isEnabledFor(logging.DEBUG):
            self.scannet_evaluator.print_results(ap_results, logging.getLogger(__name__))

        # Пытаемся получить текущую итерацию и лосс из хранилища Detectron2
        try:
            storage = get_event_storage()
            iteration = storage.iter
            try:
                total_loss = storage.history("total_loss").latest()
            except Exception:
                total_loss = -1.0
        except Exception:
            iteration = -1
            total_loss = -1.0

        metrics_for_csv = metrics.copy()
        metrics_for_csv["iteration"] = iteration
        metrics_for_csv["total_loss"] = total_loss
        
        df = pd.DataFrame([metrics_for_csv])
        csv_path = os.path.join(self.cfg.OUTPUT_DIR, "metrics_comparison.csv")
        
        # Переставляем важные колонки в начало для удобства
        cols = ["iteration", "total_loss"] + [c for c in df.columns if c not in ["iteration", "total_loss"]]
        df = df[cols]

        df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
        logging.getLogger(__name__).info(f"Метрики (iter {iteration}, PQ {metrics['PQ']:.2f}, mAP@50 {metrics['mAP@50']:.2f}) записаны в {csv_path}")
        
        # Возвращаем в формате D2 для BestCheckpointer
        # Ключ должен соответствовать тому, что мы будем мониторить
        res = {f"strawberry_3d/{k}": v for k, v in metrics.items()}
        return res


# -------------------------------------------------------------------------
# 4. Trainer Override
# -------------------------------------------------------------------------
class AMPTrainerWithClipping(AMPTrainer):
    def run_step(self):
        """
        Кастомный шаг обучения с поддержкой Gradient Clipping для AMP.
        """
        assert self.model.training, "[AMPTrainerWithClipping] model was not in training mode!"
        
        # Замеряем время загрузки данных
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start
        
        # 1. Считаем потери внутри autocast
        with torch.cuda.amp.autocast():
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())
        
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
        
        # Передаем и лоссы, и время загрузки данных
        self._write_metrics(loss_dict, data_time)

class MyTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # Оборачиваем в DDP только при multi-GPU (world_size > 1)
        # При одиночном запуске DDP требует init_process_group, которого нет
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(model, device_ids=[comm.get_local_rank()])
        
        # Используем наш кастомный трейнер с обрезкой градиентов
        if cfg.SOLVER.AMP.ENABLED:
            self._trainer = AMPTrainerWithClipping(model, data_loader, optimizer)
        else:
            self._trainer = SimpleTrainer(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, trainer=weakref.proxy(self))
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
                    logger = logging.getLogger("odin_strawberry")
                    logger.warning(f"!!! TIME LIMIT REACHED ({elapsed/3600:.2f}h). Stopping training gracefully... !!!")
                    self.trainer.iter = self.trainer.max_iter # Сигнал к остановке основного цикла

        max_time = getattr(self.cfg, "MAX_TIME_HOURS", 11.5)
        all_hooks.append(TimeLimitHook(max_time))

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
        return all_hooks

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        Переопределяем планировщик, чтобы увеличить Warmup (разминку) для стабильности.
        """
        from detectron2.solver import build_lr_scheduler
        
        # Увеличиваем Warmup до 500 шагов
        cfg.defrost()
        cfg.SOLVER.WARMUP_ITERS = 500
        cfg.freeze()
        
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Запускаем инференс (валидацию) с использованием AMP (Mixed Precision).
        Это позволяет экономить память точно так же, как при обучении.
        """
        # Принудительно очищаем кэш перед валидацией
        torch.cuda.empty_cache()
        
        # Оборачиваем инференс в autocast, если AMP включен в конфиге
        if cfg.SOLVER.AMP.ENABLED:
            with autocast():
                return super().test(cfg, model, evaluators)
        else:
            return super().test(cfg, model, evaluators)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return [Strawberry3DEvaluator(dataset_name, output_folder, cfg)]

    @classmethod
    def build_train_loader(cls, cfg):
        dataset_name = cfg.DATASETS.TRAIN[0]
        dataset_dict = DatasetCatalog.get(dataset_name)
        mapper = StrawberryDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dict)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        dataset_dict = DatasetCatalog.get(dataset_name)
        mapper = StrawberryDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, mapper=mapper, dataset=dataset_dict)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        # Implementation mirrors original train_odin.py
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.SOLVER.BASE_LR)
        return optimizer


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # Defaults for Strawberry ODIN execution
    dataset_dir = args.dataset_dir
    splits_file = args.splits_file
    register_strawberry_datasets(dataset_dir, splits_file)
    
    cfg.DATASETS.TRAIN = ("strawberry_train",)
    cfg.DATASETS.TEST = ("strawberry_val",)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = NUM_CLASSES
    
    # Конфигурация размера батча и кол-ва кадров
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    
    # Расчет макс. итераций на основе эпох, если задано
    dataset_len = len(DatasetCatalog.get(cfg.DATASETS.TRAIN[0]))
    steps_per_epoch = dataset_len // args.batch_size
    if args.num_epochs > 0:
        cfg.SOLVER.MAX_ITER = int(args.num_epochs * steps_per_epoch)
        print(f"Calculated MAX_ITER: {cfg.SOLVER.MAX_ITER} for {args.num_epochs} epochs")
    else:
        cfg.SOLVER.MAX_ITER = args.max_iter

    # Настройка периодов (Eval и Checkpoint)
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
    parser.add_argument("--visualize", action="store_true", help="Enable 3D visualization dump")
    parser.add_argument("--max_time", type=float, default=11.5, help="Stop training after N hours")
    
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
