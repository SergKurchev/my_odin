import sys
import os
import torch
import numpy as np
from pathlib import Path
import json

# Мокаем Detectron2, чтобы запустить локально на Windows без его установки
import sys
from unittest.mock import MagicMock
sys.modules['detectron2'] = MagicMock()
sys.modules['detectron2.utils'] = MagicMock()
sys.modules['detectron2.utils.comm'] = MagicMock()
sys.modules['detectron2.evaluation'] = MagicMock()

class MockDatasetEvaluator:
    def reset(self): pass
    def process(self, inputs, outputs): pass
    def evaluate(self): pass

sys.modules['detectron2.evaluation.evaluator'] = MagicMock()
sys.modules['detectron2.evaluation.evaluator'].DatasetEvaluator = MockDatasetEvaluator

# Add path to import my_train_odin
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from my_train_odin import Strawberry3DEvaluator
from generate_sample_viewer import build_html

class DummyCfg:
    OUTPUT_DIR = "./output_odin"
    class INPUT:
        SIZE_DIVISIBILITY = 32

def main():
    print("Инициализация Strawberry3DEvaluator...")
    cfg = DummyCfg()
    evaluator = Strawberry3DEvaluator("strawberry_val", "./output_odin", cfg)
    
    # 1. Подготовим "фейковые" данные в формате, в котором они приходят из D2
    # Для этого загрузим один семпл локально
    dataset_dir = r"C:\Users\NeverGonnaGiveYouUp\OneDrive\Рабочий стол\study_materials\Skoltech\projects\StrawPick\NBV_article\SegPointNetsTest\multiview_dataset"
    sample_id = "sample_00000"
    s_dir = Path(dataset_dir) / sample_id
    
    with open(s_dir / "cameras.json", "r") as f:
        cameras_data = json.load(f)
    with open(s_dir / "color_map.json", "r") as f:
        color_map = json.load(f)

    # Загружаем только 3 кадра для быстрого теста
    images = []
    depths = []
    poses = []
    intrinsics = []
    
    from PIL import Image
    for cam in cameras_data[:3]:
        fi = cam["frame_index"]
        name = f"{fi:05d}"
        
        rgb_p = s_dir / "rgb" / f"{name}.png"
        depth_p = s_dir / "depth" / f"{name}.npy"
        
        if not rgb_p.exists() or not depth_p.exists():
            continue
            
        # D2 format: (C, H, W)
        img = np.asarray(Image.open(rgb_p)).transpose(2, 0, 1)
        images.append(torch.from_numpy(img))
        
        # Depth format: (H, W) flipped vertically as in mapper
        Z_full = np.load(depth_p)[::-1, :].copy()
        depths.append(torch.from_numpy(Z_full))
        
        # Poses and intrinsics
        from test_vis import quat_to_rotmat
        R_mat = quat_to_rotmat(*cam["rotation"])
        t_vec = np.array(cam["position"], dtype=np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R_mat
        pose[:3, 3] = t_vec
        poses.append(torch.from_numpy(pose))
        
        intr = cam["intrinsics"]
        intr_mat = np.array([
            [intr["fx"], 0, intr["cx"], 0],
            [0, intr["fy"], intr["cy"], 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        intrinsics.append(torch.from_numpy(intr_mat))

    class DummyInstances:
        def __init__(self):
            H, W = images[0].shape[1:]
            self.image_size = (H, W)
            # Фейковые 3 инстанса (GT)
            # Допустим, классы: 0 (Ripe), 1 (Unripe), 2 (Half-ripe)
            self.gt_classes = torch.tensor([0, 1, 2], dtype=torch.int64)
            self.instance_ids = torch.tensor([10, 20, 30], dtype=torch.int64)
            
            # Маски (просто квадраты для теста)
            masks = torch.zeros((3, H, W), dtype=torch.bool)
            masks[0, 100:200, 100:200] = True # Ripe
            masks[1, 200:300, 200:300] = True # Unripe
            masks[2, 300:400, 300:400] = True # Half-ripe
            self.gt_masks = masks

        def __len__(self):
            return 3

    # Подготовим inputs (D2 format)
    inputs = [{
        "image_id": sample_id,
        "images": images,
        "depths": depths,
        "poses": poses,
        "intrinsics": intrinsics,
        "color_map": color_map,
        "instances_all": [DummyInstances()] * 3 # фейковые GT для 3 кадров
    }]
    
    # Подготовим outputs (формат, который выдает ODIN: pred_masks и pred_classes)
    # В ODIN pred_classes = labels_per_image + 1 (т.е. 1-indexed)
    # Ripe (0) -> 1
    # Unripe (1) -> 2
    # Half-ripe (2) -> 3
    outputs = [{
        "pred_masks": torch.zeros((3, 50000)), # mock
        "pred_classes": torch.tensor([1, 2, 3], dtype=torch.int64) # ODIN выдаст 1, 2, 3
    }]
    
    # 2. Вызываем process
    print("Запускаем process()...")
    
    # Временно подменим _parse_pred и _parse_gt, чтобы не лезть в сложные функции маппинга 3D
    evaluator._parse_gt = lambda x: {}
    
    def fake_parse_pred(out):
        # На самом деле evaluator.processed_preds хранит:
        # num_pred_instances = pred_masks.shape[0]
        # pred_classes = pred_data.get("pred_classes")
        
        # Сделаем фейковые 3D маски для наших 3 инстансов на спроецированных точках.
        # Поскольку мы не проецируем сейчас реально, сделаем маску случайной длины (например, для 100000 точек)
        # В my_train_odin.py: 
        # m = pred_masks[inst_idx] > 0
        
        # Для теста: H_padded = 480, W_padded = 640 (при div=32). Total points = 3 * 480 * 640 = ~900000
        total_pts = 3 * 480 * 640
        masks = torch.zeros((3, total_pts), dtype=torch.bool)
        
        # Выделим блоки для наших классов
        masks[0, 100000:150000] = True
        masks[1, 200000:250000] = True
        masks[2, 300000:350000] = True
        
        return {
            "pred_masks": masks,
            "pred_classes": torch.tensor([1, 2, 3], dtype=torch.int64) # 1-indexed (Ripe, Unripe, Half-ripe)
        }
        
    evaluator._parse_pred = fake_parse_pred

    evaluator.process(inputs, outputs)
    
    # 3. Вызываем evaluate (здесь происходит генерация HTML)
    print("Запускаем evaluate() и генерацию HTML...")
    # Так как мы подменили функции, вызовем только ту часть, которая генерирует HTML
    # Установим целевые сэмплы, чтобы он точно сгенерировал HTML для нашего sample_id
    evaluator.evaluate()

    print("Тест завершен! Проверь папку output_odin на наличие HTML файла.")

if __name__ == "__main__":
    main()
