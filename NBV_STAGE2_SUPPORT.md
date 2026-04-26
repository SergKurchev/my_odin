# NBV Stage2 Dataset Support

## Overview

ODIN теперь поддерживает два типа датасетов с автоматическим определением формата:

1. **Strawberry Dataset** - оригинальный датасет для классификации клубники (3 класса)
2. **NBV Stage2 Dataset** - датасет с примитивами для Next-Best-View задач (24 класса)

## Dataset Format Differences

### Strawberry Dataset
```json
// cameras.json - LIST format
[
  {
    "frame_index": 0,
    "intrinsics": {"fx": 886.8, "fy": 886.8, "cx": 512.0, "cy": 512.0},
    "position": [0.152, 0.667, 0.450],
    "rotation": [-0.057, 0.918, -0.363, -0.144]
  },
  ...
]

// color_map.json - DICT format
{
  "1": {
    "instance_id": 1,
    "category_id": 2,
    "ripeness": "half_ripe",
    "color": [1, 0, 2]
  },
  ...
}
```

**Classes**: 3 (Ripe=0, Unripe=1, Half-ripe=2)

### NBV Stage2 Dataset
```json
// cameras.json - DICT format
{
  "00000": {
    "position": [0.730, 0.043, 0.810],
    "target": [0.510, 0.045, 0.325],
    "up": [0, 0, 1],
    "rotation": [-0.150, -0.149, -0.688, 0.693],
    "intrinsics": {
      "fx": 193.98, "fy": 193.98,
      "cx": 112.0, "cy": 112.0,
      "width": 224, "height": 224,
      "near": 0.1, "far": 10.0, "fov_deg": 60.0
    }
  },
  ...
}

// color_map.json - LIST format
[
  {
    "color": [255, 0, 0],
    "instance_id": 0,
    "category_id": 6,              // Primitive ID (1-8)
    "category_name": "target_object",
    "texture_type": "green"        // Texture: red/mixed/green
  },
  {
    "color": [0, 255, 0],
    "instance_id": 9,
    "category_id": 9,               // Robot (excluded from training)
    "category_name": "robot"
  },
  ...
]
```

**Classes**: 24 (8 primitives × 3 textures)
- Primitives: cube(1), sphere(2), cylinder(3), cone(4), torus(5), capsule(6), ellipsoid(7), pyramid(8)
- Textures: red, mixed, green
- Robot (category_id=9) is treated as background and excluded

**Class Mapping**:
```
Class  0: cube_red        Class  8: cylinder_green   Class 16: capsule_mixed
Class  1: cube_mixed      Class  9: cone_red         Class 17: capsule_green
Class  2: cube_green      Class 10: cone_mixed       Class 18: ellipsoid_red
Class  3: sphere_red      Class 11: cone_green       Class 19: ellipsoid_mixed
Class  4: sphere_mixed    Class 12: torus_red        Class 20: ellipsoid_green
Class  5: sphere_green    Class 13: torus_mixed      Class 21: pyramid_red
Class  6: cylinder_red    Class 14: torus_green      Class 22: pyramid_mixed
Class  7: cylinder_mixed  Class 15: capsule_red      Class 23: pyramid_green
```

## Implementation Details

### 1. Auto-Detection (`detect_dataset_type()`)

Функция автоматически определяет тип датасета по формату `cameras.json` и `color_map.json`:

```python
def detect_dataset_type(dataset_dir: str) -> str:
    """
    Returns: "strawberry" or "nbv_stage2"
    """
    # NBV Stage2: cameras is dict, color_map is list
    # Strawberry: cameras is list, color_map is dict
```

### 2. Specialized Data Loaders

**`get_nbv_stage2_dataset_dicts()`**:
- Конвертирует `cameras` dict → list
- Обрабатывает `color_map` как список
- Поддерживает поля `texture_type` и `category_name`

**`get_strawberry_dataset_dicts()`**:
- Обрабатывает `cameras` как список
- Обрабатывает `color_map` как словарь
- Поддерживает поле `ripeness`

### 3. Universal Dataset Mapper

`StrawberryDatasetMapper` теперь принимает параметр `dataset_type`:

```python
class StrawberryDatasetMapper:
    def __init__(self, cfg, is_train: bool, dataset_type: str = "strawberry"):
        if dataset_type == "nbv_stage2":
            self.categories = NBV_CATEGORIES  # 9 classes
        else:
            self.categories = CATEGORIES  # 3 classes
```

### 4. Automatic Registration

В `setup()` функции:

```python
dataset_type = detect_dataset_type(dataset_dir)
print(f"Detected dataset type: {dataset_type}")

if dataset_type == "nbv_stage2":
    register_nbv_stage2_datasets(dataset_dir, splits_file)
    cfg.DATASETS.TRAIN = ("nbv_stage2_train",)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(NBV_CATEGORIES)
else:
    register_strawberry_datasets(dataset_dir, splits_file)
    cfg.DATASETS.TRAIN = ("strawberry_train",)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = NUM_CLASSES
```

## Usage

### Training

Просто укажите путь к датасету - тип определится автоматически:

```bash
python my_train_odin.py \
  --config-file configs/scannet_context/3d.yaml \
  --dataset_dir /path/to/nbv_stage2_dataset \
  --splits_file splits.json \
  --num_epochs 15 \
  --batch_size 2
```

### Kaggle Notebook

Ноутбук `kaggle_nbv_stage3_train_odin.ipynb` теперь не требует указания `MODEL.SEM_SEG_HEAD.NUM_CLASSES`:

```python
train_cmd = [
    VENV_PYTHON, "my_odin/my_train_odin.py",
    "--dataset_dir", DATASET_DIR,
    "--splits_file", SPLITS_FILE,
    # NUM_CLASSES определяется автоматически!
]
```

## Fixed Issues

### 1. `AttributeError: 'list' object has no attribute 'values'`
**Причина**: Код ожидал `color_map` как словарь, но NBV Stage2 использует список.

**Решение**: Добавлена проверка типа во всех местах:
- `my_train_odin.py` (2 места)
- `test_vis.py` (1 место)
- `generate_sample_viewer.py` (2 места)

### 2. `AttributeError: 'dict' object has no attribute '__len__'`
**Причина**: Код пытался использовать `len(cameras)` на словаре.

**Решение**: Специализированный loader конвертирует dict → list.

### 3. Wrong NUM_CLASSES
**Причина**: Хардкод `NUM_CLASSES=3` для всех датасетов.

**Решение**: Автоматическое определение на основе типа датасета.

## Files Modified

1. **my_train_odin.py**:
   - Добавлены `NBV_CATEGORIES`, `get_nbv_stage2_dataset_dicts()`, `detect_dataset_type()`
   - Обновлен `StrawberryDatasetMapper` для поддержки обоих типов
   - Автоматическая регистрация датасета в `setup()`

2. **test_vis.py**:
   - Добавлена проверка типа `color_map`

3. **generate_sample_viewer.py**:
   - Добавлена проверка типа `color_map` в `build_pointcloud()` и `build_html()`
   - Удален мертвый код (unreachable после `return`)

4. **kaggle_nbv_stage3_train_odin.ipynb**:
   - Удален хардкод `MODEL.SEM_SEG_HEAD.NUM_CLASSES`
   - Обновлены комментарии

## Testing

Протестировано на:
- ✅ Strawberry Dataset (локально)
- ✅ NBV Stage2 Dataset (локально: `C:\Users\...\NBV_with_obstacles_and_robot\dataset\primitives\stage2`)
- ⏳ NBV Stage2 Dataset (Kaggle: `sergeykurchev/nbv-stage2-dataset`)

## Future Work

- [ ] Добавить поддержку NBV Stage3 (с препятствиями)
- [ ] Создать unified evaluator для обоих типов датасетов
- [ ] Добавить визуализацию для NBV датасетов (текущая визуализация заточена под strawberry)
