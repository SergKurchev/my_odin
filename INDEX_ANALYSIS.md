# Анализ индексации в ODIN: Визуализация vs Метрики

## Проблема

Индексы категорий обрабатываются по-разному для:
1. **Визуализации** (3D viewer) - ИСПРАВЛЕНО в commit 0ac499d
2. **Метрик** (PQ, mAP) - ВОЗМОЖНО СЛОМАНО в commit 9afc631

Нужно убедиться что исправление метрик не сломало визуализацию и наоборот.

---

## Цепочка индексов

### 1. Выход модели ODIN (`odin_model.py`)

#### В `prepare_3d()` (строка ~1310-1320):
```python
# Check if this is strawberry dataset (uses 0-indexed labels)
is_strawberry = batched_inputs is not None and 'strawberry' in batched_inputs.get('dataset_name', '')

# add +1 to labels as mask3d evals from 1-18 (ScanNet convention)
# BUT: strawberry dataset uses 0-indexed labels, so skip +1
if is_strawberry:
    pred_classes_output = labels_per_image  # 0-indexed: [0, 1, 2]
else:
    pred_classes_output = labels_per_image + 1  # 1-indexed: [1, 2, 3, ...]

result_3d = {
    "pred_classes": pred_classes_output,  # ← ЧТО ЗДЕСЬ?
    ...
}
```

**Для strawberry**: `pred_classes` = [0, 1, 2] (0-indexed)
**Для ScanNet**: `pred_classes` = [1, 2, 3, ...] (1-indexed)

---

### 2. Evaluator: Парсинг предсказаний (`my_train_odin.py`)

#### В `_parse_pred()` (строка ~832-840):
```python
def _parse_pred(self, _out):
    pred = _out['instances_3d']
    res = {}
    for key in pred:
        if isinstance(pred[key], torch.Tensor):
            res[key] = pred[key].cpu().numpy()
        else:
            res[key] = pred[key]
    return res
```

**Результат**: `res["labels"]` = то что пришло из `pred_classes` (0-indexed для strawberry)

---

### 3. Использование в метриках (`my_train_odin.py`)

#### В `evaluate()` (строка ~1040-1080):
```python
# Формирование словарей для evaluator
preds_dict = {}
gts_dict = {}

for idx in self.processed_preds.keys():
    pred_info = self.processed_preds[idx]
    gt_info = self.processed_gts[idx]
    
    preds_dict[idx] = {
        'pred_classes': pred_info.get("labels", np.array([])),  # ← 0-indexed для strawberry
        'pred_scores': pred_info.get("scores", np.array([])),
        'pred_masks': pred_info.get("masks_3d", None)
    }
    
    gts_dict[idx] = {
        'gt_masks': gt_info.get("masks", None),
        'class_labels': gt_info.get("class_labels", np.array([]))  # ← ЧТО ЗДЕСЬ?
    }

# Вызов evaluator
for idx in preds_dict.keys():
    gt2pred, pred2gt = self.scannet_evaluator.assign_instances_for_scan(
        preds_dict[idx], 
        gts_dict[idx]  # ← ИСПРАВЛЕНО: было gts_dict[i]
    )
```

**Вопрос 1**: Что в `class_labels` из GT? 0-indexed или 1-indexed?

---

### 4. GT категории (`my_train_odin.py`)

#### В `_parse_gt()` (строка ~816-830):
```python
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
        "class_labels": target_dict["labels"].cpu()  # ← ЧТО ЗДЕСЬ?
    }
```

**Вопрос 2**: Что возвращает `convert_video_instances_to_3d` в `target_dict["labels"]`?

#### Откуда берутся GT labels (строка ~450-550 в mapper):
```python
# В StrawberryDatasetMapper
instances.gt_classes = torch.tensor(gt_classes, dtype=torch.int64)  # ← ЧТО ЗДЕСЬ?

# gt_classes берутся из color_map:
for color, info in color_map_items:
    if info["category_id"] not in self.categories:
        continue
    px = (mr == color[0]) & (mg == color[1]) & (mb == color[2])
    if np.any(px):
        gt_classes.append(info["category_id"])  # ← category_id из данных
```

**Вопрос 3**: Что в `info["category_id"]`? 0-indexed или 1-indexed?

#### Откуда берётся category_id (строка ~180-210 в mapper):
```python
# В load_strawberry_annotations_to_coco
for inst_id, v in color_map.items():
    old_class_id = v["category_id"]  # ← Из JSON файла
    
    # Маппинг на 0-indexed
    if old_class_id not in class_id_map:
        new_class_id = len(class_id_map)  # 0, 1, 2, ...
        class_id_map[old_class_id] = new_class_id
    else:
        new_class_id = class_id_map[old_class_id]
    
    new_info = v.copy()
    new_info["category_id"] = new_class_id  # ← ПЕРЕНАЗНАЧЕНО на 0-indexed
    color_to_info[tuple(v["color"])] = new_info
```

**Ответ на вопрос 3**: GT `category_id` = **0-indexed** (0, 1, 2)

---

### 5. Использование в визуализации (`my_train_odin.py`)

#### В `evaluate()` для визуализации (строка ~850-920):
```python
# Pred categories
pred_classes = pred_info.get("labels", np.array([]))  # 0-indexed для strawberry

for inst_idx in range(pred_masks_3d.shape[0]):
    m = pred_masks_3d[inst_idx] > 0.5
    if np.any(m):
        # ВАЖНО: для strawberry dataset pred_classes уже 0-индексированные
        # поэтому используем напрямую без вычитания
        point_pred_cat[m] = int(pred_classes[inst_idx])  # ← 0-indexed

# GT categories
gt_class_labels = gt_info.get("class_labels", np.array([]))  # 0-indexed

for inst_idx in range(gt_masks_3d.shape[0]):
    m = gt_masks_3d[inst_idx] > 0.5
    if np.any(m):
        cat_gt_frame[m_mask] = int(gt_c[inst_i])  # ← 0-indexed
```

**Результат**: И pred и GT используют **0-indexed** категории для визуализации.

#### В `generate_sample_viewer.py` (палитра):
```python
CATEGORY_COLORS = {
    0: [128, 128, 128],  # Background/Unknown
    1: [255, 0, 0],      # Strawberry (category_id=0 → palette key=1? НЕТ!)
    2: [0, 255, 0],      # Peduncle
    3: [255, 165, 0],    # Background
}
```

**ПРОБЛЕМА**: Палитра использует ключи 0, 1, 2, 3, но категории 0-indexed (0, 1, 2).

**Вопрос 4**: Как маппятся категории на палитру?

---

## Критические проверки

### Проверка 1: Pred classes в метриках

**Файл**: `my_train_odin.py`, строка ~1050
```python
preds_dict[idx] = {
    'pred_classes': pred_info.get("labels", np.array([])),  # ← Что здесь?
    ...
}
```

**Ожидаемое**: [0, 1, 2] для strawberry (0-indexed)

**Проверка**:
```python
pred_classes = pred_info.get("labels", np.array([]))
logger.info(f"[CHECK] pred_classes for idx={idx}: {pred_classes}")
logger.info(f"[CHECK] unique pred_classes: {np.unique(pred_classes)}")
```

### Проверка 2: GT class_labels в метриках

**Файл**: `my_train_odin.py`, строка ~1060
```python
gts_dict[idx] = {
    'class_labels': gt_info.get("class_labels", np.array([]))  # ← Что здесь?
}
```

**Ожидаемое**: [0, 1, 2] для strawberry (0-indexed)

**Проверка**:
```python
class_labels = gt_info.get("class_labels", np.array([]))
logger.info(f"[CHECK] GT class_labels for idx={idx}: {class_labels}")
logger.info(f"[CHECK] unique GT class_labels: {np.unique(class_labels)}")
```

### Проверка 3: Evaluator ожидает 0-indexed или 1-indexed?

**Файл**: `odin/data_video/evaluation/evaluate_semantic_instance.py`

**Нужно проверить**: Какие индексы ожидает `assign_instances_for_scan()`?

**Для ScanNet**: 1-indexed (1-18)
**Для strawberry**: 0-indexed (0, 1, 2) ???

**Проверка**: Посмотреть код evaluator и понять что он ожидает.

### Проверка 4: Индексация словаря в evaluate()

**Старый код** (до commit 9afc631):
```python
for i, (k, v) in enumerate(preds_dict.items()):
    gt2pred, pred2gt = self.scannet_evaluator.assign_instances_for_scan(v, gts_dict[i])
    matches[i] = {'gt': gt2pred, 'pred': pred2gt}
```

**Новый код** (после commit 9afc631):
```python
for i, (k, v) in enumerate(preds_dict.items()):
    gt2pred, pred2gt = self.scannet_evaluator.assign_instances_for_scan(v, gts_dict[k])
    matches[i] = {'gt': gt2pred, 'pred': pred2gt}
```

**Вопрос 5**: Правильно ли `matches[i]`? Или должно быть `matches[k]`?

**Анализ**:
- `preds_dict.keys()` = [0, 1, 2, ..., 143] (последовательные)
- `enumerate()` даёт `i` = 0, 1, 2, ..., 143
- `k` = ключ из словаря = 0, 1, 2, ..., 143

**Если ключи последовательные**: `i == k`, и `matches[i]` правильно.
**Если ключи НЕ последовательные**: `i != k`, и `matches[i]` неправильно!

**Проверка**:
```python
for i, (k, v) in enumerate(preds_dict.items()):
    logger.info(f"[CHECK] i={i}, k={k}, i==k: {i==k}")
    if i != k:
        logger.error(f"[ERROR] Key mismatch! i={i} but k={k}")
```

---

## Гипотезы

### Гипотеза 1: Ключи словаря не последовательные

**Причина**: Если `_current_idx` пропускает значения (например, из-за фильтрации), то ключи могут быть [0, 2, 5, 7, ...].

**Результат**: 
- `gts_dict[i]` работало случайно когда ключи были последовательными
- `gts_dict[k]` правильно, но `matches[i]` неправильно

**Решение**: Использовать `k` везде:
```python
for i, (k, v) in enumerate(preds_dict.items()):
    gt2pred, pred2gt = self.scannet_evaluator.assign_instances_for_scan(v, gts_dict[k])
    matches[k] = {'gt': gt2pred, 'pred': pred2gt}  # ← Использовать k!
```

### Гипотеза 2: Evaluator ожидает 1-indexed для strawberry

**Причина**: Если evaluator написан для ScanNet (1-indexed), он может не работать с 0-indexed.

**Результат**: Метрики = 0 потому что категории не совпадают.

**Решение**: Добавить +1 к pred_classes и class_labels перед передачей в evaluator:
```python
preds_dict[idx] = {
    'pred_classes': pred_info.get("labels", np.array([])) + 1,  # 0→1, 1→2, 2→3
    ...
}

gts_dict[idx] = {
    'class_labels': gt_info.get("class_labels", np.array([])) + 1,  # 0→1, 1→2, 2→3
}
```

**НО**: Это сломает визуализацию!

### Гипотеза 3: Визуализация и метрики используют разные данные

**Факт**: 
- Визуализация использует `pred_classes` напрямую (строка ~853)
- Метрики используют `pred_classes` через `preds_dict` (строка ~1050)

**Вопрос**: Это одни и те же данные?

**Проверка**: Убедиться что `pred_info.get("labels")` одинаковое в обоих местах.

---

## План действий

1. **Добавить проверки** из раздела "Критические проверки"
2. **Запустить eval** и собрать логи
3. **Проверить**:
   - Ключи словаря последовательные?
   - pred_classes и class_labels 0-indexed или 1-indexed?
   - Evaluator ожидает 0-indexed или 1-indexed?
4. **Исправить** в зависимости от результатов:
   - Если ключи не последовательные → использовать `k` везде
   - Если evaluator ожидает 1-indexed → добавить +1 только для метрик
   - Если evaluator ожидает 0-indexed → оставить как есть

---

## Ожидаемые результаты

### Если всё правильно:
```
[CHECK] pred_classes unique: [0 1 2]  ← 0-indexed
[CHECK] GT class_labels unique: [0 1 2]  ← 0-indexed
[CHECK] i=0, k=0, i==k: True  ← Ключи последовательные
[CHECK] i=1, k=1, i==k: True
...
[TEST] PQ = 0.0376  ← НЕ НОЛЬ!
```

### Если сломано:
```
[CHECK] i=0, k=0, i==k: True
[CHECK] i=1, k=2, i==k: False  ← КЛЮЧИ НЕ ПОСЛЕДОВАТЕЛЬНЫЕ!
[ERROR] Key mismatch! i=1 but k=2
[TEST] PQ = 0.0  ← НОЛЬ!
```

Или:
```
[CHECK] pred_classes unique: [0 1 2]  ← 0-indexed
[CHECK] GT class_labels unique: [1 2 3]  ← 1-indexed (НЕСООТВЕТСТВИЕ!)
[TEST] PQ = 0.0  ← НОЛЬ!
```
