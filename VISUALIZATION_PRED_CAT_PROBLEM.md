# Проблема: Неправильные цвета категорий в визуализации Pred Cat

## Симптомы

В 3D визуализациях (HTML viewer) предсказанные категории (Pred Cat) отображаются с неправильными цветами:
- **Красный** (strawberry) → отображается как **зелёный**
- **Зелёный** (peduncle) → отображается как **оранжевый**  
- **Оранжевый** (background) → отображается как **фон**

GT (Ground Truth) категории отображаются правильно.

## История проблемы

### Предыдущие исправления (commits c581f97, 819a05b, db3a8c7):
1. Исправлена индексация палитры категорий (0-indexed вместо 1-indexed)
2. Исправлена маскировка фона в визуализациях
3. Исправлена двойная индексация в category palette

### Текущий статус:
- GT категории: ✅ Правильные цвета
- Pred категории: ❌ Неправильные цвета (сдвиг на 1)

## Категории и цвета

### Определение категорий (my_train_odin.py, строки ~30-50):

```python
STRAWBERRY_CATEGORIES = [
    {"id": 1, "name": "strawberry", "color": [255, 0, 0]},      # Красный
    {"id": 2, "name": "peduncle", "color": [0, 255, 0]},        # Зелёный
    {"id": 3, "name": "background", "color": [255, 165, 0]},    # Оранжевый
]

NUM_CLASSES = len(STRAWBERRY_CATEGORIES)  # 3
```

**Важно**: 
- `id` начинается с 1 (не с 0!)
- `NUM_CLASSES = 3`
- В модели ODIN классы индексируются от 0 до NUM_CLASSES-1 (0, 1, 2)

### Маппинг категорий:

**В данных (color_map)**:
- `category_id: 1` → strawberry (красный)
- `category_id: 2` → peduncle (зелёный)
- `category_id: 3` → background (оранжевый)

**В модели ODIN** (после обработки):
- `class_id: 0` → strawberry
- `class_id: 1` → peduncle
- `class_id: 2` → background

**Преобразование**: `class_id = category_id - 1`

---

## Код визуализации

### 1. Генерация визуализаций (my_train_odin.py, строки ~750-900)

#### Подготовка данных для build_html:

```python
# Line ~820-850
for idx, vis_info in self.vis_data.items():
    pred_info = self.processed_preds.get(idx, {})
    gt_info = self.processed_gts.get(idx, {})
    
    # Извлечение предсказаний
    pred_masks_3d = pred_info.get("masks_3d", None)  # (N_pred, num_points)
    pred_classes = pred_info.get("labels", np.array([]))  # (N_pred,)
    pred_scores = pred_info.get("scores", np.array([]))
    
    # Извлечение GT
    gt_masks_3d = gt_info.get("masks", None)  # (N_gt, num_points)
    gt_labels = gt_info.get("labels", np.array([]))  # (num_points,)
    gt_class_labels = gt_info.get("class_labels", np.array([]))  # (N_gt,)
```

#### Создание point clouds с категориями:

```python
# Line ~860-920
# GT point cloud
point_gt_cat = np.zeros(num_points, dtype=np.int32)
if gt_masks_3d is not None:
    for inst_idx in range(gt_masks_3d.shape[0]):
        m = gt_masks_3d[inst_idx] > 0.5
        if np.any(m):
            # GT categories are already correct (1-indexed in data)
            point_gt_cat[m] = int(gt_class_labels[inst_idx])

# Pred point cloud  
point_pred_cat = np.zeros(num_points, dtype=np.int32)
if pred_masks_3d is not None:
    for inst_idx in range(pred_masks_3d.shape[0]):
        m = pred_masks_3d[inst_idx] > 0.5
        if np.any(m):
            # CRITICAL: pred_classes are 0-indexed, need to convert to 1-indexed
            point_pred_cat[m] = int(pred_classes[inst_idx]) + 1  # ← ПРОВЕРИТЬ ЭТО!
```

**Вопрос**: Правильно ли здесь `+1`? Или нужно `-1`?

### 2. HTML Viewer (generate_sample_viewer.py)

#### Палитра категорий:

```python
# Line ~50-60
CATEGORY_COLORS = {
    0: [128, 128, 128],  # Background/Unknown - серый
    1: [255, 0, 0],      # Strawberry - красный
    2: [0, 255, 0],      # Peduncle - зелёный
    3: [255, 165, 0],    # Background - оранжевый
}
```

**Важно**: Палитра использует ключи 0, 1, 2, 3 (0-indexed + background на 0)

#### Применение цветов к point cloud:

```python
# Line ~200-220
def colorize_by_category(points, categories):
    """
    points: (N, 3) numpy array
    categories: (N,) numpy array of category IDs
    Returns: (N, 3) numpy array of RGB colors
    """
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    for cat_id, color in CATEGORY_COLORS.items():
        mask = categories == cat_id
        colors[mask] = color
    return colors
```

**Логика**: 
- Если `categories[i] = 1`, то цвет будет `[255, 0, 0]` (красный)
- Если `categories[i] = 2`, то цвет будет `[0, 255, 0]` (зелёный)
- Если `categories[i] = 3`, то цвет будет `[255, 165, 0]` (оранжевый)

---

## Анализ проблемы

### Гипотеза 1: Неправильная конверсия в my_train_odin.py

**Текущий код** (строка ~811):
```python
point_pred_cat[m] = int(pred_classes[inst_idx]) + 1
```

**Проблема**: 
- `pred_classes` содержит 0-indexed классы (0, 1, 2)
- Добавляем +1 → получаем (1, 2, 3)
- Но если модель выдаёт уже 1-indexed классы, то получится (2, 3, 4)!

**Проверка**: Нужно посмотреть что именно содержится в `pred_classes`:
- Если там 0, 1, 2 → нужен `+1`
- Если там 1, 2, 3 → НЕ нужен `+1`

### Гипотеза 2: Модель выдаёт 1-indexed классы

**Где формируются pred_classes**:

#### В evaluator (_parse_pred, строка ~732-740):
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

`pred['labels']` берётся напрямую из `_out['instances_3d']['labels']`.

#### В odin_model.py (inference_3d):

**Для eval_normal** (строка ~1060-1080):
```python
processed_results = self.eval_normal(
    mask_cls_results, mask_pred_results, batched_inputs,
    scannet_gt_target_dicts, scannet_p2v, num_classes,
    scannet_idxs, scannet_segments_batched
)
```

**В eval_normal** (odin_model.py, строка ~900-950):
```python
def eval_normal(self, mask_cls, mask_pred, batched_inputs, ...):
    # mask_cls: (B, Q, num_classes+1)  ← +1 для no-object класса
    # mask_pred: (B, Q, num_points)
    
    # Argmax по классам
    scores, labels = F.softmax(mask_cls, dim=-1).max(-1)  # (B, Q)
    
    # labels содержит индексы от 0 до num_classes
    # где num_classes - это no-object класс
    
    # Фильтрация по score
    keep = labels != num_classes  # Убираем no-object
    
    result = {
        "labels": labels[keep],  # ← ЭТО И ЕСТЬ pred_classes
        "scores": scores[keep],
        "masks_3d": mask_pred[keep],
    }
```

**Важно**: 
- `labels` получается через `argmax` по `num_classes+1` классам
- Индексы: 0, 1, 2, ..., num_classes
- `num_classes` - это no-object класс
- Реальные классы: 0, 1, 2 (для 3 категорий)

**Вывод**: `pred_classes` содержит **0-indexed** классы (0, 1, 2), НЕ 1-indexed!

### Гипотеза 3: Ошибка в предыдущем исправлении

**Commit 819a05b** исправил GT категории, но возможно сломал Pred:

```python
# Было (неправильно):
point_pred_cat[m] = int(pred_classes[inst_idx]) - 1

# Стало (правильно?):
point_pred_cat[m] = int(pred_classes[inst_idx]) + 1
```

Но если `pred_classes` уже 0-indexed, то `+1` даёт правильный результат (1, 2, 3).

**Проблема**: Возможно, в коде есть ещё одно место где делается `+1` или `-1`?

---

## Места для проверки

### 1. my_train_odin.py - создание point_pred_cat

**Строка ~811**:
```python
point_pred_cat[m] = int(pred_classes[inst_idx]) + 1
```

**Проверить**: 
- Вывести `pred_classes` в лог
- Убедиться что там 0, 1, 2 (не 1, 2, 3)

### 2. generate_sample_viewer.py - палитра категорий

**Строка ~50-60**:
```python
CATEGORY_COLORS = {
    0: [128, 128, 128],  # Background/Unknown
    1: [255, 0, 0],      # Strawberry
    2: [0, 255, 0],      # Peduncle
    3: [255, 165, 0],    # Background
}
```

**Проверить**: Правильно ли маппинг?

### 3. odin_model.py - формирование labels

**Строка ~900-950** (eval_normal):
```python
scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
keep = labels != num_classes
result["labels"] = labels[keep]
```

**Проверить**: 
- Что содержится в `labels` после argmax?
- Правильно ли фильтруется no-object класс?

---

## Debug план

### Шаг 1: Добавить логирование pred_classes

В `my_train_odin.py`, строка ~810:
```python
if pred_masks_3d is not None:
    print(f"[DEBUG VIS] pred_classes for {sample_id}: {pred_classes}")
    print(f"[DEBUG VIS] unique pred_classes: {np.unique(pred_classes)}")
    
    for inst_idx in range(pred_masks_3d.shape[0]):
        m = pred_masks_3d[inst_idx] > 0.5
        if np.any(m):
            print(f"[DEBUG VIS] inst {inst_idx}: pred_class={pred_classes[inst_idx]}, "
                  f"assigned cat={int(pred_classes[inst_idx]) + 1}")
            point_pred_cat[m] = int(pred_classes[inst_idx]) + 1
```

### Шаг 2: Проверить GT категории

В `my_train_odin.py`, строка ~860:
```python
if gt_masks_3d is not None:
    print(f"[DEBUG VIS] gt_class_labels for {sample_id}: {gt_class_labels}")
    print(f"[DEBUG VIS] unique gt_class_labels: {np.unique(gt_class_labels)}")
    
    for inst_idx in range(gt_masks_3d.shape[0]):
        m = gt_masks_3d[inst_idx] > 0.5
        if np.any(m):
            print(f"[DEBUG VIS] GT inst {inst_idx}: class={gt_class_labels[inst_idx]}")
            point_gt_cat[m] = int(gt_class_labels[inst_idx])
```

### Шаг 3: Проверить что попадает в HTML

В `my_train_odin.py`, перед вызовом `build_html`:
```python
print(f"[DEBUG VIS] point_gt_cat unique: {np.unique(point_gt_cat)}")
print(f"[DEBUG VIS] point_pred_cat unique: {np.unique(point_pred_cat)}")
```

### Шаг 4: Проверить палитру в generate_sample_viewer.py

Добавить в `colorize_by_category`:
```python
def colorize_by_category(points, categories):
    print(f"[DEBUG PALETTE] unique categories: {np.unique(categories)}")
    print(f"[DEBUG PALETTE] CATEGORY_COLORS keys: {list(CATEGORY_COLORS.keys())}")
    
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    for cat_id, color in CATEGORY_COLORS.items():
        mask = categories == cat_id
        count = np.sum(mask)
        if count > 0:
            print(f"[DEBUG PALETTE] cat_id={cat_id} → color={color}, count={count}")
        colors[mask] = color
    return colors
```

---

## Ожидаемые результаты

### Если всё правильно:

**pred_classes** (из модели):
```
[0, 1, 2]  # 0-indexed
```

**point_pred_cat** (после +1):
```
[1, 2, 3]  # 1-indexed для палитры
```

**CATEGORY_COLORS** применяет:
- `cat_id=1` → `[255, 0, 0]` (красный) → strawberry ✅
- `cat_id=2` → `[0, 255, 0]` (зелёный) → peduncle ✅
- `cat_id=3` → `[255, 165, 0]` (оранжевый) → background ✅

### Если проблема:

**Возможный сценарий 1**: `pred_classes` уже 1-indexed
```
pred_classes = [1, 2, 3]
point_pred_cat = [2, 3, 4]  # После +1
# Результат: strawberry → зелёный (cat_id=2), peduncle → оранжевый (cat_id=3)
```

**Решение**: Убрать `+1` в строке 811.

**Возможный сценарий 2**: Палитра неправильная
```
CATEGORY_COLORS = {
    0: [128, 128, 128],
    1: [0, 255, 0],      # ← Должен быть красный!
    2: [255, 165, 0],    # ← Должен быть зелёный!
    3: [255, 0, 0],      # ← Должен быть оранжевый!
}
```

**Решение**: Исправить палитру.

---

## Вопрос для следующего агента

**Проблема**: В 3D визуализациях предсказанные категории (Pred Cat) отображаются с неправильными цветами - сдвиг на 1 категорию. GT категории отображаются правильно.

**Задача**:
1. Добавить debug логирование по плану выше (Шаги 1-4)
2. Запустить evaluation и проверить логи
3. Определить где именно происходит ошибка:
   - Неправильная конверсия в `my_train_odin.py` (строка ~811)?
   - Неправильная палитра в `generate_sample_viewer.py`?
   - Модель выдаёт неправильные индексы?
4. Исправить проблему
5. Убрать debug логи
6. Запушить изменения в git

**Файлы для проверки**:
- `my_train_odin.py` - строки ~800-920 (создание point clouds)
- `generate_sample_viewer.py` - строки ~50-60 (палитра), ~200-220 (colorize)
- `odin/odin_model.py` - строки ~900-950 (eval_normal, формирование labels)

**Ожидаемый результат**:
- Pred Cat: strawberry → красный, peduncle → зелёный, background → оранжевый
- GT Cat: без изменений (уже правильно)
