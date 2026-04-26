# Архитектура модели ODIN

## Общий обзор

ODIN (Omni-Dimensional INstance segmentation) — это унифицированная модель для 2D и 3D сегментации, которая обрабатывает последовательности posed RGB-D изображений (изображения с известными позами камер и картами глубины). Модель способна выполнять как семантическую, так и инстанс-сегментацию на 2D изображениях и 3D сценах.

### Ключевые особенности архитектуры:
- **Унифицированный подход**: одна модель для 2D и 3D задач
- **Multi-view fusion**: агрегация информации из множественных видов сцены
- **Query-based segmentation**: использование learnable queries для предсказания масок объектов
- **Трансформерная архитектура**: основана на Mask2Former с расширениями для 3D

---

## Архитектура модели: Основные компоненты

```
Input: RGB-D Sequence (B, V, 3+1, H, W)
   ↓
[1] Backbone (ResNet50 / Swin Transformer)
   ↓
[2] Cross-View Attention (опционально)
   ↓
[3] Pixel Decoder (MSDeformAttn)
   ↓
[4] Transformer Decoder
   ↓
Output: Masks + Class Logits
```

---

## 1. Входные данные

### Формат входа
- **RGB изображения**: `(B, V, 3, H, W)` — батч из B сцен, каждая с V видами
- **Depth карты**: `(B, V, 1, H, W)` — карты глубины для каждого вида
- **Camera poses**: `(B, V, 4, 4)` — матрицы трансформации камер (world-to-camera)
- **Camera intrinsics**: `(B, V, 3, 3)` — внутренние параметры камер

### Backprojection (3D координаты)
Модуль `backproject_depth` (`odin/modeling/backproject/backproject.py`) преобразует 2D пиксели в 3D координаты:

```python
xyz = unproject(intrinsics, poses, depths)  # (B, V, H, W, 3)
```

**Процесс:**
1. Для каждого пикселя (x, y) с глубиной z:
   ```
   cam_x = (x - px) * z / fx
   cam_y = -(y - py) * z / fy  # Y-flip для Unity координат
   cam_z = z
   ```
2. Трансформация в мировые координаты: `world_coords = pose @ cam_coords`

Результат: `xyz` — 3D координаты каждого пикселя в мировой системе координат.

---

## 2. Backbone (Feature Extraction)

### Архитектуры
- **ResNet50**: стандартный backbone из Detectron2
- **Swin Transformer**: иерархический vision transformer

### Выходы backbone
Многомасштабные признаки на разных уровнях:
- `res2`: stride 4, размер H/4 × W/4
- `res3`: stride 8, размер H/8 × W/8
- `res4`: stride 16, размер H/16 × W/16
- `res5`: stride 32, размер H/32 × W/32

**Формат**: `features[res_i] = (B*V, C_i, H_i, W_i)`

### Cross-View Attention в Backbone (опционально)

Если `MODEL.CROSS_VIEW_BACKBONE=True`, backbone содержит слои cross-view attention, которые позволяют признакам из разных видов взаимодействовать уже на ранних стадиях.

**Реализация**: PANet-style connections с cross-attention между видами.

---

## 3. Cross-View Attention Module

**Файл**: `odin/modeling/meta_arch/cross_view_attention.py`  
**Класс**: `CrossViewPAnet`

Это ключевой модуль для агрегации информации из множественных видов сцены.

### Архитектура

```
Input: feature_list [(B*V, C, H, W)], xyz_list [(B*V, H, W, 3)]
   ↓
Reshape: (B, V*H*W, C) и (B, V*H*W, 3)
   ↓
[Опционально] Voxelization: scatter_mean по voxel grid
   ↓
KNN Search: найти K ближайших соседей для каждой точки
   ↓
For each layer (num_layers раз):
   ├─ Positional Encoding (3D coordinates → features)
   ├─ Cross-Attention (query: текущая точка, keys: K соседей)
   ├─ FFN (Feed-Forward Network)
   └─ LayerNorm
   ↓
Output: aggregated features [(B*V, C, H, W)]
```

### Детали реализации

**1. Voxelization (опционально)**
Если `INPUT.VOXELIZE=True`, точки группируются в воксели для уменьшения вычислительной сложности:
```python
feature = scatter_mean(feature[b], p2v[b], dim=0)  # усреднение по вокселям
```

**2. KNN Search**
Используется `pointops.queryandgroup` для поиска K ближайших соседей:
```python
knn_points_feats, idx = pointops.queryandgroup(
    nsample, xyz, xyz, feature, None, 
    batch_offset, batch_offset, 
    use_xyz=True, return_indx=True
)
```
- `nsample`: количество соседей (обычно 16)
- Возвращает признаки и координаты K соседей для каждой точки

**3. Positional Encoding**
3D координаты кодируются в признаковое пространство:
- **Learned Embedding**: `PositionEmbeddingLearned` — простое линейное преобразование
- **MLP Encoding**: `PositionEmbeddingLearnedMLP` — многослойный перцептрон

```python
query_pe = self.encode_pe(torch.zeros_like(xyz[:, None]))  # PE для query
knn_pe = self.encode_pe(knn_points)  # PE для K соседей
```

**4. Cross-Attention Layers**
Для каждого слоя:
```python
output = self.cross_view_attention_layers[i](
    tgt=output.permute(1, 0, 2),      # query: текущие признаки
    memory=key,                        # keys: признаки K соседей
    query_pos=query_pe,                # позиционное кодирование query
    pos=knn_pe,                        # позиционное кодирование keys
)
output = self.ffn_layers[i](output)
output = self.layer_norms[i](output)
```

**Механизм**: каждая точка "смотрит" на свои K ближайших соседей в 3D пространстве и агрегирует их информацию через attention.

---

## 4. Pixel Decoder

**Файл**: `odin/modeling/pixel_decoder/msdeformattn.py`  
**Класс**: `MSDeformAttnPixelDecoder`

Pixel Decoder — это decoder-часть архитектуры, которая:
1. Принимает многомасштабные признаки от backbone
2. Применяет Multi-Scale Deformable Attention
3. Выполняет upsampling для получения высокоразрешенных признаков

### Архитектура

```
Input: features {res2, res3, res4, res5}
   ↓
[MSDeformAttn Encoder]
   ├─ Multi-scale deformable attention
   ├─ Обработка признаков на разных масштабах
   └─ Encoder layers (обычно 6 слоев)
   ↓
[Decoder with Upsampling]
   ├─ Lateral connections (как в FPN)
   ├─ Upsampling: res5 → res4 → res3 → res2
   └─ [Опционально] Cross-view attention на каждом уровне
   ↓
Output: 
   - mask_features: (B*V, C, H/4, W/4) — признаки для scoring масок
   - multi_scale_features: [res5, res4, res3] — признаки для decoder
```

### Multi-Scale Deformable Attention

**Идея**: вместо фиксированной сетки attention, модель обучается смещениям (offsets) для выборки наиболее релевантных позиций на разных масштабах.

**Компоненты**:
- `MSDeformAttn`: модуль deformable attention
- Обрабатывает признаки на 3-4 масштабах одновременно
- Каждая точка может "смотреть" на разные масштабы с learnable offsets

### PANet-style Decoder (опционально)

Если `MODEL.PIXEL_DECODER_PANET=True`, decoder использует PANet-style connections:
- Bottom-up pathway: res2 → res5
- Top-down pathway: res5 → res2
- Lateral connections между уровнями
- Cross-view attention на каждом уровне (если `MODEL.CROSS_VIEW_CONTEXTUALIZE=True`)

---

## 5. Transformer Decoder

**Файл**: `odin/modeling/transformer_decoder/odin_transformer_decoder.py`  
**Класс**: `ODINMultiScaleMaskedTransformerDecoder`

Transformer Decoder — это query-based модуль, который генерирует предсказания масок и классов.

### Архитектура

```
Input: 
   - mask_features: (B*V, C, H/4, W/4)
   - multi_scale_features: [(B*V, C, H_i, W_i)] для i=3,4,5
   - [Опционально] mask_features_xyz: (B, N, 3) — 3D координаты
   ↓
Initialize Queries:
   - query_feat: (num_queries, C) — learnable content queries
   - query_embed: (num_queries, C) — learnable positional queries
   ↓
For each decoder layer (обычно 6-9 слоев):
   ├─ Self-Attention: queries attend to each other
   ├─ Cross-Attention: queries attend to multi_scale_features
   ├─ FFN: feed-forward network
   └─ Промежуточные предсказания (для auxiliary losses)
   ↓
Output для каждого query:
   - class_logits: (B, num_queries, num_classes+1)
   - mask_embeddings: (B, num_queries, mask_dim)
   ↓
Mask Prediction:
   mask_pred = mask_embeddings @ mask_features
   ↓
Final Output:
   - pred_logits: (B, num_queries, num_classes+1)
   - pred_masks: (B, num_queries, V, H/4, W/4) для 2D
   - pred_masks: (B, num_queries, N) для 3D
```

### Queries

**Learnable Queries**: модель обучает фиксированное количество queries (обычно 100-200), каждый из которых "специализируется" на обнаружении определенных типов объектов.

```python
self.query_feat = nn.Embedding(num_queries, hidden_dim)   # content
self.query_embed = nn.Embedding(num_queries, hidden_dim)  # position
```

### Decoder Layers

Каждый слой decoder состоит из:

**1. Self-Attention**
```python
output = self.transformer_self_attention_layers[i](
    output, 
    tgt_mask=None,
    query_pos=query_embed
)
```
Queries взаимодействуют друг с другом, обмениваясь информацией.

**2. Cross-Attention**
```python
output = self.transformer_cross_attention_layers[i](
    tgt=output,
    memory=src,  # multi_scale_features
    query_pos=query_embed,
    pos=pos_embed
)
```
Queries "смотрят" на признаки изображения на разных масштабах.

**3. Feed-Forward Network**
```python
output = self.transformer_ffn_layers[i](output)
```

### Mask Prediction

**2D Masks**:
```python
mask_embed = self.mask_embed(output)  # (B, Q, mask_dim)
outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
```

**3D Masks** (если `decoder_3d=True`):
```python
# Интерполяция признаков на 3D точки
mask_features_3d = interpolate_feats_3d(
    mask_features, 
    mask_features_xyz,  # 3D координаты
    scannet_pc          # целевые 3D точки
)
# Предсказание масок на 3D точках
outputs_mask_3d = torch.einsum("bqc,bnc->bqn", mask_embed, mask_features_3d)
```

### Class Prediction

```python
outputs_class = self.class_embed(output)  # (B, Q, num_classes+1)
```

Последний класс (+1) — это "no object" класс для queries, которые не обнаружили объект.

---

## 6. Loss Functions и Training

**Файл**: `odin/modeling/criterion.py`  
**Класс**: `ODINSetCriterion`

### Hungarian Matching

Перед вычислением loss, модель выполняет bipartite matching между предсказаниями и ground truth:

**Matcher** (`odin/modeling/matcher.py`):
```python
# Вычисление cost matrix
cost_class = classification_cost(pred_logits, gt_classes)
cost_mask = mask_cost(pred_masks, gt_masks)
cost_dice = dice_cost(pred_masks, gt_masks)

cost_matrix = cost_class + cost_mask + cost_dice

# Hungarian algorithm
indices = linear_sum_assignment(cost_matrix)
```

### Loss Components

**1. Classification Loss**
```python
loss_ce = F.cross_entropy(pred_logits, target_classes)
```

**2. Mask Loss (Binary Cross-Entropy)**
```python
loss_mask = F.binary_cross_entropy_with_logits(pred_masks, target_masks)
```

**3. Dice Loss**
```python
numerator = 2 * (pred_masks * target_masks).sum()
denominator = pred_masks.sum() + target_masks.sum()
loss_dice = 1 - (numerator + 1) / (denominator + 1)
```

### Point Sampling для эффективности

Вместо вычисления loss на всех пикселях, модель сэмплирует точки:
- **Importance sampling**: больше точек на границах объектов (высокая uncertainty)
- **Random sampling**: случайные точки для регуляризации

```python
# Выбор uncertain точек
uncertainty = calculate_uncertainty(pred_masks)
point_coords = get_uncertain_point_coords(
    coords, uncertainty, 
    important_sample_ratio=0.75, 
    num_points=12544  # для 2D
)
```

### Auxiliary Losses

Модель вычисляет loss на каждом промежуточном слое decoder:
```python
for aux_outputs in predictions['aux_outputs']:
    losses += compute_loss(aux_outputs, targets)
```

Это помогает обучению глубоких слоев.

---

## 7. Inference Pipeline

### 2D Inference

```python
# Forward pass
predictions = model(images, depths, poses, intrinsics)

# Получение предсказаний
pred_logits = predictions['pred_logits']  # (B, Q, num_classes+1)
pred_masks = predictions['pred_masks']    # (B, Q, V, H, W)

# Фильтрация по confidence
scores = pred_logits.softmax(-1)[:, :, :-1]  # исключаем "no object"
max_scores, labels = scores.max(-1)

# Threshold
keep = max_scores > 0.5
final_masks = pred_masks[keep]
final_labels = labels[keep]
```

### 3D Inference

```python
# Forward pass с decoder_3d=True
predictions = model(
    images, depths, poses, intrinsics,
    scannet_pc=point_cloud,  # целевые 3D точки
    decoder_3d=True
)

# Получение 3D масок
pred_masks_3d = predictions['pred_scannet_masks']  # (B, Q, N)

# Post-processing
# 1. Non-maximum suppression
# 2. Фильтрация по confidence
# 3. Агрегация по вокселям (если использовалась voxelization)
```

---

## 8. Multi-Task Training (2D + 3D)

ODIN поддерживает совместное обучение на 2D и 3D данных:

**Конфигурация**:
```yaml
MULTI_TASK_TRAINING: True
DATASETS.TRAIN_3D: ['scannet_train']
DATASETS.TRAIN_2D: ['coco_train']
```

**Процесс**:
1. Чередование батчей из 2D и 3D датасетов
2. Для 2D: `decoder_3d=False`, loss только на 2D масках
3. Для 3D: `decoder_3d=True`, loss на 2D и 3D масках
4. Общий backbone и pixel decoder
5. Раздельные или общие decoder heads (в зависимости от конфигурации)

---

## 9. Ключевые технические детали

### Voxelization

Для уменьшения количества точек используется voxelization:
```python
# Группировка точек в воксели
p2v = voxelize(xyz, voxel_size=[0.02, 0.04, 0.08, 0.16])

# Усреднение признаков по вокселям
features_voxelized = scatter_mean(features, p2v, dim=0)
```

**Multi-scale voxelization**: используются разные размеры вокселей для разных уровней признаков.

### Data Augmentation

**2D Augmentations**:
- Random scaling (0.1 - 2.0)
- Random cropping
- Color jittering (если `INPUT.STRONG_AUGS=True`)
- Horizontal flipping

**3D Augmentations** (если `INPUT.AUGMENT_3D=True`):
- Random rotation (±π/24 по X, Y; ±π по Z)
- Random scaling (0.9 - 1.1)
- Random translation

```python
xyz, scannet_pc = rotation_augmentation_fast(xyz, scannet_pc)
xyz, scannet_pc = scale_augmentations_fast(xyz, scannet_pc)
```

### Camera Drop

Если `INPUT.CAMERA_DROP=True`, случайно удаляются некоторые виды во время обучения:
- Помогает модели быть устойчивой к неполным данным
- Предотвращает overfitting на конкретные конфигурации камер

### Gradient Clipping

```python
# Full model gradient clipping
torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
```

### Mixed Precision Training

Используется Automatic Mixed Precision (AMP) для ускорения:
```python
with autocast(dtype=torch.float16):
    loss_dict = model(data)
```

---

## 10. Архитектурные варианты

### Backbone Variants

**ResNet50**:
- Стандартный выбор
- Быстрее, меньше памяти
- Хорошо работает на небольших датасетах

**Swin Transformer**:
- Более мощный backbone
- Лучшие результаты на больших датасетах
- Требует больше памяти и вычислений

### Decoder Variants

**Standard Decoder**:
- Простой FPN-style decoder
- Быстрый inference

**PANet Decoder** (`MODEL.PIXEL_DECODER_PANET=True`):
- Дополнительные bottom-up connections
- Лучше для мелких объектов
- Медленнее

**With Cross-View Attention** (`MODEL.CROSS_VIEW_CONTEXTUALIZE=True`):
- Cross-view fusion на каждом уровне decoder
- Лучше использует multi-view информацию
- Значительно медленнее

---

## 11. Вычислительная сложность

### Memory Requirements

**Для обучения** (примерные значения):
- ResNet50, batch_size=6, 25 frames: ~24GB GPU memory
- Swin-B, batch_size=6, 25 frames: ~32GB GPU memory

**Оптимизации**:
- Voxelization: уменьшает количество точек в 5-10 раз
- Gradient checkpointing: уменьшает memory на 30-40%
- Mixed precision: уменьшает memory на 20-30%

### Inference Speed

**2D Inference**:
- ResNet50: ~0.3s per image (single GPU)
- Swin-B: ~0.5s per image

**3D Inference**:
- ResNet50: ~2-5s per scene (зависит от количества frames)
- Swin-B: ~3-8s per scene

**Bottlenecks**:
- Cross-view attention: O(N²) complexity
- KNN search: O(N log N)
- Deformable attention: O(N × K × L) где K=points, L=levels

---

## 12. Сравнение с другими архитектурами

### vs Mask2Former (2D only)
- **ODIN**: расширяет Mask2Former на 3D
- **Добавлено**: cross-view attention, 3D backprojection, 3D decoder
- **Результат**: unified 2D-3D model

### vs Mask3D (3D only)
- **Mask3D**: работает только с 3D point clouds
- **ODIN**: использует RGB-D sequences, может делать 2D и 3D
- **Преимущество ODIN**: использует 2D признаки + 3D геометрию

### vs PointGroup, SoftGroup (3D only)
- **PointGroup/SoftGroup**: специализированы на 3D point clouds
- **ODIN**: более универсальный, но может быть медленнее на чистых 3D задачах

---

## Заключение

ODIN — это сложная архитектура, которая объединяет:
1. **2D vision**: мощные CNN/Transformer backbones
2. **3D geometry**: backprojection и point cloud operations
3. **Multi-view fusion**: cross-view attention для агрегации информации
4. **Query-based segmentation**: transformer decoder для гибких предсказаний

Ключевая идея: использовать 2D признаки (богатые семантикой) + 3D геометрию (точные пространственные отношения) для получения лучших результатов на обеих задачах.
