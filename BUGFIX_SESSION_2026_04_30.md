# Сессия отладки — 30.04.2026

## Контекст

Во время тренировки на Kaggle при первом вызове `evaluate()` возникала ошибка:

```
File "my_train_odin.py", line 901, in evaluate
    point_pred_cat[m] = int(pred_classes[inst_idx])
IndexError: index 203267 is out of bounds for axis 0 with size 20
```

---

## Проделанная работа

### Bug 1 — IndexError в `evaluate()` ✅ ИСПРАВЛЕН (коммит `93d2350`)

**Причина:**  
ODIN-модель возвращает `pred_masks` в форме `[NumPoints, NumInstances]`  
(см. `odin/odin_model.py`, строка 1321: `masks.flatten(1).permute(1, 0)`).

Но код в `evaluate()` читал:
```python
num_pred_instances = pred_masks.shape[0]   # ← это NumPoints (203267!), не инстансы
m = pred_masks[inst_idx]                   # ← строка по точке, не инстанс
```

Цикл шёл до 203267, а `pred_classes[203267]` при размере 20 → выход за границы.

**Фикс:**
```python
# Было:
num_pred_instances = pred_masks.shape[0]
num_pts_total      = pred_masks.shape[1]
m = pred_masks[inst_idx] > 0

# Стало:
num_pts_total      = pred_masks.shape[0]   # axis-0 = точки
num_pred_instances = pred_masks.shape[1]   # axis-1 = инстансы
m = pred_masks[:, inst_idx] > 0            # столбец = маска инстанса
```

**Статус:** активен в репозитории.

---

### Bug 2 — global_indices в визуализации ⚠️ ИСПРАВЛЕН, ЗАТЕМ ОТМЕНЁН (коммит `94ad671` → revert `0577092`)

**Проблема:**  
`pred_masks` содержит только **valid-точки** (model делает `pred_masks[:, valids]` в `prepare_3d`).  
Старый код пытался индексировать по **пиксельным** координатам:

```python
global_indices = camera_idx * (H_padded * W_padded) + rows * W_padded + cols
valid_global = global_indices < len(point_pred_inst)
inst_pred[valid_global] = point_pred_inst[global_indices[valid_global]]
```

Из-за несовпадения пространств (valid-точки ≠ все пиксели) предсказания в HTML-визуализации были бы некорректными. Защита `valid_global` предотвращала крэш, но большинство индексов отфильтровывалось → pred-метки были бы все `-1`.

**Почему отменён:**  
Пользователь отметил, что визуализации **работали раньше**. Исследование истории git показало:

- В коммите до `51ef57b` в `_parse_pred` было явное транспонирование:
  ```python
  res['pred_masks'] = res['pred_masks'].T  # [NumInstances, NumPoints]
  ```
- Это транспонирование было **отменено** в `51ef57b` (Revert).
- После revert форма стала `[NumPoints, NumInstances]`, но visualization loop не обновили.
- Это создало latent bug, который не проявлялся пока evaluate() крашился раньше (на Bug 1).

**Решено:** отложить разбор до следующей сессии.  
**Статус:** код вернулся к `global_indices`-логике (как до обеих правок).

---

## Текущее состояние репозитория

| Коммит | Статус | Описание |
|--------|--------|----------|
| `0577092` | HEAD | Revert global_indices fix |
| `94ad671` | отменён | global_indices fix (был спорным) |
| `93d2350` | ✅ активен | **IndexError fix** — главная причина крэша |
| `3462596` | ✅ активен | Твой коммит: save pkl по порядку, не по hardcoded ID |

---

## Открытые вопросы (к следующей сессии)

### 1. Визуализация pred-масок в HTML — правильная ли она?

**Вопрос:** Работает ли `global_indices`-подход на самом деле?  

**Гипотеза A (визуализация работает):**  
Возможно, `valids` в `prepare_3d` включает все пиксели (depth > 0 для всех пикселей кадра), и тогда `num_pts_total ≈ num_frames * H_padded * W_padded`. Тогда `global_indices` случайно совпадает.

**Гипотеза B (визуализация пустая/некорректная):**  
`valids` отфильтровывает многие пиксели → пространства несовместимы → pred-метки в HTML всегда -1 (фоновый цвет).

**Что нужно проверить:**  
Запустить с дебаг-принтом в `prepare_3d`, чтобы узнать, сколько точек отфильтровывает `valids`. Можно добавить:
```python
print(f"[DEBUG prepare_3d] valids sum: {valids.sum()}, total: {valids.numel()}")
```

---

### 2. История транспонирования — почему делали `.T`?

В `_parse_pred` раньше было:
```python
res['pred_masks'] = res['pred_masks'].T  # [NumInstances, NumPoints]
```
Затем это отменили (коммит `51ef57b`). 

**Вопрос:** Зачем был revert? Был ли `.T` нужен для метрик (scannet evaluator) или только для визуализации?

Нужно проверить, что именно ожидает `scannet_evaluator.assign_instances_for_scan(pred, gt)`:  
- Если ожидает `pred_masks` в форме `[NumInstances, NumPoints]` → нужно вернуть `.T` обратно  
- Если ожидает `[NumPoints, NumInstances]` → revert был правильным

---

### 3. Правильность метрик (PQ, mAP)

Связано с вопросом 2. Если форма `pred_masks` влияет на `assign_instances_for_scan`, то текущие метрики могут быть неверными. Нулевые метрики в прошлых запусках могли быть связаны именно с этим.

---

## Рекомендуемый план на следующую сессию

1. Добавить дебаг-принт в `prepare_3d` → узнать реальный `valids`-count
2. Проверить `evaluate_semantic_instance.py` → узнать ожидаемую форму `pred_masks`
3. Принять решение: нужен ли `.T` в `_parse_pred`, или нет
4. Если визуализация действительно была правильной до всех манипуляций с транспонированием — понять механизм
5. При необходимости: вернуть правильный `global_indices`-fix или оставить как есть
