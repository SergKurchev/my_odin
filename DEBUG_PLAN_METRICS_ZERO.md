# План отладки: Почему метрики PQ/mAP стали 0 после добавления сохранения raw output

## Факты

### До изменений (работало):
```csv
iteration,PQ,mAP@50,mean_uncertainty
287,0.0376,0.000152,0.0  ← PQ и mAP работали
431,1.796,0.978,0.0      ← метрики росли
```

### После изменений (сломалось):
```csv
iteration,PQ,mAP@50,mean_uncertainty
143,0.0,0.0,0.0249  ← PQ=0, но uncertainty работает
287,0.0,0.0,0.0775  ← PQ=0, но uncertainty работает
```

### Изменения которые могли сломать:
1. **Commit 8b8c16c** - "Save raw model output to pickle for visualization samples"
2. **Последующие debug коммиты** - добавление логирования
3. **Commit 85862ec** - "Fix critical bugs" (транспонирование и индексация) - **ОТКАТИТЬ!**

---

## План действий

### Шаг 1: Откат неправильных исправлений

**Действие**: Откатить commit 85862ec (транспонирование и изменение индексации)

**Команда**:
```bash
git revert 85862ec
# или
git reset --hard 8b8c16c  # если нет других важных коммитов после
```

**Проверка**: 
- Код вернулся к состоянию после добавления сохранения raw output
- Транспонирование удалено
- Индексация словаря восстановлена

---

### Шаг 2: Сравнение кода до/после

**Действие**: Сравнить метод `process()` в двух версиях:

**Версия ДО** (последний рабочий коммит перед 8b8c16c):
```bash
git show HEAD~3:my_train_odin.py | grep -A 100 "def process"
```

**Версия ПОСЛЕ** (текущая):
```bash
git show HEAD:my_train_odin.py | grep -A 100 "def process"
```

**Что искать**:
1. Изменился ли порядок операций?
2. Добавились ли `return` или `continue` которые пропускают код?
3. Изменились ли условия `if is_target:`?
4. Правильно ли инкрементируется `self._current_idx`?

---

### Шаг 3: Проверка критических точек

#### 3.1 Проверить инкремент `_current_idx`

**Проблема**: Если `_current_idx` не инкрементируется, то `processed_preds` и `processed_gts` перезаписываются.

**Где проверить** (строка ~760 в старой версии):
```python
self._current_idx += 1  # ← ЭТО ДОЛЖНО БЫТЬ В КОНЦЕ process()!
```

**Диагностика**:
```python
# В конце process(), перед инкрементом:
logger.info(f"[DEBUG] Before increment: _current_idx={self._current_idx}")
self._current_idx += 1
logger.info(f"[DEBUG] After increment: _current_idx={self._current_idx}")
```

#### 3.2 Проверить что данные сохраняются в словари

**Где проверить** (строки ~654-657 в старой версии):
```python
self.processed_gts[idx] = self._parse_gt(_in)
self.processed_preds[idx] = self._parse_pred(_out)
```

**Диагностика**:
```python
gt_data = self._parse_gt(_in)
pred_data = self._parse_pred(_out)

logger.info(f"[DEBUG] Storing GT at idx={idx}, keys={gt_data.keys()}")
logger.info(f"[DEBUG] Storing Pred at idx={idx}, keys={pred_data.keys()}")

self.processed_gts[idx] = gt_data
self.processed_preds[idx] = pred_data

logger.info(f"[DEBUG] Total stored: gts={len(self.processed_gts)}, preds={len(self.processed_preds)}")
```

#### 3.3 Проверить что `evaluate()` получает данные

**Где проверить** (строка ~1040 в старой версии):
```python
def evaluate(self):
    logger.info(f"[DEBUG EVAL] Starting evaluation")
    logger.info(f"[DEBUG EVAL] processed_preds keys: {list(self.processed_preds.keys())}")
    logger.info(f"[DEBUG EVAL] processed_gts keys: {list(self.processed_gts.keys())}")
    logger.info(f"[DEBUG EVAL] Total samples: {len(self.processed_preds)}")
```

---

### Шаг 4: Гипотезы и проверки

#### Гипотеза 1: Блок сохранения raw output выполняется для ВСЕХ сэмплов, не только target

**Проблема**: Если `is_target` всегда True, то код сохранения выполняется для всех → замедление → таймаут?

**Проверка**:
```python
is_target = any(ts in sample_id for ts in target_samples)
logger.info(f"[DEBUG] sample_id='{sample_id}', is_target={is_target}")
```

**Ожидаемое**: `is_target=True` только для 00000, 00003, 00005

#### Гипотеза 2: Exception в блоке сохранения прерывает process()

**Проблема**: Если pickle.dump падает с exception, то код после него не выполняется.

**Проверка**: Посмотреть есть ли `try-except` вокруг сохранения:
```python
if is_target:
    try:
        # ... сохранение ...
    except Exception as e:
        logger.error(f"[ERROR] Failed to save: {e}")
        # НО ПРОДОЛЖАЕМ ДАЛЬШЕ!
```

**Важно**: После `except` должен быть код парсинга GT/Pred!

#### Гипотеза 3: Добавление кода сдвинуло `_current_idx += 1`

**Проблема**: Если инкремент `_current_idx` оказался внутри `if is_target:`, то он выполняется только для target сэмплов.

**Проверка**: Убедиться что инкремент в конце, вне всех `if`:
```python
def process(self, inputs, outputs):
    for _in, _out in zip(inputs, outputs):
        idx = self._current_idx
        
        # ... весь код ...
        
        # ЭТО ДОЛЖНО БЫТЬ В САМОМ КОНЦЕ, ВНЕ ВСЕХ IF!
        self._current_idx += 1
```

#### Гипотеза 4: Изменился порядок операций

**Проблема**: Если сохранение raw output происходит ДО парсинга, и там есть `del _out`, то парсинг получает пустой объект.

**Проверка**: Убедиться что порядок правильный:
```python
# 1. Сохранение raw output (если is_target)
if is_target:
    # ... сохранение _out ...

# 2. Парсинг (для ВСЕХ сэмплов)
self.processed_gts[idx] = self._parse_gt(_in)
self.processed_preds[idx] = self._parse_pred(_out)  # ← _out должен быть целым!

# 3. Uncertainty (для ВСЕХ сэмплов)
# ...

# 4. Инкремент (для ВСЕХ сэмплов)
self._current_idx += 1
```

---

### Шаг 5: Минимальный воспроизводимый тест

**Действие**: Создать минимальный тест который проверяет что метрики считаются:

```python
# В конце evaluate():
logger.info(f"[TEST] Final metrics:")
logger.info(f"[TEST] PQ = {metrics.get('PQ', 0)}")
logger.info(f"[TEST] mAP = {metrics.get('mAP', 0)}")
logger.info(f"[TEST] SQ = {metrics.get('SQ', 0)}")
logger.info(f"[TEST] RQ = {metrics.get('RQ', 0)}")

# Если все 0, то проблема в evaluate()
# Если не 0, то проблема в передаче метрик наверх
```

---

## Порядок выполнения

1. **Откатить commit 85862ec** (транспонирование и индексация)
2. **Добавить debug логи** из Шага 3 (инкремент, сохранение, evaluate)
3. **Запустить eval** и собрать логи
4. **Проанализировать логи** по гипотезам из Шага 4
5. **Найти где именно ломается** (process или evaluate)
6. **Исправить проблему**
7. **Убрать debug логи**
8. **Запушить**

---

## Ожидаемые результаты логов

### Если всё работает правильно:

```
[DEBUG] sample_id='00000_part0', is_target=False
[DEBUG] Storing GT at idx=0
[DEBUG] Storing Pred at idx=0
[DEBUG] Before increment: _current_idx=0
[DEBUG] After increment: _current_idx=1

[DEBUG] sample_id='00000_part1', is_target=True  ← только для target
[SAVE RAW OUTPUT] Saved raw model output to ...
[DEBUG] Storing GT at idx=1
[DEBUG] Storing Pred at idx=1
[DEBUG] Before increment: _current_idx=1
[DEBUG] After increment: _current_idx=2

...

[DEBUG EVAL] Starting evaluation
[DEBUG EVAL] Total samples: 144
[DEBUG EVAL] processed_preds keys: [0, 1, 2, ..., 143]
[DEBUG EVAL] processed_gts keys: [0, 1, 2, ..., 143]
[TEST] PQ = 0.0376  ← НЕ НОЛЬ!
[TEST] mAP = 0.000152
```

### Если сломано:

**Вариант 1**: Данные не сохраняются
```
[DEBUG] Total stored: gts=0, preds=0  ← ПУСТО!
[DEBUG EVAL] Total samples: 0
[TEST] PQ = 0.0
```

**Вариант 2**: Индексы перезаписываются
```
[DEBUG] Before increment: _current_idx=0
[DEBUG] After increment: _current_idx=0  ← НЕ ИНКРЕМЕНТИРУЕТСЯ!
[DEBUG] Total stored: gts=1, preds=1  ← Только 1 сэмпл вместо 144!
```

**Вариант 3**: Exception прерывает process
```
[ERROR] Failed to save: ...
# Дальше логов нет → код после exception не выполняется
```

---

## Критические проверки

- [ ] `_current_idx` инкрементируется для ВСЕХ сэмплов (не только target)
- [ ] `processed_gts` и `processed_preds` заполняются для ВСЕХ сэмплов
- [ ] Блок сохранения raw output в `try-except` и не прерывает process
- [ ] `_out` не модифицируется/удаляется перед `_parse_pred()`
- [ ] `evaluate()` получает 144 сэмпла (не 0, не 1)
- [ ] Транспонирование и индексация откачены (код как был до 85862ec)
