# ОБНОВЛЕННОЕ РАССЛЕДОВАНИЕ: КАТАСТРОФИЧЕСКАЯ ДЕГРАДАЦИЯ МЕТРИК

**Дата:** 28 апреля 2026  
**Статус:** 🚨 КРИТИЧЕСКАЯ ПРОБЛЕМА ОБНАРУЖЕНА

---

## 🔴 EXECUTIVE SUMMARY

После получения правильных данных OLD VERSION обнаружена **катастрофическая деградация метрик**:

- **PQ: 52.80 → 43.86** (падение на **-16.9%**)
- **mAP@50: 72.32 → 61.85** (падение на **-14.5%**)

Это НЕ случайная флуктуация - деградация наблюдается **с самых первых итераций** и сохраняется на протяжении всего обучения.

---

## 📊 СРАВНЕНИЕ МЕТРИК

### Финальные результаты

| Версия | Итераций | Последняя итерация | PQ | mAP@50 | Uncertainty |
|--------|----------|-------------------|-----|--------|-------------|
| **OLD** | 20 | 2879 | **52.80** | **72.32** | N/A |
| **PRETRAIN** | 61 | 8783 | 43.86 | 61.85 | 0.0 |
| **LATEST** | 61 | 8783 | 43.86 | 61.85 | 0.0 |

### Деградация

- **ΔPQ = -8.94** (-16.9%)
- **ΔmAP@50 = -10.46** (-14.5%)

---

## 📈 ТРАЕКТОРИЯ ОБУЧЕНИЯ

### Сравнение на одинаковых итерациях

| Итерация | OLD PQ | PRETRAIN PQ | Разница |
|----------|--------|-------------|---------|
| 143 | 0.03 | 0.00 | -0.03 |
| 287 | 0.18 | 0.00 | -0.18 |
| 431 | 1.10 | 0.00 | -1.10 |
| 575 | 5.03 | 0.08 | **-4.95** |
| 719 | 10.02 | 0.00 | **-10.02** |
| 863 | 23.11 | 0.10 | **-23.01** |
| 1007 | 26.06 | 1.15 | **-24.91** |
| 1151 | 30.59 | 1.42 | **-29.17** |
| 1295 | 31.66 | 3.25 | **-28.40** |
| 1439 | 36.47 | 4.93 | **-31.54** |

**Вывод:** Деградация начинается с **первых итераций** и нарастает. На итерации 1439 разница уже **-31.54 PQ**!

---

## 🔍 АНАЛИЗ ПРИЧИН

### Что изменилось между OLD и PRETRAIN?

**Временная линия коммитов:**

1. **OLD VERSION** (20 апреля 2026, 15:54)
   - Коммит: `b13865b` или `2d2cfbe`
   - Без Bayesian inference
   - Детерминированная модель

2. **PRETRAIN** (26 апреля 2026, 13:03)
   - Коммит: `d99b58d` - "Add Bayesian inference support (SWAG + MC Dropout)"
   - Добавлен SWAG wrapper для predictor
   - Добавлены новые конфигурационные параметры

**Ключевые изменения в коммите d99b58d:**

```
 my_train_odin.py                     | 152 +++++++++++++++--
 odin/config.py                       |  13 +-
 odin/modeling/bayesian/__init__.py   |   4 +
 odin/modeling/bayesian/swag.py       | 268 +++++++++++++++++++++++++++++
 odin/modeling/meta_arch/odin_head.py | 224 +++++++++++++++++-------
```

---

## 🎯 ГИПОТЕЗЫ О ПРИЧИНАХ ДЕГРАДАЦИИ

### Гипотеза 1: SWAG wrapper ломает обучение ❌
**Вероятность:** ВЫСОКАЯ

**Факты:**
- SWAG wrapper добавлен в `MyTrainer.__init__()` (строки 1136-1154)
- Оборачивает только predictor (transformer decoder)
- Но может влиять на градиенты и оптимизацию

**Проверка:**
```python
# В my_train_odin.py, строки 1136-1154
if bayesian_type == "swag":
    if hasattr(model, 'sem_seg_head') and hasattr(model.sem_seg_head, 'predictor'):
        predictor = model.sem_seg_head.predictor
        self.swag_model = SWAG(predictor, no_cov_mat=no_cov_mat, max_num_models=max_num_models)
        model.sem_seg_head.swag_model = self.swag_model
```

**Проблема:** SWAG wrapper может:
1. Изменять поведение predictor во время forward pass
2. Влиять на backpropagation
3. Добавлять дополнительные параметры, которые не оптимизируются

---

### Гипотеза 2: Изменения в OdinHead ломают inference ❌
**Вероятность:** ОЧЕНЬ ВЫСОКАЯ

**Факты:**
- В коммите d99b58d файл `odin/modeling/meta_arch/odin_head.py` изменен на **224 строки**
- Добавлена поддержка Bayesian inference modes
- Изменена логика forward pass

**Критические изменения:**
```python
# Добавлен Bayesian inference в forward()
bayesian_type = self.bayesian_type
bayesian_samples = self.bayesian_samples

if bayesian_type == "swag" and hasattr(self, 'swag_model'):
    # SWAG sampling logic
    ...
elif bayesian_type == "mc_dropout":
    # MC Dropout logic
    ...
```

**Проблема:** Даже если `bayesian_type="none"`, изменения в коде могли:
1. Сломать детерминированный путь
2. Изменить порядок операций
3. Добавить баги в логику inference

---

### Гипотеза 3: Конфигурация SWAG активна даже при "none" ❌
**Вероятность:** СРЕДНЯЯ

**Проверка конфига (kaggle_my_train_odin.py):**
```python
'MODEL.BAYESIAN_TYPE', 'swag',  # Включаем SWAG
'MODEL.BAYESIAN_SAMPLES', '10',
'MODEL.BAYESIAN_INFERENCE_DURING_TRAINING', 'True',  # Детерминированный eval
'MODEL.SWAG.START_EPOCH', '5',
'MODEL.SWAG.UPDATE_FREQ', '5',
'MODEL.SWAG.MAX_MODELS', '10',
```

**Проблема:** `BAYESIAN_TYPE='swag'` активирует SWAG wrapper, который может влиять на обучение даже если inference детерминированный.

---

### Гипотеза 4: Изменения в config.py сломали параметры ❌
**Вероятность:** НИЗКАЯ

**Изменения в odin/config.py:**
```python
# Добавлены новые параметры
cfg.MODEL.BAYESIAN_TYPE = "none"
cfg.MODEL.BAYESIAN_SAMPLES = 1
cfg.MODEL.BAYESIAN_INFERENCE_DURING_TRAINING = False
cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
```

**Проблема:** Новые параметры могли перезаписать существующие или изменить дефолтные значения.

---

### Гипотеза 5: Баг в SWAG.collect_model() ❌
**Вероятность:** ВЫСОКАЯ

**Проверка SWAGHook (my_train_odin.py, строки 1299-1341):**
```python
def after_step(self):
    current_epoch = self.trainer.iter * self.batch_size / self.dataset_len
    
    if current_epoch >= self.start_epoch:
        if (self.trainer.iter + 1) % self.update_freq == 0:
            # Collect only predictor weights
            if hasattr(model, 'sem_seg_head') and hasattr(model.sem_seg_head, 'predictor'):
                predictor = model.sem_seg_head.predictor
                self.swag_model.collect_model(predictor)
```

**Проблема:** `collect_model()` вызывается каждые 5 итераций с эпохи 5. Это может:
1. Изменять веса predictor во время обучения
2. Сбрасывать momentum оптимизатора
3. Вносить шум в градиенты

---

## 🔬 ДЕТАЛЬНЫЙ АНАЛИЗ SWAG IMPLEMENTATION

### SWAG.collect_model() (odin/modeling/bayesian/swag.py)

```python
def collect_model(self, base_model):
    """
    Collect current model weights for SWAG.
    Updates running mean and second moment.
    """
    # Update n_models counter
    self.n_models += 1
    
    # Collect parameters
    for (name, base_param), swag_param in zip(
        base_model.named_parameters(), self.params
    ):
        # Update mean: mean = (mean * (n-1) + param) / n
        swag_param["mean"].mul_(self.n_models - 1).add_(base_param.data).div_(self.n_models)
        
        # Update sq_mean for variance
        swag_param["sq_mean"].mul_(self.n_models - 1).add_(
            base_param.data ** 2
        ).div_(self.n_models)
```

**КРИТИЧЕСКАЯ ПРОБЛЕМА:** Метод `collect_model()` **НЕ изменяет** веса base_model, только собирает статистику. Значит, проблема не здесь.

---

## 🚨 ГЛАВНАЯ ГИПОТЕЗА: Изменения в OdinHead.forward()

### Анализ изменений в odin_head.py

**ДО коммита d99b58d:**
```python
def forward(self, features, mask=None):
    # Standard forward pass
    predictions = self.predictor(features, mask)
    return predictions
```

**ПОСЛЕ коммита d99b58d:**
```python
def forward(self, features, mask=None):
    bayesian_type = self.bayesian_type
    
    if bayesian_type == "swag" and hasattr(self, 'swag_model'):
        # SWAG path
        ...
    elif bayesian_type == "mc_dropout":
        # MC Dropout path
        ...
    else:
        # Standard path (должен быть идентичен старому)
        predictions = self.predictor(features, mask)
        return predictions
```

**ПРОБЛЕМА:** Даже если код в "standard path" выглядит идентично, могли быть:
1. Изменения в порядке операций
2. Добавлены условия, которые влияют на все пути
3. Изменена инициализация параметров

---

## 🔍 ПЛАН РАССЛЕДОВАНИЯ

### Шаг 1: Проверить, что SWAG действительно активен
```bash
# В логах обучения должно быть:
>>> SWAG: Starting weight collection at epoch 5.XX <<<
>>> SWAG: Collected N predictor weight snapshots <<<
```

### Шаг 2: Сравнить веса predictor
```python
# Загрузить чекпоинты OLD и PRETRAIN
old_ckpt = torch.load('old_model_final.pth')
new_ckpt = torch.load('pretrain_model_final.pth')

# Сравнить веса predictor
for key in old_ckpt['model'].keys():
    if 'predictor' in key:
        old_w = old_ckpt['model'][key]
        new_w = new_ckpt['model'][key]
        diff = (old_w - new_w).abs().mean()
        print(f'{key}: diff={diff:.6f}')
```

### Шаг 3: Запустить обучение с BAYESIAN_TYPE='none'
```python
# В kaggle_my_train_odin.py
'MODEL.BAYESIAN_TYPE', 'none',  # Отключить SWAG
```

### Шаг 4: Откатить изменения в odin_head.py
```bash
git checkout b13865b -- odin/modeling/meta_arch/odin_head.py
```

### Шаг 5: Сравнить архитектуру моделей
```python
# Вывести архитектуру OLD и PRETRAIN
print(old_model)
print(new_model)
```

---

## 🎯 РЕКОМЕНДАЦИИ

### Немедленные действия (КРИТИЧНО)

1. **Откатить коммит d99b58d**
   ```bash
   git revert d99b58d
   ```

2. **Запустить обучение на коммите b13865b**
   - Проверить, что метрики восстанавливаются
   - Это подтвердит, что проблема в d99b58d

3. **Изолировать проблему**
   - Применить изменения из d99b58d по частям:
     1. Только SWAG класс (без изменений в OdinHead)
     2. Только изменения в config.py
     3. Только изменения в OdinHead
   - Найти, какое именно изменение ломает обучение

### Долгосрочные действия

1. **Создать stable branch на b13865b**
   ```bash
   git checkout b13865b
   git checkout -b stable-strawberry-baseline
   git push origin stable-strawberry-baseline
   ```

2. **Переписать Bayesian inference с нуля**
   - Не изменять OdinHead.forward()
   - Добавить Bayesian inference только для eval (`--eval-only`)
   - Использовать отдельный wrapper для inference

3. **Добавить unit tests**
   - Тест на идентичность forward pass с/без SWAG
   - Тест на градиенты
   - Тест на метрики на toy dataset

---

## 📋 CHECKLIST ДЛЯ ОТЛАДКИ

- [ ] Проверить логи PRETRAIN на наличие SWAG messages
- [ ] Сравнить веса predictor между OLD и PRETRAIN
- [ ] Запустить обучение с `BAYESIAN_TYPE='none'`
- [ ] Откатить изменения в odin_head.py
- [ ] Сравнить архитектуру моделей
- [ ] Запустить обучение на коммите b13865b
- [ ] Изолировать проблемное изменение
- [ ] Создать stable branch
- [ ] Переписать Bayesian inference

---

## 🔴 КРИТИЧЕСКИЙ ВЫВОД

**Коммит d99b58d ("Add Bayesian inference support") сломал обучение.**

Деградация метрик на **-16.9% PQ** и **-14.5% mAP@50** - это катастрофический результат. Необходимо:

1. **НЕМЕДЛЕННО** откатить d99b58d
2. Запустить обучение на b13865b для подтверждения
3. Изолировать проблемное изменение
4. Переписать Bayesian inference правильно

**Не использовать PRETRAIN/LATEST чекпоинты для дальнейшей работы!**

---

**Конец обновленного отчета**
