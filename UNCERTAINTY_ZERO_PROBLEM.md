# Проблема: Uncertainty всегда равна 0.0

## Симптомы
В файле `metrics_comparison (22).csv` колонка `mean_uncertainty` содержит только нули:
```
iteration,mean_uncertainty
143,0.0
287,0.0
431,0.0
...
3887,0.0
```

## Математические формулы (Kendall & Gal 2017)

### Bayesian Deep Learning для Uncertainty Estimation

При использовании MC Dropout или SWAG мы получаем T сэмплов из posterior распределения весов:
- θ₁, θ₂, ..., θ_T ~ p(θ|D)

Для каждого сэмпла θₜ модель выдаёт распределение вероятностей классов:
- p(y|x, θₜ) - categorical distribution over C classes

### 1. Predictive Entropy (Total Uncertainty)
Энтропия усреднённого предсказания:

```
H[y|x,D] = -∑ₖ p̄ₖ log(p̄ₖ)

где p̄ₖ = (1/T) ∑ₜ p(y=k|x,θₜ)
```

**Физический смысл**: Общая неопределённость модели в предсказании класса.

### 2. Expected Entropy (Aleatoric Uncertainty)
Ожидаемая энтропия отдельных предсказаний:

```
E_θ[H[y|x,θ]] = (1/T) ∑ₜ H[y|x,θₜ]
               = -(1/T) ∑ₜ ∑ₖ p(y=k|x,θₜ) log p(y=k|x,θₜ)
```

**Физический смысл**: Неопределённость, присущая данным (шум, неоднозначность).

### 3. Mutual Information (Epistemic Uncertainty)
Разница между общей и алеаторной неопределённостью:

```
I[y;θ|x,D] = H[y|x,D] - E_θ[H[y|x,θ]]
```

**Физический смысл**: Неопределённость из-за незнания весов модели. Уменьшается с ростом данных.

---

## Код: Где вычисляется Uncertainty

### 1. Модель ODIN Head (`odin/modeling/meta_arch/odin_head.py`)

#### Условие активации Bayesian inference:
```python
# Line ~140-145
use_bayesian = (
    (not self.training or self.bayesian_during_training) and
    self.bayesian_type in ["swag", "dropout"] and
    self.bayesian_samples > 1
)
```

**Параметры**:
- `self.training` - режим модели (train/eval)
- `self.bayesian_during_training` - флаг `MODEL.BAYESIAN_INFERENCE_DURING_TRAINING`
- `self.bayesian_type` - тип Bayesian inference (`MODEL.BAYESIAN_TYPE`)
- `self.bayesian_samples` - количество сэмплов (`MODEL.BAYESIAN_SAMPLES`)

#### Вычисление uncertainty (если `use_bayesian=True`):
```python
# Line ~160-200
all_logits = []  # Shape: (T, B, Q, C)
for sample_idx in range(self.bayesian_samples):
    # Sample weights from SWAG or enable dropout
    if self.bayesian_type == "swag":
        self.swag_model.sample(scale=1.0, cov=True)
    
    # Forward pass
    outputs_sample = self.predictor(...)
    all_logits.append(outputs_sample["pred_logits"])

# Stack: (T, B, Q, C)
all_logits = torch.stack(all_logits, dim=0)

# Convert to probabilities
all_probs = F.softmax(all_logits, dim=-1)  # (T, B, Q, C)

# 1. Predictive Entropy
mean_probs = all_probs.mean(dim=0)  # (B, Q, C)
predictive_entropy = -torch.sum(
    mean_probs * torch.log(mean_probs + 1e-8), 
    dim=-1
)  # (B, Q)

# 2. Expected Entropy (Aleatoric)
sample_entropies = -torch.sum(
    all_probs * torch.log(all_probs + 1e-8), 
    dim=-1
)  # (T, B, Q)
expected_entropy = sample_entropies.mean(dim=0)  # (B, Q)

# 3. Mutual Information (Epistemic)
mutual_information = predictive_entropy - expected_entropy  # (B, Q)

outputs["uncertainty"] = {
    "predictive_entropy": predictive_entropy,
    "expected_entropy": expected_entropy,
    "mutual_information": mutual_information,
}
```

### 2. Передача uncertainty в результаты (`odin/odin_model.py`)

#### Критическое место (commit afde5c3):
```python
# Line ~1050-1070
def forward(self, batched_inputs):
    ...
    outputs = self.sem_seg_head(...)
    
    # ВАЖНО: Сохранить uncertainty ДО удаления outputs
    uncertainty = outputs.get("uncertainty", None)
    
    # Удаляем тяжёлые тензоры
    del outputs
    
    # Обработка результатов
    processed_results = self.inference_3d(...)
    
    # Добавляем uncertainty обратно
    if uncertainty is not None:
        for i, result in enumerate(processed_results):
            result["uncertainty"] = uncertainty
    
    # Fallback: если uncertainty нет, добавляем pred_logits
    for i, result in enumerate(processed_results):
        if "uncertainty" not in result:
            result["pred_logits"] = mask_cls_results[i:i+1]
    
    return processed_results
```

### 3. Извлечение uncertainty в evaluator (`my_train_odin.py`)

#### В методе `process()`:
```python
# Line ~660-682
def process(self, inputs, outputs):
    for _in, _out in zip(inputs, outputs):
        # Извлекаем uncertainty из выхода модели
        if 'uncertainty' in _out and 'predictive_entropy' in _out['uncertainty']:
            # Bayesian uncertainty
            predictive_entropy = _out['uncertainty']['predictive_entropy']  # [B, Q]
            mean_entropy = predictive_entropy.mean().item()
            self.uncertainties.append(mean_entropy)
            print(f"[UNCERTAINTY] Using Bayesian uncertainty: {mean_entropy:.6f}")
        
        elif 'pred_logits' in _out:
            # Fallback: deterministic entropy
            logits = _out['pred_logits']  # (B, num_queries, num_classes)
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            mean_entropy = entropy.mean().item()
            self.uncertainties.append(mean_entropy)
            print(f"[UNCERTAINTY] Using deterministic entropy: {mean_entropy:.6f}")
        
        else:
            print(f"[UNCERTAINTY] WARNING: No uncertainty or pred_logits in output!")
```

#### В методе `evaluate()`:
```python
# Line ~1050-1070
def evaluate(self):
    ...
    # Усреднение uncertainty по всем сэмплам
    if len(self.uncertainties) > 0:
        mean_uncertainty = np.mean(self.uncertainties)
    else:
        mean_uncertainty = 0.0
    
    metrics["mean_uncertainty"] = mean_uncertainty
    return metrics
```

---

## Конфигурационные параметры

### В `my_train_odin.py` (setup):
```python
# Line ~1780-1790
cfg.MODEL.BAYESIAN_TYPE = "swag"  # или "dropout"
cfg.MODEL.BAYESIAN_SAMPLES = 10   # Количество MC сэмплов
cfg.MODEL.BAYESIAN_INFERENCE_DURING_TRAINING = True  # Включить во время обучения

# SWAG параметры
cfg.MODEL.SWAG.START_EPOCH = 5
cfg.MODEL.SWAG.UPDATE_FREQ = 5
cfg.MODEL.SWAG.MAX_MODELS = 10
cfg.MODEL.SWAG.RANK = 20
cfg.MODEL.SWAG.NO_COV_MAT = False
```

### В командной строке (`kaggle_my_train_odin.py`):
```python
'MODEL.BAYESIAN_TYPE', 'swag',
'MODEL.BAYESIAN_SAMPLES', '10',
'MODEL.BAYESIAN_INFERENCE_DURING_TRAINING', 'True',
'MODEL.SWAG.START_EPOCH', '5',
'MODEL.SWAG.UPDATE_FREQ', '5',
'MODEL.SWAG.MAX_MODELS', '10',
'MODEL.SWAG.RANK', '20',
'MODEL.SWAG.NO_COV_MAT', 'False',
```

---

## Диагностика: Почему uncertainty = 0?

### Гипотеза 1: Bayesian inference не активируется
**Проверка**: Ищем в логах:
```
[UNCERTAINTY] Using Bayesian uncertainty: X.XXXXXX
```

Если этого нет, значит `use_bayesian = False` в `odin_head.py`.

**Возможные причины**:
1. `self.bayesian_samples <= 1`
2. `self.bayesian_type` не равен "swag" или "dropout"
3. `self.training = True` и `self.bayesian_during_training = False`
4. SWAG модель не инициализирована (`self.swag_model is None`)

### Гипотеза 2: Uncertainty вычисляется, но не передаётся
**Проверка**: Добавить debug в `odin_head.py`:
```python
if use_bayesian:
    print(f"[DEBUG] Computed uncertainty: {outputs['uncertainty']['predictive_entropy'].mean().item()}")
```

Если uncertainty вычисляется, но в evaluator приходит 0, значит проблема в передаче.

### Гипотеза 3: Uncertainty удаляется вместе с outputs
**Статус**: Исправлено в commit `afde5c3`, но нужно проверить что изменения применены.

### Гипотеза 4: SWAG не собирает статистику
**Проверка**: SWAG начинает работать только после `START_EPOCH=5`.

Если обучение прерывается раньше, SWAG не активен.

**Проверка в логах**:
```
[SWAG] Collecting model at epoch X
```

### Гипотеза 5: Evaluator не извлекает uncertainty
**Проверка**: Ищем в логах:
```
[UNCERTAINTY] WARNING: No uncertainty or pred_logits in output!
```

Если это есть, значит `_out` не содержит ни `uncertainty`, ни `pred_logits`.

---

## План отладки

### Шаг 1: Проверить активацию Bayesian inference
Добавить в `odin/modeling/meta_arch/odin_head.py` (line ~145):
```python
use_bayesian = (...)
print(f"[DEBUG BAYESIAN] use_bayesian={use_bayesian}, training={self.training}, "
      f"bayesian_during_training={self.bayesian_during_training}, "
      f"bayesian_type={self.bayesian_type}, bayesian_samples={self.bayesian_samples}, "
      f"swag_model={self.swag_model is not None}")
```

### Шаг 2: Проверить вычисление uncertainty
Добавить в `odin/modeling/meta_arch/odin_head.py` (после вычисления):
```python
if use_bayesian:
    pred_ent = outputs["uncertainty"]["predictive_entropy"].mean().item()
    exp_ent = outputs["uncertainty"]["expected_entropy"].mean().item()
    mut_inf = outputs["uncertainty"]["mutual_information"].mean().item()
    print(f"[DEBUG UNCERTAINTY] pred_ent={pred_ent:.6f}, exp_ent={exp_ent:.6f}, mut_inf={mut_inf:.6f}")
```

### Шаг 3: Проверить передачу в odin_model.py
Добавить в `odin/odin_model.py` (после сохранения uncertainty):
```python
uncertainty = outputs.get("uncertainty", None)
if uncertainty is not None:
    print(f"[DEBUG ODIN_MODEL] Saved uncertainty before del outputs: {uncertainty['predictive_entropy'].mean().item():.6f}")
else:
    print(f"[DEBUG ODIN_MODEL] No uncertainty in outputs! Keys: {outputs.keys()}")
```

### Шаг 4: Проверить извлечение в evaluator
Уже есть debug вывод в `my_train_odin.py` line ~666-681.

### Шаг 5: Проверить SWAG статистику
Добавить в `my_train_odin.py` в SWAGHook:
```python
def after_step(self):
    if self.should_collect():
        print(f"[DEBUG SWAG] Collecting model at iter {self.trainer.iter}, epoch {current_epoch:.2f}")
        self.swag_model.collect_model(self.trainer.model)
```

---

## Текущий статус

### Что исправлено:
1. ✅ Bayesian inference logic (`use_bayesian` условие) - commit `5d5ee59`
2. ✅ Сохранение uncertainty перед `del outputs` - commit `afde5c3`
3. ✅ Fallback на `pred_logits` если uncertainty нет - commit `afde5c3`

### Что НЕ проверено:
1. ❓ Активируется ли `use_bayesian=True` во время eval?
2. ❓ Собирает ли SWAG статистику (нужно дождаться epoch 5)?
3. ❓ Передаётся ли uncertainty из `odin_head.py` → `odin_model.py` → `evaluator`?

---

## Вопрос для следующего агента

**Проблема**: В CSV файле `mean_uncertainty` всегда равна 0.0, хотя:
- Bayesian inference настроен (`MODEL.BAYESIAN_TYPE=swag`, `MODEL.BAYESIAN_SAMPLES=10`)
- Код для вычисления uncertainty есть в `odin/modeling/meta_arch/odin_head.py`
- Код для извлечения uncertainty есть в `my_train_odin.py` evaluator

**Задача**: 
1. Добавить debug логирование по плану выше (Шаги 1-5)
2. Запустить обучение и найти где именно теряется uncertainty
3. Исправить проблему и убедиться что `mean_uncertainty > 0` в CSV

**Файлы для проверки**:
- `odin/modeling/meta_arch/odin_head.py` - вычисление uncertainty
- `odin/odin_model.py` - передача uncertainty в результаты
- `my_train_odin.py` - извлечение uncertainty в evaluator
- `odin/config.py` - конфигурационные параметры

**Ожидаемый результат**: 
В логах должны появиться сообщения вида:
```
[DEBUG BAYESIAN] use_bayesian=True, ...
[DEBUG UNCERTAINTY] pred_ent=0.XXXXX, exp_ent=0.XXXXX, mut_inf=0.XXXXX
[UNCERTAINTY] Using Bayesian uncertainty: 0.XXXXX
```

И в CSV файле `mean_uncertainty` должна быть > 0.
