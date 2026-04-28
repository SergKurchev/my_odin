# ПЛАН МИГРАЦИИ: От b13865b к Стабильной Версии с Bayesian Inference

**Дата:** 28 апреля 2026  
**Цель:** Создать стабильную ветку с сохранением метрик + добавлением критически важных фич

---

## 📋 АНАЛИЗ КОММИТОВ (b13865b → HEAD)

### КАТЕГОРИЗАЦИЯ КОММИТОВ

#### ✅ КРИТИЧЕСКИ ВАЖНЫЕ (Must Have)
Эти коммиты решают реальные проблемы и должны быть включены:

1. **b229c31** (23 апр, 22:34) - "Fix disk overflow: aggressive checkpoint and visualization cleanup"
   - **Проблема:** Переполнение диска из-за накопления чекпоинтов
   - **Решение:** CheckpointCleanupHook (хранит только последние 2 чекпоинта)
   - **Статус:** ✅ КРИТИЧНО - без этого Kaggle падает

2. **4a69382** (23 апр, 22:34) - "Fix TimeLimitHook: use sys.exit after saving"
   - **Проблема:** Обучение не останавливается по таймауту
   - **Решение:** sys.exit(0) после сохранения чекпоинта
   - **Статус:** ✅ ВАЖНО - но нужно улучшить (sys.exit не работает)

3. **69f2f78** (24 апр, 00:52) - "Fix argparse conflict: remove duplicate --max_time"
   - **Проблема:** Дублирование аргумента --max_time
   - **Решение:** Удален дубликат
   - **Статус:** ✅ КРИТИЧНО - без этого argparse падает

4. **c581f97** (25 апр, 10:30) - "Fix visualization background masking and category indices"
   - **Проблема:** Неправильные цвета в визуализации (смещение классов)
   - **Решение:** 0-indexed вместо 1-indexed для инстансов
   - **Статус:** ✅ КРИТИЧНО - правильные визуализации

5. **e9772b8** (24 апр, 09:58) - "Fix CSV logging: support dynamic column addition"
   - **Проблема:** CSV не поддерживает новые колонки
   - **Решение:** Динамическое добавление колонок
   - **Статус:** ✅ ВАЖНО - для расширяемости метрик

6. **3edd802** (22 апр, 14:07) - "Auto-sync previous Kaggle outputs and increase warmup"
   - **Проблема:** Resume не работает из Kaggle Input
   - **Решение:** sync_previous_output() функция
   - **Статус:** ✅ ВАЖНО - для продолжения обучения

7. **78f0739** (22 апр, 13:51) - "Fix NaN crash: reduced LR to 5e-5 and added WARMUP_FACTOR"
   - **Проблема:** NaN в лоссах
   - **Решение:** LR=5e-5, WARMUP_FACTOR=0.001
   - **Статус:** ⚠️ ПРОВЕРИТЬ - может влиять на метрики

#### ❌ ПРОБЛЕМНЫЕ (Must Avoid)
Эти коммиты ломают обучение:

8. **d99b58d** (26 апр, 00:47) - "Add Bayesian inference support (SWAG + MC Dropout)"
   - **Проблема:** Деградация метрик на -17%
   - **Изменения:** 224 строки в odin_head.py + SWAG класс
   - **Статус:** ❌ СЛОМАНО - нужно переписать

9. **78a31dc** (23 апр, 20:22) - "Implement Bayesian classification with MC Dropout"
   - **Проблема:** Первая попытка Bayesian inference (возможно тоже сломана)
   - **Статус:** ❌ ПРОВЕРИТЬ - возможно тоже ломает

10. **890158d** (26 апр, 16:00) - "Fix logger undefined error in SWAG initialization"
    - **Проблема:** Фикс для сломанного d99b58d
    - **Статус:** ❌ НЕ НУЖЕН - если откатим d99b58d

#### 🔧 ПОЛЕЗНЫЕ (Nice to Have)
Улучшения, которые не критичны:

11. **46f0520** (22 апр, 12:13) - "Auto-calculate SOLVER.STEPS based on epochs"
    - **Улучшение:** Автоматический расчет LR decay steps
    - **Статус:** ✅ ПОЛЕЗНО

12. **7c73a54** (22 апр, 12:20) - "Fix speed calculation for variable-length samples"
    - **Улучшение:** Правильный расчет FPS
    - **Статус:** ✅ ПОЛЕЗНО

13. **db3a8c7** (24 апр, 09:51) - "Fix double indexing bug in 3D visualization"
    - **Улучшение:** Фикс визуализации
    - **Статус:** ✅ ПОЛЕЗНО (но c581f97 уже исправляет)

14. **819a05b** (25 апр, 10:09) - "Fix 3D viewer category palette mismatch"
    - **Улучшение:** Фикс визуализации (промежуточный)
    - **Статус:** ⚠️ ПРОПУСТИТЬ - c581f97 финальный фикс

#### 📝 ДОКУМЕНТАЦИЯ (Optional)
15. **18cb1c1**, **9c5b1ac**, **253cf51** - Обновления README
    - **Статус:** ✅ МОЖНО ВКЛЮЧИТЬ

#### 🆕 НОВЫЕ ФИЧИ (Separate Branch)
16. **82c6842** (26 апр, 15:58) - "Add NBV Stage2 dataset support with 24 classes"
    - **Фича:** Поддержка нового датасета
    - **Статус:** ✅ ПОЛЕЗНО - но в отдельной ветке

17. **62e4370** (27 апр, 00:46) - "Fix NBV Stage2 loader"
    - **Фикс:** Для NBV Stage2
    - **Статус:** ✅ НУЖЕН - если используем NBV Stage2

#### 🐛 ТЕХНИЧЕСКИЕ ФИКСЫ (After d99b58d)
18. **72f45e6**, **0680363**, **355c7b1**, **3d3a94e** - Фиксы для SWAG
    - **Статус:** ❌ НЕ НУЖНЫ - если откатим d99b58d

#### 🔄 КОНФИГУРАЦИЯ (Check Impact)
19. **3d0ea94** (22 апр, 09:49) - "change config, made CAMERA_DROP: True"
    - **Изменение:** CAMERA_DROP=True (аугментация)
    - **Статус:** ⚠️ ПРОВЕРИТЬ - может влиять на метрики

20. **c3ca406** (22 апр, 12:11) - "Revert NUM_OBJECT_QUERIES and SOLVER.STEPS"
    - **Изменение:** Откат параметров к оригиналу
    - **Статус:** ✅ ВАЖНО - для совместимости с весами

21. **4b0d4e4** (22 апр, 12:08) - "Adjust architecture parameters"
    - **Изменение:** 100 queries, MLP PE
    - **Статус:** ⚠️ ПРОВЕРИТЬ - может влиять на метрики

---

## 🎯 СТРАТЕГИЯ МИГРАЦИИ

### Фаза 1: Создание Stable Baseline (НЕМЕДЛЕННО)

**Цель:** Стабильная ветка с рабочими метриками + критичные фиксы

**Базовый коммит:** `b13865b` (20 апр, 15:20)

**Cherry-pick коммиты (в порядке применения):**

```bash
# 1. Создаем stable branch
git checkout b13865b
git checkout -b stable-strawberry-v1

# 2. Критичные фиксы (22 апреля)
git cherry-pick 3edd802  # Auto-sync previous Kaggle outputs
git cherry-pick 78f0739  # Fix NaN crash (ПРОВЕРИТЬ ВЛИЯНИЕ!)
git cherry-pick 7c73a54  # Fix speed calculation
git cherry-pick 46f0520  # Auto-calculate SOLVER.STEPS
git cherry-pick c3ca406  # Revert NUM_OBJECT_QUERIES (важно!)

# 3. Disk overflow fix (23 апреля) - КРИТИЧНО!
git cherry-pick b229c31  # Fix disk overflow + cleanup hooks
git cherry-pick 4a69382  # Fix TimeLimitHook (но нужно улучшить)

# 4. Argparse fix (24 апреля)
git cherry-pick 69f2f78  # Fix argparse conflict

# 5. Visualization fixes (24-25 апреля)
git cherry-pick db3a8c7  # Fix double indexing bug
git cherry-pick c581f97  # Fix visualization (финальный фикс)

# 6. CSV logging fix (24 апреля)
git cherry-pick e9772b8  # Fix CSV logging

# 7. Документация (опционально)
git cherry-pick 253cf51  # add link on kaggle
git cherry-pick 9c5b1ac  # Add Kaggle notebook metrics
git cherry-pick 18cb1c1  # Add inference metrics

# 8. Push stable branch
git push origin stable-strawberry-v1
```

**ВАЖНО:** После каждого cherry-pick проверять, что код компилируется!

---

### Фаза 2: Тестирование Stable Baseline

**Запустить обучение на Kaggle:**

```python
# В kaggle_my_train_odin.py
# Использовать stable-strawberry-v1 branch
# Параметры:
--num_epochs 10
--eval_period 144
--checkpoint_period 144
--max_time 6
--batch_size 1
--num_frames 5
--image_size 640
--lr 0.00005  # Или 0.0001 - ПРОВЕРИТЬ!
```

**Ожидаемый результат:**
- PQ ≥ 50 (близко к 52.80)
- mAP@50 ≥ 70 (близко к 72.32)
- Нет переполнения диска
- Автоостановка через 6 часов работает

**Если метрики ниже:**
- Откатить 78f0739 (LR=5e-5) → вернуть LR=1e-4
- Откатить 3d0ea94 (CAMERA_DROP=True) → вернуть False
- Откатить 4b0d4e4 (architecture changes)

---

### Фаза 3: Добавление Bayesian Inference (ПРАВИЛЬНО)

**Создать новую ветку от stable-strawberry-v1:**

```bash
git checkout stable-strawberry-v1
git checkout -b feature/bayesian-inference-v2
```

**Принципы реализации:**

1. **НЕ ИЗМЕНЯТЬ odin_head.py forward pass**
   - Оставить детерминированный путь нетронутым
   - Bayesian inference только через wrapper

2. **SWAG только для eval**
   - Собирать статистику весов во время обучения
   - Использовать SWAG sampling только в `--eval-only` режиме

3. **MC Dropout через отдельный wrapper**
   - Не изменять основную модель
   - Активировать dropout только для inference

**Архитектура:**

```python
# Новый файл: odin/modeling/bayesian/bayesian_wrapper.py

class BayesianInferenceWrapper:
    """
    Wrapper для Bayesian inference без изменения основной модели.
    """
    def __init__(self, model, method='none', n_samples=1):
        self.model = model
        self.method = method
        self.n_samples = n_samples
        self.swag = None
        
        if method == 'swag':
            # Инициализация SWAG для predictor
            self.swag = SWAG(model.sem_seg_head.predictor)
    
    def forward(self, *args, **kwargs):
        if self.method == 'none':
            # Детерминированный путь (БЕЗ ИЗМЕНЕНИЙ!)
            return self.model(*args, **kwargs)
        
        elif self.method == 'mc_dropout':
            # MC Dropout inference
            return self._mc_dropout_forward(*args, **kwargs)
        
        elif self.method == 'swag':
            # SWAG inference
            return self._swag_forward(*args, **kwargs)
    
    def _mc_dropout_forward(self, *args, **kwargs):
        # Активируем dropout
        self.model.train()  # Включаем dropout
        
        # Множественные проходы
        outputs = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                out = self.model(*args, **kwargs)
                outputs.append(out)
        
        # Агрегация
        return self._aggregate_outputs(outputs)
    
    def _swag_forward(self, *args, **kwargs):
        # Сэмплируем веса из SWAG posterior
        outputs = []
        for _ in range(self.n_samples):
            # Sample weights
            self.swag.sample(scale=1.0)
            
            with torch.no_grad():
                out = self.model(*args, **kwargs)
                outputs.append(out)
        
        # Восстанавливаем оригинальные веса
        self.swag.restore_original_weights()
        
        # Агрегация
        return self._aggregate_outputs(outputs)
```

**Использование:**

```python
# В my_train_odin.py

# Обучение (детерминированное)
trainer = MyTrainer(cfg)
trainer.train()

# Eval с Bayesian inference (отдельный запуск)
if args.eval_only and cfg.MODEL.BAYESIAN_TYPE != 'none':
    model = trainer.build_model(cfg)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    
    # Оборачиваем в Bayesian wrapper
    bayesian_model = BayesianInferenceWrapper(
        model, 
        method=cfg.MODEL.BAYESIAN_TYPE,
        n_samples=cfg.MODEL.BAYESIAN_SAMPLES
    )
    
    # Eval
    results = trainer.test(cfg, bayesian_model)
```

**Преимущества:**
- ✅ Детерминированный путь не изменен
- ✅ Метрики сохраняются
- ✅ Bayesian inference изолирован
- ✅ Легко тестировать и отлаживать

---

### Фаза 4: Дополнительные Улучшения

**После успешного тестирования Фазы 3:**

1. **Улучшить TimeLimitHook**
   ```python
   # Вместо sys.exit(0):
   self.trainer.storage.iter = self.trainer.max_iter
   ```

2. **Исправить Resume из Input**
   ```python
   # В sync_previous_output():
   # Копировать last_checkpoint файл
   ```

3. **Добавить NBV Stage2 support** (опционально)
   ```bash
   git cherry-pick 82c6842  # Add NBV Stage2 dataset
   git cherry-pick 62e4370  # Fix NBV Stage2 loader
   ```

4. **Добавить unit tests**
   ```python
   # tests/test_bayesian_inference.py
   def test_deterministic_path_unchanged():
       # Проверить, что forward pass идентичен
       pass
   ```

---

## 📊 ПЛАН ТЕСТИРОВАНИЯ

### Тест 1: Stable Baseline (Фаза 2)
**Цель:** Подтвердить, что метрики восстановлены

**Запуск:**
```bash
git checkout stable-strawberry-v1
# Запустить на Kaggle
```

**Критерии успеха:**
- PQ ≥ 50
- mAP@50 ≥ 70
- Нет переполнения диска
- Автоостановка работает

**Если провал:**
- Откатить подозрительные коммиты (78f0739, 3d0ea94, 4b0d4e4)
- Повторить тест

---

### Тест 2: Bayesian Inference (Фаза 3)
**Цель:** Проверить, что Bayesian inference работает без деградации

**Запуск 2.1: Детерминированный (baseline)**
```bash
git checkout feature/bayesian-inference-v2
# Запустить с BAYESIAN_TYPE='none'
```

**Критерии успеха:**
- PQ ≥ 50 (идентично Тесту 1)
- mAP@50 ≥ 70 (идентично Тесту 1)

**Запуск 2.2: MC Dropout**
```bash
# Запустить с BAYESIAN_TYPE='mc_dropout', BAYESIAN_SAMPLES=10
```

**Критерии успеха:**
- PQ ≥ 48 (допустимо небольшое снижение из-за dropout)
- Uncertainty > 0 (должна быть ненулевая)

**Запуск 2.3: SWAG**
```bash
# 1. Обучение с SWAG collection
# 2. Eval с SWAG sampling
```

**Критерии успеха:**
- PQ ≥ 48
- Uncertainty > 0
- SWAG статистика собирается корректно

---

## 🚀 ROADMAP

### Неделя 1 (28 апр - 4 мая)
- [x] Расследование завершено
- [ ] Создать stable-strawberry-v1 branch
- [ ] Запустить Тест 1 на Kaggle
- [ ] Подтвердить метрики ≥ 50 PQ

### Неделя 2 (5-11 мая)
- [ ] Реализовать BayesianInferenceWrapper
- [ ] Запустить Тест 2.1 (детерминированный)
- [ ] Запустить Тест 2.2 (MC Dropout)

### Неделя 3 (12-18 мая)
- [ ] Запустить Тест 2.3 (SWAG)
- [ ] Добавить unit tests
- [ ] Улучшить TimeLimitHook и Resume

### Неделя 4 (19-25 мая)
- [ ] Добавить NBV Stage2 support
- [ ] Финальное тестирование
- [ ] Документация

---

## ⚠️ РИСКИ И МИТИГАЦИЯ

### Риск 1: Метрики не восстанавливаются в Тесте 1
**Вероятность:** СРЕДНЯЯ

**Причина:** Один из cherry-picked коммитов ломает обучение

**Митигация:**
1. Откатить подозрительные коммиты по одному:
   - 78f0739 (LR change)
   - 3d0ea94 (CAMERA_DROP)
   - 4b0d4e4 (architecture)
2. Повторить тест после каждого отката
3. Найти проблемный коммит

---

### Риск 2: BayesianInferenceWrapper не работает
**Вероятность:** НИЗКАЯ

**Причина:** Технические сложности с wrapping

**Митигация:**
1. Начать с простого MC Dropout (легче реализовать)
2. Добавить подробное логирование
3. Тестировать на toy dataset сначала

---

### Риск 3: SWAG требует слишком много памяти
**Вероятность:** СРЕДНЯЯ

**Причина:** Хранение статистики весов

**Митигация:**
1. Использовать low-rank approximation (rank=20)
2. Собирать статистику только для predictor (не всей модели)
3. Ограничить MAX_MODELS=10

---

## 📝 CHECKLIST

### Фаза 1: Stable Baseline
- [ ] Создать stable-strawberry-v1 branch
- [ ] Cherry-pick критичные коммиты
- [ ] Проверить компиляцию после каждого cherry-pick
- [ ] Push в remote

### Фаза 2: Тестирование
- [ ] Запустить Тест 1 на Kaggle
- [ ] Проверить PQ ≥ 50
- [ ] Проверить mAP@50 ≥ 70
- [ ] Проверить disk cleanup работает
- [ ] Проверить TimeLimitHook работает

### Фаза 3: Bayesian Inference
- [ ] Создать feature/bayesian-inference-v2 branch
- [ ] Реализовать BayesianInferenceWrapper
- [ ] Добавить MC Dropout support
- [ ] Добавить SWAG support
- [ ] Запустить Тест 2.1 (детерминированный)
- [ ] Запустить Тест 2.2 (MC Dropout)
- [ ] Запустить Тест 2.3 (SWAG)

### Фаза 4: Улучшения
- [ ] Улучшить TimeLimitHook
- [ ] Исправить Resume из Input
- [ ] Добавить NBV Stage2 support
- [ ] Добавить unit tests
- [ ] Обновить документацию

---

## 🎯 КРИТЕРИИ УСПЕХА

### Минимальные требования (MVP)
- ✅ PQ ≥ 50 (восстановление метрик)
- ✅ Нет переполнения диска
- ✅ Правильные визуализации
- ✅ Автоостановка работает

### Полные требования (Full)
- ✅ PQ ≥ 50
- ✅ Bayesian inference работает (MC Dropout + SWAG)
- ✅ Uncertainty > 0
- ✅ Resume из Input работает
- ✅ Unit tests покрывают критичные части

---

**Конец плана миграции**
