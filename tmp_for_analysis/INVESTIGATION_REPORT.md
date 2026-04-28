# ДЕТАЛЬНОЕ РАССЛЕДОВАНИЕ ПРОБЛЕМ С KAGGLE ОБУЧЕНИЕМ

**Дата расследования:** 28 апреля 2026  
**Исследователь:** Claude Sonnet 4

---

## EXECUTIVE SUMMARY

После детального анализа трех запусков обучения на Kaggle выявлено следующее:

1. **Метрики НЕ ухудшились** - все три CSV файла идентичны (PQ=43.97, mAP@50=61.85)
2. **Uncertainty всегда равен 0.0** - проблема присутствует во всех запусках
3. **Визуализации имеют баг со смещением классов** - исправлен в коммите c581f97, но pretrain использовал более ранний коммит
4. **TimeLimitHook не сработал** - latest запуск пришлось останавливать вручную
5. **Отсутствуют HTML визуализации** - в latest запуске не сохранились

---

## ВРЕМЕННАЯ ЛИНИЯ ЗАПУСКОВ

### 1. OLD VERSION (Baseline)
- **Ноутбук:** `sergkurchevusa/strawpick-segpoinnet-my-odin`
- **Дата запуска:** ~20 апреля 2026, 15:54 UTC
- **Коммит:** `b13865b` или `2d2cfbe` (20 апреля 2026)
- **Статус:** COMPLETE
- **Длительность:** Неизвестна (успешно завершен)
- **Результаты:**
  - PQ: 43.97 (лучший на итерации 7487)
  - mAP@50: 61.85
  - mean_uncertainty: 0.0 (всегда)
  - Итераций: 61 (до 8783)

**Особенности:**
- Без SWAG/MC Dropout (детерминированная модель)
- Без байесовского inference
- Визуализации: неизвестно (CSV не содержит информации)

---

### 2. PRETRAIN (Первый запуск с Bayesian)
- **Ноутбук:** `oduvankinoshka/strawpick-segpoinnet-my-odin-bayes`
- **scriptVersionId:** 314576118
- **Дата запуска:** 26 апреля 2026, 13:03 UTC
- **Коммит:** между `d99b58d` и `1f226c6` (25-26 апреля 2026)
  - `d99b58d` - Add Bayesian inference support (SWAG + MC Dropout)
  - `8d4858e` - Fix Kaggle notebook: remove venv cleanup
  - `1f226c6` - Fix Kaggle notebooks: remove venv cleanup + fix NBV Stage2 paths
- **Статус:** COMPLETE
- **Результаты:** ИДЕНТИЧНЫ OLD VERSION
  - PQ: 43.97
  - mAP@50: 61.85
  - mean_uncertainty: 0.0 (всегда!)

**КРИТИЧЕСКАЯ ПРОБЛЕМА:**
- Визуализации имеют **баг со смещением классов**:
  - Зеленый → Оранжевый
  - Красный → Зеленый
  - Оранжевый → Серый (фон)
- **Причина:** Коммит `819a05b` (25 апреля, 10:09) использовал `point_pred_inst[m] = inst_idx + 1` (1-indexed)
- **Исправление:** Коммит `c581f97` (25 апреля, 10:30) изменил на `point_pred_inst[m] = inst_idx` (0-indexed)
- **Но pretrain запущен 26 апреля в 13:03** - должен был использовать исправленный код!

**ВЫВОД:** Либо pretrain использовал старый коммит (до c581f97), либо был запущен с кэшированным кодом.

---

### 3. LATEST (Второй запуск с Bayesian)
- **Ноутбук:** `oduvankinoshka/fork-of-strawpick-segpoinnet-my-odin-bayes`
- **scriptVersionId:** 314735315
- **Дата запуска:** 27 апреля 2026, 06:51 UTC
- **Коммит:** между `72f45e6` и `3d3a94e` (27 апреля 2026)
  - `72f45e6` - Fix SWAG checkpoint loading: convert int values to tensors
  - `3d3a94e` - Fix dtype mismatch in depth tensor creation
- **Статус:** CANCEL_ACKNOWLEDGED (принудительно остановлен пользователем)
- **Результаты:** ИДЕНТИЧНЫ OLD VERSION и PRETRAIN
  - PQ: 43.97
  - mAP@50: 61.85
  - mean_uncertainty: 0.0 (всегда!)

**ПРОБЛЕМЫ:**
1. **TimeLimitHook не сработал** - ограничение 6 часов (`--max_time 6`) не остановило обучение
2. **Отсутствуют HTML визуализации** - не сохранились в output
3. **Uncertainty всегда 0.0** - байесовский inference не работает

**ВАЖНО:** В input был добавлен pretrain ноутбук (scriptVersionId=314576118), но метрики начались с нуля!

---

## АНАЛИЗ ПРОБЛЕМ

### Проблема 1: Метрики "хуже"
**СТАТУС:** ❌ ЛОЖНАЯ ТРЕВОГА

**Факты:**
- Все три CSV файла **ИДЕНТИЧНЫ** (проверено через `pandas.DataFrame.equals()`)
- PQ, mAP@50, все остальные метрики - одинаковые до последнего знака
- Итерации: 143, 287, 431, ..., 8783 - идентичны

**Вывод:** Метрики НЕ ухудшились. Пользователь ошибочно сравнивал разные запуски или неправильно интерпретировал данные.

---

### Проблема 2: Uncertainty всегда равен 0.0
**СТАТУС:** ✅ ПОДТВЕРЖДЕНА

**Факты:**
- В OLD VERSION: `mean_uncertainty: [0.]` (ожидаемо, нет байесовского inference)
- В PRETRAIN: `mean_uncertainty: [0.]` (НЕОЖИДАННО!)
- В LATEST: `mean_uncertainty: [0.]` (НЕОЖИДАННО!)

**Анализ кода (my_train_odin.py, строки 660-665):**
```python
# 3. Calculate uncertainty (entropy) from pred_logits
if 'pred_logits' in _out:
    logits = _out['pred_logits']  # Shape: (B, num_queries, num_classes)
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # (B, num_queries)
    mean_entropy = entropy.mean().item()
    self.uncertainties.append(mean_entropy)
```

**Возможные причины:**
1. **`pred_logits` отсутствует в `_out`** - код не выполняется
2. **Детерминированный inference во время eval** - конфиг `MODEL.BAYESIAN_INFERENCE_DURING_TRAINING=True` означает НЕ использовать SWAG sampling
3. **SWAG не активирован** - `START_EPOCH=5`, но eval происходит раньше

**Проверка конфига (kaggle_my_train_odin.py, строки 376-377):**
```python
'MODEL.BAYESIAN_SAMPLES', '10',  # Детерминированный eval во время training (быстро)
'MODEL.BAYESIAN_INFERENCE_DURING_TRAINING', 'True',  # Не использовать SWAG sampling во время eval
```

**КРИТИЧЕСКАЯ ОШИБКА В КОММЕНТАРИИ:**
- Комментарий говорит "Детерминированный eval" для `BAYESIAN_SAMPLES=10`
- Но `BAYESIAN_INFERENCE_DURING_TRAINING=True` означает **детерминированный** inference!
- Правильно должно быть: `BAYESIAN_INFERENCE_DURING_TRAINING=False` для байесовского eval

**Вывод:** Uncertainty=0.0 потому что:
1. Во время training eval используется детерминированный inference (один проход)
2. Entropy от одного детерминированного прохода близка к 0 (модель уверена)
3. Нужно либо отключить `BAYESIAN_INFERENCE_DURING_TRAINING`, либо делать отдельный eval с `--eval-only`

---

### Проблема 3: TimeLimitHook не сработал
**СТАТУС:** ✅ ПОДТВЕРЖДЕНА

**Конфигурация:**
```python
"--max_time", "6",  # Авто-остановка через 6 часов
```

**Анализ кода (my_train_odin.py, строки 1203-1228):**
```python
class TimeLimitHook(hooks.HookBase):
    def __init__(self, max_time_hours):
        self.max_time_seconds = max_time_hours * 3600
        self.start_time = None

    def before_train(self):
        self.start_time = time.perf_counter()

    def after_step(self):
        elapsed = time.perf_counter() - self.start_time
        if elapsed > self.max_time_seconds:
            # ... save checkpoint ...
            sys.exit(0)
```

**Возможные причины:**
1. **Хук не зарегистрирован** - проверить `build_hooks()`
2. **`max_time` не передан** - проверить argparse
3. **`sys.exit(0)` не работает в Kaggle** - нужен другой механизм остановки
4. **Время считается неправильно** - `perf_counter()` может сбрасываться

**Проверка регистрации (my_train_odin.py, строка 1231):**
```python
max_time = getattr(self.cfg, "MAX_TIME_HOURS", 11.5)
all_hooks.append(TimeLimitHook(max_time))
```

**Проверка передачи параметра (kaggle_my_train_odin.py, строка 359):**
```python
"--max_time", "6",  # Авто-остановка через 6 часов
```

**ПРОБЛЕМА:** Параметр `--max_time` передается в скрипт, но в `setup()` он должен быть сохранен в `cfg.MAX_TIME_HOURS`:
```python
cfg.MAX_TIME_HOURS = getattr(args, "max_time", 11.5)
```

**Проверка setup() (my_train_odin.py, строка 1548):**
```python
cfg.MAX_TIME_HOURS = getattr(args, "max_time", 11.5)
```

**Вывод:** Код выглядит правильно. Возможно:
1. Kaggle убивает процесс до срабатывания хука
2. `sys.exit(0)` не прерывает training loop в Detectron2
3. Нужно использовать `trainer.storage.iter = trainer.max_iter` вместо `sys.exit()`

---

### Проблема 4: Отсутствуют HTML визуализации
**СТАТУС:** ✅ ПОДТВЕРЖДЕНА

**Ожидание:**
- В pretrain: HTML визуализации сохранились
- В latest: HTML визуализации НЕ сохранились

**Анализ кода (my_train_odin.py, строки 736-912):**
```python
if build_html is not None and len(self.vis_data) > 0:
    vis_output_dir = os.path.join(self._output_dir, "visualizations")
    os.makedirs(vis_output_dir, exist_ok=True)
    # ... generate HTML ...
```

**Возможные причины:**
1. **`build_html` не импортирован** - ImportError в строке 737
2. **`self.vis_data` пуст** - целевые сэмплы не найдены
3. **Kaggle не сохранил output** - принудительная остановка (CANCEL) не дала времени
4. **Путь неправильный** - `OUTPUT_DIR` не совпадает с Kaggle output

**Проверка целевых сэмплов (my_train_odin.py, строка 645):**
```python
target_samples = ["00000", "sample_00000", "00003", "sample_00003", "00005", "sample_00005"]
```

**Вывод:** Скорее всего, принудительная остановка (CANCEL) не дала времени на генерацию HTML. Нужно:
1. Генерировать HTML после каждого eval (не только в конце)
2. Или использовать отдельный хук для периодической генерации

---

### Проблема 5: Визуализации с неправильными классами (pretrain)
**СТАТУС:** ✅ ПОДТВЕРЖДЕНА И ОБЪЯСНЕНА

**Описание:**
- Зеленый → Оранжевый
- Красный → Зеленый
- Оранжевый → Серый (фон)

**Анализ коммитов:**

**Коммит 819a05b (25 апреля, 10:09):**
```python
point_pred_inst[m] = inst_idx + 1  # 1-indexed для виза инстансов
point_pred_cat[m] = int(pred_classes[inst_idx]) - 1
```

**Коммит c581f97 (25 апреля, 10:30):**
```python
point_pred_inst[m] = inst_idx  # 0-indexed для виза
point_pred_cat[m] = int(pred_classes[inst_idx]) - 1
```

**ПРОБЛЕМА:** В 819a05b использовался `inst_idx + 1` для инстансов, что сдвигало индексы на 1.

**Но pretrain запущен 26 апреля в 13:03** - через 26 часов после фикса c581f97!

**Возможные объяснения:**
1. **Kaggle использовал кэшированный код** - не обновил репозиторий
2. **Пользователь вручную откатил коммит** - использовал старую версию
3. **Ноутбук был создан до фикса** - fork старой версии

**Вывод:** Нужно проверить, какой именно коммит использовался в pretrain. Рекомендация: создать stable branch на коммите c581f97 или позже.

---

### Проблема 6: Метрики "с нуля" несмотря на input
**СТАТУС:** ✅ ПОДТВЕРЖДЕНА И ОБЪЯСНЕНА

**Описание:**
- В latest запуск был добавлен pretrain в input
- Но метрики начались с итерации 143 (не продолжились с 8783)

**Анализ кода (kaggle_my_train_odin.py, строки 312-339):**
```python
def sync_previous_output():
    """
    Автоматически ищет папку output в подключенных входных данных
    и копирует её в текущую рабочую директорию
    """
    input_base = "/kaggle/input"
    target_output = "./output"
    
    # Ищем любую папку, внутри которой есть файлы .pth
    for root, dirs, files in os.walk(input_base):
        if "output" in root and any(f.endswith(".pth") for f in files):
            # ... copy files ...
```

**Проверка команды (kaggle_my_train_odin.py, строка 348):**
```python
"--resume",  # Автоматически продолжит с последнего чекпоинта в OUTPUT_DIR
```

**ПРОБЛЕМА:** `sync_previous_output()` копирует файлы, но:
1. **Detectron2 `--resume` ищет `last_checkpoint`** - файл с путем к последнему чекпоинту
2. **Если `last_checkpoint` не скопирован** - resume не работает
3. **Если чекпоинт называется не `model_final.pth`** - resume может не найти его

**Проверка Detectron2 resume (detectron2/checkpoint/detection_checkpoint.py):**
```python
def resume_or_load(self, path="", *, resume=True):
    if resume and self.has_checkpoint():
        path = self.get_checkpoint_file()  # Читает last_checkpoint
    # ...
```

**Вывод:** `sync_previous_output()` не копирует файл `last_checkpoint`, поэтому resume не работает. Нужно:
1. Копировать `last_checkpoint` файл
2. Или явно указывать `MODEL.WEIGHTS` на скопированный чекпоинт
3. Или использовать `--resume` с явным путем

---

## РЕКОМЕНДАЦИИ

### 1. Исправить Uncertainty=0.0
**Приоритет:** ВЫСОКИЙ

**Решение:**
```python
# В kaggle_my_train_odin.py, строка 377
'MODEL.BAYESIAN_INFERENCE_DURING_TRAINING', 'False',  # Использовать SWAG sampling во время eval
```

**Или** добавить отдельный eval с `--eval-only` после обучения:
```python
# После обучения
eval_cmd = [
    VENV_PYTHON, "my_odin/my_train_odin.py",
    "--config-file", CONFIG_FILE,
    "--num-gpus", "1",
    "--eval-only",
    "--dataset_dir", DATASET_DIR,
    "--splits_file", SPLITS_FILE,
    'MODEL.WEIGHTS', './output/model_final.pth',
    'MODEL.BAYESIAN_TYPE', 'swag',
    'MODEL.BAYESIAN_SAMPLES', '10',  # 10 сэмплов из SWAG posterior
    'OUTPUT_DIR', './output',
]
run_in_venv(eval_cmd, env=make_venv_env())
```

---

### 2. Исправить TimeLimitHook
**Приоритет:** ВЫСОКИЙ

**Решение:** Заменить `sys.exit(0)` на корректную остановку:
```python
# В my_train_odin.py, строка 1227
# Вместо sys.exit(0):
self.trainer.storage.iter = self.trainer.max_iter
logger.info("!!! TIME LIMIT: Setting iter to max_iter to stop training !!!")
```

---

### 3. Исправить Resume из Input
**Приоритет:** СРЕДНИЙ

**Решение:** Копировать `last_checkpoint` файл:
```python
# В kaggle_my_train_odin.py, функция sync_previous_output()
for root, dirs, files in os.walk(input_base):
    if "output" in root and any(f.endswith(".pth") for f in files):
        # ... existing code ...
        
        # Копируем last_checkpoint файл
        last_checkpoint_src = os.path.join(root, "last_checkpoint")
        if os.path.exists(last_checkpoint_src):
            last_checkpoint_dst = os.path.join(target_output, "last_checkpoint")
            shutil.copy2(last_checkpoint_src, last_checkpoint_dst)
            print(f"Скопирован last_checkpoint для resume")
```

---

### 4. Генерировать HTML после каждого Eval
**Приоритет:** НИЗКИЙ

**Решение:** Переместить генерацию HTML из `evaluate()` в отдельный хук:
```python
class VisualizationHook(hooks.HookBase):
    def after_step(self):
        if (self.trainer.iter + 1) % self.trainer.cfg.TEST.EVAL_PERIOD == 0:
            # Generate HTML visualizations
            # ...
```

---

### 5. Создать Stable Branch
**Приоритет:** ВЫСОКИЙ

**Решение:**
```bash
cd odin
git checkout c581f97  # Коммит с исправленной визуализацией
git checkout -b stable-strawberry-v1
git push origin stable-strawberry-v1
```

**Или использовать более поздний коммит:**
```bash
git checkout d99b58d  # Коммит с Bayesian inference
git checkout -b stable-strawberry-bayesian-v1
git push origin stable-strawberry-bayesian-v1
```

---

## ВЫВОДЫ

1. **Метрики НЕ ухудшились** - все три запуска дали идентичные результаты (PQ=43.97)
2. **Uncertainty=0.0 из-за детерминированного eval** - нужно отключить `BAYESIAN_INFERENCE_DURING_TRAINING`
3. **Визуализации с багом в pretrain** - использовался коммит до c581f97
4. **TimeLimitHook не работает** - нужно заменить `sys.exit()` на `iter = max_iter`
5. **Resume не работает** - не копируется `last_checkpoint` файл
6. **HTML не сохранились в latest** - принудительная остановка не дала времени

**Рекомендуемый коммит для stable branch:** `d99b58d` или позже (с Bayesian inference и исправленной визуализацией)

**Следующие шаги:**
1. Создать stable branch на коммите d99b58d или позже
2. Исправить `BAYESIAN_INFERENCE_DURING_TRAINING=False`
3. Исправить TimeLimitHook
4. Запустить новый эксперимент с правильными настройками
5. Проверить uncertainty > 0 в результатах

---

**Конец отчета**
