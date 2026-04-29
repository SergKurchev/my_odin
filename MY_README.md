# MY_README.md

Данный документ описывает воркфлоу работы `my_train_odin.py` в рамках протокола `models_evaluation_protocol.md` для Strawberry Multi-view Dataset.

## 0. Стабильная версия (Stable Baseline)
**Текущий рабочий ноутбук на  kaggle:** [`strawpick-segpoinnet-my-odin`](https://www.kaggle.com/code/sergkurchevusa/strawpick-segpoinnet-my-odin/output?scriptVersionId=313026137)

**Метрики:**
- `iteration`: 2879
- `total_loss`: 7.2639570569153875
- `PQ`: 52.79569217686208
- `SQ`: 69.03304133622852
- `RQ`: 76.47887323943662
- `mAP`: 32.64766504558833
- `mAP@50`: 72.3151450339084
- `mAP@25`: 81.93239412340795

**Метрики инференса:**
- `Inference Speed`: 0.4387 sec/sample (~2.27 FPS)
- `Условия инференса`: 
  - 16 ГБ VRAM
  - 5 кадров на сэмпл
  - Разрешение каждого кадра: 640x640 px
  - `batch size` = 1

---

## 🟢 Активная обученная модель (Active Trained Model)

> **Статус:** Модель обучена и готова к инференсу.

**Ноутбук на Kaggle:** [`strawpick-segpoinnet-my-odin-bayes`](https://www.kaggle.com/code/sergeistwpk/strawpick-segpoinnet-my-odin-bayes/settings?scriptVersionId=315363362)

**Коммит обучения:** `f9b4b1d` — на данном коммите была обучена модель.

**Коммит инференса:** `f9b4b1d` — инференс модели также работает на данном коммите.

> Оба процесса (обучение и инференс) верифицированы на коммите **`f9b4b1d`** (`Fix zero metrics: convert pred_classes to 1-indexed for evaluator`).

---

## 1. Назначение `my_train_odin.py`
Этот скрипт – адаптированная версия `train_odin.py`. В нём настроены:
- Парсинг специфичной файловой структуры `multiview_dataset`.
- Формирование батчей данных для библиотеки `detectron2` и модели MaskFormer/ODIN.
- Сбор метрик производительности (Train_Speed, Inference_Speed) и качества сегментации (PQ, SQ, RQ, mAP) из 3D-пространства.
- Модуль логирования метрик в CSV (`metrics_comparison.csv`).

## 2. Описание пайплайна (Workflow)

### Этап 1. Регистрация данных (Dataset registration)
Функция `get_strawberry_dataset_dicts()`:
1. Открывает файл `splits.json` по пути `/kaggle/input/datasets/sergeykurchev/strawpick-sint-pointnetseg-test/splits.json`
2. По ключам `"train"`, `"val"`, `"test"` получает массивы `sample_ids` (напр., `"00000"`).
3. Парсит директорию `sample_00000`, читает `cameras.json` и `color_map.json`.
4. Собирает полные абсолютные пути к:
   - RGB-изображениям (`rgb/*.png`)
   - Картам глубин (`depth/*.npy`)
   - Маскам сегментации (`masks/*.png`)
5. Вычисляет матрицы камеры: `intrinsics` (матрица 3x3) и `poses` (матрица 4x4, `R | t`).
6. **Нарезка кадров:** Исходная папка содержит 20 кадров. Чтобы снизить потребление VRAM (GPU памяти) и увеличить общее количество обучающих примеров (сэмплов), мы разбиваем видео на фрагменты (`chunks`) строго по 5 кадров. Если было 100 исходных сэмплов по 20 кадров, на выходе получается 400 сэмплов по 5 кадров. Сэмплы генерируются с идентификаторами `sample_00000_part0`, `_part1` и т.д.
7. Формирует словарь формата `detectron2` и регистрирует датасет в `DatasetCatalog`.

### Этап 2. Сборщик датасета на лету (Dataset Mapper)
Класс `StrawberryDatasetMapper` вызывается Data Loader-ом для каждой эпохи:
1. **Сэмплирование:** Согласно `cfg.INPUT.SAMPLING_FRAME_NUM` выбирает кадры из сцены.
2. **Чтение изображений:** Загружаются RGB-тензоры.
3. **Чтение масок:** Используется `color_map`, из Flat Unlit `masks/*.png` вырезаются пиксели согласно конкретным `category_id` (0=Ripe, 1=Unripe, 2=Half-ripe). Класс 3 (Цветоножка/Фон) игнорируется.
4. **Формат ODIN:** Генератор возвращает объект, содержащий ключи `images`, `depths`, `poses`, `intrinsics`, `instances_all`.
5. **Backprojection:** Используя функцию `backprojector_dataloader`, собираются облака точек `multi_scale_xyz` для 3D декодера ODIN.

### Этап 3. Эвалюация (3D Evaluator)
Класс `Strawberry3DEvaluator`:
1. На этапе тестирования/валидации собирает предсказания `outputs` и истинные значения `inputs`.
2. В методе `evaluate()` рассчитывает (или аггрегирует):
   - **PQ, SQ, RQ** по формулам IoU 3D для вокселей (согласно SemanticKITTI API Protocol).
   - **mAP, mAP@50, mAP@25**
   - Скорость работы батчей (Training и Inference Speed).
3. **Вывод:** Обновляет файл `output/metrics_comparison.csv`, добавляя в него новую строку со значениями для последней эпохи.

### Этап 4. Внешний скрипт запуска Kaggle (`kaggle_my_train_odin.py` / `.ipynb`)
Рабочий процесс Kaggle сводится к запусканию скрипта ноутбука без вмешательства в его код:
1. Клонирует исходники ODIN с GitHub (`https://github.com/SergKurchev/my_odin`).
2. Создает изолированное `venv` окружение.
3. Устанавливает тяжелые зависимости (`pytorch3d`, `detectron2`).
4. Автоматически запускает `my_train_odin.py` передав ему захардкоженные пути к датасету.

### Настройки обучения и конфигурации (Training Config)
В `kaggle_my_train_odin.py` жёстко задан вызов скрипта обучения.
Используемый конфиг базовой архитектуры: 
- `odin/configs/scannet_context/maskformer2_R50_bs16_50ep.yaml` (Основан на ResNet-50).

**Однако параметры переопределяются напрямую в `my_train_odin.py`:**
- **Размер батча (Batch Size):** `cfg.SOLVER.IMS_PER_BATCH = 1`
- **Размер сэмпла (Frame Num):** `cfg.INPUT.SAMPLING_FRAME_NUM = 5`

Эти изменения критически важны для успешного запуска ODIN в Kaggle с 1 GPU на T4 (15.2 GB), чтобы предотвратить Out-of-Memory (OOM) ошибки. Выходные метрики и веса сохраняются в папке `output`.

## 3. Байесовская классификация (Bayesian Inference)
В модель интегрирован механизм **SWAG (Stochastic Weight Averaging-Gaussian)** и **Monte Carlo Dropout** для оценки неопределенности:
1. **MC Dropout**: В голову классификации добавлен слой Dropout, который остается активным во время инференса (`F.dropout(..., training=True)`).
2. **SWAG**: Собирает статистику весов модели во время обучения и сэмплирует из апостериорного распределения при инференсе.
3. **Вероятностная оценка**: Модель выполняет несколько проходов (по умолчанию 5-10) и усредняет результаты в пространстве вероятностей.
4. **Зачем это нужно**: Это позволяет получить калиброванные вероятности классов. Высокая энтропия предсказания служит сигналом для NBV-планировщика, что объект требует дополнительного обследования.
5. **Настройка**: Количество сэмплов регулируется параметром `MODEL.BAYESIAN_SAMPLES`.

## 4. Поддержка NBV Stage2 Dataset (24 класса)

### Автоматическое определение типа датасета
Код автоматически определяет тип датасета по формату `cameras.json` и `color_map.json`:
- **Strawberry Dataset**: 3 класса (Ripe, Unripe, Half-ripe)
- **NBV Stage2 Dataset**: 24 класса (8 примитивов × 3 текстуры)

### NBV Stage2 Class Mapping (24 класса)

NBV Stage2 содержит 8 геометрических примитивов с 3 типами текстур, что дает **24 уникальных класса**:

**Примитивы (Primitive IDs 1-8)**:
1. cube (куб)
2. sphere (сфера)
3. cylinder (цилиндр)
4. cone (конус)
5. torus (тор)
6. capsule (капсула)
7. ellipsoid (эллипсоид)
8. pyramid (пирамида)

**Текстуры (Texture Types)**:
- `red` - красная текстура
- `mixed` - смешанная градиентная текстура
- `green` - зеленая текстура

**Маппинг классов (Class ID 0-23)**:
```
Class  0: cube_red          Class  8: cylinder_green    Class 16: capsule_mixed
Class  1: cube_mixed        Class  9: cone_red          Class 17: capsule_green
Class  2: cube_green        Class 10: cone_mixed        Class 18: ellipsoid_red
Class  3: sphere_red        Class 11: cone_green        Class 19: ellipsoid_mixed
Class  4: sphere_mixed      Class 12: torus_red         Class 20: ellipsoid_green
Class  5: sphere_green      Class 13: torus_mixed       Class 21: pyramid_red
Class  6: cylinder_red      Class 14: torus_green       Class 22: pyramid_mixed
Class  7: cylinder_mixed    Class 15: capsule_red       Class 23: pyramid_green
```

**Важно**: Робот (`category_id=9` в исходных данных) исключается из обучения и рассматривается как фон.

### Конверсия классов в data loader

В `get_nbv_stage2_dataset_dicts()` происходит автоматическая конверсия:
```python
# Исходный формат в color_map.json:
{
  "category_id": 6,        # Primitive ID (capsule)
  "texture_type": "green"  # Texture type
}

# Конвертируется в:
{
  "category_id": 17        # Unified class ID (capsule_green)
}
```

Формула конверсии: `class_id = (primitive_id - 1) * 3 + texture_index`

Где `texture_index`: red=0, mixed=1, green=2

### Использование

Просто укажите путь к NBV Stage2 датасету - тип определится автоматически:

```bash
python my_train_odin.py \
  --dataset_dir /path/to/nbv-stage2-dataset \
  --splits_file splits.json \
  --num_epochs 15
```

Код автоматически:
- Определит NBV Stage2 формат
- Создаст 24 класса
- Исключит робота из обучения
- Установит `MODEL.SEM_SEG_HEAD.NUM_CLASSES = 24`

