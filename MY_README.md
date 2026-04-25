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
В модель интегрирован механизм **Monte Carlo Dropout** для оценки неопределенности:
1. **MC Dropout**: В голову классификации добавлен слой Dropout, который остается активным во время инференса (`F.dropout(..., training=True)`).
2. **Вероятностная оценка**: Модель выполняет несколько проходов (по умолчанию 5) и усредняет результаты в пространстве вероятностей.
3. **Зачем это нужно**: Это позволяет получить калиброванные вероятности классов. Высокая энтропия предсказания служит сигналом для NBV-планировщика, что объект требует дополнительного обследования.
4. **Настройка**: Количество сэмплов регулируется параметром `MODEL.BAYESIAN_SAMPLES`.
