# Байесовские подходы в ODIN - Руководство пользователя

## Обзор

ODIN теперь поддерживает три режима байесовского вывода для оценки неопределенности:

1. **"none"** - Детерминированный inference (без uncertainty estimation)
2. **"mc_dropout"** - MC Dropout с исправленной реализацией
3. **"swag"** - SWAG (Stochastic Weight Averaging-Gaussian)

## Быстрый старт

### 1. Детерминированный режим (по умолчанию)

```bash
python my_train_odin.py \
  --config-file configs/scannet_context/3d.yaml \
  --eval-only \
  MODEL.BAYESIAN_TYPE none \
  MODEL.BAYESIAN_SAMPLES 1
```

### 2. MC Dropout

```bash
python my_train_odin.py \
  --config-file configs/scannet_context/3d.yaml \
  --eval-only \
  MODEL.BAYESIAN_TYPE mc_dropout \
  MODEL.BAYESIAN_SAMPLES 10 \
  MODEL.MASK_FORMER.DROPOUT 0.1
```

**Что изменилось**: Теперь dropout действительно активен во время inference (исправлена оригинальная реализация).

### 3. SWAG

#### Шаг 1: Обучение с SWAG

```bash
python my_train_odin.py \
  --config-file configs/scannet_context/3d.yaml \
  --num_epochs 20 \
  MODEL.BAYESIAN_TYPE swag \
  MODEL.BAYESIAN_SAMPLES 1 \
  MODEL.BAYESIAN_INFERENCE_DURING_TRAINING False \
  MODEL.SWAG.START_EPOCH 10 \
  MODEL.SWAG.UPDATE_FREQ 5 \
  MODEL.SWAG.MAX_MODELS 20 \
  MODEL.SWAG.RANK 20 \
  MODEL.SWAG.NO_COV_MAT False
```

**Важно**: 
- `BAYESIAN_SAMPLES=1` - детерминированный eval во время обучения (быстро)
- `BAYESIAN_INFERENCE_DURING_TRAINING=False` - не использовать SWAG sampling во время eval (рекомендуется)

#### Шаг 2: Inference с SWAG

```bash
python my_train_odin.py \
  --config-file configs/scannet_context/3d.yaml \
  --eval-only \
  MODEL.BAYESIAN_TYPE swag \
  MODEL.BAYESIAN_SAMPLES 10 \
  MODEL.SWAG.SCALE 1.0
```

## Конфигурационные параметры

### Общие параметры

- `MODEL.BAYESIAN_TYPE` - Тип байесовского вывода: "none", "mc_dropout", "swag"
- `MODEL.BAYESIAN_SAMPLES` - Количество сэмплов для MC методов (default: 1)
- `MODEL.BAYESIAN_INFERENCE_DURING_TRAINING` - Использовать Bayesian inference во время eval в процессе обучения (default: False)
  - `False` (рекомендуется): Быстрый детерминированный eval во время training
  - `True`: Полный Bayesian inference во время eval (медленнее, но с uncertainty estimation)

### SWAG параметры

- `MODEL.SWAG.START_EPOCH` - Эпоха начала сбора статистики (default: 10)
- `MODEL.SWAG.UPDATE_FREQ` - Частота обновления статистики в итерациях (default: 5)
- `MODEL.SWAG.MAX_MODELS` - Максимальное количество снимков весов (default: 20)
- `MODEL.SWAG.SCALE` - Масштаб для sampling из posterior (default: 1.0)
- `MODEL.SWAG.RANK` - Ранг для low-rank covariance (default: 20)
- `MODEL.SWAG.NO_COV_MAT` - Использовать только diagonal covariance (default: False)

## Архитектура

### Файлы

1. **odin/config.py** - Конфигурационные параметры
2. **odin/modeling/bayesian/swag.py** - SWAG модуль
3. **odin/modeling/meta_arch/odin_head.py** - Inference логика
4. **my_train_odin.py** - Training hooks и checkpointing

### Как это работает

#### MC Dropout
- Dropout слои активируются во время inference
- Модель запускается N раз с разными dropout масками
- Предсказания усредняются по softmax вероятностям

#### SWAG
- **Scope**: Применяется только к **predictor** (transformer decoder + class_embed), не ко всей модели
- **Training**: Собирает статистику весов predictor (mean, variance, deviations)
- **Inference**: Сэмплирует веса predictor из N(w_SWA, Σ_SWAG) и усредняет предсказания
- **Covariance**: Σ = (1/2)(diag(σ²) + (1/(K-1))DD^T)
- **Эффективность**: ~10-20M параметров вместо ~100M+ (вся модель)

## Примеры использования в Notebook

См. `kaggle_my_train_odin.ipynb` - ячейки с примерами для всех трех режимов.

## Сохранение и загрузка

### SWAG State

SWAG статистика сохраняется автоматически:
- Файл: `OUTPUT_DIR/swag_state.pth`
- Частота: каждый checkpoint
- Загрузка: автоматически при `--resume`

### Checkpoint структура

```
output/
├── model_final.pth       # Веса модели
├── swag_state.pth        # SWAG статистика (если SWAG включен)
└── model_best.pth        # Лучшая модель по PQ
```

## Производительность

| Метод | Скорость | Память | Точность uncertainty |
|-------|----------|--------|---------------------|
| none | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | - |
| mc_dropout | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| swag | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## Troubleshooting

### SWAG не собирает статистику
- Проверьте, что `MODEL.BAYESIAN_TYPE = "swag"`
- Убедитесь, что текущая эпоха >= `SWAG.START_EPOCH`
- Проверьте логи: должно быть сообщение "SWAG: Starting weight collection"

### MC Dropout дает одинаковые результаты
- Убедитесь, что `MODEL.BAYESIAN_TYPE = "mc_dropout"`
- Проверьте, что `MODEL.MASK_FORMER.DROPOUT > 0`
- Новая реализация исправляет эту проблему

### SWAG inference падает с ошибкой
- Убедитесь, что модель обучена с SWAG (есть `swag_state.pth`)
- Проверьте, что `MODEL.BAYESIAN_TYPE = "swag"` при inference

## Дополнительная информация

Подробное математическое описание методов см. в `BAYESIAN_APPROACHES.md`.

## Авторы

Реализация SWAG и рефакторинг байесовских методов: Claude Code (2026)
Оригинальная архитектура ODIN: Kate Group
