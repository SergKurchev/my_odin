# 🚀 QUICK START: Быстрое Тестирование (1 час вместо 6)

**Обновлено:** 28 апреля 2026, 16:30  
**Время до первых результатов:** 1 час!

---

## ⚡ ПОЧЕМУ 1 ЧАС ДОСТАТОЧНО?

Из анализа OLD VERSION мы знаем, что на **итерации 863 (~1 час)**:
- **OLD:** PQ=23.11, mAP@50=30.33 ✅
- **PRETRAIN:** PQ=0.10, mAP@50=0.00 ❌

**Разница -23.01 уже видна через 1 час!**

**Критерий успеха:** PQ ≥ 20 на итерации 863

---

## 🎯 ПЛАН НА СЕГОДНЯ (16:30-18:00)

### Шаг 1: Создать Stable Branch (10 минут)

```bash
cd "C:\Users\NeverGonnaGiveYouUp\OneDrive\Рабочий стол\study_materials\Skoltech\projects\StrawPick\NBV_article\my_ODIN_for_strawberry\odin"

git checkout b13865b
git checkout -b stable-strawberry-v1

# Cherry-pick критичные фиксы
git cherry-pick 3edd802  # Auto-sync Kaggle outputs
git cherry-pick 78f0739  # Fix NaN crash (LR=5e-5) ⚠️
git cherry-pick 7c73a54  # Fix speed calculation
git cherry-pick 46f0520  # Auto-calculate SOLVER.STEPS
git cherry-pick c3ca406  # Revert NUM_OBJECT_QUERIES
git cherry-pick b229c31  # Fix disk overflow 🔥 КРИТИЧНО!
git cherry-pick 4a69382  # Fix TimeLimitHook
git cherry-pick 69f2f78  # Fix argparse conflict
git cherry-pick db3a8c7  # Fix double indexing
git cherry-pick c581f97  # Fix visualization 🔥 КРИТИЧНО!
git cherry-pick e9772b8  # Fix CSV logging

git push origin stable-strawberry-v1
```

**Если cherry-pick конфликтует:**
```bash
git cherry-pick --abort
# Пропустить и продолжить
```

---

### Шаг 2: Обновить Kaggle Notebook (5 минут)

**Файл:** `kaggle_my_train_odin.py`

**Изменение 1: Добавить поддержку branch**

```python
# Строка ~22
CONFIG = {
    "ODIN_DIR": "my_odin",
    "ODIN_REPO_URL": "https://github.com/SergKurchev/my_odin.git",
    "ODIN_BRANCH": "stable-strawberry-v1",  # ДОБАВИТЬ!
    "ODIN_WEIGHTS_URL": "https://huggingface.co/katefgroup/odin/resolve/main/scannet_resnet_47.8_73.3_32k_1.5k.pth",
    # ...
}
```

**Изменение 2: Клонирование с branch**

```python
# Строка ~130, в функции install_odin_dependencies()
if not os.path.exists(CONFIG["ODIN_DIR"]):
    print("\n0. Cloning ODIN...")
    subprocess.run([
        "git", "clone", "-q", 
        "-b", CONFIG["ODIN_BRANCH"],  # ДОБАВИТЬ!
        CONFIG["ODIN_REPO_URL"], 
        CONFIG["ODIN_DIR"]
    ], check=True)
```

**Изменение 3: Параметры для 1-часового теста**

```python
# Строка ~343
train_cmd = [
    VENV_PYTHON, "my_odin/my_train_odin.py",
    "--config-file", CONFIG_FILE,
    "--num-gpus", "1",
    "--dist-url", "tcp://127.0.0.1:23456",
    "--resume",
    "--visualize",
    "--dataset_dir", DATASET_DIR,
    "--splits_file", SPLITS_FILE,
    
    # БЫСТРЫЙ ТЕСТ: 1 час
    "--num_epochs", "3",           # Было: 10
    "--eval_period", "144",
    "--checkpoint_period", "144",
    "--max_time", "1.2",           # Было: 6
    
    # Параметры обучения
    "--image_size", "640",
    "--batch_size", "1",
    "--num_frames", "5",
    "--lr", "0.0001",              # ВЕРНУТЬ 1e-4 (было 5e-5)
    
    # Config overrides
    'MODEL.WEIGHTS', CONFIG["ODIN_WEIGHTS_PATH"],
    'OUTPUT_DIR', './output',
    'SOLVER.AMP.ENABLED', 'True',
    
    # БЕЗ SWAG! (закомментировать все SWAG параметры)
    # 'MODEL.BAYESIAN_TYPE', 'swag',
    # 'MODEL.BAYESIAN_SAMPLES', '10',
    # ...
]
```

---

### Шаг 3: Запустить на Kaggle (1 час)

1. Загрузить обновленный `kaggle_my_train_odin.py` на Kaggle
2. Запустить ноутбук
3. Ждать ~1 час

---

### Шаг 4: Проверить Результаты (5 минут)

**Открыть:** `output/metrics_comparison.csv`

**Найти строку с iteration=863:**

```csv
iteration,total_loss,PQ,mAP@50,...
863,XX.XX,YY.YY,ZZ.ZZ,...
```

**Критерии:**

| PQ на iter 863 | Статус | Действие |
|----------------|--------|----------|
| **≥ 20** | ✅ SUCCESS! | Переходим к Bayesian Inference |
| **15-20** | ⚠️ Медленно | Запустить еще на 1 час (до iter 1151) |
| **10-15** | ⚠️ Проблема | Запустить Тест 1.1 (откат LR) |
| **< 10** | ❌ Сломано | Запустить Тест 1.1 (откат LR) |

---

## 📊 ОЖИДАЕМЫЕ МЕТРИКИ (через 1 час)

### Здоровая траектория (как OLD):

| Итерация | PQ | mAP@50 | Статус |
|----------|-----|--------|--------|
| 143 | 0.03 | 0.00 | Начало |
| 287 | 0.18 | 0.10 | Разогрев |
| 431 | 1.10 | 0.15 | Рост |
| 575 | 5.03 | 3.61 | Критическая точка |
| 719 | 10.02 | 10.97 | Хороший знак |
| **863** | **23.11** | **30.33** | **SUCCESS!** ✅ |

### Сломанная траектория (как PRETRAIN):

| Итерация | PQ | mAP@50 | Статус |
|----------|-----|--------|--------|
| 143 | 0.00 | 0.00 | Начало |
| 287 | 0.00 | 0.00 | Нет роста! |
| 431 | 0.00 | 0.00 | Нет роста! |
| 575 | 0.08 | 0.10 | Очень медленно |
| 719 | 0.00 | 0.00 | Откат! |
| **863** | **0.10** | **0.00** | **СЛОМАНО** ❌ |

**Если видишь сломанную траекторию → останавливай тест досрочно!**

---

## 🔧 ЕСЛИ ТЕСТ 1.0 ПРОВАЛИЛСЯ (PQ < 20)

### Тест 1.1: Откатить LR change (еще 1 час)

```bash
cd odin
git checkout stable-strawberry-v1
git revert 78f0739  # Откатить LR=5e-5
git push origin stable-strawberry-v1
```

**В kaggle_my_train_odin.py:**
```python
"--lr", "0.0001",  # Явно указать 1e-4
```

**Запустить тест на 1 час.**

**Ожидание:** PQ ≥ 20 на итерации 863

---

### Тест 1.2: Минимальный baseline (если 1.1 не помог)

```bash
cd odin
git checkout b13865b
git checkout -b stable-minimal

# ТОЛЬКО критичные фиксы
git cherry-pick b229c31  # Disk cleanup
git cherry-pick c581f97  # Visualization fix

git push origin stable-minimal
```

**Обновить CONFIG["ODIN_BRANCH"] = "stable-minimal"**

**Запустить тест на 1 час.**

**Ожидание:** PQ ≥ 20 (должно работать, это почти чистый OLD)

---

## 📅 TIMELINE НА СЕГОДНЯ

| Время | Действие | Длительность |
|-------|----------|--------------|
| **16:30** | Создать stable branch | 10 мин |
| **16:40** | Обновить Kaggle notebook | 5 мин |
| **16:45** | Запустить Тест 1.0 | 1 час |
| **17:45** | Проверить результаты | 5 мин |
| **17:50** | Если успех → 🎉 | - |
| **17:50** | Если провал → Тест 1.1 | 1 час |
| **18:50** | Проверить результаты Теста 1.1 | 5 мин |

**Итого:** 2-3 часа до финального результата (вместо 6-12 часов!)

---

## 🎯 КРИТЕРИИ ПРИНЯТИЯ РЕШЕНИЙ

### После Теста 1.0:

**PQ ≥ 20:**
```
✅ SUCCESS! Stable baseline работает!
→ Завтра начинаем Bayesian Inference
```

**15 ≤ PQ < 20:**
```
⚠️ Работает, но медленнее
→ Запустить еще на 1 час (до iter 1151)
→ Ожидание: PQ ≥ 30
```

**PQ < 15:**
```
❌ Что-то не так
→ Запустить Тест 1.1 (откат LR)
```

---

### После Теста 1.1:

**PQ ≥ 20:**
```
✅ Проблема была в LR!
→ Используем LR=1e-4
→ Завтра начинаем Bayesian Inference
```

**PQ < 20:**
```
❌ Проблема не в LR
→ Запустить Тест 1.2 (минимальный baseline)
```

---

## 🚨 БЫСТРАЯ ДИАГНОСТИКА

### Смотреть на total_loss:

```
iteration: 143, total_loss: ~38-45 ✅ (нормально)
iteration: 287, total_loss: ~34-38 ✅ (снижается)
iteration: 575, total_loss: ~30-32 ✅ (продолжает снижаться)
iteration: 863, total_loss: ~23-26 ✅ (хороший знак)
```

**Если total_loss не снижается → проблема с оптимизацией!**

---

### Смотреть на траекторию PQ:

```
Iter 287: PQ > 0.10 ✅ (начинает учиться)
Iter 575: PQ > 3.00 ✅ (хороший знак)
Iter 863: PQ > 20.00 ✅ (SUCCESS!)
```

**Если PQ < 1.0 на итерации 575 → останавливай тест!**

---

## 📋 CHECKLIST НА СЕГОДНЯ

- [ ] 16:30 - Создать stable-strawberry-v1 branch
- [ ] 16:40 - Обновить kaggle_my_train_odin.py
- [ ] 16:45 - Запустить Тест 1.0 на Kaggle
- [ ] 17:45 - Проверить метрики на iter 863
- [ ] 17:50 - Если PQ ≥ 20 → SUCCESS! 🎉
- [ ] 17:50 - Если PQ < 20 → Запустить Тест 1.1

---

## 📚 ДОКУМЕНТАЦИЯ

- **Этот файл:** Быстрый старт на сегодня
- **FAST_TESTING_PLAN.md:** Полный план с быстрыми тестами
- **MIGRATION_PLAN.md:** Детальный план на 4 недели
- **FINAL_REPORT.md:** Результаты расследования

---

## 🎓 КЛЮЧЕВЫЕ МОМЕНТЫ

1. **1 час достаточно** для оценки (iter 863)
2. **PQ ≥ 20** = успех
3. **Disk cleanup критичен** (b229c31)
4. **Visualization fix критичен** (c581f97)
5. **LR может влиять** (проверить в Тесте 1.1)

---

**Начинай прямо сейчас! Через 1.5 часа узнаешь результат! 🚀**
