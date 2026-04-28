# 🚀 QUICK START: Восстановление Метрик + Bayesian Inference

**Статус:** ✅ План готов к выполнению  
**Время:** ~4 недели  
**Приоритет:** КРИТИЧЕСКИЙ

---

## 🎯 ЦЕЛЬ

Создать стабильную версию с:
1. ✅ Метриками как в OLD (PQ ≥ 50, mAP@50 ≥ 70)
2. ✅ Правильными визуализациями
3. ✅ Disk cleanup (нет переполнения)
4. ✅ Bayesian inference (SWAG + MC Dropout)
5. ✅ Uncertainty > 0

---

## ⚡ НЕМЕДЛЕННЫЕ ДЕЙСТВИЯ (Сегодня)

### Шаг 1: Создать Stable Branch (10 минут)

```bash
cd odin
git checkout b13865b
git checkout -b stable-strawberry-v1

# Cherry-pick критичные фиксы
git cherry-pick 3edd802  # Auto-sync Kaggle outputs
git cherry-pick 78f0739  # Fix NaN crash (LR=5e-5)
git cherry-pick 7c73a54  # Fix speed calculation
git cherry-pick 46f0520  # Auto-calculate SOLVER.STEPS
git cherry-pick c3ca406  # Revert NUM_OBJECT_QUERIES
git cherry-pick b229c31  # Fix disk overflow (КРИТИЧНО!)
git cherry-pick 4a69382  # Fix TimeLimitHook
git cherry-pick 69f2f78  # Fix argparse conflict
git cherry-pick db3a8c7  # Fix double indexing
git cherry-pick c581f97  # Fix visualization (КРИТИЧНО!)
git cherry-pick e9772b8  # Fix CSV logging

# Push
git push origin stable-strawberry-v1
```

**ВАЖНО:** Если cherry-pick конфликтует - пропустить и разобраться позже.

---

### Шаг 2: Запустить Тест на Kaggle (30 минут)

**Обновить kaggle_my_train_odin.py:**

```python
# Строка с клонированием репозитория
CONFIG = {
    "ODIN_DIR": "my_odin",
    "ODIN_REPO_URL": "https://github.com/SergKurchev/my_odin.git",
    "ODIN_BRANCH": "stable-strawberry-v1",  # ДОБАВИТЬ!
    # ...
}

# В функции install_odin_dependencies():
if not os.path.exists(CONFIG["ODIN_DIR"]):
    print("\n0. Cloning ODIN...")
    subprocess.run([
        "git", "clone", "-q", "-b", CONFIG["ODIN_BRANCH"],  # ДОБАВИТЬ -b!
        CONFIG["ODIN_REPO_URL"], CONFIG["ODIN_DIR"]
    ], check=True)
```

**Параметры обучения:**

```python
train_cmd = [
    # ...
    "--num_epochs", "10",
    "--eval_period", "144",
    "--checkpoint_period", "144",
    "--max_time", "6",
    "--batch_size", "1",
    "--num_frames", "5",
    "--image_size", "640",
    "--lr", "0.0001",  # ВЕРНУТЬ 1e-4 (было 5e-5)
    
    # БЕЗ SWAG!
    # 'MODEL.BAYESIAN_TYPE', 'none',  # Закомментировать все SWAG параметры
]
```

**Запустить на Kaggle и ждать результатов.**

---

## 📊 ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ

### Тест 1: Stable Baseline (Сегодня-Завтра)

**Критерии успеха:**
- ✅ PQ ≥ 50 (целевое: 52.80)
- ✅ mAP@50 ≥ 70 (целевое: 72.32)
- ✅ Нет переполнения диска
- ✅ Правильные визуализации (цвета не смещены)
- ✅ Автоостановка через 6 часов

**Если метрики ниже 50:**

1. **Откатить LR change:**
   ```bash
   git revert 78f0739
   # Вернуть LR=1e-4 в kaggle скрипте
   ```

2. **Проверить CAMERA_DROP:**
   ```bash
   # Если был применен 3d0ea94, откатить:
   git revert 3d0ea94
   ```

3. **Повторить тест**

---

## 🔬 СЛЕДУЮЩИЕ ШАГИ (После успеха Теста 1)

### Неделя 2: Bayesian Inference (Правильная реализация)

**Создать новую ветку:**

```bash
git checkout stable-strawberry-v1
git checkout -b feature/bayesian-inference-v2
```

**Реализовать BayesianInferenceWrapper** (см. MIGRATION_PLAN.md, Фаза 3)

**Ключевые принципы:**
1. ❌ НЕ изменять odin_head.py
2. ✅ Wrapper поверх модели
3. ✅ Bayesian inference только для eval
4. ✅ Детерминированный путь нетронут

---

## 📋 QUICK CHECKLIST

### Сегодня (28 апреля)
- [ ] Создать stable-strawberry-v1 branch
- [ ] Cherry-pick критичные коммиты
- [ ] Push в remote
- [ ] Обновить kaggle_my_train_odin.py (добавить branch)
- [ ] Запустить Тест 1 на Kaggle

### Завтра (29 апреля)
- [ ] Проверить результаты Теста 1
- [ ] Если PQ < 50 → откатить LR change
- [ ] Если PQ ≥ 50 → SUCCESS! 🎉

### Неделя 2 (5-11 мая)
- [ ] Реализовать BayesianInferenceWrapper
- [ ] Тест 2.1: Детерминированный (должен быть = Тест 1)
- [ ] Тест 2.2: MC Dropout (PQ ≥ 48, Uncertainty > 0)

### Неделя 3 (12-18 мая)
- [ ] Тест 2.3: SWAG (PQ ≥ 48, Uncertainty > 0)
- [ ] Unit tests
- [ ] Улучшения (TimeLimitHook, Resume)

---

## 🚨 КРИТИЧНЫЕ МОМЕНТЫ

### 1. Disk Overflow Fix (b229c31)
**БЕЗ ЭТОГО KAGGLE УПАДЕТ!**

Проверить, что в коде есть:
```python
class CheckpointCleanupHook(hooks.HookBase):
    def __init__(self, output_dir, keep_last=2):
        # Хранит только последние 2 чекпоинта
```

### 2. Visualization Fix (c581f97)
**БЕЗ ЭТОГО ЦВЕТА НЕПРАВИЛЬНЫЕ!**

Проверить, что в коде:
```python
point_pred_inst[m] = inst_idx  # 0-indexed (НЕ inst_idx + 1!)
```

### 3. LR Value
**МОЖЕТ ВЛИЯТЬ НА МЕТРИКИ!**

- OLD использовал: LR=1e-4
- 78f0739 изменил на: LR=5e-5

Если метрики низкие → вернуть 1e-4

---

## 📞 ЕСЛИ ЧТО-ТО ПОШЛО НЕ ТАК

### Проблема: Cherry-pick конфликтует

**Решение:**
```bash
# Пропустить проблемный коммит
git cherry-pick --abort

# Продолжить с остальными
# Вернуться к проблемному позже
```

### Проблема: Метрики низкие (PQ < 50)

**Решение:**
1. Откатить 78f0739 (LR change)
2. Откатить 3d0ea94 (CAMERA_DROP) если был применен
3. Откатить 4b0d4e4 (architecture) если был применен
4. Повторить тест

### Проблема: Kaggle падает с OOM

**Решение:**
```python
# Уменьшить параметры:
"--image_size", "512",  # Было 640
"--num_frames", "3",    # Было 5
```

---

## 🎓 КЛЮЧЕВЫЕ ИНСАЙТЫ

1. **Коммит d99b58d сломал обучение** (-17% метрик)
2. **Проблема в изменениях odin_head.py** (224 строки)
3. **Нужен wrapper подход** для Bayesian inference
4. **Disk cleanup критичен** для Kaggle
5. **Visualization fix критичен** для правильных цветов

---

## 📚 ДОКУМЕНТАЦИЯ

- **Полный план:** `MIGRATION_PLAN.md`
- **Детальное расследование:** `FINAL_REPORT.md`
- **Краткая сводка:** `CRITICAL_SUMMARY.md`

---

**Начинай с Шага 1! Удачи! 🚀**
