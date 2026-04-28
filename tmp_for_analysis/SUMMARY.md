# КРАТКАЯ СВОДКА РАССЛЕДОВАНИЯ

## ЧТО ПРОИЗОШЛО

### ❌ ЛОЖНЫЕ ПРОБЛЕМЫ
1. **"Метрики хуже"** - НЕТ! Все три CSV файла идентичны (PQ=43.97, mAP@50=61.85)

### ✅ РЕАЛЬНЫЕ ПРОБЛЕМЫ

1. **Uncertainty всегда 0.0**
   - Причина: `MODEL.BAYESIAN_INFERENCE_DURING_TRAINING=True` → детерминированный eval
   - Решение: Изменить на `False` или делать отдельный `--eval-only`

2. **TimeLimitHook не сработал (6 часов)**
   - Причина: `sys.exit(0)` не останавливает Detectron2 training loop
   - Решение: Заменить на `self.trainer.storage.iter = self.trainer.max_iter`

3. **Визуализации с неправильными классами (pretrain)**
   - Причина: Использовался коммит до c581f97 (баг с 1-indexed инстансами)
   - Решение: Использовать коммит c581f97 или позже

4. **HTML визуализации не сохранились (latest)**
   - Причина: Принудительная остановка (CANCEL) не дала времени
   - Решение: Генерировать HTML после каждого eval, не только в конце

5. **Resume не работает из Input**
   - Причина: Не копируется файл `last_checkpoint`
   - Решение: Добавить копирование `last_checkpoint` в `sync_previous_output()`

## ВРЕМЕННАЯ ЛИНИЯ

| Запуск | Дата | Коммит | Статус | PQ | Uncertainty |
|--------|------|--------|--------|----|----|
| OLD | 20 апр | b13865b | ✅ COMPLETE | 43.97 | 0.0 |
| PRETRAIN | 26 апр | d99b58d-1f226c6 | ✅ COMPLETE | 43.97 | 0.0 |
| LATEST | 27 апр | 72f45e6-3d3a94e | ❌ CANCEL | 43.97 | 0.0 |

## КЛЮЧЕВЫЕ КОММИТЫ

- `819a05b` (25 апр, 10:09) - Баг с 1-indexed инстансами
- `c581f97` (25 апр, 10:30) - **ИСПРАВЛЕНИЕ** визуализации (0-indexed)
- `d99b58d` (26 апр, 00:47) - **Добавлен Bayesian inference (SWAG + MC Dropout)**
- `72f45e6` (27 апр, 09:42) - Fix SWAG checkpoint loading

## РЕКОМЕНДАЦИИ

### 1. Создать Stable Branch
```bash
git checkout d99b58d  # Или позже
git checkout -b stable-strawberry-bayesian-v1
git push origin stable-strawberry-bayesian-v1
```

### 2. Исправить Uncertainty
В `kaggle_my_train_odin.py`, строка 377:
```python
'MODEL.BAYESIAN_INFERENCE_DURING_TRAINING', 'False',  # Было True
```

### 3. Исправить TimeLimitHook
В `my_train_odin.py`, строка 1227:
```python
# Вместо sys.exit(0):
self.trainer.storage.iter = self.trainer.max_iter
```

### 4. Исправить Resume
В `kaggle_my_train_odin.py`, функция `sync_previous_output()`:
```python
# Добавить копирование last_checkpoint
last_checkpoint_src = os.path.join(root, "last_checkpoint")
if os.path.exists(last_checkpoint_src):
    shutil.copy2(last_checkpoint_src, os.path.join(target_output, "last_checkpoint"))
```

## СЛЕДУЮЩИЕ ШАГИ

1. ✅ Создать stable branch на d99b58d или позже
2. ✅ Применить исправления (uncertainty, time limit, resume)
3. ✅ Запустить новый эксперимент
4. ✅ Проверить uncertainty > 0 в результатах
5. ✅ Проверить автоостановку через 6 часов
6. ✅ Проверить HTML визуализации

---

**Полный отчет:** См. `INVESTIGATION_REPORT.md`
