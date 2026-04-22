#%% [markdown]
# # Обучение ODIN на синтетическом датасете Strawberry Multi-view
# Этот скрипт настраивает окружение (собирает CUDA ядра и тяжелые зависимости) 
# в изолированном виртуальном окружении и запускает обучение.

#%% [code]
import os
import shutil
# Чистим старое окружение перед запуском (запрос пользователя)
if os.path.exists("venv"):
    print("Cleaning up old venv...")
    shutil.rmtree("venv")

import subprocess
import sys
import urllib.request

# Базовая конфигурация проекта
CONFIG = {
    "ODIN_DIR": "my_odin",
    "ODIN_REPO_URL": "https://github.com/SergKurchev/my_odin.git",
    # Публичные веса из katefgroup/odin (открытый репозиторий!)
    "ODIN_WEIGHTS_URL": "https://huggingface.co/katefgroup/odin/resolve/main/scannet_resnet_47.8_73.3_32k_1.5k.pth",
    "ODIN_WEIGHTS_PATH": "my_odin/models/odin_scannet_context.pth",
    "M2F_WEIGHTS_URL": "https://huggingface.co/katefgroup/odin/resolve/main/m2f_coco.pkl",
    "M2F_WEIGHTS_PATH": "my_odin/models/model_final_5c90d4.pkl",
}

#%% [markdown]
# ## 1. Создание виртуального окружения (VENV) и установка зависимостей
# Kaggle предоставляет базовый контейнер, но мы создадим виртуальное окружение, 
# чтобы точно контролировать версии PyTorch (2.2.0), CUDA (12.1) и NumPy.

#%% [code]
# 1. Создаем venv (даже если ensurepip нет, структура папок создастся)
if not os.path.exists("venv"):
    print("Устанавливаю Python 3.10...")
    subprocess.run(["apt-get", "update", "-y"], check=False)
    subprocess.run(["apt-get", "install", "-y", "python3.10", "python3.10-venv", "python3.10-dev", "python3.10-distutils"], check=False)

    print("Создаю виртуальное окружение на Python 3.10...")
    # Флаг --without-pip предотвращает попытки использовать отсутствующий ensurepip
    subprocess.run(["python3.10", "-m", "venv", "venv", "--without-pip"], check=True)

    VENV_PYTHON_TMP = os.path.abspath("venv/bin/python")

    # Прямая установка pip через официальный скрипт
    print("Загружаю get-pip.py для ручной установки...")
    urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", "get-pip.py")

    print("Устанавливаю pip в виртуальное окружение...")
    subprocess.run([VENV_PYTHON_TMP, "get-pip.py"], check=True)

    # Чистим за собой
    if os.path.exists("get-pip.py"):
        os.remove("get-pip.py")
    print("Pip успешно установлен!")

# Теперь пути будут корректными
VENV_PYTHON = os.path.abspath("venv/bin/python")
VENV_PIP = os.path.abspath("venv/bin/pip")


def make_venv_env(extra=None):
    """
    Создаёт словарь переменных окружения с правильными путями к venv.

    КЛЮЧЕВОЕ ОТЛИЧИЕ от системной среды (reference):
    В системной среде pip/python берутся из PATH и уже знают о системных пакетах.
    В venv — нужно явно прописать VIRTUAL_ENV и PATH, иначе subprocess-ы,
    запущенные из подпроцессов (например, build isolation в pip),
    не будут знать, что они внутри venv и не найдут torch.
    """
    env = os.environ.copy()
    venv_dir = os.path.abspath("venv")
    env["VIRTUAL_ENV"] = venv_dir
    env["PATH"] = os.path.join(venv_dir, "bin") + ":" + env.get("PATH", "")
    # Убираем PYTHONPATH системы — он может мешать изоляции
    env.pop("PYTHONPATH", None)
    if extra:
        env.update(extra)
    return env


def clean_build_artifacts(directory):
    """
    Очищает старые артефакты сборки перед компиляцией.

    Ref-функция это делает явно — без очистки повторный запуск
    может завершиться ошибкой из-за конфликта старых .egg файлов.
    """
    import shutil
    import glob
    for d in ["build", "dist"]:
        path = os.path.join(directory, d)
        if os.path.exists(path):
            print(f"   Cleaning {path}...")
            shutil.rmtree(path)
    for egg in glob.glob(os.path.join(directory, "*.egg-info")):
        print(f"   Cleaning {egg}...")
        shutil.rmtree(egg)


def run_in_venv(cmd, cwd=None, env=None, check=True):
    """Обертка для запуска команд внутри созданного виртуального окружения."""
    if cmd[0] == "pip":
        cmd[0] = VENV_PIP
    elif cmd[0] == "python":
        cmd[0] = VENV_PYTHON

    print(f"Выполняю: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, env=env, check=check)


def install_odin_dependencies():
    """Установка зависимостей ODIN в виртуальное окружение."""
    print("\n" + "=" * 80)
    print("Installing ODIN Dependencies into VENV")
    print("=" * 80)

    venv_env = make_venv_env()

    # 0. Клонирование репозитория
    if not os.path.exists(CONFIG["ODIN_DIR"]):
        print("\n0. Cloning ODIN...")
        subprocess.run(["git", "clone", "-q", CONFIG["ODIN_REPO_URL"], CONFIG["ODIN_DIR"]], check=True)

    # 1. PyTorch 2.2.0 + CUDA 12.1
    print("\n1. Installing PyTorch 2.2.0...")
    run_in_venv([
        "pip", "install", "-q", "torch==2.2.0", "torchvision==0.17.0",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ], env=venv_env)

    run_in_venv([
        "pip", "install", "-q", "torch-scatter",
        "-f", "https://data.pyg.org/whl/torch-2.2.0+cu121.html"
    ], env=venv_env)

    # 2. NumPy < 2 и Pillow
    print("\n2. Installing NumPy < 2 + Pillow...")
    run_in_venv(["pip", "install", "-q", "numpy<2", "--force-reinstall"], env=venv_env)
    run_in_venv(["pip", "install", "-q", "Pillow>=10.2.0"], env=venv_env)

    # 3. Clean requirements (через Python — надежнее чем sed)
    print("\n3. Cleaning ODIN requirements...")
    req_path = os.path.join(CONFIG["ODIN_DIR"], "requirements.txt")

    with open(req_path, 'r') as f:
        lines = f.readlines()

    with open(req_path, 'w') as f:
        for line in lines:
            line_clean = line.strip().lower()
            # Пропускаем все проблемные пакеты
            if any(x in line_clean for x in ["waspinator", "detectron2", "pytorch3d"]):
                print(f"   Skipping: {line.strip()}")
                continue
            # Фикс для старого PyYAML
            if "pyyaml==5.3.1" in line_clean:
                f.write("pyyaml>=5.4.1\n")
            else:
                f.write(line)

    # 4. Build tools + Modern COCO API (нужны перед requirements.txt)
    print("\n4. Installing Build Tools & Modern dependencies...")
    run_in_venv(["pip", "install", "-q", "cython", "setuptools", "wheel", "pycocotools"], env=venv_env)

    # 5. ODIN requirements + ninja, fvcore, iopath
    print("\n5. Installing ODIN requirements...")
    run_in_venv(["pip", "install", "-q", "-r", req_path], env=venv_env)
    # Фикс: откатываем transformers до версии, совместимой с Torch 2.2.0, чтобы не было ошибки "PyTorch not found"
    run_in_venv(["pip", "install", "-q", "transformers==4.38.2"], env=venv_env)
    run_in_venv(["pip", "install", "-q", "ninja", "fvcore", "iopath"], env=venv_env)

    # 6. Detectron2
    # КЛЮЧЕВОЕ ОТЛИЧИЕ от системной среды:
    # В системной среде pip видит torch в системных пакетах Kaggle.
    # В нашем venv pip должен использовать torch из venv,
    # поэтому нужен --no-build-isolation (запрет временного пустого build env)
    # + явно прокидываем venv_env с правильным VIRTUAL_ENV и PATH.
    print("\n6. Installing Detectron2...")
    run_in_venv([
        "pip", "install", "-q", "--no-build-isolation",
        "git+https://github.com/facebookresearch/detectron2.git"
    ], env=venv_env)

    # 7. PyTorch3D
    print("\n7. Installing PyTorch3D (with CUDA)...")
    cuda_env = make_venv_env({
        "FORCE_CUDA": "1",
        "TORCH_CUDA_ARCH_LIST": "6.0;7.0;7.5;8.0;8.6",
    })
    run_in_venv([
        "pip", "install", "-q", "--no-build-isolation",
        "git+https://github.com/facebookresearch/pytorch3d.git"
    ], env=cuda_env)

    # 8. Критически важно: переустановка правильных версий NumPy и OpenCV
    # (другие пакеты могут их обновить в процессе установки)
    print("\n8. Re-installing correct NumPy + OpenCV...")
    run_in_venv(["pip", "uninstall", "-y", "-q", "numpy"], env=venv_env)
    run_in_venv(["pip", "install", "-q", "numpy==1.26.4"], env=venv_env)
    run_in_venv(["pip", "install", "-q", "opencv-python-headless==4.8.0.76"], env=venv_env)

    # 9. Компиляция CUDA-ядер pointops2
    # КЛЮЧЕВОЕ ОТЛИЧИЕ:
    # Ref-функция делает os.chdir() + setup.py install --user.
    # --user ставит в ~/.local, откуда системный python видит, но venv — нет!
    # Мы передаём cwd= и VENV_PYTHON напрямую, БЕЗ --user.
    # Очистка артефактов — как в ref-функции.
    print("\n9. Compiling CUDA kernels (pointops2)...")
    pointops_dir = os.path.abspath(os.path.join(CONFIG["ODIN_DIR"], "libs", "pointops2"))
    clean_build_artifacts(pointops_dir)
    run_in_venv(["python", "setup.py", "install"], cwd=pointops_dir, env=cuda_env)

    # 10. Компиляция CUDA-ядер deformable attention
    print("\n10. Compiling CUDA kernels (deformable attention)...")
    deform_dir = os.path.abspath(os.path.join(CONFIG["ODIN_DIR"], "odin", "modeling", "pixel_decoder", "ops"))
    clean_build_artifacts(deform_dir)
    run_in_venv(["python", "setup.py", "build", "install"], cwd=deform_dir, env=cuda_env)

    print("\n" + "=" * 80)
    print("Installation complete!")
    print("=" * 80)


def download_weights():
    """Скачивание предобученных весов с публичных URL (katefgroup/odin)."""
    print("\n" + "=" * 80)
    print("Downloading Weights")
    print("=" * 80)

    os.makedirs(os.path.dirname(CONFIG["ODIN_WEIGHTS_PATH"]), exist_ok=True)

    def download_file(url, dest_path, min_size_mb=50):
        """Скачивает файл и проверяет целостность по размеру."""
        size_mb = os.path.getsize(dest_path) / (1024 * 1024) if os.path.exists(dest_path) else 0
        if size_mb >= min_size_mb:
            print(f"   ✓ Файл уже скачан ({size_mb:.0f} MB): {dest_path}")
            return
        if os.path.exists(dest_path):
            print(f"   ⚠️  Файл повреждён ({size_mb:.1f} MB), удаляю...")
            os.remove(dest_path)
        print(f"   Скачиваю {url} ...")
        ret = os.system(f"wget --tries=3 --retry-connrefused -c '{url}' -O '{dest_path}'")
        if ret != 0:
            print(f"   ❌ wget завершился с кодом {ret}, пробую curl...")
            os.system(f"curl -L --retry 3 '{url}' -o '{dest_path}'")
        size_mb = os.path.getsize(dest_path) / (1024 * 1024) if os.path.exists(dest_path) else 0
        print(f"   ✓ Скачано: {dest_path} ({size_mb:.0f} MB)")
        if size_mb < min_size_mb:
            raise RuntimeError(f"Файл слишком мал ({size_mb:.1f} MB) — скачивание провалилось!")

    print("\n1. Downloading ODIN weights (ScanNet ResNet50)...")
    download_file(CONFIG["ODIN_WEIGHTS_URL"], CONFIG["ODIN_WEIGHTS_PATH"], min_size_mb=100)

    print("\n2. Downloading Mask2Former weights (M2F COCO ResNet)...")
    download_file(CONFIG["M2F_WEIGHTS_URL"], CONFIG["M2F_WEIGHTS_PATH"], min_size_mb=50)

    print("\nWeights ready!")


# Запускаем сборку
install_odin_dependencies()
download_weights()

#%% [markdown]
# ## 3. Запуск обучения (my_train_odin.py)
# Теперь запустим обучение, используя наш venv-python интерпретатор напрямую.

#%% [code]
# ── Диагностика датасета (запускается перед обучением) ────────────────────────
import json, os
from pathlib import Path

DATASET_DIR = "/kaggle/input/datasets/sergeykurchev/strawpick-sint-pointnetseg-test/multiview_dataset/multiview_dataset"
SPLITS_FILE = "/kaggle/input/datasets/sergeykurchev/strawpick-sint-pointnetseg-test/splits.json"

print("=== Диагностика путей ===")
print(f"DATASET_DIR exists: {os.path.exists(DATASET_DIR)}")
print(f"SPLITS_FILE exists: {os.path.exists(SPLITS_FILE)}")

if os.path.exists(DATASET_DIR):
    entries = os.listdir(DATASET_DIR)
    print(f"\nПервые 10 элементов в DATASET_DIR ({len(entries)} всего):")
    for e in sorted(entries)[:10]:
        print(f"  {e}/")

if os.path.exists(SPLITS_FILE):
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    for split_name, ids in splits.items():
        print(f"\nСплит '{split_name}': {len(ids)} элементов")
        print(f"  Первые 5 ID: {ids[:5]}")
        
        # Проверим, какой из форматов путей реально существует
        if ids:
            sid = ids[0]
            p1 = Path(DATASET_DIR) / str(sid)
            p2 = Path(DATASET_DIR) / f"sample_{str(sid).zfill(5)}"
            p3 = Path(DATASET_DIR) / f"{str(sid).zfill(5)}"
            print(f"  Пробую путь '{p1}': {'✓' if p1.exists() else '✗'}")
            print(f"  Пробую путь '{p2}': {'✓' if p2.exists() else '✗'}")
            print(f"  Пробую путь '{p3}': {'✓' if p3.exists() else '✗'}")

#%% [code]
CONFIG_FILE = "my_odin/configs/scannet_context/3d.yaml"

train_cmd = [
    VENV_PYTHON, "my_odin/my_train_odin.py",
    "--config-file", CONFIG_FILE,
    "--num-gpus", "1",
    "--dist-url", "tcp://127.0.0.1:23456",
    "--resume",                # Автоматически продолжит с последнего чекпоинта в OUTPUT_DIR
    "--visualize",             # Генерация HTML-визуализаций для проверочных сэмплов
    "--dataset_dir", DATASET_DIR,
    "--splits_file", SPLITS_FILE,
    
    # ПАРАМЕТРЫ ОБУЧЕНИЯ
    "--num_epochs", "10",      # Общее кол-во эпох
    "--eval_period", "144",     # Валидация каждую эпоху (если в эпохе ~144 итерации)
    "--checkpoint_period", "144", # Сохранение чекпоинта каждую эпоху
    
    # ОГРАНИЧЕНИЕ ВРЕМЕНИ (ВАЖНО ДЛЯ KAGGLE)
    "--max_time", "6",       # Авто-остановка через 6 часов (чтобы успеть сохранить веса)
    
    # ПАРАМЕТРЫ РЕСУРСОВ (Memory optimization)
    "--image_size", "640",    
    "--batch_size", "1",
    "--num_frames", "5",        # 5 кадров при 640px могут потребовать много VRAM
    "--lr", "0.0001",
    
    # ТЕХНИЧЕСКИЕ ПАРАМЕТРЫ (Config Overrides)
    # Важно: эти параметры БЕЗ черточек должны идти в самом конце списка
    "MODEL.WEIGHTS", CONFIG["ODIN_WEIGHTS_PATH"],
    "OUTPUT_DIR", "./output",
    "SOLVER.AMP.ENABLED", "True",
    "MODEL.MASK_FORMER.DEC_LAYERS", "4",
    
    # Исправление несовпадения весов предобученной модели (ScanNet -> Strawberry)
    "MODEL.MASK_FORMER.NUM_OBJECT_QUERIES", "100", # Соответствует весам в чекпоинте (было 20)
    "USE_MLP_POSITIONAL_ENCODING", "True",         # Соответствует Linear слою в чекпоинте (было Conv1d)
    "SOLVER.STEPS", "[]",                          # Убираем предупреждение о шагах обучения
]

print("Starting training script...")
# Используем run_in_venv с пробросом venv_env, чтобы окружение было полностью корректным
venv_env = make_venv_env()
run_in_venv(train_cmd, env=venv_env)

#%% [markdown]
# Вывод результатов (метрики) сохраняется в `output/metrics_comparison.csv`
