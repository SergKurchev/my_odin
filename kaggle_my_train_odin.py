#%% [markdown]
# # Обучение ODIN на синтетическом датасете Strawberry Multi-view
# Этот скрипт настраивает окружение (собирает CUDA ядра и тяжелые зависимости) 
# в изолированном виртуальном окружении и запускает обучение.

#%% [code]
import os
import subprocess
import sys

# Базовая конфигурация проекта
CONFIG = {
    "ODIN_DIR": "my_odin",
    "ODIN_REPO_URL": "https://github.com/SergKurchev/my_odin.git",
    "ODIN_WEIGHTS_URL": "https://huggingface.co/ayushjain1144/odin/resolve/main/odin_scannet_context.pth",
    "ODIN_WEIGHTS_PATH": "my_odin/models/odin_scannet_context.pth",
    "M2F_WEIGHTS_URL": "https://huggingface.co/ayushjain1144/odin/resolve/main/model_final_5c90d4.pkl",
    "M2F_WEIGHTS_PATH": "my_odin/models/model_final_5c90d4.pkl",
}

#%% [markdown]
# ## 1. Создание виртуального окружения (VENV) и установка зависимостей
# Kaggle предоставляет базовый контейнер, но мы создадим виртуальное окружение, 
# чтобы точно контролировать версии PyTorch (2.2.0), CUDA (12.1) и NumPy.

#%% [code]
# 1. Создаем venv
if not os.path.exists("venv"):
    print("Создаю виртуальное окружение...")
    subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)

# Формируем полные абсолютные пути к pip и python внутри venv
VENV_PYTHON = os.path.abspath("venv/bin/python")
VENV_PIP = os.path.abspath("venv/bin/pip")

def run_in_venv(cmd, cwd=None, env=None, check=True):
    """Обертка для запуска команд внутри созданного виртуального окружения."""
    if cmd[0] == "pip":
        cmd[0] = VENV_PIP
    elif cmd[0] == "python":
        cmd[0] = VENV_PYTHON
    
    print(f"Выполняю: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, env=env, check=check)


def install_odin_dependencies():
    """Установка точных зависимостей из требований пользователя."""
    print("\n" + "=" * 80)
    print("Installing ODIN Dependencies into VENV")
    print("=" * 80)
    
    # 0. Клонирование репозитория
    if not os.path.exists(CONFIG["ODIN_DIR"]):
        print("\n0. Cloning ODIN...")
        subprocess.run(["git", "clone", "-q", CONFIG["ODIN_REPO_URL"], CONFIG["ODIN_DIR"]], check=True)
        
    # 1. PyTorch 2.2.0 + CUDA 12.1
    print("\n1. Installing PyTorch 2.2.0...")
    run_in_venv([
        "pip", "install", "-q", "torch==2.2.0", "torchvision==0.17.0",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ])
    
    run_in_venv([
        "pip", "install", "-q", "torch-scatter",
        "-f", "https://data.pyg.org/whl/torch-2.2.0+cu121.html"
    ])
    
    # 2. NumPy < 2
    print("\n2. Installing NumPy < 2...")
    run_in_venv(["pip", "install", "-q", "numpy<2", "--force-reinstall"])
    
    # 4. Clean requirements
    print("\n4. Cleaning ODIN requirements...")
    req_path = os.path.join(CONFIG["ODIN_DIR"], "requirements.txt")
    subprocess.run(["sed", "-i", "s/pyyaml==5.3.1/pyyaml>=5.4.1/gi", req_path], check=True)
    subprocess.run(["sed", "-i", "/detectron2/d", req_path], check=True)
    subprocess.run(["sed", "-i", "/pytorch3d/d", req_path], check=True)
    
    # 5. Install ODIN requirements
    print("\n5. Installing ODIN requirements...")
    run_in_venv(["pip", "install", "-q", "-r", req_path])
    run_in_venv(["pip", "install", "-q", "ninja", "fvcore", "iopath"])
    
    # 6. Detectron2
    print("\n6. Installing Detectron2...")
    run_in_venv(["pip", "install", "-q", "git+https://github.com/facebookresearch/detectron2.git"])
    
    # 7. PyTorch3D
    print("\n7. Installing PyTorch3D...")
    env = os.environ.copy()
    env["FORCE_CUDA"] = "1"
    run_in_venv(["pip", "install", "-q", "git+https://github.com/facebookresearch/pytorch3d.git"], env=env)
    
    # 8. Fix NumPy
    print("\n8. Re-installing NumPy 1.26.4...")
    run_in_venv(["pip", "uninstall", "-y", "-q", "numpy"])
    run_in_venv(["pip", "install", "-q", "numpy==1.26.4"])
    
    # 9. Compile CUDA kernels
    print("\n9. Compiling CUDA kernels...")
    pointops_dir = os.path.join(CONFIG["ODIN_DIR"], "libs", "pointops2")
    env["TORCH_CUDA_ARCH_LIST"] = "6.0;7.0;7.5;8.0;8.6;8.9;9.0"
    
    # pointops2 setup
    run_in_venv(["python", "setup.py", "install", "--user"], cwd=pointops_dir, env=env)
    
    # deform ops setup
    deform_dir = os.path.join(CONFIG["ODIN_DIR"], "odin", "modeling", "pixel_decoder", "ops")
    run_in_venv(["python", "setup.py", "build", "install", "--user"], cwd=deform_dir, env=env)
    
    print("\n" + "=" * 80)
    print("Installation complete!")
    print("=" * 80)

def download_weights():
    """Скачивание предобученных весов."""
    print("\n" + "=" * 80)
    print("Downloading Weights")
    print("=" * 80)
    
    os.makedirs(os.path.dirname(CONFIG["ODIN_WEIGHTS_PATH"]), exist_ok=True)
    
    if not os.path.exists(CONFIG["ODIN_WEIGHTS_PATH"]):
        print("\n1. Downloading ODIN weights...")
        os.system(f"wget -q {CONFIG['ODIN_WEIGHTS_URL']} -O {CONFIG['ODIN_WEIGHTS_PATH']}")
        print(f"   Downloaded: {CONFIG['ODIN_WEIGHTS_PATH']}")
    
    if not os.path.exists(CONFIG["M2F_WEIGHTS_PATH"]):
        print("\n2. Downloading M2F weights...")
        os.system(f"wget -q {CONFIG['M2F_WEIGHTS_URL']} -O {CONFIG['M2F_WEIGHTS_PATH']}")
        print(f"   Downloaded: {CONFIG['M2F_WEIGHTS_PATH']}")
    
    print("\nWeights ready!")

# Запускаем сборку
install_odin_dependencies()
download_weights()

#%% [markdown]
# ## 3. Запуск обучения (my_train_odin.py)
# Теперь запустим обучение, используя наш venv-python интерпретатор напрямую.

#%% [code]
DATASET_DIR = "/kaggle/input/datasets/sergeykurchev/strawpick-sint-pointnetseg-test/multiview_dataset"
SPLITS_FILE = "/kaggle/input/datasets/sergeykurchev/strawpick-sint-pointnetseg-test/splits.json"

CONFIG_FILE = "my_odin/odin/configs/scannet_context/maskformer2_R50_bs16_50ep.yaml"

train_cmd = [
    VENV_PYTHON, "my_odin/odin/my_train_odin.py",
    "--config-file", CONFIG_FILE,
    "--num-gpus", "1",
    "--dataset_dir", DATASET_DIR,
    "--splits_file", SPLITS_FILE,
    # Если хотим использовать веса ResNet (закомментировано)
    # "MODEL.WEIGHTS", "detectron2://ImageNetPretrained/torchvision/R-50.pkl", 
    # Если хотим использовать готовые веса (раскомментировано)
    "MODEL.WEIGHTS", CONFIG["ODIN_WEIGHTS_PATH"], 
    "OUTPUT_DIR", "./output"
]

print("Starting training script...")
subprocess.run(train_cmd, check=True)

#%% [markdown]
# Вывод результатов (метрики) сохраняется в `output/metrics_comparison.csv`
