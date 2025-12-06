"""
Configuration file for the NTO ML competition baseline.
"""

from pathlib import Path
import sys

try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ PyTorch CUDA is available. GPU count: {torch.cuda.device_count()}")
        print(f"   GPU 0: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ PyTorch CUDA is NOT available. Using CPU.")
except ImportError:
    torch = None
    print("❌ PyTorch not installed. Using CPU.")

from . import constants

# --- DIRECTORIES ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

# --- GPU CONFIG ---
USE_GPU = True  # Включить использование GPU
GPU_ID = 0  # ID GPU устройства

# Определяем устройство для PyTorch (BERT)
if torch is not None and torch.cuda.is_available() and USE_GPU:
    BERT_DEVICE = f"cuda:{GPU_ID}"
    # Настраиваем видимые GPU
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    print(f"✅ Using GPU {GPU_ID} for PyTorch: {torch.cuda.get_device_name(GPU_ID)}")
else:
    BERT_DEVICE = "cpu"
    print("⚠️  Using CPU for PyTorch")

# Определяем устройство для LightGBM
if USE_GPU:
    # Проверяем наличие GPU для LightGBM
    try:
        import lightgbm as lgb
        # Тестовая проверка GPU для LightGBM
        LGB_DEVICE = "gpu"
        print("✅ LightGBM will use GPU")
    except ImportError:
        LGB_DEVICE = "cpu"
        print("⚠️  LightGBM not installed, will use CPU")
else:
    LGB_DEVICE = "cpu"
    print("⚠️  GPU disabled in config, using CPU for LightGBM")

# --- PARAMETERS ---
N_SPLITS = 5
RANDOM_STATE = 42
TARGET = constants.COL_RELEVANCE

# --- TEMPORAL SPLIT CONFIG ---
TEMPORAL_SPLIT_RATIO = 0.8

# --- TRAINING CONFIG ---
EARLY_STOPPING_ROUNDS = 50
MODEL_FILENAME_PATTERN = "lgb_fold_{fold}.txt"
MODEL_FILENAME = "lgb_model.txt"

# --- TF-IDF PARAMETERS ---
TFIDF_MAX_FEATURES = 500
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 2)

# --- BERT PARAMETERS ---
BERT_MODEL_NAME = constants.BERT_MODEL_NAME
BERT_BATCH_SIZE = 32 if BERT_DEVICE.startswith("cuda") else 8  # Больше на GPU
BERT_MAX_LENGTH = 512
BERT_EMBEDDING_DIM = 768

# --- FEATURES ---
CAT_FEATURES = [
    constants.COL_USER_ID,
    constants.COL_BOOK_ID,
    constants.COL_GENDER,
    constants.COL_AGE,
    constants.COL_AUTHOR_ID,
    constants.COL_PUBLICATION_YEAR,
    constants.COL_LANGUAGE,
    constants.COL_PUBLISHER,
]

# --- MODEL PARAMETERS ---
# LightGBM с поддержкой GPU
LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "num_leaves": 31,
    "verbose": -1,
    "n_jobs": -1,
    "seed": RANDOM_STATE,
    "boosting_type": "gbdt",
    # GPU параметры
    "device": LGB_DEVICE,
    "gpu_platform_id": 0,
    "gpu_device_id": GPU_ID,
    # Оптимизации памяти
    "max_bin": 255,
    "force_row_wise": True,
}

LGB_FIT_PARAMS = {
    "eval_metric": "multi_logloss",
    "callbacks": [],
}

print("=" * 60)
print("Configuration loaded:")
print(f"  PyTorch device: {BERT_DEVICE}")
print(f"  LightGBM device: {LGB_DEVICE}")
print(f"  USE_GPU: {USE_GPU}")
print("=" * 60)