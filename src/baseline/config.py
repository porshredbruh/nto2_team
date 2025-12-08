"""
Configuration file for the NTO ML competition baseline.
"""

from pathlib import Path
import sys
import os

try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ PyTorch CUDA is available. GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
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
USE_GPU = True
USE_MULTI_GPU = False  # Использовать несколько GPU
GPU_IDS = [0, 1]  # Обе видеокарты A4000
NUM_GPUS = len(GPU_IDS) if USE_MULTI_GPU and torch and torch.cuda.is_available() else 1

# Настраиваем видимые GPU
if torch and torch.cuda.is_available() and USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPU_IDS))
    print(f"✅ Using GPUs: {GPU_IDS}")
    
    # Проверяем доступность GPU
    available_gpus = list(range(torch.cuda.device_count()))
    print(f"   Available GPUs: {available_gpus}")
    
    # Определяем устройство для BERT (используем первую GPU)
    BERT_DEVICE = f"cuda:{GPU_IDS[0]}" if GPU_IDS[0] < torch.cuda.device_count() else "cuda:0"
    
    # Для LightGBM multi-GPU
    if USE_MULTI_GPU and len(available_gpus) > 1:
        LGB_DEVICE = "gpu"
        print(f"✅ LightGBM will use {len(available_gpus)} GPUs")
    else:
        LGB_DEVICE = "gpu"
        print(f"✅ LightGBM will use 1 GPU")
else:
    BERT_DEVICE = "cpu"
    LGB_DEVICE = "cpu"
    print("⚠️  Using CPU for all computations")

# --- PARAMETERS ---
N_SPLITS = 5
RANDOM_STATE = 42
TARGET = constants.COL_RELEVANCE

# --- TEMPORAL SPLIT CONFIG ---
TEMPORAL_SPLIT_RATIO = 0.8

# --- TRAINING CONFIG ---
EARLY_STOPPING_ROUNDS = 100
MODEL_FILENAME_PATTERN = "lgb_fold_{fold}.txt"
MODEL_FILENAME = "lgb_model.txt"
CATBOOST_MODEL_FILENAME = "catboost_model.cbm"

# --- TF-IDF PARAMETERS ---
TFIDF_MAX_FEATURES = 1000  # Увеличили для лучшего качества
TFIDF_MIN_DF = 3
TFIDF_MAX_DF = 0.9
TFIDF_NGRAM_RANGE = (1, 3)  # Добавили триграммы

# --- BERT PARAMETERS ---
BERT_MODEL_NAME = constants.BERT_MODEL_NAME
BERT_BATCH_SIZE = 48 if BERT_DEVICE.startswith("cuda") else 8
BERT_MAX_LENGTH = 256  # Уменьшили для скорости
BERT_EMBEDDING_DIM = 768

# --- FEATURES ---
CAT_FEATURES = [
    constants.COL_USER_ID,
    constants.COL_BOOK_ID,
    constants.COL_GENDER,
    constants.COL_AUTHOR_ID,
    constants.COL_PUBLICATION_YEAR,
    constants.COL_LANGUAGE,
    constants.COL_PUBLISHER,
]

# --- LIGHTGBM PARAMETERS ---
# Базовые параметры с поддержкой multi-GPU
LGB_PARAMS_BASE = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "num_leaves": 255,
    "max_depth": -1,
    "min_child_samples": 20,
    "min_child_weight": 0.001,
    "min_split_gain": 0.0,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "lambda_l1": 0.3,
    "lambda_l2": 0.3,
    "verbose": -1,
    "seed": RANDOM_STATE,
    "boosting_type": "gbdt",
    "extra_trees": False,
    "path_smooth": 0.5,
    "max_bin": 127,  # Уменьшили с 511 до 255 для GPU
    "min_data_in_bin": 3,  # Добавили для стабильности
}

# Multi-GPU параметры
if LGB_DEVICE == "gpu":
    LGB_PARAMS = LGB_PARAMS_BASE.copy()
    LGB_PARAMS.update({
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": GPU_IDS[0],
        "force_row_wise": True,
        "gpu_use_dp": False,  # Отключаем double precision для скорости
    })
    
    if USE_MULTI_GPU and NUM_GPUS > 1:
        LGB_PARAMS.update({
            "gpu_use_dp": True,
            "num_gpu": NUM_GPUS,
        })
else:
    LGB_PARAMS = LGB_PARAMS_BASE.copy()
    LGB_PARAMS.update({
        "device": "cpu",
        "n_jobs": -1,
    })

LGB_FIT_PARAMS = {
    "eval_metric": "multi_logloss",
    "callbacks": [],
    "class_weight": {0: 1.0, 1: 2.0, 2: 3.0},  # Веса классов
}

# --- CATBOOST PARAMETERS ---
CATBOOST_PARAMS = {
    'iterations': 3000,
    'learning_rate': 0.03,
    'depth': 10,
    'loss_function': 'MultiClass',
    'verbose': 100,
    'random_seed': RANDOM_STATE,
    'early_stopping_rounds': 100,
    'use_best_model': True,
    'class_weights': [1, 2, 3],
    'l2_leaf_reg': 5,
    'bootstrap_type': 'Bayesian',
    'bagging_temperature': 1,
    'random_strength': 1,
}

if torch and torch.cuda.is_available() and USE_GPU:
    CATBOOST_PARAMS.update({
        'task_type': 'GPU',
        'devices': ','.join(map(str, range(min(NUM_GPUS, torch.cuda.device_count())))),
    })

print("=" * 60)
print("Enhanced Configuration loaded:")
print(f"  PyTorch device: {BERT_DEVICE}")
print(f"  LightGBM device: {LGB_DEVICE}")
print(f"  Num GPUs: {NUM_GPUS}")
print(f"  Multi-GPU: {USE_MULTI_GPU}")
print("=" * 60)