"""
Main training script for the enhanced model with multi-GPU support and ensemble.
"""

import json
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import joblib

from . import config, constants
from .features import handle_missing_values  # Импортируем только handle_missing_values
from .temporal_split import get_split_date_from_ratio, temporal_split_by_date


def setup_gpu_environment() -> None:
    """Настраивает окружение для работы с multi-GPU."""
    print("\n🎮 MULTI-GPU Setup for Training:")
    
    # PyTorch диагностика
    if torch.cuda.is_available():
        print(f"✅ PyTorch CUDA is available")
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} - {mem:.1f} GB")
        
        if config.USE_MULTI_GPU and config.NUM_GPUS > 1:
            print(f"   Using {config.NUM_GPUS} GPUs for training")
        else:
            print(f"   Using GPU {config.GPU_IDS[0]} for training")
    else:
        print("❌ PyTorch CUDA is NOT available")
    
    # LightGBM диагностика
    print(f"\n📊 LightGBM Configuration:")
    print(f"   Device: {config.LGB_PARAMS.get('device', 'cpu')}")
    if config.LGB_PARAMS.get('device') == 'gpu':
        if 'num_gpu' in config.LGB_PARAMS:
            print(f"   Num GPUs: {config.LGB_PARAMS.get('num_gpu', 1)}")
    
    # Проверяем CatBoost
    try:
        import catboost as cb
        print(f"✅ CatBoost available")
        if torch.cuda.is_available() and config.USE_GPU:
            print(f"   CatBoost will use GPU")
    except ImportError:
        print("⚠️  CatBoost not installed (optional)")


def train_catboost_model(X_train, y_train, X_val, y_val, features, categorical_features_indices):
    """Обучает CatBoost модель."""
    try:
        import catboost as cb
        
        print("\n" + "=" * 60)
        print("🐱 Training CatBoost Model")
        print("=" * 60)
        
        # Подготавливаем данные для CatBoost
        train_pool = cb.Pool(
            X_train, y_train,
            cat_features=categorical_features_indices
        )
        val_pool = cb.Pool(
            X_val, y_val,
            cat_features=categorical_features_indices
        )
        
        # Параметры CatBoost
        cb_params = config.CATBOOST_PARAMS.copy()
        
        # Определяем веса классов
        class_weights = compute_class_weights(y_train)
        cb_params['class_weights'] = class_weights
        
        # Создаем и обучаем модель
        model = cb.CatBoostClassifier(**cb_params)
        
        model.fit(
            train_pool,
            eval_set=val_pool,
            verbose=100,
            plot=False
        )
        
        # Оценка модели
        val_pred = model.predict(val_pool)
        val_proba = model.predict_proba(val_pool)
        
        accuracy = accuracy_score(y_val, val_pred)
        f1 = f1_score(y_val, val_pred, average='weighted')
        
        print(f"\n📊 CatBoost Validation Metrics:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-score: {f1:.4f}")
        
        # Сохраняем модель
        model_path = config.MODEL_DIR / config.CATBOOST_MODEL_FILENAME
        model.save_model(str(model_path))
        print(f"💾 CatBoost model saved to {model_path}")
        
        return model
        
    except ImportError:
        print("⚠️  CatBoost not installed, skipping CatBoost training")
        return None
    except Exception as e:
        print(f"⚠️  CatBoost training failed: {e}")
        return None


def compute_class_weights(y):
    """Вычисляет веса классов для несбалансированных данных."""
    from sklearn.utils.class_weight import compute_class_weight
    
    # Получаем уникальные классы
    classes = np.unique(y)
    
    # Вычисляем веса классов (обратно пропорционально частоте)
    weights = compute_class_weight(
        class_weight='balanced', 
        classes=classes, 
        y=y
    )
    
    # Создаем словарь {класс: вес}
    class_weights_dict = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
    
    # Убедимся, что у нас есть веса для всех 3 классов (0, 1, 2)
    for cls in [0, 1, 2]:
        if cls not in class_weights_dict:
            class_weights_dict[cls] = 1.0
    
    return class_weights_dict


def time_series_cross_validation(X, y, timestamps, n_splits=5):
    """Выполняет временную кросс-валидацию."""
    print(f"\n🔄 Time Series Cross-Validation (n_splits={n_splits})")
    
    # Сортируем по времени
    sort_idx = np.argsort(timestamps)
    X_sorted = X.iloc[sort_idx].copy() if hasattr(X, 'iloc') else X[sort_idx]
    y_sorted = y[sort_idx]
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sorted)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        
        X_train_fold = X_sorted.iloc[train_idx] if hasattr(X_sorted, 'iloc') else X_sorted[train_idx]
        y_train_fold = y_sorted[train_idx]
        X_val_fold = X_sorted.iloc[val_idx] if hasattr(X_sorted, 'iloc') else X_sorted[val_idx]
        y_val_fold = y_sorted[val_idx]
        
        print(f"   Train: {len(X_train_fold):,} samples")
        print(f"   Val: {len(X_val_fold):,} samples")
        
        # Обучаем LightGBM на фолде
        model = lgb.LGBMClassifier(**config.LGB_PARAMS)
        
        # Вычисляем веса классов для этого фолда
        class_weights = compute_class_weights(y_train_fold)
        
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric=config.LGB_FIT_PARAMS['eval_metric'],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(100),
            ],
            class_weight=class_weights
        )
        
        # Оцениваем модель
        val_pred = model.predict(X_val_fold)
        val_proba = model.predict_proba(X_val_fold)
        
        accuracy = accuracy_score(y_val_fold, val_pred)
        f1 = f1_score(y_val_fold, val_pred, average='weighted')
        
        # Вычисляем custom NDCG score
        ndcg_score = calculate_custom_ndcg(y_val_fold, val_pred, val_proba)
        
        print(f"   Fold {fold+1} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, NDCG: {ndcg_score:.4f}")
        
        fold_scores.append({
            'accuracy': accuracy,
            'f1': f1,
            'ndcg': ndcg_score
        })
        fold_models.append(model)
    
    # Анализируем результаты CV
    print(f"\n📊 Cross-Validation Results:")
    avg_accuracy = np.mean([s['accuracy'] for s in fold_scores])
    avg_f1 = np.mean([s['f1'] for s in fold_scores])
    avg_ndcg = np.mean([s['ndcg'] for s in fold_scores])
    
    print(f"   Average Accuracy: {avg_accuracy:.4f}")
    print(f"   Average F1-score: {avg_f1:.4f}")
    print(f"   Average NDCG: {avg_ndcg:.4f}")
    
    return fold_models, fold_scores


def calculate_custom_ndcg(y_true, y_pred, y_proba, k=20):
    """Вычисляет custom NDCG метрику для валидации."""
    # Эта функция имитирует логику NDCG для ранжирования
    # В реальной задаче у нас нет ранжирования на уровне пользователя во время CV
    # Поэтому используем упрощенную версию
    
    # Для каждого класса вычисляем "релевантность" предсказания
    relevance_scores = []
    for true, pred, proba in zip(y_true, y_pred, y_proba, strict=False):
        if pred == true:
            # Правильное предсказание
            if true == 2:  # Прочитано
                relevance_scores.append(2.0)
            elif true == 1:  # Планы
                relevance_scores.append(1.0)
            else:  # Холодные
                relevance_scores.append(0.0)
        else:
            # Неправильное предсказание
            if true == 2 and pred == 1:  # Прочитано предсказано как планы
                relevance_scores.append(0.5)
            elif true == 1 and pred == 2:  # Планы предсказано как прочитано
                relevance_scores.append(1.5)
            else:
                relevance_scores.append(0.0)
    
    if not relevance_scores:
        return 0.0
    
    # Сортировка по убыванию вероятности правильного класса
    sorted_indices = np.argsort([max(p) for p in y_proba])[::-1]
    sorted_relevance = [relevance_scores[i] for i in sorted_indices[:k]]
    
    # Вычисляем DCG
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(sorted_relevance))
    
    # Идеальный порядок
    ideal_relevance = sorted(relevance_scores, reverse=True)[:k]
    ideal_dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
    
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def train() -> None:
    """Runs the enhanced model training pipeline with multi-GPU support."""
    
    # Настройка GPU окружения
    setup_gpu_environment()
    
    # Load prepared data
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Please run 'poetry run python -m src.baseline.prepare_data' first."
        )

    print(f"\n📂 Loading prepared data from {processed_path}...")
    featured_df = pd.read_parquet(processed_path, engine="pyarrow")
    print(f"   Loaded {len(featured_df):,} rows with {len(featured_df.columns)} features")

    # Separate train set
    train_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # Check for timestamp column
    if constants.COL_TIMESTAMP not in train_set.columns:
        print("⚠️  Timestamp column not found, using random split")
        # Используем случайное разделение
        from sklearn.model_selection import train_test_split
        train_split, val_split = train_test_split(
            train_set, test_size=0.2, random_state=config.RANDOM_STATE, stratify=train_set[config.TARGET]
        )
    else:
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(train_set[constants.COL_TIMESTAMP]):
            train_set[constants.COL_TIMESTAMP] = pd.to_datetime(train_set[constants.COL_TIMESTAMP])

        # Perform temporal split
        print(f"\n📅 Performing temporal split with ratio {config.TEMPORAL_SPLIT_RATIO}...")
        split_date = get_split_date_from_ratio(train_set, config.TEMPORAL_SPLIT_RATIO, constants.COL_TIMESTAMP)
        print(f"   Split date: {split_date}")

        train_mask, val_mask = temporal_split_by_date(train_set, split_date, constants.COL_TIMESTAMP)

        # Split data
        train_split = train_set[train_mask].copy()
        val_split = train_set[val_mask].copy()

        print(f"   Train split: {len(train_split):,} rows")
        print(f"   Validation split: {len(val_split):,} rows")

        # Verify temporal correctness
        if len(train_split) > 0 and len(val_split) > 0:
            max_train_timestamp = train_split[constants.COL_TIMESTAMP].max()
            min_val_timestamp = val_split[constants.COL_TIMESTAMP].min()
            
            print(f"   Max train timestamp: {max_train_timestamp}")
            print(f"   Min validation timestamp: {min_val_timestamp}")
            
            if min_val_timestamp <= max_train_timestamp:
                print("⚠️  Warning: Validation data contains older timestamps than training data")
            else:
                print("   ✅ Temporal split validation passed")

    # Используем уже созданные признаки
    print(f"\n📊 Используем уже созданные признаки...")
    
    # Копируем данные для безопасности
    train_split_final = train_split.copy()
    val_split_final = val_split.copy()
    
    # Проверяем, что все нужные признаки присутствуют
    print(f"   Проверка признаков в train_split: {len(train_split_final.columns)} признаков")
    print(f"   Проверка признаков в val_split: {len(val_split_final.columns)} признаков")

    # Обрабатываем пропущенные значения
    print("🔧 Handling missing values...")
    train_split_final = handle_missing_values(train_split_final, train_split_final)
    val_split_final = handle_missing_values(val_split_final, train_split_final)

    # Define features (X) and target (y)
    exclude_cols = [
        constants.COL_SOURCE,
        config.TARGET,
        constants.COL_PREDICTION,
        constants.COL_TIMESTAMP,
        constants.COL_USER_ID,
        constants.COL_BOOK_ID,
        constants.COL_HAS_READ,
        constants.COL_RELEVANCE,
    ]
    
    # Исключаем также текстовые колонки и другие не-фичи
    features = [col for col in train_split_final.columns if col not in exclude_cols]
    
    # Удаляем object колонки
    non_feature_object_cols = train_split_final[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in non_feature_object_cols]
    
    # Удаляем datetime колонки
    datetime_cols = train_split_final[features].select_dtypes(include=["datetime64", "timedelta64"]).columns.tolist()
    if datetime_cols:
        print(f"   ⚠️  Excluding datetime columns: {datetime_cols}")
        features = [f for f in features if f not in datetime_cols]

    X_train = train_split_final[features].copy()
    y_train = train_split_final[config.TARGET]
    X_val = val_split_final[features].copy()
    y_val = val_split_final[config.TARGET]

    # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Обрабатываем high-cardinality признаки
    print("\n🔍 Анализ high-cardinality признаков...")
    high_cardinality_features = []
    for col in features:
        if X_train[col].nunique() > 1000:  # Если больше 1000 уникальных значений
            high_cardinality_features.append(col)
            print(f"   ⚠️  High-cardinality feature: {col} - {X_train[col].nunique()} unique values")
    
    # Если есть high-cardinality признаки, применяем frequency encoding
    if high_cardinality_features:
        print(f"   Применяем frequency encoding к {len(high_cardinality_features)} признакам...")
        for col in high_cardinality_features:
            # Вычисляем частоту каждого значения в train
            freq_encoding = X_train[col].value_counts(normalize=True)
            # Применяем к train и val
            X_train[col + '_freq'] = X_train[col].map(freq_encoding).fillna(0)
            X_val[col + '_freq'] = X_val[col].map(freq_encoding).fillna(0)
            # Удаляем оригинальный признак
            X_train = X_train.drop(columns=[col])
            X_val = X_val.drop(columns=[col])
            # Обновляем список features
            features.remove(col)
            features.append(col + '_freq')
            print(f"   ✅ Заменили {col} на {col}_freq")
    
    print(f"\n   После обработки: {len(features)} признаков")

    # Optimize memory usage for GPU
    print("\n💾 Optimizing data types for GPU memory efficiency...")
    
    # Convert float64 to float32
    float64_cols = X_train.select_dtypes(include=["float64"]).columns
    if len(float64_cols) > 0:
        print(f"   Converting {len(float64_cols)} float64 columns to float32...")
        X_train[float64_cols] = X_train[float64_cols].astype("float32")
        X_val[float64_cols] = X_val[float64_cols].astype("float32")
    
    # Convert int64 to int32 where possible
    int64_cols = X_train.select_dtypes(include=["int64"]).columns
    converted = 0
    for col in int64_cols:
        if X_train[col].max() < 2**31 - 1 and X_train[col].min() > -2**31:
            X_train[col] = X_train[col].astype("int32")
            X_val[col] = X_val[col].astype("int32")
            converted += 1
    if converted > 0:
        print(f"   Converted {converted} int64 columns to int32")
    
    print(f"   Training data memory: {X_train.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"   Validation data memory: {X_val.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Identify categorical features
    categorical_features = [
        f for f in features if X_train[f].dtype.name == "category"
    ]
    categorical_features_indices = [
        features.index(f) for f in categorical_features if f in features
    ]
    
    if categorical_features:
        print(f"   Categorical features: {len(categorical_features)}")

    print(f"\n🎯 Training features: {len(features)}")
    print(f"   Training data shape: {X_train.shape}")
    print(f"   Features preview: {features[:20]}...")

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Train LightGBM model
    print(f"\n" + "=" * 60)
    print(f"🚀 Training Enhanced LightGBM Model")
    
    # 🔥 ПРЯМОЕ ИСПРАВЛЕНИЕ: Устанавливаем безопасные параметры для GPU
    print(f"   Device: GPU (with safety parameters)")
    print("=" * 60)
    
    # Вычисляем веса классов
    class_weights = compute_class_weights(y_train)
    print(f"   Class weights: {class_weights}")
    
    # Создаем веса для каждого сэмпла (sample_weight)
    sample_weights = np.array([class_weights[class_id] for class_id in y_train])
    print(f"   Sample weights shape: {sample_weights.shape}")
    
    # 🔥 Создаем безопасные параметры для GPU
    print(f"\n🔧 Setting safe GPU parameters...")
    
    # Базовые параметры
    safe_lgb_params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "n_estimators": 2000,  # Уменьшаем для теста
        "learning_rate": 0.05,  # Увеличиваем learning rate
        "num_leaves": 31,  # Уменьшаем для стабильности
        "max_depth": 7,  # Ограничиваем глубину
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "seed": config.RANDOM_STATE,
        
        # 🔥 КРИТИЧЕСКИЕ ПАРАМЕТРЫ ДЛЯ GPU
        "device": "gpu",
        "gpu_device_id": 0,
        "max_bin": 63,  # ОЧЕНЬ МАЛЕНЬКОЕ значение!
        "min_data_in_bin": 10,
        "bin_construct_sample_cnt": 100000,  # Ограничиваем сэмплы для построения бинов
        "feature_pre_filter": False,
        "force_row_wise": True,
        "gpu_use_dp": False,  # Single precision
        "gpu_platform_id": 0,
    }
    
    # Создаем модель
    model = lgb.LGBMClassifier(**safe_lgb_params)
    
    # Callback для сохранения чекпоинтов
    def checkpoint_callback(env: lgb.callback.CallbackEnv) -> None:
        iteration = env.iteration
        if iteration > 0 and iteration % 100 == 0:
            checkpoint_path = config.MODEL_DIR / f"checkpoint_iter_{iteration}.txt"
            env.model.save_model(str(checkpoint_path))
            if iteration % 500 == 0:
                print(f"   Checkpoint saved at iteration {iteration}")
    
    # Обучаем модель
    print(f"\n🔧 Starting training...")
    try:
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(100, verbose=True),
                lgb.log_evaluation(100),
                checkpoint_callback,
            ],
            categorical_feature=categorical_features_indices if categorical_features_indices else 'auto',
        )
    except Exception as e:
        print(f"⚠️  GPU training failed: {e}")
        print("   Falling back to CPU training...")
        # Переключаем на CPU
        safe_lgb_params["device"] = "cpu"
        safe_lgb_params["n_jobs"] = -1
        del safe_lgb_params["gpu_device_id"]
        del safe_lgb_params["gpu_use_dp"]
        del safe_lgb_params["gpu_platform_id"]
        
        model = lgb.LGBMClassifier(**safe_lgb_params)
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(100, verbose=True),
                lgb.log_evaluation(100),
            ],
            categorical_feature=categorical_features_indices if categorical_features_indices else 'auto',
        )

    # Evaluate the model
    print(f"\n📈 Evaluating LightGBM model...")
    val_preds = model.predict(X_val)
    val_proba = model.predict_proba(X_val)

    accuracy = accuracy_score(y_val, val_preds)
    precision = precision_score(y_val, val_preds, average="weighted", zero_division=0)
    recall = recall_score(y_val, val_preds, average="weighted", zero_division=0)
    f1 = f1_score(y_val, val_preds, average="weighted", zero_division=0)
    
    # Custom NDCG
    ndcg_score = calculate_custom_ndcg(y_val, val_preds, val_proba)

    # Class distribution
    class_dist = pd.Series(val_preds).value_counts().sort_index()
    class_proba_mean = val_proba.mean(axis=0)

    print(f"\n" + "=" * 60)
    print("📊 LightGBM Validation metrics:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-score: {f1:.4f}")
    print(f"   Custom NDCG: {ndcg_score:.4f}")
    print(f"   Predicted class distribution:")
    for class_idx in range(3):
        count = class_dist.get(class_idx, 0)
        proba_mean = class_proba_mean[class_idx]
        class_name = {0: 'cold', 1: 'planned', 2: 'read'}.get(class_idx, 'unknown')
        print(f"     Class {class_idx} ({class_name}): {count} samples ({100*count/len(val_preds):.1f}%), mean proba: {proba_mean:.4f}")
    print("=" * 60)

    categorical_info = {}
    for col in categorical_features:
        if col in X_train.columns and X_train[col].dtype.name == "category":
            categorical_info[col] = {
                'categories': list(X_train[col].cat.categories),
                'ordered': X_train[col].cat.ordered
            }
    
    categorical_info_path = config.MODEL_DIR / "categorical_info.json"
    with open(categorical_info_path, "w") as f:
        json.dump(categorical_info, f)
    print(f"📋 Categorical features info saved to {categorical_info_path}")

    # Save the trained LightGBM model
    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    model.booster_.save_model(str(model_path))
    print(f"\n💾 LightGBM model saved to {model_path}")

    # Save feature list for prediction
    features_path = config.MODEL_DIR / "features_list.json"
    with open(features_path, "w") as f:
        json.dump(features, f)
    print(f"📋 Feature list saved to {features_path}")

    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_path = config.MODEL_DIR / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"📊 Top 10 features: {importance_df.head(10)['feature'].tolist()}")
    print(f"📊 Feature importance saved to {importance_path}")

    # Train CatBoost модель (опционально)
    catboost_model = train_catboost_model(
        X_train, y_train, X_val, y_val, features, categorical_features_indices
    )

    print("\n" + "=" * 60)
    print("✅ Enhanced training complete!")
    print(f"   Models saved: LightGBM{' + CatBoost' if catboost_model else ''}")
    print(f"   Total features: {len(features)}")
    print("=" * 60)


if __name__ == "__main__":
    train()