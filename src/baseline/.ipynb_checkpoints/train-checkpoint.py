"""
Main training script for the LightGBM model with GPU support.
"""

import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score

from . import config, constants
from .features import add_aggregate_features, handle_missing_values
from .temporal_split import get_split_date_from_ratio, temporal_split_by_date


def setup_gpu_environment() -> None:
    """Настраивает окружение для работы с GPU."""
    print("\n🎮 GPU Setup for Training:")
    
    # PyTorch диагностика
    if torch.cuda.is_available():
        print(f"✅ PyTorch CUDA is available")
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"      Memory: {mem:.1f} GB")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU_ID)
        print(f"   Using GPU {config.GPU_ID} for PyTorch")
    else:
        print("❌ PyTorch CUDA is NOT available")
    
    # LightGBM диагностика
    print(f"\n📊 LightGBM Configuration:")
    print(f"   Device in params: {config.LGB_PARAMS.get('device', 'cpu')}")
    print(f"   USE_GPU in config: {config.USE_GPU}")
    
    # Тест LightGBM GPU
    if config.USE_GPU:
        print(f"   Testing LightGBM GPU support...")
        try:
            # Простой тест на маленьких данных
            X_test = np.random.rand(100, 10).astype(np.float32)
            y_test = np.random.randint(0, 3, 100)
            
            params = {
                "objective": "multiclass",
                "num_class": 3,
                "device": "gpu",
                "gpu_device_id": config.GPU_ID,
                "verbosity": -1,
                "seed": config.RANDOM_STATE
            }
            
            test_model = lgb.LGBMClassifier(**params)
            test_model.fit(X_test, y_test)
            print(f"   ✅ LightGBM GPU test passed")
        except Exception as e:
            print(f"   ❌ LightGBM GPU test failed: {e}")
            print(f"   Falling back to CPU")
            config.LGB_PARAMS["device"] = "cpu"
            print(f"   Updated device to: cpu")


def train() -> None:
    """Runs the model training pipeline with temporal split and GPU support."""
    
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
        raise ValueError(
            f"Timestamp column '{constants.COL_TIMESTAMP}' not found in train set. "
            "Make sure data was prepared with timestamp preserved."
        )

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
    max_train_timestamp = train_split[constants.COL_TIMESTAMP].max()
    min_val_timestamp = val_split[constants.COL_TIMESTAMP].min()
    print(f"   Max train timestamp: {max_train_timestamp}")
    print(f"   Min validation timestamp: {min_val_timestamp}")

    if min_val_timestamp <= max_train_timestamp:
        raise ValueError(
            f"Temporal split validation failed: min validation timestamp ({min_val_timestamp}) "
            f"is not greater than max train timestamp ({max_train_timestamp})."
        )
    print("   ✅ Temporal split validation passed")

    # Compute aggregate features on train split only
    print(f"\n📊 Computing aggregate features on train split only...")
    train_split_with_agg = add_aggregate_features(train_split.copy(), train_split)
    val_split_with_agg = add_aggregate_features(val_split.copy(), train_split)

    # Handle missing values
    print("🔧 Handling missing values...")
    train_split_final = handle_missing_values(train_split_with_agg, train_split)
    val_split_final = handle_missing_values(val_split_with_agg, train_split)

    # Define features (X) and target (y)
    exclude_cols = [
        constants.COL_SOURCE,
        config.TARGET,
        constants.COL_PREDICTION,
        constants.COL_TIMESTAMP,
    ]
    features = [col for col in train_split_final.columns if col not in exclude_cols]

    # Exclude any remaining object columns
    non_feature_object_cols = train_split_final[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in non_feature_object_cols]

    X_train = train_split_final[features].copy()
    y_train = train_split_final[config.TARGET]
    X_val = val_split_final[features].copy()
    y_val = val_split_final[config.TARGET]

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

    # Identify categorical features for LightGBM
    categorical_features = [
        f for f in features if train_split_final[f].dtype.name == "category"
    ]
    if categorical_features:
        print(f"   Categorical features: {len(categorical_features)}")

    print(f"\n🎯 Training features: {len(features)}")
    print(f"   Training data shape: {X_train.shape}")

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Train model with GPU
    print(f"\n" + "=" * 60)
    print(f"🚀 Training LightGBM model on {config.LGB_PARAMS.get('device', 'cpu').upper()}...")
    print("   Classes: 0=cold candidates, 1=planned books, 2=read books")
    print("=" * 60)
    
    model = lgb.LGBMClassifier(**config.LGB_PARAMS)

    # Create callback for saving checkpoints
    def checkpoint_callback(env: lgb.callback.CallbackEnv) -> None:
        """Save model checkpoint every 50 iterations."""
        iteration = env.iteration
        if iteration > 0 and iteration % 100 == 0:
            checkpoint_path = config.MODEL_DIR / f"checkpoint_iter_{iteration}.txt"
            env.model.save_model(str(checkpoint_path))
            if iteration % 500 == 0:
                print(f"   Checkpoint saved at iteration {iteration}")

    # Update fit params
    fit_params = config.LGB_FIT_PARAMS.copy()
    fit_params["callbacks"] = [
        lgb.early_stopping(
            stopping_rounds=config.EARLY_STOPPING_ROUNDS,
            verbose=True,
        ),
        lgb.log_evaluation(period=100),
        checkpoint_callback,
    ]

    # Convert categorical feature names to column indices
    categorical_feature_indices = [
        features.index(f) for f in categorical_features if f in features
    ]

    # Train the model
    print(f"\n🔧 Starting training...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=fit_params["eval_metric"],
        callbacks=fit_params["callbacks"],
        categorical_feature=categorical_feature_indices if categorical_feature_indices else "auto",
    )

    # Evaluate the model
    print(f"\n📈 Evaluating model...")
    val_preds = model.predict(X_val)
    val_proba = model.predict_proba(X_val)

    accuracy = accuracy_score(y_val, val_preds)
    precision = precision_score(y_val, val_preds, average="weighted", zero_division=0)
    recall = recall_score(y_val, val_preds, average="weighted", zero_division=0)

    # Class distribution
    class_dist = pd.Series(val_preds).value_counts().sort_index()
    class_proba_mean = val_proba.mean(axis=0)

    print(f"\n" + "=" * 60)
    print("📊 Validation metrics:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision (weighted): {precision:.4f}")
    print(f"   Recall (weighted): {recall:.4f}")
    print(f"   Predicted class distribution:")
    for class_idx in range(3):
        count = class_dist.get(class_idx, 0)
        proba_mean = class_proba_mean[class_idx]
        print(f"     Class {class_idx}: {count} samples ({100*count/len(val_preds):.1f}%), mean proba: {proba_mean:.4f}")
    print("=" * 60)

    # Save the trained model
    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    model.booster_.save_model(str(model_path))
    print(f"\n💾 Model saved to {model_path}")

    # Save feature list for prediction
    features_path = config.MODEL_DIR / "features_list.json"
    with open(features_path, "w") as f:
        json.dump(features, f)
    print(f"📋 Feature list saved to {features_path}")

    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"   Device used: {config.LGB_PARAMS.get('device', 'cpu')}")
    print("=" * 60)


if __name__ == "__main__":
    train()