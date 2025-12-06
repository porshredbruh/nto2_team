"""
Inference script to generate predictions for the test set.

Computes aggregate features on all train data and applies them to test set,
then generates predictions using the trained model.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config, constants
from .features import add_aggregate_features, handle_missing_values


def predict() -> None:
    """Generates and saves predictions for the test set.

    This script loads prepared data from data/processed/, computes aggregate features
    on all train data, applies them to test set, and generates predictions using
    the trained model.

    Note: Data must be prepared first using prepare_data.py, and model must be trained
    using train.py
    """
    # Load prepared data
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Please run 'poetry run python -m src.baseline.prepare_data' first."
        )

    print(f"Loading prepared data from {processed_path}...")
    featured_df = pd.read_parquet(processed_path, engine="pyarrow")
    print(f"Loaded {len(featured_df):,} rows with {len(featured_df.columns)} features")

    # Separate train and test sets
    train_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()
    test_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TEST].copy()

    print(f"Train set: {len(train_set):,} rows")
    print(f"Test set: {len(test_set):,} rows")

    # Compute aggregate features on ALL train data (to use for test predictions)
    print("\nComputing aggregate features on all train data...")
    print("Note: Using both has_read=1 and has_read=0 records for interaction counts...")
    test_set_with_agg = add_aggregate_features(test_set.copy(), train_set)

    # Handle missing values (use train_set for fill values)
    print("Handling missing values...")
    test_set_final = handle_missing_values(test_set_with_agg, train_set)

    # Define features (exclude source, target, prediction, timestamp columns)
    exclude_cols = [
        constants.COL_SOURCE,
        config.TARGET,
        constants.COL_PREDICTION,
        constants.COL_TIMESTAMP,
        constants.COL_HAS_READ,
    ]
    features = [col for col in test_set_final.columns if col not in exclude_cols]

    # Exclude any remaining object columns that are not model features
    non_feature_object_cols = test_set_final[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in non_feature_object_cols]

    X_test = test_set_final[features]
    print(f"Prediction features: {len(features)}")

    # Load trained model
    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. " "Please run 'poetry run python -m src.baseline.train' first."
        )

    print(f"\nLoading model from {model_path}...")
    model = lgb.Booster(model_file=str(model_path))

    # Generate predictions
    print("Generating predictions...")
    test_preds = model.predict(X_test)

    # Clip predictions to be within the valid rating range [0, 10]
    clipped_preds = np.clip(test_preds, constants.PREDICTION_MIN_VALUE, constants.PREDICTION_MAX_VALUE)

    # Create submission file
    submission_df = test_set[[constants.COL_USER_ID, constants.COL_BOOK_ID]].copy()
    submission_df[constants.COL_PREDICTION] = clipped_preds

    # Ensure submission directory exists
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = config.SUBMISSION_DIR / constants.SUBMISSION_FILENAME

    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission file created at: {submission_path}")
    print(f"Predictions: min={clipped_preds.min():.4f}, max={clipped_preds.max():.4f}, mean={clipped_preds.mean():.4f}")
    
    # Выводим статистику по новым признакам
    print("\nNew features statistics:")
    new_features = [
        'user_total_interactions', 'user_to_read_count', 'user_read_ratio',
        'book_total_interactions', 'book_to_read_count', 'book_read_ratio', 'book_popularity'
    ]
    
    for feature in new_features:
        if feature in test_set_final.columns:
            print(f"  {feature}: mean={test_set_final[feature].mean():.3f}, "
                  f"min={test_set_final[feature].min():.3f}, max={test_set_final[feature].max():.3f}")


if __name__ == "__main__":
    predict()