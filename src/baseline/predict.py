"""
Enhanced inference script with ensemble predictions.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from . import config, constants
from .data_processing import expand_candidates, load_and_merge_data
from .features import add_aggregate_features, handle_missing_values


def predict_with_ensemble() -> None:
    """Generates ensemble predictions for the test set."""
    
    # Load targets and candidates
    print("Loading targets and candidates...")
    targets_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.TARGETS_FILENAME,
        dtype={constants.COL_USER_ID: "int32"},
    )
    candidates_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.CANDIDATES_FILENAME,
        dtype={constants.COL_USER_ID: "int32"},
    )

    print(f"Targets: {len(targets_df):,} users")
    print(f"Candidates: {len(candidates_df):,} users")

    # Expand candidates into pairs
    print("\nExpanding candidates...")
    candidates_pairs_df = expand_candidates(candidates_df)

    # Load prepared data for base features
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Please run 'poetry run python -m src.baseline.prepare_data' first."
        )

    print(f"Loading prepared data from {processed_path}...")
    featured_df = pd.read_parquet(processed_path, engine="pyarrow")

    # Get train data for computing aggregates
    train_df = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # Load metadata for candidates
    print("Loading metadata...")
    _, _, _, book_genres_df, descriptions_df = load_and_merge_data()
    user_data_df = pd.read_csv(config.RAW_DATA_DIR / constants.USER_DATA_FILENAME)
    book_data_df = pd.read_csv(config.RAW_DATA_DIR / constants.BOOK_DATA_FILENAME)

    # Merge metadata with candidates
    print("Merging metadata with candidates...")
    candidates_with_meta = candidates_pairs_df.merge(user_data_df, on=constants.COL_USER_ID, how="left")
    book_data_df = book_data_df.drop_duplicates(subset=[constants.COL_BOOK_ID])
    candidates_with_meta = candidates_with_meta.merge(book_data_df, on=constants.COL_BOOK_ID, how="left")

    # Add base features from prepared data
    print("Adding base features...")
    
    # Identify feature columns from prepared data
    exclude_base_cols = [
        constants.COL_USER_ID,
        constants.COL_BOOK_ID,
        constants.COL_SOURCE,
        constants.COL_TIMESTAMP,
        constants.COL_HAS_READ,
        config.TARGET,
        constants.COL_PREDICTION,
        constants.COL_GENDER,
        constants.COL_AGE,
        constants.COL_AUTHOR_ID,
        constants.COL_PUBLICATION_YEAR,
        constants.COL_LANGUAGE,
        constants.COL_PUBLISHER,
        constants.COL_AVG_RATING,
    ]
    
    feature_cols = [col for col in featured_df.columns if col not in exclude_base_cols]
    
    # Get a representative row for each book
    book_features_df = featured_df[[constants.COL_BOOK_ID] + feature_cols].drop_duplicates(
        subset=[constants.COL_BOOK_ID]
    )

    # Merge book features
    cols_to_drop = [col for col in feature_cols if col in candidates_with_meta.columns]
    if cols_to_drop:
        candidates_with_meta = candidates_with_meta.drop(columns=cols_to_drop)

    candidates_with_meta = candidates_with_meta.merge(
        book_features_df, on=constants.COL_BOOK_ID, how="left"
    )

    # Add text features
    print("Adding text features...")
    tfidf_cols = [col for col in featured_df.columns if col.startswith("tfidf_")]
    bert_cols = [col for col in featured_df.columns if col.startswith("bert")]
    text_feature_cols = tfidf_cols + bert_cols

    if text_feature_cols:
        book_text_features = featured_df[[constants.COL_BOOK_ID] + text_feature_cols].drop_duplicates(
            subset=[constants.COL_BOOK_ID]
        )
        candidates_with_meta = candidates_with_meta.merge(
            book_text_features, on=constants.COL_BOOK_ID, how="left"
        )

    # Compute aggregate features on ALL train data
    print("\nComputing aggregate features on all train data...")
    candidates_with_agg = add_aggregate_features(candidates_with_meta.copy(), train_df)

    # Handle missing values
    print("Handling missing values...")
    candidates_final = handle_missing_values(candidates_with_agg, train_df)

    # Load feature list saved during training
    features_path = config.MODEL_DIR / "features_list.json"
    if features_path.exists():
        print("Loading feature list from training...")
        with open(features_path, "r") as f:
            features = json.load(f)
        print(f"Loaded {len(features)} features from training")
    else:
        print("Warning: Feature list not found, using all available features")
        exclude_cols = exclude_base_cols + [
            constants.COL_USER_ID,
            constants.COL_BOOK_ID,
        ]
        features = [col for col in candidates_final.columns if col not in exclude_cols]

    # Ensure all features exist
    missing_features = [f for f in features if f not in candidates_final.columns]
    if missing_features:
        print(f"Warning: Missing {len(missing_features)} features, adding defaults")
        for feat in missing_features:
            candidates_final[feat] = 0.0

    # Keep only necessary columns
    keep_cols = features + [constants.COL_USER_ID, constants.COL_BOOK_ID]
    candidates_final = candidates_final[keep_cols]

    # Критически важная часть: правильная обработка категориальных признаков
    print("\n🎯 Critical step: Preparing categorical features for LightGBM...")
    
    # Загружаем информацию о категориальных признаках из обучающих данных
    train_categorical_info = {}
    for col in features:
        if col in train_df.columns and train_df[col].dtype.name == "category":
            # Сохраняем категории из обучающих данных
            train_categories = list(train_df[col].cat.categories)
            train_categorical_info[col] = train_categories
            print(f"   Found categorical feature: {col} with {len(train_categories)} categories")
    
    # Преобразуем категориальные признаки в тестовых данных
    for col in features:
        if col in train_categorical_info:
            # Этот признак был категориальным при обучении
            train_categories = train_categorical_info[col]
            
            if col not in candidates_final.columns:
                print(f"   ⚠️  Categorical feature {col} not in test data, adding default")
                candidates_final[col] = train_categories[0] if train_categories else "missing"
            else:
                # Преобразуем в строки и заменяем значения, которых нет в обучающих категориях
                candidates_final[col] = candidates_final[col].astype(str)
                
                # Находим значения, которых нет в обучающих категориях
                unique_vals = set(candidates_final[col].unique())
                train_vals_set = set(train_categories)
                unknown_vals = unique_vals - train_vals_set
                
                if unknown_vals:
                    print(f"   ⚠️  Feature {col} has {len(unknown_vals)} unknown values, replacing with first category")
                    # Заменяем неизвестные значения первой категорией
                    candidates_final.loc[candidates_final[col].isin(unknown_vals), col] = train_categories[0] if train_categories else "missing"
                
                # Создаем категориальный тип с теми же категориями, что и в обучающих данных
                candidates_final[col] = pd.Categorical(
                    candidates_final[col],
                    categories=train_categories,
                    ordered=False
                )
        else:
            # Признак не был категориальным при обучении
            if col in candidates_final.columns:
                if candidates_final[col].dtype.name == "category":
                    # Если в тестовых данных он категориальный, но не был при обучении - преобразуем в строку
                    candidates_final[col] = candidates_final[col].astype(str)
                elif candidates_final[col].dtype.name == "object":
                    # Object тип также преобразуем в строку
                    candidates_final[col] = candidates_final[col].astype(str)
    
    print(f"✅ Categorical features prepared for LightGBM")

    X_test = candidates_final[features]
    print(f"Prediction features: {len(features)}")
    print(f"Test data shape: {X_test.shape}")
    
    # Проверяем типы данных перед предсказанием
    print("\n🔍 Checking data types before prediction...")
    dtypes_summary = X_test.dtypes.value_counts()
    for dtype, count in dtypes_summary.items():
        print(f"   {dtype}: {count} columns")
    
    # Убедимся, что нет datetime типов
    datetime_cols = X_test.select_dtypes(include=['datetime64', 'timedelta64']).columns
    if len(datetime_cols) > 0:
        print(f"   ⚠️  Found datetime columns: {list(datetime_cols)}")
        # Преобразуем их в числовые (timestamp)
        for col in datetime_cols:
            X_test[col] = X_test[col].astype(np.int64) // 10**9

    # Load models for ensemble
    print("\n🤖 Loading ensemble models...")
    
    # Load LightGBM model
    lgb_path = config.MODEL_DIR / config.MODEL_FILENAME
    if not lgb_path.exists():
        raise FileNotFoundError(f"LightGBM model not found at {lgb_path}")
    
    print(f"Loading LightGBM model from {lgb_path}...")
    lgb_model = lgb.Booster(model_file=str(lgb_path))
    
    # Try to load CatBoost model
    cb_path = config.MODEL_DIR / config.CATBOOST_MODEL_FILENAME
    use_catboost = False
    cb_model = None
    
    if cb_path.exists():
        try:
            import catboost as cb
            print(f"Loading CatBoost model from {cb_path}...")
            cb_model = cb.CatBoostClassifier()
            cb_model.load_model(str(cb_path))
            use_catboost = True
            print("✅ CatBoost model loaded")
        except Exception as e:
            print(f"⚠️  Failed to load CatBoost model: {e}")
    else:
        print("ℹ️  CatBoost model not found, using LightGBM only")

    # Generate predictions
    print("\nGenerating ensemble predictions...")
    
    # LightGBM predictions - ОЧЕНЬ ВАЖНО: преобразуем данные в правильный формат
    print("Preparing data for LightGBM prediction...")
    
    # Для LightGBM нужно явно указать категориальные признаки
    # Получаем индексы категориальных признаков
    categorical_indices = []
    for i, col in enumerate(features):
        if col in train_categorical_info:
            categorical_indices.append(i)
    
    print(f"   LightGBM will use {len(categorical_indices)} categorical features")
    
    # Преобразуем DataFrame в numpy массив для LightGBM
    # LightGBM может работать с pandas DataFrame, но лучше явно указать категориальные признаки
    X_test_for_lgb = X_test.copy()
    
    # Для LightGBM нужно преобразовать категориальные признаки в целочисленные коды
    for col_idx, col in enumerate(features):
        if col in train_categorical_info:
            # Уже категориальный тип, LightGBM сам преобразует в коды
            pass
    
    # Делаем предсказание LightGBM
    print("Making LightGBM predictions...")
    try:
        lgb_proba_all = lgb_model.predict(X_test_for_lgb)
        lgb_proba_all = np.array(lgb_proba_all)
        if lgb_proba_all.ndim == 1:
            lgb_proba_all = lgb_proba_all.reshape(-1, 3)
        print(f"   LightGBM predictions shape: {lgb_proba_all.shape}")
    except Exception as e:
        print(f"❌ LightGBM prediction failed: {e}")
        # Попробуем альтернативный подход
        print("   Trying alternative prediction method...")
        X_test_array = X_test_for_lgb.values.astype(np.float32)
        lgb_proba_all = lgb_model.predict(X_test_array)
        lgb_proba_all = np.array(lgb_proba_all)
        if lgb_proba_all.ndim == 1:
            lgb_proba_all = lgb_proba_all.reshape(-1, 3)
        print(f"   LightGBM predictions shape: {lgb_proba_all.shape}")
    
    if use_catboost and cb_model is not None:
        # CatBoost predictions
        print("Generating CatBoost predictions...")
        
        # Prepare data for CatBoost
        # CatBoost требует особого формата категориальных признаков
        X_test_cb = X_test.copy()
        categorical_features_for_cb = [col for col in features if col in train_categorical_info]
        categorical_indices_cb = [features.index(f) for f in categorical_features_for_cb if f in features]
        
        # CatBoost требует, чтобы категориальные признаки были строковыми
        for col in categorical_features_for_cb:
            if col in X_test_cb.columns:
                X_test_cb[col] = X_test_cb[col].astype(str).fillna("missing")
        
        try:
            cb_proba_all = cb_model.predict_proba(X_test_cb)
            print(f"   CatBoost predictions shape: {cb_proba_all.shape}")
            
            # Ensemble weights (можно настроить)
            lgb_weight = 0.6
            cb_weight = 0.4
            
            ensemble_proba = lgb_weight * lgb_proba_all + cb_weight * cb_proba_all
            print(f"   Ensemble weights: LightGBM={lgb_weight}, CatBoost={cb_weight}")
        except Exception as e:
            print(f"⚠️  CatBoost prediction failed: {e}")
            print("   Using LightGBM predictions only")
            ensemble_proba = lgb_proba_all
    else:
        ensemble_proba = lgb_proba_all
        print("Using LightGBM predictions only")
    
    # Enhanced ranking score calculation
    print("Calculating enhanced ranking scores...")
    
    # Веса классов
    class_weights = np.array([0.0, 1.0, 3.0])  # cold: 0, planned: 1, read: 3
    
    # Базовый скор
    ranking_scores = np.sum(ensemble_proba * class_weights, axis=1)
    
    # Добавляем поправку на уверенность модели
    confidence = np.max(ensemble_proba, axis=1)
    ranking_scores = ranking_scores * (0.3 + 0.7 * confidence)
    
    # Добавляем поправку на разницу между классами 2 и 1
    proba_diff = ensemble_proba[:, 2] - ensemble_proba[:, 1]
    ranking_scores = ranking_scores * (1.0 + 0.5 * np.tanh(proba_diff))
    
    candidates_final["prediction"] = ranking_scores

    # Rank candidates for each user and select top-K
    print("\nRanking candidates for each user...")
    submission_rows = []

    for user_id in targets_df[constants.COL_USER_ID]:
        user_candidates = candidates_final[candidates_final[constants.COL_USER_ID] == user_id].copy()

        if len(user_candidates) == 0:
            # No candidates for this user - empty list
            book_id_list = ""
        else:
            # Sort by prediction probability (descending)
            user_candidates = user_candidates.sort_values("prediction", ascending=False)

            # Select top-K, where K = min(20, num_candidates)
            k = min(constants.MAX_RANKING_LENGTH, len(user_candidates))
            top_books = user_candidates.head(k)

            # Create comma-separated string of book_ids
            book_id_list = ",".join([str(int(book_id)) for book_id in top_books[constants.COL_BOOK_ID]])

        submission_rows.append({constants.COL_USER_ID: user_id, constants.COL_BOOK_ID_LIST: book_id_list})

    # Create submission DataFrame
    submission_df = pd.DataFrame(submission_rows)

    # Ensure submission directory exists
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = config.SUBMISSION_DIR / constants.SUBMISSION_FILENAME

    # Save submission
    submission_df.to_csv(submission_path, index=False)
    print(f"\n✅ Submission file created at: {submission_path}")
    print(f"   Submission shape: {submission_df.shape}")

    # Print statistics
    non_empty = submission_df[submission_df[constants.COL_BOOK_ID_LIST] != ""].shape[0]
    avg_books = submission_df[constants.COL_BOOK_ID_LIST].apply(
        lambda x: len(x.split(",")) if x else 0
    ).mean()
    
    print(f"📊 Submission statistics:")
    print(f"   Users with recommendations: {non_empty}/{len(submission_df)}")
    print(f"   Average books per user: {avg_books:.2f}")
    
    # Сохраняем также предсказания для анализа
    predictions_path = config.SUBMISSION_DIR / "predictions_with_scores.parquet"
    candidates_final[["user_id", "book_id", "prediction"]].to_parquet(predictions_path)
    print(f"   Detailed predictions saved to: {predictions_path}")


if __name__ == "__main__":
    predict_with_ensemble()