"""
Enhanced data preparation script with multi-GPU support.
"""

import numpy as np
import pandas as pd
import torch

from . import config, constants
from .data_processing import load_and_merge_data
from .features import create_features_enhanced


def prepare_data() -> None:
    """Processes raw data and saves prepared features with enhanced pipeline."""
    print("=" * 60)
    print("ENHANCED Data Preparation Pipeline")
    print("=" * 60)
    
    # Диагностика GPU
    print("\n🎮 GPU Diagnostics:")
    if torch.cuda.is_available():
        print(f"✅ CUDA is available")
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"      Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        
        if config.USE_MULTI_GPU and config.NUM_GPUS > 1:
            print(f"   Using {config.NUM_GPUS} GPUs")
    else:
        print("❌ CUDA is NOT available")
    
    print(f"\n📱 Using device for BERT: {config.BERT_DEVICE}")
    print(f"📱 Using device for LightGBM: {config.LGB_PARAMS.get('device', 'cpu')}")
    
    # Load and merge raw data
    print("\n" + "=" * 60)
    print("📊 Loading data...")
    merged_df, targets_df, candidates_df, book_genres_df, descriptions_df = load_and_merge_data()

    # Apply enhanced feature engineering
    print("\n" + "=" * 60)
    print("🔧 ENHANCED Feature Engineering...")
    
    # Определяем, использовать ли BERT
    include_bert = True
    
    featured_df = create_features_enhanced(
        merged_df, 
        book_genres_df, 
        descriptions_df, 
        include_aggregates=True, 
        include_bert=include_bert,
        include_temporal=True,
        include_sequences=True,
        include_collaborative=True
    )

    # Ensure processed directory exists
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Define the output path
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    # Оптимизация типов данных перед сохранением
    print("\n💾 Optimizing data types before saving...")
    
    # Преобразуем float64 в float32
    float64_cols = featured_df.select_dtypes(include=['float64']).columns
    if len(float64_cols) > 0:
        print(f"   Converting {len(float64_cols)} float64 columns to float32...")
        for col in float64_cols:
            try:
                featured_df[col] = featured_df[col].astype(np.float32)
            except:
                # Если не получается, оставляем как есть
                pass
    
    # Преобразуем int64 в int32 где возможно
    int64_cols = featured_df.select_dtypes(include=['int64']).columns
    converted = 0
    for col in int64_cols:
        try:
            if featured_df[col].max() < 2**31 - 1 and featured_df[col].min() > -2**31:
                featured_df[col] = featured_df[col].astype(np.int32)
                converted += 1
        except:
            pass
    if converted > 0:
        print(f"   Converted {converted} int64 columns to int32")
    
    # Save processed data as parquet
    print(f"\n💾 Saving processed data to {processed_path}...")
    
    try:
        # Пробуем сохранить с разными опциями
        featured_df.to_parquet(
            processed_path, 
            index=False, 
            engine="pyarrow", 
            compression="snappy"
        )
        print("✅ Processed data saved successfully!")
    except Exception as e:
        print(f"⚠️  Error saving with pyarrow: {e}")
        print("   Saving as pickle instead...")
        # Сохраняем как pickle в качестве запасного варианта
        pickle_path = processed_path.with_suffix('.pkl')
        featured_df.to_pickle(pickle_path)
        print(f"✅ Data saved as pickle: {pickle_path}")

    # Print statistics
    train_rows = len(featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN])
    total_features = len(featured_df.columns)
    
    # Анализ признаков
    bert_features = len([col for col in featured_df.columns if col.startswith("bert")])
    tfidf_features = len([col for col in featured_df.columns if col.startswith("tfidf_")])
    svd_features = len([col for col in featured_df.columns if col.startswith(("user_svd_", "item_svd_", "cf_score_"))])
    temporal_features = len([col for col in featured_df.columns if any(keyword in col for keyword in 
                                                                      ['activity', 'interaction', 'hour', 'day', 'temporal'])])
    
    print("\n" + "=" * 60)
    print("✅ ENHANCED data preparation complete!")
    print(f"   Train rows: {train_rows:,}")
    print(f"   Total features: {total_features}")
    print(f"   Feature breakdown:")
    print(f"     - BERT features: {bert_features}")
    print(f"     - TF-IDF features: {tfidf_features}")
    print(f"     - Collaborative features: {svd_features}")
    print(f"     - Temporal features: {temporal_features}")
    print(f"     - Aggregate features: ~50+")
    print(f"   Output file: {processed_path}")
    if hasattr(featured_df, 'memory_usage'):
        print(f"   Memory: {featured_df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    prepare_data()