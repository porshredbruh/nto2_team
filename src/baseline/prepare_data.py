"""
Data preparation script that processes raw data and saves it to processed directory.
"""

import torch
from . import config, constants
from .data_processing import load_and_merge_data
from .features import create_features


def prepare_data() -> None:
    """Processes raw data and saves prepared features."""
    print("=" * 60)
    print("Data Preparation Pipeline")
    print("=" * 60)
    
    # Диагностика GPU
    print("\n🎮 GPU Diagnostics:")
    if torch.cuda.is_available():
        print(f"✅ CUDA is available")
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"      Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ CUDA is NOT available")
        print("   PyTorch was likely installed without CUDA support")
        print("   To install PyTorch with CUDA, run:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print(f"\n📱 Using device for BERT: {config.BERT_DEVICE}")
    print(f"📱 Using device for LightGBM: {config.LGB_PARAMS.get('device', 'cpu')}")
    
    # Проверка доступности GPU для LightGBM
    try:
        import lightgbm as lgb
        print(f"✅ LightGBM version: {lgb.__version__}")
        # Проверка поддержки GPU в LightGBM
        if config.USE_GPU:
            try:
                # Создаем тестовые данные для проверки GPU
                import numpy as np
                X_test = np.random.rand(100, 10)
                y_test = np.random.randint(0, 3, 100)
                params = {"device": "gpu", "verbosity": -1}
                test_model = lgb.LGBMClassifier(**params)
                test_model.fit(X_test, y_test)
                print("✅ LightGBM GPU test passed")
            except Exception as e:
                print(f"⚠️  LightGBM GPU test failed: {e}")
                print("   LightGBM may not be compiled with GPU support")
    except ImportError:
        print("❌ LightGBM not installed")

    # Load and merge raw data
    print("\n" + "=" * 60)
    print("📊 Loading data...")
    merged_df, targets_df, candidates_df, book_genres_df, descriptions_df = load_and_merge_data()

    # Apply feature engineering WITH BERT
    print("\n" + "=" * 60)
    print("🔧 Feature Engineering...")
    
    # Определяем, использовать ли BERT (в зависимости от наличия GPU)
    # На CPU BERT очень медленный, можно отключить для тестирования
    include_bert = True  # Всегда включаем, т.к. это важные фичи
    
    featured_df = create_features(
        merged_df, 
        book_genres_df, 
        descriptions_df, 
        include_aggregates=False, 
        include_bert=include_bert
    )

    # Ensure processed directory exists
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Define the output path
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    # Save processed data as parquet
    print(f"\n💾 Saving processed data to {processed_path}...")
    featured_df.to_parquet(processed_path, index=False, engine="pyarrow", compression="snappy")
    print("✅ Processed data saved successfully!")

    # Print statistics
    train_rows = len(featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN])
    total_features = len(featured_df.columns)
    
    # Посчитать BERT фичи
    bert_features = len([col for col in featured_df.columns if col.startswith("bert_")])
    tfidf_features = len([col for col in featured_df.columns if col.startswith("tfidf_")])

    print("\n" + "=" * 60)
    print("✅ Data preparation complete!")
    print(f"   Train rows: {train_rows:,}")
    print(f"   Total features: {total_features}")
    print(f"     - BERT features: {bert_features}")
    print(f"     - TF-IDF features: {tfidf_features}")
    print(f"   Output file: {processed_path}")
    print(f"   GPU used for BERT: {'Yes' if include_bert and config.BERT_DEVICE.startswith('cuda') else 'No'}")
    print("=" * 60)


if __name__ == "__main__":
    prepare_data()