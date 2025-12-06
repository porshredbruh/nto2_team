"""
Feature engineering script with GPU optimizations.
"""

import gc
import time
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from . import config, constants


def cleanup_gpu_memory() -> None:
    """Очищает память GPU."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def add_bert_features_gpu(
    df: pd.DataFrame, 
    descriptions_df: pd.DataFrame,
    device: Optional[str] = None
) -> pd.DataFrame:
    """Adds BERT embeddings using GPU with memory optimizations."""
    
    if device is None:
        device = config.BERT_DEVICE
        
    print(f"\n🤖 Adding BERT embeddings using {device.upper()}...")
    
    # Если устройство CPU, уменьшаем batch size
    batch_size = config.BERT_BATCH_SIZE if device.startswith("cuda") else 4
    
    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path = config.MODEL_DIR / constants.BERT_EMBEDDINGS_FILENAME

    # Check if embeddings are already cached
    if embeddings_path.exists():
        print(f"   📂 Loading cached BERT embeddings from {embeddings_path}")
        try:
            embeddings_dict = joblib.load(embeddings_path)
            print(f"   ✅ Loaded {len(embeddings_dict)} embeddings from cache")
        except Exception as e:
            print(f"   ⚠️  Failed to load cache: {e}, recomputing...")
            embeddings_dict = None
    else:
        embeddings_dict = None
    
    if embeddings_dict is None:
        print(f"   ⚙️  Computing BERT embeddings on {device}...")
        
        # Очищаем память перед началом
        cleanup_gpu_memory()
        
        # Load tokenizer and model
        print(f"   📥 Loading BERT model: {config.BERT_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        model = AutoModel.from_pretrained(config.BERT_MODEL_NAME)
        model.to(device)
        model.eval()
        model.requires_grad_(False)  # Отключаем градиенты для экономии памяти
        
        # Подготовка описаний
        all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
        all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")
        
        # Get unique books
        unique_books = all_descriptions.drop_duplicates(subset=[constants.COL_BOOK_ID])
        book_ids = unique_books[constants.COL_BOOK_ID].to_numpy()
        descriptions = unique_books[constants.COL_DESCRIPTION].to_numpy().tolist()
        
        # Initialize embeddings dictionary
        embeddings_dict = {}
        
        # Process in batches with progress bar
        num_batches = (len(descriptions) + batch_size - 1) // batch_size
        
        print(f"   📊 Processing {len(descriptions)} books in {num_batches} batches (batch_size={batch_size})")
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="   BERT Progress", unit="batch"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(descriptions))
                batch_descriptions = descriptions[start_idx:end_idx]
                batch_book_ids = book_ids[start_idx:end_idx]
                
                # Skip empty batches
                if not batch_descriptions:
                    continue
                
                # Tokenize batch
                encoded = tokenizer(
                    batch_descriptions,
                    padding=True,
                    truncation=True,
                    max_length=config.BERT_MAX_LENGTH,
                    return_tensors="pt",
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                
                # Get embeddings
                outputs = model(**encoded)
                
                # Mean pooling
                attention_mask = encoded["attention_mask"]
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                
                sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask_expanded, dim=1)
                sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask
                
                # Move to CPU and store
                batch_embeddings = mean_pooled.cpu().numpy()
                
                for book_id, embedding in zip(batch_book_ids, batch_embeddings, strict=False):
                    embeddings_dict[book_id] = embedding
                
                # Очистка памяти после каждого батча
                del encoded, outputs, mean_pooled, batch_embeddings
                cleanup_gpu_memory()
                
                # Пауза для предотвращения перегрева
                if device.startswith("cuda"):
                    time.sleep(0.05)
        
        # Save embeddings
        print(f"   💾 Saving BERT embeddings to {embeddings_path}")
        joblib.dump(embeddings_dict, embeddings_path, compress=3)
        
        # Очищаем память после завершения
        del model, tokenizer
        cleanup_gpu_memory()
        print(f"   ✅ BERT embeddings computed and saved ({len(embeddings_dict)} embeddings)")
    
    # Map embeddings to DataFrame
    print(f"   🔗 Mapping embeddings to {len(df)} rows...")
    df_book_ids = df[constants.COL_BOOK_ID].to_numpy()
    embeddings_list = []
    
    missing_count = 0
    for book_id in df_book_ids:
        if book_id in embeddings_dict:
            embeddings_list.append(embeddings_dict[book_id])
        else:
            embeddings_list.append(np.zeros(config.BERT_EMBEDDING_DIM))
            missing_count += 1
    
    if missing_count > 0:
        print(f"   ⚠️  {missing_count} books without BERT embeddings (using zeros)")
    
    embeddings_array = np.array(embeddings_list)
    
    # Create BERT features DataFrame
    bert_feature_names = [f"bert_{i}" for i in range(config.BERT_EMBEDDING_DIM)]
    bert_df = pd.DataFrame(embeddings_array, columns=bert_feature_names, index=df.index)
    
    # Concatenate with main DataFrame
    df_with_bert = pd.concat([df.reset_index(drop=True), bert_df.reset_index(drop=True)], axis=1)
    
    print(f"   ✅ Added {len(bert_feature_names)} BERT features")
    return df_with_bert


def add_text_features(df: pd.DataFrame, train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """Adds TF-IDF features from book descriptions."""
    print("\n📝 Adding text features (TF-IDF)...")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer_path = config.MODEL_DIR / constants.TFIDF_VECTORIZER_FILENAME

    # Get unique books from train set
    train_books = train_df[constants.COL_BOOK_ID].unique()
    train_descriptions = descriptions_df[descriptions_df[constants.COL_BOOK_ID].isin(train_books)].copy()
    train_descriptions[constants.COL_DESCRIPTION] = train_descriptions[constants.COL_DESCRIPTION].fillna("")

    # Load or fit vectorizer
    if vectorizer_path.exists():
        print(f"   📂 Loading existing vectorizer from {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
    else:
        print(f"   ⚙️  Fitting TF-IDF vectorizer on {len(train_descriptions)} training descriptions...")
        vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            ngram_range=config.TFIDF_NGRAM_RANGE,
        )
        vectorizer.fit(train_descriptions[constants.COL_DESCRIPTION])
        joblib.dump(vectorizer, vectorizer_path)
        print(f"   💾 Vectorizer saved to {vectorizer_path}")

    # Transform all descriptions
    all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
    all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")
    description_map = dict(
        zip(all_descriptions[constants.COL_BOOK_ID], all_descriptions[constants.COL_DESCRIPTION], strict=False)
    )

    df_descriptions = df[constants.COL_BOOK_ID].map(description_map).fillna("")
    tfidf_matrix = vectorizer.transform(df_descriptions)

    # Convert to DataFrame
    tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf_feature_names,
        index=df.index,
    )

    df_with_tfidf = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    print(f"   ✅ Added {len(tfidf_feature_names)} TF-IDF features")
    return df_with_tfidf


def add_interaction_feature(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Adds binary interaction feature."""
    print("\n🤝 Adding interaction feature...")

    interaction_pairs = set(
        zip(train_df[constants.COL_USER_ID], train_df[constants.COL_BOOK_ID], strict=False)
    )

    df[constants.F_USER_BOOK_INTERACTION] = df.apply(
        lambda row: 1
        if (row[constants.COL_USER_ID], row[constants.COL_BOOK_ID]) in interaction_pairs
        else 0,
        axis=1,
    ).astype("int8")

    interaction_count = df[constants.F_USER_BOOK_INTERACTION].sum()
    print(f"   ✅ Interactions found: {interaction_count:,} / {len(df):,} ({100 * interaction_count / len(df):.1f}%)")
    return df


def add_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates aggregate features."""
    print("\n📊 Adding aggregate features...")

    # User aggregates
    user_agg = train_df.groupby(constants.COL_USER_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    user_agg.columns = [
        constants.COL_USER_ID,
        constants.F_USER_MEAN_RATING,
        constants.F_USER_RATINGS_COUNT,
    ]

    # Book aggregates
    book_agg = train_df.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    book_agg.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_MEAN_RATING,
        constants.F_BOOK_RATINGS_COUNT,
    ]

    # Author aggregates
    author_agg = train_df.groupby(constants.COL_AUTHOR_ID)[config.TARGET].agg(["mean"]).reset_index()
    author_agg.columns = [constants.COL_AUTHOR_ID, constants.F_AUTHOR_MEAN_RATING]

    # Merge aggregates
    df = df.merge(user_agg, on=constants.COL_USER_ID, how="left")
    df = df.merge(book_agg, on=constants.COL_BOOK_ID, how="left")
    df = df.merge(author_agg, on=constants.COL_AUTHOR_ID, how="left")
    
    print(f"   ✅ Added user, book, and author aggregates")
    return df


def add_genre_features(df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    """Adds genre count feature."""
    print("\n📚 Adding genre features...")
    genre_counts = book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID].count().reset_index()
    genre_counts.columns = [constants.COL_BOOK_ID, constants.F_BOOK_GENRES_COUNT]
    df_with_genres = df.merge(genre_counts, on=constants.COL_BOOK_ID, how="left")
    print(f"   ✅ Added genre count feature")
    return df_with_genres


def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values."""
    print("\n🔧 Handling missing values...")

    global_mean = train_df[config.TARGET].mean()
    age_median = df[constants.COL_AGE].median()
    df[constants.COL_AGE] = df[constants.COL_AGE].fillna(age_median)

    # Fill aggregate features
    if constants.F_USER_MEAN_RATING in df.columns:
        df[constants.F_USER_MEAN_RATING] = df[constants.F_USER_MEAN_RATING].fillna(global_mean)
    if constants.F_BOOK_MEAN_RATING in df.columns:
        df[constants.F_BOOK_MEAN_RATING] = df[constants.F_BOOK_MEAN_RATING].fillna(global_mean)
    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        df[constants.F_AUTHOR_MEAN_RATING] = df[constants.F_AUTHOR_MEAN_RATING].fillna(global_mean)

    if constants.F_USER_RATINGS_COUNT in df.columns:
        df[constants.F_USER_RATINGS_COUNT] = df[constants.F_USER_RATINGS_COUNT].fillna(0)
    if constants.F_BOOK_RATINGS_COUNT in df.columns:
        df[constants.F_BOOK_RATINGS_COUNT] = df[constants.F_BOOK_RATINGS_COUNT].fillna(0)

    df[constants.COL_AVG_RATING] = df[constants.COL_AVG_RATING].fillna(global_mean)
    df[constants.F_BOOK_GENRES_COUNT] = df[constants.F_BOOK_GENRES_COUNT].fillna(0)

    # Fill text features
    tfidf_cols = [col for col in df.columns if col.startswith("tfidf_")]
    for col in tfidf_cols:
        df[col] = df[col].fillna(0.0)

    bert_cols = [col for col in df.columns if col.startswith("bert_")]
    for col in bert_cols:
        df[col] = df[col].fillna(0.0)

    # Fill categorical features
    for col in config.CAT_FEATURES:
        if col in df.columns:
            if df[col].dtype.name in ("category", "object") and df[col].isna().any():
                df[col] = df[col].astype(str).fillna(constants.MISSING_CAT_VALUE).astype("category")
            elif pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].isna().any():
                df[col] = df[col].fillna(constants.MISSING_NUM_VALUE)

    print(f"   ✅ Missing values handled")
    return df


def create_features(
    df: pd.DataFrame,
    book_genres_df: pd.DataFrame,
    descriptions_df: pd.DataFrame,
    include_aggregates: bool = False,
    include_bert: bool = True,
) -> pd.DataFrame:
    """Runs the full feature engineering pipeline."""
    print("\n" + "=" * 60)
    print("🚀 Starting feature engineering pipeline...")
    print("=" * 60)
    
    # Очищаем память GPU перед началом
    cleanup_gpu_memory()
    
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # Add features in sequence
    df = add_interaction_feature(df, train_df)
    
    if include_aggregates:
        df = add_aggregate_features(df, train_df)
    
    df = add_genre_features(df, book_genres_df)
    df = add_text_features(df, train_df, descriptions_df)
    
    if include_bert:
        df = add_bert_features_gpu(df, descriptions_df, config.BERT_DEVICE)
    else:
        print("\n⚠️  BERT features disabled (include_bert=False)")
    
    df = handle_missing_values(df, train_df)

    # Convert categorical columns
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print("\n" + "=" * 60)
    print("✅ Feature engineering complete.")
    print(f"   Final shape: {df.shape}")
    print("=" * 60)
    
    # Очищаем память после завершения
    cleanup_gpu_memory()
    
    return df