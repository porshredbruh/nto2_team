"""
Feature engineering script with GPU optimizations and enhanced features.
"""

import gc
import time
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from scipy import sparse

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
    
    # Используем меньший batch size для стабильности
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
        model.requires_grad_(False)
        
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
                
                # Очистка памяти
                del encoded, outputs, mean_pooled, batch_embeddings
                cleanup_gpu_memory()
                
                if device.startswith("cuda"):
                    time.sleep(0.01)
        
        # Save embeddings
        print(f"   💾 Saving BERT embeddings to {embeddings_path}")
        joblib.dump(embeddings_dict, embeddings_path, compress=3)
        
        # Очищаем память
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
    
    # Reduce dimensionality for efficiency
    if embeddings_array.shape[0] > 1000:
        print(f"   🔧 Reducing BERT dimensionality from {embeddings_array.shape[1]} to 128...")
        pca = PCA(n_components=128, random_state=config.RANDOM_STATE)
        embeddings_array = pca.fit_transform(embeddings_array)
        bert_feature_names = [f"bert_pca_{i}" for i in range(128)]
    else:
        bert_feature_names = [f"bert_{i}" for i in range(embeddings_array.shape[1])]
    
    bert_df = pd.DataFrame(embeddings_array, columns=bert_feature_names, index=df.index)
    
    # Concatenate with main DataFrame
    df_with_bert = pd.concat([df.reset_index(drop=True), bert_df.reset_index(drop=True)], axis=1)
    
    print(f"   ✅ Added {len(bert_feature_names)} BERT features")
    return df_with_bert


def add_text_features(df: pd.DataFrame, train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """Adds TF-IDF features from book descriptions."""
    print("\n📝 Adding enhanced text features (TF-IDF)...")

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
        print(f"   ⚙️  Fitting enhanced TF-IDF vectorizer on {len(train_descriptions)} training descriptions...")
        
        # Используем базовые стоп-слова для русского, если доступны
        try:
            # Пробуем загрузить русские стоп-слова из nltk
            import nltk
            from nltk.corpus import stopwords
            try:
                nltk.download('stopwords', quiet=True)
                russian_stop_words = stopwords.words('russian')
                print(f"   Using {len(russian_stop_words)} Russian stop words")
            except:
                russian_stop_words = None
                print("   ⚠️  Could not load Russian stop words, using default")
        except ImportError:
            russian_stop_words = None
            print("   ⚠️  nltk not installed, using default parameters")
        
        vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            stop_words=russian_stop_words if russian_stop_words else None,
            sublinear_tf=True,
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

    # Convert to dense array immediately to avoid sparse issues
    print(f"   Converting sparse TF-IDF matrix to dense array...")
    tfidf_array = tfidf_matrix.toarray().astype(np.float32)
    
    # Convert to DataFrame
    tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_array.shape[1])]
    tfidf_df = pd.DataFrame(
        tfidf_array,
        columns=tfidf_feature_names,
        index=df.index,
    )

    df_with_tfidf = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    print(f"   ✅ Added {len(tfidf_feature_names)} TF-IDF features (dense format)")
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
    """Calculates enhanced aggregate features."""
    print("\n📊 Adding enhanced aggregate features...")

    # User aggregates - более детальная статистика
    user_agg = train_df.groupby(constants.COL_USER_ID).agg({
        config.TARGET: ['mean', 'count', 'std', lambda x: (x == 2).sum()],  # 2 = прочитанные
        constants.COL_HAS_READ: 'mean'
    }).reset_index()
    
    # Создаем имена столбцов
    user_agg_columns = [
        constants.COL_USER_ID,
        constants.F_USER_MEAN_RATING,
        constants.F_USER_RATINGS_COUNT,
        'user_std_rating',
        'user_read_count',
        'user_read_ratio'
    ]
    
    user_agg.columns = user_agg_columns

    # Book aggregates
    book_agg = train_df.groupby(constants.COL_BOOK_ID).agg({
        config.TARGET: ['mean', 'count', 'std', lambda x: (x == 2).sum()],
        constants.COL_HAS_READ: 'mean'
    }).reset_index()
    
    book_agg_columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_MEAN_RATING,
        constants.F_BOOK_RATINGS_COUNT,
        'book_std_rating',
        'book_read_count',
        'book_read_ratio'
    ]
    
    book_agg.columns = book_agg_columns

    # Author aggregates
    author_agg = train_df.groupby(constants.COL_AUTHOR_ID).agg({
        config.TARGET: ['mean', 'count', 'std'],
    }).reset_index()
    
    author_agg_columns = [
        constants.COL_AUTHOR_ID,
        constants.F_AUTHOR_MEAN_RATING,
        'author_ratings_count',
        'author_std_rating'
    ]
    
    author_agg.columns = author_agg_columns

    # Merge aggregates
    df = df.merge(user_agg, on=constants.COL_USER_ID, how="left")
    df = df.merge(book_agg, on=constants.COL_BOOK_ID, how="left")
    df = df.merge(author_agg, on=constants.COL_AUTHOR_ID, how="left")
    
    # Добавляем производные признаки ТОЛЬКО ПОСЛЕ того, как все столбцы созданы
    # Проверяем существование столбцов перед использованием
    if constants.F_USER_RATINGS_COUNT in df.columns and constants.F_BOOK_RATINGS_COUNT in df.columns:
        df['user_book_popularity'] = df[constants.F_USER_RATINGS_COUNT] * df[constants.F_BOOK_RATINGS_COUNT]
    
    if constants.F_USER_MEAN_RATING in df.columns and constants.F_AUTHOR_MEAN_RATING in df.columns:
        df['user_author_affinity'] = df[constants.F_USER_MEAN_RATING] * df[constants.F_AUTHOR_MEAN_RATING]
    
    print(f"   ✅ Added enhanced user, book, and author aggregates")
    return df


def add_temporal_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет временные признаки на основе истории взаимодействий."""
    print("\n⏰ Adding temporal features...")
    
    # Для пользователей
    if constants.COL_TIMESTAMP in train_df.columns:
        train_df = train_df.copy()
        train_df[constants.COL_TIMESTAMP] = pd.to_datetime(train_df[constants.COL_TIMESTAMP])
        
        user_temporal = train_df.groupby(constants.COL_USER_ID).agg({
            constants.COL_TIMESTAMP: ['min', 'max', 'count'],
            constants.COL_HAS_READ: 'mean',
            constants.COL_RELEVANCE: lambda x: (x == 2).sum()  # количество прочитанных
        })
        user_temporal.columns = ['user_first_interaction', 'user_last_interaction', 
                                 'user_total_interactions', 'user_planned_ratio', 'user_read_count']
        
        # Вычисляем время активности пользователя (в днях)
        user_temporal['user_activity_days'] = (user_temporal['user_last_interaction'] - 
                                              user_temporal['user_first_interaction']).dt.days + 1
        user_temporal['user_avg_days_between'] = user_temporal['user_activity_days'] / \
                                                user_temporal['user_total_interactions'].clip(lower=1)
        
        user_temporal['user_read_ratio'] = user_temporal['user_read_count'] / user_temporal['user_total_interactions'].clip(lower=1)
        
        # Преобразуем datetime в Unix timestamp (числа) для совместимости с LightGBM
        user_temporal['user_first_interaction_ts'] = user_temporal['user_first_interaction'].astype(np.int64) // 10**9
        user_temporal['user_last_interaction_ts'] = user_temporal['user_last_interaction'].astype(np.int64) // 10**9
        
        # Удаляем оригинальные datetime колонки
        user_temporal = user_temporal.drop(columns=['user_first_interaction', 'user_last_interaction'])
        
        df = df.merge(user_temporal, on=constants.COL_USER_ID, how='left')
    
    # Для книг
    if constants.COL_TIMESTAMP in train_df.columns:
        book_temporal = train_df.groupby(constants.COL_BOOK_ID).agg({
            constants.COL_TIMESTAMP: ['min', 'max', 'count'],
            constants.COL_HAS_READ: 'mean',
            constants.COL_RELEVANCE: lambda x: (x == 2).sum()
        })
        book_temporal.columns = ['book_first_interaction', 'book_last_interaction',
                                 'book_total_interactions', 'book_planned_ratio', 'book_read_count']
        
        book_temporal['book_read_ratio'] = book_temporal['book_read_count'] / book_temporal['book_total_interactions'].clip(lower=1)
        
        # Преобразуем datetime в Unix timestamp (числа)
        book_temporal['book_first_interaction_ts'] = book_temporal['book_first_interaction'].astype(np.int64) // 10**9
        book_temporal['book_last_interaction_ts'] = book_temporal['book_last_interaction'].astype(np.int64) // 10**9
        
        # Удаляем оригинальные datetime колонки
        book_temporal = book_temporal.drop(columns=['book_first_interaction', 'book_last_interaction'])
        
        df = df.merge(book_temporal, on=constants.COL_BOOK_ID, how='left')
    
    print(f"   ✅ Added temporal features")
    return df


def add_interaction_sequence_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет признаки последовательностей взаимодействий."""
    print("\n🔄 Adding interaction sequence features...")
    
    if constants.COL_TIMESTAMP in train_df.columns:
        train_df_sorted = train_df.sort_values([constants.COL_USER_ID, constants.COL_TIMESTAMP]).copy()
        
        # Признаки паттернов чтения
        user_sequence_stats = train_df_sorted.groupby(constants.COL_USER_ID).agg({
            constants.COL_TIMESTAMP: lambda x: (x.diff().mean().total_seconds() / 3600 
                                                if len(x) > 1 else 0),
            constants.COL_HAS_READ: lambda x: ((x == 1) & (x.shift() == 0)).sum()  # переходы planned -> read
        })
        user_sequence_stats.columns = ['user_avg_hours_between', 'user_conversions']
        
        # Частота по дням недели и часам
        train_df_sorted['hour'] = train_df_sorted[constants.COL_TIMESTAMP].dt.hour
        train_df_sorted['dayofweek'] = train_df_sorted[constants.COL_TIMESTAMP].dt.dayofweek
        
        # Наиболее активный час
        user_active_hour = train_df_sorted.groupby([constants.COL_USER_ID, 'hour']).size().groupby(
            level=0).idxmax().apply(lambda x: x[1] if isinstance(x, tuple) else 0)
        user_active_hour.name = 'user_active_hour'
        
        # Наиболее активный день недели
        user_active_day = train_df_sorted.groupby([constants.COL_USER_ID, 'dayofweek']).size().groupby(
            level=0).idxmax().apply(lambda x: x[1] if isinstance(x, tuple) else 0)
        user_active_day.name = 'user_active_day'
        
        # Объединяем все признаки пользователя
        user_sequence_stats = user_sequence_stats.join(user_active_hour).join(user_active_day)
        df = df.merge(user_sequence_stats, on=constants.COL_USER_ID, how='left')
    
    print(f"   ✅ Added sequence features")
    return df


def add_collaborative_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет коллаборативные признаки через матричные разложения."""
    print("\n👥 Adding collaborative filtering features...")
    
    try:
        # Создаем разреженную матрицу взаимодействий
        from scipy import sparse
        
        # Создаем матрицу user-item с релевантностью
        interaction_matrix = train_df.pivot_table(
            index=constants.COL_USER_ID,
            columns=constants.COL_BOOK_ID,
            values=constants.COL_RELEVANCE,
            fill_value=0,
            aggfunc='mean'  # средняя релевантность
        )
        
        # Заполняем пропуски нулями
        interaction_matrix = interaction_matrix.fillna(0)
        
        # Преобразуем в разреженную матрицу
        sparse_matrix = sparse.csr_matrix(interaction_matrix.values)
        
        # Уменьшаем размерность с помощью SVD
        n_components = min(50, min(sparse_matrix.shape) - 1)
        if n_components > 5:
            svd = TruncatedSVD(n_components=n_components, random_state=config.RANDOM_STATE)
            user_factors = svd.fit_transform(sparse_matrix)
            
            # Создаем DataFrame с факторами пользователей
            user_factors_df = pd.DataFrame(
                user_factors,
                index=interaction_matrix.index,
                columns=[f'user_svd_{i}' for i in range(n_components)]
            )
            
            # Факторы предметов
            item_factors = svd.components_.T
            
            # Для экономии памяти, не создаем полный DataFrame для книг
            # Вместо этого создадим словарь для быстрого доступа
            item_factors_dict = dict(zip(interaction_matrix.columns, item_factors))
            
            # Добавляем факторы пользователей
            df = df.merge(user_factors_df, on=constants.COL_USER_ID, how='left')
            
            # Добавляем факторы книг и вычисляем скалярное произведение
            for i in range(min(10, n_components)):  # Берем только первые 10 компонент
                user_col = f'user_svd_{i}'
                df[f'item_svd_{i}'] = df[constants.COL_BOOK_ID].map(
                    lambda x: item_factors_dict.get(x, [0]*n_components)[i] if x in item_factors_dict else 0
                )
                df[f'cf_score_{i}'] = df[user_col] * df[f'item_svd_{i}']
            
            explained_variance = svd.explained_variance_ratio_.sum()
            print(f"   SVD explained variance: {explained_variance:.3f}")
        
    except Exception as e:
        print(f"   ⚠️  Collaborative features failed: {e}")
    
    print(f"   ✅ Added collaborative features")
    return df


def add_genre_features(df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    """Adds enhanced genre count feature."""
    print("\n📚 Adding enhanced genre features...")
    
    # Базовая статистика по жанрам
    genre_counts = book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID].agg(['count', 'nunique']).reset_index()
    genre_counts.columns = [constants.COL_BOOK_ID, constants.F_BOOK_GENRES_COUNT, 'book_unique_genres']
    
    # Популярность жанров (сколько раз книга встречалась в жанре)
    genre_popularity = book_genres_df.groupby(constants.COL_GENRE_ID).size().reset_index(name='genre_popularity')
    book_genres_with_pop = book_genres_df.merge(genre_popularity, on=constants.COL_GENRE_ID)
    book_genre_stats = book_genres_with_pop.groupby(constants.COL_BOOK_ID).agg({
        'genre_popularity': ['mean', 'max', 'min']
    }).reset_index()
    book_genre_stats.columns = [constants.COL_BOOK_ID, 'genre_pop_mean', 'genre_pop_max', 'genre_pop_min']
    
    # Объединяем все признаки жанров
    df_with_genres = df.merge(genre_counts, on=constants.COL_BOOK_ID, how='left')
    df_with_genres = df_with_genres.merge(book_genre_stats, on=constants.COL_BOOK_ID, how='left')
    
    print(f"   ✅ Added enhanced genre features")
    return df_with_genres


def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values with enhanced strategy."""
    print("\n🔧 Handling missing values with enhanced strategy...")

    # Глобальные статистики из train
    global_mean = train_df[config.TARGET].mean() if len(train_df) > 0 else 0
    age_median = train_df[constants.COL_AGE].median() if len(train_df) > 0 else 30
    
    # Заполнение базовых признаков
    df[constants.COL_AGE] = df[constants.COL_AGE].fillna(age_median)
    
    # Список числовых признаков для заполнения медианой
    numeric_features = [
        constants.F_USER_MEAN_RATING,
        constants.F_BOOK_MEAN_RATING,
        constants.F_AUTHOR_MEAN_RATING,
        constants.COL_AVG_RATING,
        'user_std_rating',
        'book_std_rating',
        'author_std_rating',
        'user_read_ratio',
        'book_read_ratio',
        'user_planned_ratio',
        'book_planned_ratio',
    ]
    
    for feature in numeric_features:
        if feature in df.columns:
            train_median = train_df[feature].median() if feature in train_df.columns else global_mean
            df[feature] = df[feature].fillna(train_median)
    
    # Список count признаков для заполнения 0
    count_features = [
        constants.F_USER_RATINGS_COUNT,
        constants.F_BOOK_RATINGS_COUNT,
        constants.F_BOOK_GENRES_COUNT,
        'user_read_count',
        'book_read_count',
        'author_ratings_count',
        'user_total_interactions',
        'book_total_interactions',
        'user_conversions',
    ]
    
    for feature in count_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna(0)
    
    # Заполнение временных признаков
    temporal_features = [
        'user_activity_days',
        'user_avg_days_between',
        'user_avg_hours_between',
    ]
    
    for feature in temporal_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna(df[feature].median() if df[feature].notna().any() else 0)
    
    # Заполнение категориальных признаков
    for col in config.CAT_FEATURES:
        if col in df.columns:
            if df[col].dtype.name in ("category", "object") and df[col].isna().any():
                df[col] = df[col].astype(str).fillna(constants.MISSING_CAT_VALUE).astype("category")
            elif pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].isna().any():
                df[col] = df[col].fillna(constants.MISSING_NUM_VALUE)
    
    # Заполнение text и BERT признаков
    tfidf_cols = [col for col in df.columns if col.startswith("tfidf_")]
    for col in tfidf_cols:
        df[col] = df[col].fillna(0.0)
    
    bert_cols = [col for col in df.columns if col.startswith("bert")]
    for col in bert_cols:
        df[col] = df[col].fillna(0.0)
    
    # Заполнение collaborative features
    svd_cols = [col for col in df.columns if col.startswith(("user_svd_", "item_svd_", "cf_score_"))]
    for col in svd_cols:
        df[col] = df[col].fillna(0.0)
    
    print(f"   ✅ Missing values handled")
    return df


def create_features_enhanced(
    df: pd.DataFrame,
    book_genres_df: pd.DataFrame,
    descriptions_df: pd.DataFrame,
    include_aggregates: bool = True,
    include_bert: bool = True,
    include_temporal: bool = True,
    include_sequences: bool = True,
    include_collaborative: bool = True,
) -> pd.DataFrame:
    """Enhanced feature engineering pipeline."""
    print("\n" + "=" * 60)
    print("🚀 ENHANCED Feature Engineering Pipeline")
    print("=" * 60)
    
    cleanup_gpu_memory()
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()
    
    # 1. Базовые признаки взаимодействия
    print("\n1️⃣  Adding basic interaction features...")
    df = add_interaction_feature(df, train_df)
    
    # 2. Агрегированные признаки
    if include_aggregates:
        print("\n2️⃣  Adding aggregate features...")
        df = add_aggregate_features(df, train_df)
    
    # 3. Временные признаки
    if include_temporal and constants.COL_TIMESTAMP in train_df.columns:
        print("\n3️⃣  Adding temporal features...")
        df = add_temporal_features(df, train_df)
    
    # 4. Признаки последовательностей
    if include_sequences and constants.COL_TIMESTAMP in train_df.columns:
        print("\n4️⃣  Adding sequence features...")
        df = add_interaction_sequence_features(df, train_df)
    
    # 5. Коллаборативные признаки
    if include_collaborative:
        print("\n5️⃣  Adding collaborative features...")
        df = add_collaborative_features(df, train_df)
    
    # 6. Признаки жанров
    print("\n6️⃣  Adding genre features...")
    df = add_genre_features(df, book_genres_df)
    
    # 7. Текстовые признаки
    print("\n7️⃣  Adding text features...")
    df = add_text_features(df, train_df, descriptions_df)
    
    # 8. BERT эмбеддинги
    if include_bert:
        print("\n8️⃣  Adding BERT features...")
        df = add_bert_features_gpu(df, descriptions_df, config.BERT_DEVICE)
    
    # 9. Обработка пропусков
    print("\n9️⃣  Handling missing values...")
    df = handle_missing_values(df, train_df)
    
    # 10. Конвертация категориальных признаков
    print("\n🔟  Converting categorical features...")
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")
    
    print("\n" + "=" * 60)
    print(f"✅ ENHANCED features complete!")
    print(f"   Final shape: {df.shape}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
    print("=" * 60)
    
    cleanup_gpu_memory()
    return df