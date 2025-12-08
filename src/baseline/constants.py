"""
Project-wide constants with enhancements.
"""

# --- FILENAMES ---
TRAIN_FILENAME = "train.csv"
TARGETS_FILENAME = "targets.csv"
CANDIDATES_FILENAME = "candidates.csv"
USER_DATA_FILENAME = "users.csv"
BOOK_DATA_FILENAME = "books.csv"
BOOK_GENRES_FILENAME = "book_genres.csv"
GENRES_FILENAME = "genres.csv"
BOOK_DESCRIPTIONS_FILENAME = "book_descriptions.csv"
SUBMISSION_FILENAME = "submission.csv"
TFIDF_VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"
BERT_EMBEDDINGS_FILENAME = "bert_embeddings.pkl"
BERT_MODEL_NAME = "DeepPavlov/rubert-base-cased"
PROCESSED_DATA_FILENAME = "processed_features.parquet"

# --- COLUMN NAMES ---
# Main columns
COL_USER_ID = "user_id"
COL_BOOK_ID = "book_id"
COL_TARGET = "has_read"
COL_RELEVANCE = "relevance"
COL_SOURCE = "source"
COL_PREDICTION = "rating_predict"
COL_HAS_READ = "has_read"
COL_TIMESTAMP = "timestamp"
COL_BOOK_ID_LIST = "book_id_list"

# Feature columns
F_USER_MEAN_RATING = "user_mean_rating"
F_USER_RATINGS_COUNT = "user_ratings_count"
F_BOOK_MEAN_RATING = "book_mean_rating"
F_BOOK_RATINGS_COUNT = "book_ratings_count"
F_AUTHOR_MEAN_RATING = "author_mean_rating"
F_BOOK_GENRES_COUNT = "book_genres_count"
F_USER_BOOK_INTERACTION = "f_user_book_interaction"

# Metadata columns
COL_GENDER = "gender"
COL_AGE = "age"
COL_AUTHOR_ID = "author_id"
COL_PUBLICATION_YEAR = "publication_year"
COL_LANGUAGE = "language"
COL_PUBLISHER = "publisher"
COL_AVG_RATING = "avg_rating"
COL_GENRE_ID = "genre_id"
COL_DESCRIPTION = "description"

# --- VALUES ---
VAL_SOURCE_TRAIN = "train"
VAL_SOURCE_TEST = "test"

# --- MAGIC NUMBERS ---
MISSING_CAT_VALUE = "-1"
MISSING_NUM_VALUE = -1
PREDICTION_MIN_VALUE = 0
PREDICTION_MAX_VALUE = 2
MAX_RANKING_LENGTH = 20

# --- NEW FEATURE CATEGORIES ---
TEMPORAL_FEATURES = [
    'user_first_interaction',
    'user_last_interaction', 
    'user_total_interactions',
    'user_activity_days',
    'user_avg_days_between',
    'user_read_ratio',
    'user_planned_ratio',
    'book_first_interaction',
    'book_last_interaction',
    'book_total_interactions',
    'book_read_ratio',
    'book_planned_ratio',
]

SEQUENCE_FEATURES = [
    'user_avg_hours_between',
    'user_conversions',
    'user_active_hour',
    'user_active_day',
]

COLLABORATIVE_FEATURES = [
    'user_svd_',
    'item_svd_', 
    'cf_score_',
]

# --- MODEL TYPES ---
MODEL_LIGHTGBM = "lightgbm"
MODEL_CATBOOST = "catboost"
MODEL_ENSEMBLE = "ensemble"