"""
Data loading and merging script.
"""

from typing import Any

import pandas as pd

from . import config, constants


def load_and_merge_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads raw data files and merges them into a single DataFrame.

    For Stage 2B, uses all records from train.csv (both has_read=0 and has_read=1).
    Also loads targets.csv and candidates.csv for ranking task.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            A tuple containing:
            - The merged train DataFrame (with metadata).
            - The targets DataFrame (user_id list).
            - The candidates DataFrame (user_id, book_id_list).
            - The book_genres DataFrame.
            - The book_descriptions DataFrame.
    """
    print("Loading data...")

    # Define dtypes for memory optimization
    dtype_spec: dict[str, Any] = {
        constants.COL_USER_ID: "int32",
        constants.COL_BOOK_ID: "int32",
        constants.COL_HAS_READ: "int32",
        constants.COL_GENDER: "category",
        constants.COL_AGE: "float32",
        constants.COL_AUTHOR_ID: "int32",
        constants.COL_PUBLICATION_YEAR: "float32",
        constants.COL_LANGUAGE: "category",
        constants.COL_PUBLISHER: "category",
        constants.COL_AVG_RATING: "float32",
        constants.COL_GENRE_ID: "int16",
    }

    # Load train.csv - IMPORTANT: use all records (has_read=0 and has_read=1)
    train_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.TRAIN_FILENAME,
        dtype={
            k: v
            for k, v in dtype_spec.items()
            if k in [constants.COL_USER_ID, constants.COL_BOOK_ID, constants.COL_HAS_READ]
        },
        parse_dates=[constants.COL_TIMESTAMP],
    )
    print(f"Loaded train data: {len(train_df):,} rows (using all records: has_read=0 and has_read=1)")

    # Create relevance target variable for multiclass classification
    # has_read=1 -> relevance=2 (read books)
    # has_read=0 -> relevance=1 (planned books)
    # Note: relevance=0 (cold candidates) will be created later when processing candidates
    train_df[constants.COL_RELEVANCE] = train_df[constants.COL_HAS_READ].map({1: 2, 0: 1}).astype("int8")
    print(f"Created relevance target: {train_df[constants.COL_RELEVANCE].value_counts().to_dict()}")

    # Load targets.csv
    targets_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.TARGETS_FILENAME,
        dtype={constants.COL_USER_ID: "int32"},
    )
    print(f"Loaded targets: {len(targets_df):,} users")

    # Load candidates.csv
    candidates_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.CANDIDATES_FILENAME,
        dtype={constants.COL_USER_ID: "int32"},
    )
    print(f"Loaded candidates: {len(candidates_df):,} users")

    # Load metadata
    user_data_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.USER_DATA_FILENAME,
        dtype={
            k: v
            for k, v in dtype_spec.items()
            if k in [constants.COL_USER_ID, constants.COL_GENDER, constants.COL_AGE]
        },
    )
    book_data_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.BOOK_DATA_FILENAME,
        dtype={
            k: v
            for k, v in dtype_spec.items()
            if k
            in [
                constants.COL_BOOK_ID,
                constants.COL_AUTHOR_ID,
                constants.COL_PUBLICATION_YEAR,
                constants.COL_LANGUAGE,
                constants.COL_AVG_RATING,
                constants.COL_PUBLISHER,
            ]
        },
    )
    book_genres_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.BOOK_GENRES_FILENAME,
        dtype={
            k: v for k, v in dtype_spec.items() if k in [constants.COL_BOOK_ID, constants.COL_GENRE_ID]
        },
    )
    genres_df = pd.read_csv(config.RAW_DATA_DIR / constants.GENRES_FILENAME)
    book_descriptions_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.BOOK_DESCRIPTIONS_FILENAME,
        dtype={constants.COL_BOOK_ID: "int32"},
    )

    print("Data loaded. Merging datasets...")

    # Add source column to train
    train_df[constants.COL_SOURCE] = constants.VAL_SOURCE_TRAIN

    # Join metadata
    train_df = train_df.merge(user_data_df, on=constants.COL_USER_ID, how="left")

    # Drop duplicates from book_data_df before merging
    book_data_df = book_data_df.drop_duplicates(subset=[constants.COL_BOOK_ID])
    train_df = train_df.merge(book_data_df, on=constants.COL_BOOK_ID, how="left")

    print(f"Merged train data shape: {train_df.shape}")
    return train_df, targets_df, candidates_df, book_genres_df, book_descriptions_df


def expand_candidates(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Expands candidates.csv from (user_id, book_id_list) to (user_id, book_id) pairs.

    Args:
        candidates_df: DataFrame with columns user_id and book_id_list,
            where book_id_list is a comma-separated string of book IDs.

    Returns:
        pd.DataFrame: DataFrame with columns user_id and book_id, with one row
            per (user_id, book_id) pair.
    """
    print("Expanding candidates...")

    expanded_rows = []
    for _, row in candidates_df.iterrows():
        user_id = row[constants.COL_USER_ID]
        book_id_list_str = row[constants.COL_BOOK_ID_LIST]

        # Handle empty string
        if pd.isna(book_id_list_str) or book_id_list_str == "":
            continue

        # Split by comma and clean up
        book_ids = [int(book_id.strip()) for book_id in book_id_list_str.split(",") if book_id.strip()]

        # Create rows for each book_id
        for book_id in book_ids:
            expanded_rows.append({constants.COL_USER_ID: user_id, constants.COL_BOOK_ID: book_id})

    expanded_df = pd.DataFrame(expanded_rows)
    # Convert to proper dtypes
    expanded_df[constants.COL_USER_ID] = expanded_df[constants.COL_USER_ID].astype("int32")
    expanded_df[constants.COL_BOOK_ID] = expanded_df[constants.COL_BOOK_ID].astype("int32")
    print(f"Expanded candidates: {len(candidates_df):,} users -> {len(expanded_df):,} pairs")
    return expanded_df

