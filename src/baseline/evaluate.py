"""Evaluation script for Stage 2 predictions.

This script evaluates submissions against a solution file that contains true
relevance labels and public/private split information. Note: the solution file
itself is not provided to participants, only this evaluation logic.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _validate_submission_columns(df: pd.DataFrame) -> None:
    """Validate submission file has required columns and no extra columns."""
    required_cols = {"user_id", "book_id_list"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}. Expected: {required_cols}")

    extra_cols = set(df.columns) - required_cols
    if extra_cols:
        raise ValueError(f"Found extra columns: {extra_cols}. Expected only: {required_cols}")


def _validate_submission_user_id(df: pd.DataFrame) -> None:
    """Validate submission user_id column."""
    if df["user_id"].isna().any():
        null_count = df["user_id"].isna().sum()
        raise ValueError(f"Column 'user_id' contains {null_count} null values")

    try:
        df["user_id"] = pd.to_numeric(df["user_id"], errors="raise")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Column 'user_id' contains non-numeric values: {e}") from e

    duplicates = df.duplicated(subset=["user_id"])
    if duplicates.any():
        dup_count = duplicates.sum()
        dup_examples = df[duplicates][["user_id"]].to_dict("records")[:5]
        raise ValueError(f"Found {dup_count} duplicate user_id values. Examples: {dup_examples}")


def _validate_submission_book_id_list(df: pd.DataFrame) -> None:
    """Validate submission book_id_list column format and content."""
    # Note: Empty strings and NaN are allowed (represent no predictions for a user)

    max_books = 20
    invalid_rows = []

    for idx, row in df.iterrows():
        book_list_str = row["book_id_list"]

        # Empty string is allowed (no predictions for this user)
        if pd.isna(book_list_str) or book_list_str == "":
            continue

        try:
            book_ids = [int(x.strip()) for x in str(book_list_str).split(",") if x.strip()]
        except ValueError as e:
            invalid_rows.append((idx, row["user_id"], f"Invalid format: {e}"))
            continue

        # Check for duplicates within the list
        if len(book_ids) != len(set(book_ids)):
            duplicates = [x for x in book_ids if book_ids.count(x) > 1]
            invalid_rows.append((idx, row["user_id"], f"Duplicate book_ids in list: {set(duplicates)}"))

        # Check maximum length
        if len(book_ids) > max_books:
            invalid_rows.append((idx, row["user_id"], f"List contains {len(book_ids)} books, maximum is {max_books}"))

    if invalid_rows:
        examples = [{"row": r[0], "user_id": r[1], "error": r[2]} for r in invalid_rows[:5]]
        raise ValueError(f"Found {len(invalid_rows)} invalid book_id_list entries. Examples: {examples}")


def _validate_submission_user_ids_match(df: pd.DataFrame, solution_df: pd.DataFrame) -> None:
    """Validate that submission contains exactly the same user_ids as solution."""
    solution_users = set(solution_df["user_id"].unique())
    submission_users = set(df["user_id"].unique())

    missing_users = solution_users - submission_users
    if missing_users:
        examples = list(missing_users)[:5]
        raise ValueError(f"Missing {len(missing_users)} required user_ids from solution. Examples: {examples}")

    extra_users = submission_users - solution_users
    if extra_users:
        examples = list(extra_users)[:5]
        raise ValueError(f"Found {len(extra_users)} extra user_ids not in solution. Examples: {examples}")


def validate_submission_format(df: pd.DataFrame, solution_df: pd.DataFrame) -> None:
    """Validate submission file format and content.

    Raises:
        ValueError: On any format or content error.
    """
    if df.empty:
        raise ValueError("Submission file is empty")

    _validate_submission_columns(df)
    _validate_submission_user_id(df)
    _validate_submission_book_id_list(df)
    _validate_submission_user_ids_match(df, solution_df)


def _validate_solution_columns(df: pd.DataFrame) -> None:
    """Validate solution file has required columns."""
    required_cols = {"user_id", "book_id_list_read", "book_id_list_planned", "stage"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}. Expected: {required_cols}")


def _validate_solution_stage(df: pd.DataFrame) -> None:
    """Validate solution stage column."""
    if df["stage"].isna().any():
        null_count = df["stage"].isna().sum()
        raise ValueError(f"Column 'stage' contains {null_count} null values. Must be 'public' or 'private'")

    valid_stages = {"public", "private"}
    invalid = set(df["stage"].unique()) - valid_stages
    if invalid:
        raise ValueError(f"Invalid stage values: {invalid}. Allowed: {valid_stages}")

    if "public" not in df["stage"].to_numpy():
        raise ValueError("No records with stage='public' found in solution")

    if "private" not in df["stage"].to_numpy():
        raise ValueError("No records with stage='private' found in solution")


def validate_solution_format(df: pd.DataFrame) -> None:
    """Validate solution file format and content.

    Raises:
        ValueError: On any format or content error.
    """
    if df.empty:
        raise ValueError("Solution file is empty")

    _validate_solution_columns(df)
    _validate_solution_stage(df)


def dcg_at_k(relevance_scores: list[int], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    scores_array = np.asarray(relevance_scores, dtype=float)[:k]
    if scores_array.size:
        return float(np.sum(scores_array / np.log2(np.arange(2, scores_array.size + 2))))
    return 0.0


def ndcg_at_k(relevance_scores: list[int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k with multi-level relevance.

    Args:
        relevance_scores: List of relevance scores for each position (0, 1, or 2)
        k: Number of positions to evaluate

    Returns:
        NDCG@k in range [0.0, 1.0]
    """
    if len(relevance_scores) == 0:
        return 0.0

    top_k_scores = relevance_scores[:k]

    if sum(top_k_scores) == 0:
        return 0.0

    calculated_dcg = dcg_at_k(top_k_scores, k=k)
    ideal_scores = sorted(top_k_scores, reverse=True)
    ideal_dcg = dcg_at_k(ideal_scores, k=k)

    return calculated_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def calculate_stage2_metrics(submission: pd.DataFrame, solution: pd.DataFrame) -> dict[str, float]:
    """
    Calculate metrics for Stage 2: NDCG@20 (Normalized Discounted Cumulative Gain at 20).

    The metric uses three-level relevance:
    - 2 points: book is read (book_id_list_read)
    - 1 point: book is planned (book_id_list_planned)
    - 0 points: "cold" candidate (no interaction)

    NDCG penalizes incorrect ordering: read books should rank above planned books,
    and planned books should rank above "cold" candidates.
    """
    K = 20

    def parse_book_id_list(book_list_str: str | float) -> set[int]:
        """Parse book_id_list string (comma-separated) into a set of book_ids."""
        if pd.isna(book_list_str) or book_list_str == "":
            return set()
        return {int(x.strip()) for x in str(book_list_str).split(",") if x.strip()}

    solution_grouped = solution.groupby("user_id").agg({
        "book_id_list_read": lambda x: parse_book_id_list(x.iloc[0]) if len(x) > 0 else set(),
        "book_id_list_planned": lambda x: parse_book_id_list(x.iloc[0]) if len(x) > 0 else set(),
    })

    merged_df = submission.merge(solution_grouped, on="user_id", how="inner")

    if merged_df.empty:
        return {"Score": 0.0, "NDCG@20": 0.0}

    # Validate merge completeness
    if merged_df.shape[0] != submission.shape[0]:
        missing = submission.shape[0] - merged_df.shape[0]
        missing_users = submission[
            ~submission["user_id"].isin(merged_df["user_id"])
        ]["user_id"].tolist()
        raise ValueError(f"Missing {missing} users after merge. Examples: {missing_users[:5]}")

    merged_df["book_id_list_read"] = merged_df["book_id_list_read"].apply(
        lambda x: x if isinstance(x, set) else set()
    )
    merged_df["book_id_list_planned"] = merged_df["book_id_list_planned"].apply(
        lambda x: x if isinstance(x, set) else set()
    )

    def parse_prediction_list(book_list_str: str | float) -> list[int]:
        """Parse book_id_list string (comma-separated) into a list of book_ids."""
        if pd.isna(book_list_str) or book_list_str == "":
            return []
        return [int(x.strip()) for x in str(book_list_str).split(",") if x.strip()]

    merged_df["y_pred"] = merged_df["book_id_list"].apply(parse_prediction_list)

    def calculate_relevance_scores(row: pd.Series) -> list[int]:
        """Calculate relevance scores for the predicted list."""
        y_pred = row["y_pred"]
        books_read = row["book_id_list_read"]
        books_planned = row["book_id_list_planned"]

        relevance = []
        for book_id in y_pred:
            if book_id in books_read:
                relevance.append(2)
            elif book_id in books_planned:
                relevance.append(1)
            else:
                relevance.append(0)

        return relevance

    merged_df["relevance_scores"] = merged_df.apply(calculate_relevance_scores, axis=1)

    merged_df[f"ndcg@{K}"] = merged_df["relevance_scores"].apply(
        lambda scores: ndcg_at_k(scores, k=K)
    )

    mean_ndcg = merged_df[f"ndcg@{K}"].mean()

    return {"Score": mean_ndcg, f"NDCG@{K}": mean_ndcg}


def main() -> dict[str, float]:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 predictions")
    parser.add_argument(
        "--submission",
        type=str,
        default="submission.csv",
        help="Path to submission file (default: submission.csv)",
    )
    parser.add_argument(
        "--solution",
        type=str,
        default="solution.csv",
        help="Path to solution file (default: solution.csv)",
    )
    args = parser.parse_args()

    submission_path = Path(args.submission)
    solution_path = Path(args.solution)

    if not submission_path.exists():
        print(f"Error: File not found: {submission_path}", file=sys.stderr)
        sys.exit(1)

    if not solution_path.exists():
        print(f"Error: File not found: {solution_path}", file=sys.stderr)
        sys.exit(1)

    try:
        submission = pd.read_csv(submission_path)
        solution = pd.read_csv(solution_path)
    except FileNotFoundError as e:
        print(f"Error: File not found: {e.filename}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV files: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        validate_solution_format(solution)
    except ValueError as e:
        print(f"Solution validation error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        validate_submission_format(submission, solution)
    except ValueError as e:
        print(f"Submission validation error: {e}", file=sys.stderr)
        sys.exit(1)

    solution_public = solution[solution["stage"] == "public"].copy()
    solution_private = solution[solution["stage"] == "private"].copy()

    # Filter submission to only include users from the corresponding stage
    public_user_ids = set(solution_public["user_id"].unique())
    private_user_ids = set(solution_private["user_id"].unique())

    submission_public = submission[submission["user_id"].isin(public_user_ids)].copy()
    submission_private = submission[submission["user_id"].isin(private_user_ids)].copy()

    public_metrics = calculate_stage2_metrics(submission_public, solution_public)
    private_metrics = calculate_stage2_metrics(submission_private, solution_private)

    print("--- Public ---")
    for metric, value in public_metrics.items():
        print(f"{metric}: {value:.6f}")

    print("\n--- Private ---")
    for metric, value in private_metrics.items():
        print(f"{metric}: {value:.6f}")

    return {
        "public_score": public_metrics["Score"],
        "private_score": private_metrics["Score"],
    }


if __name__ == "__main__":
    main()
