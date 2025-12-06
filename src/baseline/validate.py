"""
Script to validate the format of the submission file.
"""

import pandas as pd

from . import config, constants


def validate() -> None:
    """Validates the structure and format of the submission file.

    Performs a series of checks to ensure the submission file is valid before
    uploading, such as verifying the number of rows, checking for missing
    values, and ensuring the user/book pairs match the candidates.

    Raises:
        FileNotFoundError: If the required files do not exist.
        AssertionError: If any of the validation checks fail.
    """
    print("Validating submission file...")

    try:
        # Load required files
        targets_df = pd.read_csv(config.RAW_DATA_DIR / constants.TARGETS_FILENAME)
        candidates_df = pd.read_csv(config.RAW_DATA_DIR / constants.CANDIDATES_FILENAME)
        submission_df = pd.read_csv(config.SUBMISSION_DIR / constants.SUBMISSION_FILENAME)

        # 1. Check columns
        required_cols = {constants.COL_USER_ID, constants.COL_BOOK_ID_LIST}
        assert required_cols.issubset(
            submission_df.columns
        ), f"Missing required columns. Expected: {required_cols}, Got: {set(submission_df.columns)}"
        print("✅ Column check passed.")

        # 2. Check that all users from targets are present
        targets_set = set(targets_df[constants.COL_USER_ID])
        submission_users_set = set(submission_df[constants.COL_USER_ID])
        missing_users = targets_set - submission_users_set
        assert len(missing_users) == 0, f"Missing users in submission: {list(missing_users)[:10]}"
        print(f"✅ All {len(targets_set)} target users present in submission.")

        # 3. Check for duplicate users
        duplicate_users = submission_df[submission_df.duplicated(subset=[constants.COL_USER_ID])]
        assert len(duplicate_users) == 0, f"Found {len(duplicate_users)} duplicate users in submission."
        print("✅ No duplicate users check passed.")

        # 4. Check book_id_list format and count
        # Create a mapping of user_id -> candidate book_ids
        candidates_dict = {}
        for _, row in candidates_df.iterrows():
            user_id = row[constants.COL_USER_ID]
            book_id_list_str = row[constants.COL_BOOK_ID_LIST]
            if pd.isna(book_id_list_str) or book_id_list_str == "":
                candidates_dict[user_id] = set()
            else:
                book_ids = [int(bid.strip()) for bid in book_id_list_str.split(",") if bid.strip()]
                candidates_dict[user_id] = set(book_ids)

        # Validate each user's submission
        errors = []
        for _, row in submission_df.iterrows():
            user_id = row[constants.COL_USER_ID]
            book_id_list_str = row[constants.COL_BOOK_ID_LIST]

            # Parse book_id_list
            if pd.isna(book_id_list_str) or book_id_list_str == "":
                book_ids = []
            else:
                try:
                    book_ids = [int(bid.strip()) for bid in book_id_list_str.split(",") if bid.strip()]
                except ValueError as e:
                    errors.append(f"User {user_id}: Invalid book_id_list format - {e}")
                    continue

            # Check count
            if len(book_ids) > constants.MAX_RANKING_LENGTH:
                errors.append(
                    f"User {user_id}: Too many books ({len(book_ids)} > {constants.MAX_RANKING_LENGTH})"
                )

            # Check for duplicates
            if len(book_ids) != len(set(book_ids)):
                errors.append(f"User {user_id}: Duplicate book_ids in list")

            # Check that all book_ids are from candidates
            if user_id in candidates_dict:
                candidate_book_ids = candidates_dict[user_id]
                invalid_book_ids = set(book_ids) - candidate_book_ids
                if invalid_book_ids:
                    errors.append(
                        f"User {user_id}: Book IDs not in candidates: {list(invalid_book_ids)[:5]}"
                    )

        assert len(errors) == 0, f"Validation errors found:\n" + "\n".join(errors[:10])
        print("✅ Book ID list validation passed.")

        # 5. Print summary statistics
        print("\nSubmission summary:")
        total_recommendations = sum(
            len([b for b in row[constants.COL_BOOK_ID_LIST].split(",") if b.strip()])
            if pd.notna(row[constants.COL_BOOK_ID_LIST]) and row[constants.COL_BOOK_ID_LIST] != ""
            else 0
            for _, row in submission_df.iterrows()
        )
        avg_recommendations = total_recommendations / len(submission_df) if len(submission_df) > 0 else 0
        print(f"  - Total users: {len(submission_df)}")
        print(f"  - Average recommendations per user: {avg_recommendations:.2f}")

        print("\n✅ Validation successful! The submission file appears to be in the correct format.")

    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure the required files exist.")
    except AssertionError as e:
        print(f"Validation failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    validate()

