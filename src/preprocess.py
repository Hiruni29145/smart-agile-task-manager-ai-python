"""
preprocessing.py

Handles all dataset loading and preprocessing tasks for the
AI Task Effort Estimation project.
"""

import pandas as pd

from config import DATASET_PATH


# ==========================================================
# Dataset Loading
# ==========================================================

def load_dataset() -> pd.DataFrame:
    """
    Load the Agile task dataset from the configured Excel file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """

    df = pd.read_excel(DATASET_PATH)

    print("=" * 60)
    print("Dataset Loaded Successfully")
    print("=" * 60)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


# ==========================================================
# Dataset Validation
# ==========================================================

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Display basic dataset information.

    Args:
        df (pd.DataFrame): Dataset to validate.
    """

    print("\n" + "=" * 60)
    print("Dataset Information")
    print("=" * 60)

    df.info()

    print("\n" + "=" * 60)
    print("Missing Values")
    print("=" * 60)

    print(df.isnull().sum())


# ==========================================================
# Text Feature Engineering
# ==========================================================

def create_text_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine Task Name and Description into a single text feature.

    The generated 'text' column is used later for TF-IDF
    vectorization during model training.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with the new 'text' column.
    """

    df = df.copy()

    df["text"] = (
        df["Task Name"].fillna("").astype(str).str.strip()
        + " "
        + df["Description"].fillna("").astype(str).str.strip()
    )

    return df


# ==========================================================
# Preview Processed Data
# ==========================================================

def display_sample(df: pd.DataFrame, rows: int = 5) -> None:
    """
    Display a preview of the generated text feature.

    Args:
        df (pd.DataFrame): Dataset.
        rows (int): Number of rows to display.
    """

    print("\n" + "=" * 100)
    print("Combined Text Preview")
    print("=" * 100)

    print(
        df[
            [
                "Task Name",
                "Description",
                "text",
            ]
        ].head(rows)
    )


# ==========================================================
# Preprocessing Pipeline
# ==========================================================

def preprocess_dataset(show_preview: bool = True) -> pd.DataFrame:
    """
    Execute the complete preprocessing pipeline.

    Steps:
        1. Load dataset
        2. Validate dataset
        3. Create combined text feature
        4. Display preview (optional)

    Args:
        show_preview (bool): Display processed samples.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """

    df = load_dataset()

    validate_dataset(df)

    df = create_text_feature(df)

    if show_preview:
        display_sample(df)

    return df


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    preprocess_dataset()