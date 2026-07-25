"""
preprocess.py

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

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
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
    Display dataset information and missing values.
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
    Combine Task Name and Description into one text column.
    """

    df = df.copy()

    df["text"] = (
        df["Task Name"]
        .fillna("")
        .astype(str)
        .str.strip()
        + " "
        + df["Description"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return df


# ==========================================================
# Preview Dataset
# ==========================================================

def display_sample(
    df: pd.DataFrame,
    rows: int = 5
) -> None:
    """
    Display a preview of the processed dataset.
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
# Dataset Quality Analysis
# ==========================================================

def dataset_quality_analysis(df: pd.DataFrame) -> None:
    """
    Display useful statistics about the dataset.
    """

    print("\n" + "=" * 60)
    print("Story Point Distribution")
    print("=" * 60)

    print(
        df["Story Points"]
        .value_counts()
        .sort_index()
    )

    print("\n" + "=" * 60)
    print("Priority vs Story Points")
    print("=" * 60)

    print(
        pd.crosstab(
            df["Priority"],
            df["Story Points"]
        )
    )

    print("\n" + "=" * 60)
    print("Task Type vs Story Points")
    print("=" * 60)

    print(
        pd.crosstab(
            df["Task Type"],
            df["Story Points"]
        )
    )

    print("\n" + "=" * 60)
    print("Complexity vs Story Points")
    print("=" * 60)

    print(
        pd.crosstab(
            df["Complexity"],
            df["Story Points"]
        )
    )

    print("\n" + "=" * 60)
    print("Average Hours per Story Point")
    print("=" * 60)

    print(
        df.groupby("Story Points")["Actual Hours"]
        .agg(
            [
                "count",
                "mean",
                "min",
                "max",
            ]
        )
        .round(2)
    )

    print("\n" + "=" * 60)
    print("Story Point Statistics")
    print("=" * 60)

    print(
        df["Story Points"]
        .describe()
    )

    print("\n" + "=" * 60)
    print("Actual Hours Statistics")
    print("=" * 60)

    print(
        df["Actual Hours"]
        .describe()
    )


# ==========================================================
# Preprocessing Pipeline
# ==========================================================

def preprocess_dataset(
    show_preview: bool = True,
    show_analysis: bool = True,
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.

    Steps
    -----
    1. Load dataset
    2. Validate dataset
    3. Create combined text feature
    4. Display preview
    5. Display dataset analysis
    """

    df = load_dataset()

    validate_dataset(df)

    df = create_text_feature(df)

    if show_preview:
        display_sample(df)

    if show_analysis:
        dataset_quality_analysis(df)

    return df


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    preprocess_dataset()