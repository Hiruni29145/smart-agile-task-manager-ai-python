import pandas as pd
from pathlib import Path


# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "agile_task_dataset_ml_read.xlsx"


# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset():
    """
    Load the Excel dataset into a Pandas DataFrame.
    """

    df = pd.read_excel(DATASET_PATH)

    print("=" * 60)
    print("Dataset Loaded Successfully")
    print("=" * 60)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


# ==========================================================
# Validate Dataset
# ==========================================================

def validate_dataset(df):
    """
    Display dataset information and check for missing values.
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
# Create Training Text
# ==========================================================

def create_text_feature(df):
    """
    Combine Task Name and Description into a single text column.
    This column will later be used for TF-IDF vectorization.
    """

    df["text"] = (
        df["Task Name"].astype(str).str.strip()
        + " "
        + df["Description"].astype(str).str.strip()
    )

    return df


# ==========================================================
# Display Sample
# ==========================================================

def display_sample(df):
    """
    Display the first 5 rows of the important columns.
    """

    print("\n" + "=" * 100)
    print("Combined Text Preview")
    print("=" * 100)

    print(
        df[
            [
                "Task Name",
                "Description",
                "text"
            ]
        ].head(5)
    )


# ==========================================================
# Main
# ==========================================================

def main():

    # Load dataset
    df = load_dataset()

    # Validate dataset
    validate_dataset(df)

    # Create combined text column
    df = create_text_feature(df)

    # Display sample
    display_sample(df)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()