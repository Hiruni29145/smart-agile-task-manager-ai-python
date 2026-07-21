import pandas as pd
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset path
DATASET_PATH = PROJECT_ROOT / "dataset" / "agile_task_dataset_ml_read.xlsx"

# Read Excel dataset
df = pd.read_excel(DATASET_PATH)

print("=" * 60)
print("Dataset Loaded Successfully")
print("=" * 60)

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nColumns:")
print(df.columns.tolist())


print("\n" + "=" * 60)
print("Dataset Information")
print("=" * 60)

df.info()

print("\n" + "=" * 60)
print("Missing Values")
print("=" * 60)

print(df.isnull().sum())