import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from scipy.sparse import csr_matrix


# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "agile_task_dataset_ml_read.xlsx"
)


# ==========================================================
# Load Dataset
# ==========================================================

# Load the cleaned Agile task dataset into a Pandas DataFrame.
df = pd.read_excel(DATASET_PATH)


# ==========================================================
# Create Combined Text Feature
# ==========================================================

# Combine Task Name and Description into a single text feature.
# This text will later be converted into numerical vectors using TF-IDF.

df["text"] = (
    df["Task Name"].astype(str).str.strip()
    + " "
    + df["Description"].astype(str).str.strip()
)


# ==========================================================
# TF-IDF Vectorization
# ==========================================================

# Convert textual task information into numerical vectors.

vectorizer = TfidfVectorizer()

X_text = vectorizer.fit_transform(df["text"])


# ==========================================================
# TF-IDF Summary
# ==========================================================

print("=" * 60)
print("TF-IDF Completed")
print("=" * 60)

print()
print(f"Number of Tasks : {X_text.shape[0]}")
print(f"Number of Words : {X_text.shape[1]}")

print()
print("First 20 Vocabulary Words")
print(list(vectorizer.get_feature_names_out()[:20]))


# ==========================================================
# Display First Task
# ==========================================================

print("\n" + "=" * 60)
print("First Task")
print("=" * 60)

print(df["text"].iloc[0])


# ==========================================================
# Display TF-IDF Vector Information
# ==========================================================

print("\n" + "=" * 60)
print("TF-IDF Vector Shape")
print("=" * 60)

print(X_text[0].shape)


# ==========================================================
# Display Important TF-IDF Features
# ==========================================================

print("\n" + "=" * 60)
print("Non-zero TF-IDF Features")
print("=" * 60)

vector = X_text[0]
feature_names = vectorizer.get_feature_names_out()

for index, value in zip(vector.indices, vector.data):
    print(f"{feature_names[index]:20} : {value:.4f}")


# ==========================================================
# Priority Encoding
# ==========================================================

# Priority has a natural order.
# Low < Medium < High < Critical

priority_encoder = OrdinalEncoder(
    categories=[["Low", "Medium", "High", "Critical"]]
)

priority_encoded = priority_encoder.fit_transform(
    df[["Priority"]]
)

print("\n" + "=" * 60)
print("Priority Encoding")
print("=" * 60)

for i in range(5):
    print(
        f"{df['Priority'].iloc[i]:10} -> {int(priority_encoded[i][0])}"
    )


# ==========================================================
# Task Type Encoding
# ==========================================================

# Task Type has no natural order.
# Therefore One-Hot Encoding is used.

task_type_encoder = OneHotEncoder(
    sparse_output=True,
    handle_unknown="ignore"
)

task_type_encoded = task_type_encoder.fit_transform(
    df[["Task Type"]]
)


print("\n" + "=" * 60)
print("Task Type Categories")
print("=" * 60)

print(task_type_encoder.categories_[0])


print("\n" + "=" * 60)
print("Task Type Encoding")
print("=" * 60)

task_type_names = task_type_encoder.get_feature_names_out(
    ["Task Type"]
)

encoded_df = pd.DataFrame(
    task_type_encoded.toarray()[:5],
    columns=task_type_names
)

print(encoded_df)


# ==========================================================
# Convert Priority to Sparse Matrix
# ==========================================================

priority_sparse = csr_matrix(priority_encoded)

# ==========================================================
# Create Final Feature Matrix
# ==========================================================

X = hstack([
    X_text,
    priority_sparse,
    task_type_encoded
])

# ==========================================================
# Feature Matrix Summary
# ==========================================================

print("\n" + "=" * 60)
print("Feature Matrix")
print("=" * 60)

print(f"Rows     : {X.shape[0]}")
print(f"Features : {X.shape[1]}")

# ==========================================================
# Target Variables
# ==========================================================

y_story_points = df["Story Points"]

y_actual_hours = df["Actual Hours"]

print("\n" + "=" * 60)
print("Target Variables")
print("=" * 60)

print(f"Story Points Shape : {y_story_points.shape}")
print(f"Actual Hours Shape : {y_actual_hours.shape}")

print("\nFirst 5 Story Points")
print(y_story_points.head())

print("\nFirst 5 Actual Hours")
print(y_actual_hours.head())