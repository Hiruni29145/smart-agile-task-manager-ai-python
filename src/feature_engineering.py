"""
feature_engineering.py

Handles all feature engineering tasks for the
AI Task Effort Estimation project.

Responsibilities
----------------
- TF-IDF Vectorization
- Priority Encoding
- Task Type Encoding
- Feature Matrix Construction
"""

import pandas as pd

from scipy.sparse import csr_matrix
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from config import (
    LOWERCASE,
    STOP_WORDS,
    STRIP_ACCENTS,
    NGRAM_RANGE,
    MIN_DF,
    MAX_DF,
    SUBLINEAR_TF,
)


# ==========================================================
# TF-IDF Vectorization
# ==========================================================

def vectorize_text(df: pd.DataFrame):
    """
    Convert task text into TF-IDF vectors.
    """

    vectorizer = TfidfVectorizer(
        lowercase=LOWERCASE,
        stop_words=STOP_WORDS,
        strip_accents=STRIP_ACCENTS,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        sublinear_tf=SUBLINEAR_TF,
    )

    X_text = vectorizer.fit_transform(df["text"])

    print("\n" + "=" * 60)
    print("TF-IDF Vectorization")
    print("=" * 60)

    print(f"Tasks           : {X_text.shape[0]}")
    print(f"Vocabulary Size : {X_text.shape[1]}")

    print("\nFirst 20 Vocabulary Words\n")
    print(list(vectorizer.get_feature_names_out()[:20]))

    return X_text, vectorizer


# ==========================================================
# Priority Encoding
# ==========================================================

def encode_priority(df: pd.DataFrame):
    """
    Encode Priority using Ordinal Encoding.

    Low -> 0
    Medium -> 1
    High -> 2
    Critical -> 3
    """

    priority_encoder = OrdinalEncoder(
        categories=[
            [
                "Low",
                "Medium",
                "High",
                "Critical"
            ]
        ]
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

    return priority_encoded, priority_encoder


# ==========================================================
# Task Type Encoding
# ==========================================================

def encode_task_type(df: pd.DataFrame):
    """
    Encode Task Type using One-Hot Encoding.
    """

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
    print("Task Type Encoding Preview")
    print("=" * 60)

    task_type_names = task_type_encoder.get_feature_names_out(
        ["Task Type"]
    )

    preview = pd.DataFrame(
        task_type_encoded.toarray()[:5],
        columns=task_type_names
    )

    print(preview)

    return task_type_encoded, task_type_encoder


# ==========================================================
# Build Feature Matrix
# ==========================================================

def build_feature_matrix(
    X_text,
    priority_encoded,
    task_type_encoded
):
    """
    Combine all engineered features into one sparse matrix.

    Feature Order

    TF-IDF
        +
    Priority
        +
    Task Type
    """

    priority_sparse = csr_matrix(priority_encoded)

    X = hstack([
        X_text,
        priority_sparse,
        task_type_encoded
    ]).tocsr()

    print("\n" + "=" * 60)
    print("Final Feature Matrix")
    print("=" * 60)

    print(f"Rows     : {X.shape[0]}")
    print(f"Features : {X.shape[1]}")

    return X


# ==========================================================
# Target Variables
# ==========================================================

def get_target_variables(df: pd.DataFrame):
    """
    Extract target variables.
    """

    y_story_points = df["Story Points"]

    y_actual_hours = df["Actual Hours"]

    print("\n" + "=" * 60)
    print("Target Variables")
    print("=" * 60)

    print(f"Story Points Shape : {y_story_points.shape}")
    print(f"Actual Hours Shape : {y_actual_hours.shape}")

    return (
        y_story_points,
        y_actual_hours
    )