import sys
import os
from pathlib import Path

# Add src to path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.sparse import csr_matrix, hstack

from preprocess import preprocess_dataset
from feature_engineering import (
    vectorize_text,
    encode_priority,
    encode_task_type,
    build_feature_matrix,
    get_target_variables,
)
from trainer import split_dataset
from model_loader import load_all

def snap_to_complexity(value: float) -> str:
    """Snap to fibonacci and then to complexity."""
    fib_sequence = [1, 2, 3, 5, 8, 13, 21]
    snapped = min(fib_sequence, key=lambda x: abs(x - value))
    if snapped <= 3:
        return "Low"
    elif snapped == 5:
        return "Medium"
    elif snapped <= 13:
        return "High"
    else:
        return "Very High"

def main():
    print("Loading Dataset...")
    df = preprocess_dataset(show_preview=False)

    print("Engineering Features...")
    X_text, vectorizer = vectorize_text(df)
    priority_encoded, priority_encoder = encode_priority(df)
    task_type_encoded, task_type_encoder = encode_task_type(df)
    
    X = build_feature_matrix(X_text, priority_encoded, task_type_encoded)
    y_story_points, _ = get_target_variables(df)

    print("Splitting Data...")
    _, X_test, _, y_test = split_dataset(X, y_story_points)

    print("Loading Model...")
    artifacts = load_all()
    model = artifacts["story_point_model"]

    print("Predicting...")
    y_pred = model.predict(X_test)

    print("Converting to Complexity Categories...")
    y_test_categories = [snap_to_complexity(val) for val in y_test]
    y_pred_categories = [snap_to_complexity(val) for val in y_pred]

    labels = ["Low", "Medium", "High", "Very High"]

    cm = confusion_matrix(y_test_categories, y_pred_categories, labels=labels)

    print("Plotting Confusion Matrix...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Complexity - Confusion Matrix (XGBoost)')
    plt.ylabel('True Complexity')
    plt.xlabel('Predicted Complexity')
    
    output_path = os.path.join(os.path.dirname(__file__), 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Confusion Matrix saved to: {output_path}")

if __name__ == "__main__":
    main()
