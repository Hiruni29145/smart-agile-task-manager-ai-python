"""
predict.py

Prediction service for the
Smart Agile Task Manager AI.
"""

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse import hstack

from model_loader import load_all


# ==========================================================
# Load Models Once
# ==========================================================

artifacts = load_all()

story_point_model = artifacts["story_point_model"]
hours_model = artifacts["hours_model"]

vectorizer = artifacts["vectorizer"]
priority_encoder = artifacts["priority_encoder"]
task_type_encoder = artifacts["task_type_encoder"]


# ==========================================================
# Helpers
# ==========================================================

def snap_to_fibonacci(value: float) -> int:
    """
    Snap a predicted float value to the nearest
    Agile Fibonacci story point value.
    """
    fib_sequence = [1, 2, 3, 5, 8, 13, 21]
    return min(fib_sequence, key=lambda x: abs(x - value))


# ==========================================================
# Predict
# ==========================================================

def predict(
    task_name: str,
    description: str,
    priority: str,
    task_type: str,
):
    """
    Predict Story Points and Actual Hours.
    """

    # ----------------------------------------------
    # Create Text Feature
    # ----------------------------------------------

    text = f"{task_name.strip()} {description.strip()}"

    # ----------------------------------------------
    # TF-IDF
    # ----------------------------------------------

    X_text = vectorizer.transform([text])

    # ----------------------------------------------
    # Priority Encoding
    # ----------------------------------------------

    priority = priority_encoder.transform(
        pd.DataFrame({"Priority": [priority]})
    )

    priority = csr_matrix(priority)

    # ----------------------------------------------
    # Task Type Encoding
    # ----------------------------------------------

    task_type = task_type_encoder.transform(
        pd.DataFrame({"Task Type": [task_type]})
    )

    # ----------------------------------------------
    # Final Feature Matrix
    # ----------------------------------------------

    X = hstack([
        X_text,
        priority,
        task_type
    ])

    # ----------------------------------------------
    # Predictions
    # ----------------------------------------------

    story_points = story_point_model.predict(X)[0]
    story_points_snapped = snap_to_fibonacci(story_points)

    sp_sparse = csr_matrix([[story_points_snapped]])
    X_chained = hstack([X, sp_sparse])

    actual_hours_log = hours_model.predict(X_chained)[0]
    actual_hours = np.expm1(actual_hours_log)

    # Calculate Heuristics
    if story_points_snapped <= 3:
        complexity = "Low"
    elif story_points_snapped == 5:
        complexity = "Medium"
    elif story_points_snapped <= 13:
        complexity = "High"
    else:
        complexity = "Very High"
        
    confidence = 95 - story_points_snapped

    return {
        "Estimated Hours": f"{round(float(actual_hours), 1)}h",
        "Story Points": story_points_snapped,
        "Complexity": complexity,
        "Confidence Score": f"{confidence}%",
    }


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    prediction = predict(
        task_name="Implement JWT authentication",
        description="Users should login using JWT access and refresh tokens.",
        priority="High",
        task_type="Feature",
    )

    print("\nPrediction")
    print(prediction)