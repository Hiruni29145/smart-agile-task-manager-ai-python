"""
evaluator.py

Model evaluation utilities for the
Smart Agile Task Manager AI.

Responsibilities:
- Model Evaluation
- Performance Metrics
- Prediction Preview
"""

import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==========================================================
# Evaluate Model
# ==========================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name="Model",
    predictions=None,
):
    """
    Evaluate a trained regression model.

    Parameters
    ----------
    model
        Trained regression model.

    X_test
        Test feature matrix.

    y_test
        Actual target values.

    model_name
        Display name.
        
    predictions
        Optional pre-calculated predictions.
    """

    if predictions is None:
        predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print(f"{model_name} Evaluation")
    print("=" * 60)

    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictions": predictions,
    }


# ==========================================================
# Prediction Preview
# ==========================================================

def preview_predictions(
    y_test,
    predictions,
    rows=10,
):
    """
    Display actual vs predicted values.
    """

    preview = pd.DataFrame(
        {
            "Actual": y_test.values[:rows],
            "Predicted": predictions[:rows],
        }
    )

    preview["Difference"] = (
        preview["Predicted"]
        - preview["Actual"]
    ).round(2)

    print("\n" + "=" * 60)
    print("Prediction Preview")
    print("=" * 60)

    print(preview)