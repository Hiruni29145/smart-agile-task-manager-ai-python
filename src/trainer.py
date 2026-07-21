"""
trainer.py

Handles all model training tasks for the
AI Task Effort Estimation project.

Responsibilities:
- Train/Test Split
- Story Point Model Training
- Actual Hours Model Training
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from config import (
    TEST_SIZE,
    RANDOM_STATE,
    N_ESTIMATORS
)


# ==========================================================
# Train / Test Split
# ==========================================================

def split_dataset(X, y):
    """
    Split a dataset into training and testing sets.

    Parameters
    ----------
    X
        Feature matrix.

    y
        Target variable.

    Returns
    -------
    tuple
        X_train
        X_test
        y_train
        y_test
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print("\n" + "=" * 60)
    print("Train / Test Split")
    print("=" * 60)

    print(f"Training Samples : {X_train.shape[0]}")
    print(f"Testing Samples  : {X_test.shape[0]}")

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ==========================================================
# Story Point Model
# ==========================================================

def train_story_point_model(
    X_train,
    y_train
):
    """
    Train the Story Point prediction model.

    Parameters
    ----------
    X_train
        Training feature matrix.

    y_train
        Story Point target values.

    Returns
    -------
    RandomForestRegressor
        Trained Story Point model.
    """

    print("\n" + "=" * 60)
    print("Training Story Point Model")
    print("=" * 60)

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print("Story Point Model Training Completed.")

    return model


# ==========================================================
# Actual Hours Model
# ==========================================================

def train_hours_model(
    X_train,
    y_train
):
    """
    Train the Actual Hours prediction model.

    Parameters
    ----------
    X_train
        Training feature matrix.

    y_train
        Actual Hours target values.

    Returns
    -------
    RandomForestRegressor
        Trained Hours prediction model.
    """

    print("\n" + "=" * 60)
    print("Training Hours Model")
    print("=" * 60)

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print("Hours Model Training Completed.")

    return model