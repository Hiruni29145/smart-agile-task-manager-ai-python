"""
trainer.py

Handles all model training tasks for the
AI Task Effort Estimation project.

Responsibilities
----------------
- Train/Test Split
- Story Point Model Training
- Actual Hours Model Training
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from config import (
    TEST_SIZE,
    RANDOM_STATE,
    N_ESTIMATORS,
    MAX_DEPTH,
    MIN_SAMPLES_SPLIT,
    MIN_SAMPLES_LEAF,
    MAX_FEATURES,
    BOOTSTRAP,
    N_JOBS,
)


# ==========================================================
# Train / Test Split
# ==========================================================

def split_dataset(X, y):
    """
    Split the dataset into training and testing sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
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
        y_test,
    )


# ==========================================================
# Create Random Forest Model
# ==========================================================

def create_random_forest():
    """
    Create a tuned Random Forest model.
    """

    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        bootstrap=BOOTSTRAP,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )


# ==========================================================
# Story Point Model
# ==========================================================

def train_story_point_model(X_train, y_train):
    """
    Train the Story Point prediction model.
    """

    print("\n" + "=" * 60)
    print("Training Story Point Model")
    print("=" * 60)

    model = create_random_forest()

    model.fit(X_train, y_train)

    print("✓ Story Point Model Training Completed.")

    return model


# ==========================================================
# Hours Model
# ==========================================================

def train_hours_model(X_train, y_train):
    """
    Train the Actual Hours prediction model.
    """

    print("\n" + "=" * 60)
    print("Training Hours Model")
    print("=" * 60)

    model = create_random_forest()

    model.fit(X_train, y_train)

    print("✓ Hours Model Training Completed.")

    return model