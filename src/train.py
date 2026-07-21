"""
train.py

Main training pipeline for the
Smart Agile Task Manager AI.

Pipeline

1. Load Dataset
2. Validate Dataset
3. Create Text Feature
4. TF-IDF Vectorization
5. Encode Priority
6. Encode Task Type
7. Build Feature Matrix
8. Extract Target Variables
9. Split Dataset
10. Train Story Point Model
11. Train Hours Model
"""

from preprocess import preprocess_dataset

from feature_engineering import (
    vectorize_text,
    encode_priority,
    encode_task_type,
    build_feature_matrix,
    get_target_variables,
)

from trainer import (
    split_dataset,
    train_story_point_model,
    train_hours_model,
)

from evaluator import (
    evaluate_model,
    preview_predictions,
)

# ==========================================================
# Main Training Pipeline
# ==========================================================

def main():
    print("\n" + "=" * 60)
    print("SMART AGILE TASK MANAGER AI")
    print("Model Training Pipeline")
    print("=" * 60)

    # ======================================================
    # Dataset Preprocessing
    # ======================================================

    df = preprocess_dataset(show_preview=True)

    # ======================================================
    # Feature Engineering
    # ======================================================

    X_text, vectorizer = vectorize_text(df)

    priority_encoded, priority_encoder = encode_priority(df)

    task_type_encoded, task_type_encoder = encode_task_type(df)

    X = build_feature_matrix(
        X_text,
        priority_encoded,
        task_type_encoded,
    )

    # ======================================================
    # Target Variables
    # ======================================================

    (
        y_story_points,
        y_actual_hours,
    ) = get_target_variables(df)

    # ======================================================
    # Story Point Dataset
    # ======================================================

    (
        X_train_sp,
        X_test_sp,
        y_train_sp,
        y_test_sp,
    ) = split_dataset(
        X,
        y_story_points,
    )

    # ======================================================
    # Hours Dataset
    # ======================================================

    (
        X_train_hr,
        X_test_hr,
        y_train_hr,
        y_test_hr,
    ) = split_dataset(
        X,
        y_actual_hours,
    )

    # ======================================================
    # Train Models
    # ======================================================

    story_point_model = train_story_point_model(
        X_train_sp,
        y_train_sp,
    )

    hours_model = train_hours_model(
        X_train_hr,
        y_train_hr,
    )

    # ======================================================
    # Evaluate Story Point Model
    # ======================================================

    story_results = evaluate_model(
        story_point_model,
        X_test_sp,
        y_test_sp,
        "Story Point Model"
    )

    preview_predictions(
        y_test_sp,
        story_results["predictions"]
    )

    # ======================================================
    # Evaluate Hours Model
    # ======================================================

    hours_results = evaluate_model(
        hours_model,
        X_test_hr,
        y_test_hr,
        "Hours Model"
    )

    preview_predictions(
        y_test_hr,
        hours_results["predictions"]
    )

    # ======================================================
    # Training Complete
    # ======================================================

    print("\n" + "=" * 60)
    print("Training Pipeline Completed Successfully")
    print("=" * 60)

    print("✓ Story Point Model Trained")
    print("✓ Hours Model Trained")



# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()