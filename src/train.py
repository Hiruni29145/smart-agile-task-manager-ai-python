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

import numpy as np

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
    # Train Models (Story Points)
    # ======================================================

    sp_model_rf = train_story_point_model(
        X_train_sp,
        y_train_sp,
        model_type="rf"
    )

    sp_model_xgb = train_story_point_model(
        X_train_sp,
        y_train_sp,
        model_type="xgb"
    )

    # ======================================================
    # Train Models (Hours - with Log Transform)
    # ======================================================
    
    y_train_hr_log = np.log1p(y_train_hr)

    hr_model_rf = train_hours_model(
        X_train_hr,
        y_train_hr_log,
        model_type="rf"
    )

    hr_model_xgb = train_hours_model(
        X_train_hr,
        y_train_hr_log,
        model_type="xgb"
    )

    # ======================================================
    # Evaluate Story Point Models
    # ======================================================

    story_results_rf = evaluate_model(
        sp_model_rf,
        X_test_sp,
        y_test_sp,
        "Story Point Model (RF)"
    )
    preview_predictions(
        y_test_sp,
        story_results_rf["predictions"]
    )

    story_results_xgb = evaluate_model(
        sp_model_xgb,
        X_test_sp,
        y_test_sp,
        "Story Point Model (XGB)"
    )
    preview_predictions(
        y_test_sp,
        story_results_xgb["predictions"]
    )

    # ======================================================
    # Evaluate Hours Models (Reverse Log Transform)
    # ======================================================

    hr_preds_rf_log = hr_model_rf.predict(X_test_hr)
    hr_preds_rf = np.expm1(hr_preds_rf_log)
    
    hours_results_rf = evaluate_model(
        hr_model_rf,
        X_test_hr,
        y_test_hr,
        "Hours Model (RF)",
        predictions=hr_preds_rf
    )
    preview_predictions(
        y_test_hr,
        hours_results_rf["predictions"]
    )

    hr_preds_xgb_log = hr_model_xgb.predict(X_test_hr)
    hr_preds_xgb = np.expm1(hr_preds_xgb_log)

    hours_results_xgb = evaluate_model(
        hr_model_xgb,
        X_test_hr,
        y_test_hr,
        "Hours Model (XGB)",
        predictions=hr_preds_xgb
    )
    preview_predictions(
        y_test_hr,
        hours_results_xgb["predictions"]
    )

    # ======================================================
    # Training Complete
    # ======================================================

    print("\n" + "=" * 60)
    print("Training Pipeline Completed Successfully")
    print("=" * 60)

    print("[OK] Story Point Models Trained (RF & XGB)")
    print("[OK] Hours Models Trained (RF & XGB)")



# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()