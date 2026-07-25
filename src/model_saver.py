"""
model_saver.py

Save all trained machine learning models and preprocessors.

Artifacts Saved
---------------
- Story Point Model
- Hours Model
- TF-IDF Vectorizer
- Priority Encoder
- Task Type Encoder
"""

from pathlib import Path
import joblib
import json

from config import MODEL_DIR


# ==========================================================
# Create Models Directory
# ==========================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# Save Model
# ==========================================================

def save_model(model, filename: str):
    """
    Save a trained model using Joblib.
    """

    path = MODEL_DIR / filename

    joblib.dump(
        model,
        path
    )

    print(f"[OK] Saved {filename}")


# ==========================================================
# Save All Artifacts
# ==========================================================

def save_all(
    story_point_model,
    hours_model,
    vectorizer,
    priority_encoder,
    task_type_encoder,
    metadata=None,
):
    """
    Save every trained artifact.
    """

    print("\n" + "=" * 60)
    print("Saving Models")
    print("=" * 60)

    save_model(
        story_point_model,
        "story_point_model.pkl"
    )

    save_model(
        hours_model,
        "hours_model.pkl"
    )

    save_model(
        vectorizer,
        "vectorizer.pkl"
    )

    save_model(
        priority_encoder,
        "priority_encoder.pkl"
    )

    save_model(
        task_type_encoder,
        "task_type_encoder.pkl"
    )

    if metadata:
        metadata_path = MODEL_DIR / "model_info.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print("[OK] Saved model_info.json")

    print("\nAll models saved successfully.")