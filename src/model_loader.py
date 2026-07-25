"""
model_loader.py

Loads all trained machine learning models and preprocessing
artifacts from disk.
"""

import joblib

from config import MODEL_DIR


# ==========================================================
# Load Model
# ==========================================================

def load_model(filename: str):
    """
    Load a saved model from the models directory.
    """

    path = MODEL_DIR / filename

    model = joblib.load(path)

    print(f"[OK] Loaded {filename}")

    return model


# ==========================================================
# Load All Models
# ==========================================================

def load_all():
    """
    Load all trained models and preprocessing objects.

    Returns
    -------
    dict
        Dictionary containing all loaded artifacts.
    """

    print("\n" + "=" * 60)
    print("Loading Saved Models")
    print("=" * 60)

    artifacts = {
        "story_point_model": load_model("story_point_model.pkl"),
        "hours_model": load_model("hours_model.pkl"),
        "vectorizer": load_model("vectorizer.pkl"),
        "priority_encoder": load_model("priority_encoder.pkl"),
        "task_type_encoder": load_model("task_type_encoder.pkl"),
    }

    print("\nAll models loaded successfully.")

    return artifacts


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    load_all()