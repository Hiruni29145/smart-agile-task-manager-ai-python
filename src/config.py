"""
config.py

Central configuration for the
Smart Agile Task Manager AI project.
"""

from pathlib import Path


# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "agile_task_dataset_ml_read.xlsx"
)

MODEL_DIR = PROJECT_ROOT / "models"


# ==========================================================
# Train/Test Configuration
# ==========================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ==========================================================
# Random Forest Configuration
# ==========================================================

N_ESTIMATORS = 300
MAX_DEPTH = 50
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
MAX_FEATURES = "sqrt"
BOOTSTRAP = True
N_JOBS = -1


# ==========================================================
# TF-IDF Configuration
# ==========================================================

LOWERCASE = True
STOP_WORDS = "english"
NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_DF = 0.95
SUBLINEAR_TF = True
STRIP_ACCENTS = "unicode"