from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "dataset" / "agile_task_dataset_ml_read.xlsx"

MODEL_DIR = PROJECT_ROOT / "models"

TEST_SIZE = 0.2
RANDOM_STATE = 42

N_ESTIMATORS = 20
N_JOBS = -1