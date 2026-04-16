from pathlib import Path


ML_DIR = Path(__file__).resolve().parent
DATA_DIR = ML_DIR / "data"
MODELS_DIR = ML_DIR / "models"

MODEL_PATH = MODELS_DIR / "model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic_batches.csv"

RANDOM_SEED = 42
N_SAMPLES = 1800
TIME_SERIES_STEPS = 24

LABEL_TO_NAME = {0: "PASS", 1: "FAIL", 2: "REVIEW"}
NAME_TO_LABEL = {name: label for label, name in LABEL_TO_NAME.items()}
