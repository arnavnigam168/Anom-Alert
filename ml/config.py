from pathlib import Path


ML_DIR = Path(__file__).resolve().parent
DATA_DIR = ML_DIR / "data"
MODELS_DIR = ML_DIR / "models"

# Artifacts
MODEL_PATH = MODELS_DIR / "model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic_batches.csv"

# Reproducibility
RANDOM_SEED = 42

# Synthetic dataset generation
N_SAMPLES = 2400
TIME_SERIES_STEPS = 24
ROLLING_WINDOW = 3

# Labels
LABEL_TO_NAME = {0: "PASS", 1: "FAIL", 2: "REVIEW"}
NAME_TO_LABEL = {name: label for label, name in LABEL_TO_NAME.items()}

# Strict feature order used across preprocessing, training, prediction.
FEATURE_NAMES = [
    # Batch-level / scalar features
    "pH_mean",
    "temp_max",
    "oxygen_variance",
    # Time-series aggregate features (existing)
    "ts_mean",
    "ts_std",
    "ts_max",
    "ts_min",
    # Time-series engineered features (new)
    "ts_slope",
    "ts_max_jump",
    "ts_rolling_var_mean",
    "ts_spike_count",
]

