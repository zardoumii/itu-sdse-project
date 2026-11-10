"""Paths and hyperparameters for the training pipeline.

All paths are absolute and resolved from the project root, so the code works
no matter which directory it is run from.
"""

from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]

# Data (the metadata csvs are tracked with DVC, the images are downloaded)
DATA_DIR = PROJ_ROOT / "data"
IMAGE_DATA_DIR = DATA_DIR / "images" / "raw"
METADATA_PATH = DATA_DIR / "images" / "metadata.csv"
TEST_METADATA_PATH = DATA_DIR / "test" / "metadata.csv"
TEST_DATA_DIR = DATA_DIR / "test"

# Object store the images are downloaded from (public bucket, no credentials)
S3_ENDPOINT = "https://itudsip.hel1.your-objectstorage.com"
S3_BUCKET = "data"

# Artifacts produced by training
ARTIFACT_DIR = PROJ_ROOT / "artifacts"
PYTORCH_MODEL_PATH = ARTIFACT_DIR / "vial_cnn_model.pth"
MODEL_RESULTS_PATH = ARTIFACT_DIR / "model_results.json"
DRIFT_BASELINE_STATS_PATH = ARTIFACT_DIR / "drift_baseline_stats.json"
CLASS_NAMES_PATH = ARTIFACT_DIR / "class_names.json"
IMAGE_STATISTICS_PATH = ARTIFACT_DIR / "image_statistics.json"
INFERENCE_LOG_PATH = ARTIFACT_DIR / "inference_log.json"

# MLflow logs to a local sqlite db, so no tracking server has to be running
MLFLOW_STATE_DIR = PROJ_ROOT / "mlflow-state"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_STATE_DIR / 'mlflow.db'}"
MLFLOW_ARTIFACT_LOCATION = str(MLFLOW_STATE_DIR / "artifacts")
EXPERIMENT_NAME = "vial_image_classification"
MODEL_NAME = "cnn_model"

# Training
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
THRESHOLD = 0.5
EARLY_STOP_PATIENCE = 2
SCHEDULER_PATIENCE = 3
RANDOM_STATE = 42

# Images
IMAGE_SIZE = (64, 64)
NUM_CLASSES = 2
CLASS_NAMES = {0: "good", 1: "defective"}
TRANSFORM_MEAN = [0.485, 0.456, 0.406]
TRANSFORM_STD = [0.229, 0.224, 0.225]
