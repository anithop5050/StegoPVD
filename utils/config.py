import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # Go up to project root
PROJECT_ROOT = BASE_DIR  # Alias for clarity
DATASET_DIR = BASE_DIR / "DATASET"

MODEL_PATH = BASE_DIR / "stego_detector_model.joblib"
SCALER_PATH = BASE_DIR / "feature_scaler.joblib"
REPORT_PATH = BASE_DIR / "intensive_report.txt"
EXECUTIVE_REPORT_PATH = BASE_DIR / "model_evaluation_summary.md"

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

PATCH_SIZE = 128
FEATURE_DIMENSION = 1280  # 5 x 256-bin histograms
HIST_BINS = 256

# ============================================================================
# MODEL HYPERPARAMETERS (RandomForest classical baseline)
# ============================================================================

RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 15
RF_RANDOM_STATE = 42
RF_N_JOBS = -1

# ============================================================================
# INFERENCE AGGREGATION & THRESHOLDS
# ============================================================================

# Legacy max-patch rule: threshold at which a single patch flags the image
STEGO_THRESHOLD_MAX = 0.85
EARLY_STOP_THRESHOLD_MAX = 0.95

# New aggregation policies: reduce false positives from textured clean images
# Options: "max" (legacy), "mean", "percentile", "voting"
AGGREGATION_POLICY = "mean"
AGGREGATION_PERCENTILE = 90

# Voting/suspicious patch rules
SUSPICIOUS_PATCH_THRESHOLD = 0.70
SUSPICIOUS_RATIO_THRESHOLD = 0.30
MEAN_PROB_THRESHOLD = 0.35  # Optimized via grid search: 85% accuracy, 70% STEGO detection, 0% FPR
MAX_PROB_HARD_THRESHOLD = 0.92

STEGO_VOTE_RATIO = 0.50  # Fraction of patches that must be suspicious to flag as stego

# ============================================================================
# FILE HANDLING
# ============================================================================

MAX_UPLOAD_SIZE_MB = 10
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
INFERENCE_TIMEOUT_SECONDS = 30

# ============================================================================
# MODEL BACKEND SELECTION
# ============================================================================

# Options: "classical" (RandomForest), "deep_cnn" (PyTorch CNN)
# Current supported: "classical"
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "classical")

CLEAN_TRAIN_DIRS = [
    DATASET_DIR / "CLEAN" / "INTERNET_PNG",
    DATASET_DIR / "CLEAN" / "INTERNET_BMP",
    DATASET_DIR / "CLEAN" / "PNG",
    DATASET_DIR / "CLEAN" / "BMP",
    DATASET_DIR / "CLEAN" / "INTERNET",
]

STEGO_TRAIN_DIRS = [
    DATASET_DIR / "STEGO" / "INTERNET_PNG",
    DATASET_DIR / "STEGO" / "INTERNET_BMP",
    DATASET_DIR / "STEGO" / "PNG",
    DATASET_DIR / "STEGO" / "BMP",
]

TEST_CLEAN_DIR = DATASET_DIR / "TEST_DATA" / "CLEAN"
TEST_STEGO_DIR = DATASET_DIR / "TEST_DATA" / "STEGO"

IMAGE_EXTENSIONS = (".png", ".bmp", ".jpg", ".jpeg")
