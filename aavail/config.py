"""Filesystem paths for the demo repository; pass alternate roots for isolation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
ARTIFACT_DIR = ROOT / 'artifacts'
MODEL_PATH = ARTIFACT_DIR / 'churn_model.joblib'
FEATURES_PATH = ARTIFACT_DIR / 'customer_features.csv'
LOG_DIR = ROOT / 'logs'
