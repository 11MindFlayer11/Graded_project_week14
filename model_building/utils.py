"""Shared helpers: paths and Hugging Face Hub access.

If HF_TOKEN is not set the scripts fall back to local files, so the whole
pipeline can also be run and tested offline.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = ROOT / "model_building" / "artifacts"
RAW_CSV = DATA_DIR / "tourism.csv"
SPLIT_FILES = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]


def get_api():
    """Return an authenticated HfApi, or None when HF_TOKEN is not configured."""
    token = os.getenv("HF_TOKEN")
    if not token:
        return None
    from huggingface_hub import HfApi
    return HfApi(token=token)


def repo_ids(api):
    """Dataset / model / space repo ids (override with env vars if you wish)."""
    user = os.getenv("HF_USERNAME") or api.whoami()["name"]
    return {
        "dataset": os.getenv("HF_DATASET_REPO", f"{user}/tourism-dataset"),
        "model": os.getenv("HF_MODEL_REPO", f"{user}/tourism-wellness-model"),
        "space": os.getenv("HF_SPACE_REPO", f"{user}/tourism-wellness-app"),
    }
