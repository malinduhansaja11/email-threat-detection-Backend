# url_threat_detection/model_loader.py
import os
import gdown
import joblib

# ── Paste your real file IDs here ────────────────────────────────────────────
DRIVE_IDS = {
    "ensemble_model.pkl":  "1ckroWmF59kGv4rM2g36W5LuX4-xUEW83",
    "rf_model.pkl":        "1dOJ09Udfnr9b5mOCsQye231-Qz49Nckk",
    "xgb_model.pkl":       "1PN0-GivleS5n358gpFEiYUsMrUo3ae9a",
    "lgbm_model.pkl":      "1SVhFnABbYkqsuu5snCLlzubHl8E-QSDW",
    "feature_columns.pkl": "10sNVt1Vw9RQ_jteR1TuG682iFULmx5nK",
}

BASE      = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def download_models():
    """Download model files from Google Drive if not already present locally."""
    for filename, file_id in DRIVE_IDS.items():
        dest = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(dest):
            print(f"Downloading {filename} from Google Drive...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, dest, quiet=False)
            print(f"  ✅ Saved to {dest}")
        else:
            print(f"  ✅ {filename} already exists, skipping download.")

def load_models():
    """Download if needed, then load all models."""
    download_models()
    ensemble_model = joblib.load(os.path.join(MODEL_DIR, "ensemble_model.pkl"))
    feature_cols   = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    return ensemble_model, feature_cols