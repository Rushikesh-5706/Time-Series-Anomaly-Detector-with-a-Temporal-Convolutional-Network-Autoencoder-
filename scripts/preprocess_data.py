"""
Preprocessing pipeline for the NASA SMAP dataset.

Downloads channel P-1 telemetry data, applies MinMax normalization,
creates overlapping sliding windows, and serializes all artifacts to disk.
"""

import os
import sys
import logging
import requests
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 100))
RAW_DIR = Path(os.getenv("DATASET_RAW_DIR", "data/raw"))
PROCESSED_DIR = Path(os.getenv("DATASET_PROCESSED_DIR", "data/processed"))
CHANNEL = os.getenv("DATASET_CHANNEL", "P-1")

# The NASA SMAP telemetry data is hosted in a ZIP file on Amazon S3.
# The raw.githubusercontent URLs are deprecated and return 404.
TELEMANOM_DATA_ZIP_URL = "https://s3-us-west-2.amazonaws.com/telemanom/data.zip"


def create_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Directories verified: %s, %s", RAW_DIR, PROCESSED_DIR)


def download_npy(channel: str, is_train: bool, destination: Path) -> np.ndarray:
    if destination.exists():
        logger.info("Cache hit — skipping download: %s", destination)
        return np.load(destination, allow_pickle=True)

    # Download data.zip if it doesn't exist locally
    zip_path = RAW_DIR / "data.zip"
    if not zip_path.exists():
        logger.info("Downloading dataset archive from %s", TELEMANOM_DATA_ZIP_URL)
        response = requests.get(TELEMANOM_DATA_ZIP_URL, stream=True, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(f"Download failed with HTTP {response.status_code}")
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Archive saved to %s", zip_path)

    import zipfile
    logger.info("Extracting %s data for channel %s from archive", "train" if is_train else "test", channel)
    
    # Path inside the zip file
    internal_path = f"data/train/{channel}.npy" if is_train else f"data/test/{channel}.npy"
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        if internal_path not in z.namelist():
            raise FileNotFoundError(f"{internal_path} not found in the downloaded archive.")
        
        with z.open(internal_path) as f:
            destination.write_bytes(f.read())
            
    logger.info("Saved to %s", destination)
    return np.load(destination, allow_pickle=True)


def normalize(
    train_data: np.ndarray, test_data: np.ndarray
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    logger.info(
        "Normalization complete — train shape: %s, test shape: %s",
        train_scaled.shape,
        test_scaled.shape,
    )
    return train_scaled, test_scaled, scaler


def create_windows(data: np.ndarray, window_size: int) -> np.ndarray:
    if len(data) < window_size:
        raise ValueError(
            f"Data length ({len(data)}) is shorter than window size ({window_size})."
        )
    windows = np.array(
        [data[i : i + window_size] for i in range(len(data) - window_size + 1)]
    )
    logger.info(
        "Windowing complete — %d windows of shape (%d, %d)",
        len(windows),
        window_size,
        data.shape[1] if data.ndim > 1 else 1,
    )
    return windows


def main() -> None:
    logger.info("=== Preprocessing pipeline started ===")
    create_directories()

    train_raw_path = RAW_DIR / f"{CHANNEL}_train.npy"
    test_raw_path = RAW_DIR / f"{CHANNEL}_test.npy"

    train_data = download_npy(CHANNEL, is_train=True, destination=train_raw_path)
    test_data = download_npy(CHANNEL, is_train=False, destination=test_raw_path)

    if train_data.ndim == 1:
        train_data = train_data.reshape(-1, 1)
    if test_data.ndim == 1:
        test_data = test_data.reshape(-1, 1)

    logger.info(
        "Raw data loaded — train: %s, test: %s", train_data.shape, test_data.shape
    )

    train_scaled, test_scaled, scaler = normalize(train_data, test_data)

    scaler_path = PROCESSED_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler saved to %s", scaler_path)

    np.save(PROCESSED_DIR / "test_raw.npy", test_scaled)
    logger.info("Raw (unwindowed) test data saved to data/processed/test_raw.npy")

    train_windows = create_windows(train_scaled, WINDOW_SIZE)
    test_windows = create_windows(test_scaled, WINDOW_SIZE)

    np.save(PROCESSED_DIR / "train.npy", train_windows)
    np.save(PROCESSED_DIR / "test.npy", test_windows)

    logger.info("Train windows saved to data/processed/train.npy")
    logger.info("Test windows saved to data/processed/test.npy")
    logger.info("=== Preprocessing pipeline complete ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        sys.exit(1)
