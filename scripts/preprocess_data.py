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

SMAP_TRAIN_URL = (
    "https://raw.githubusercontent.com/khundman/telemanom/master/data/train/{channel}.npy"
)
SMAP_TEST_URL = (
    "https://raw.githubusercontent.com/khundman/telemanom/master/data/test/{channel}.npy"
)


def create_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Directories verified: %s, %s", RAW_DIR, PROCESSED_DIR)


def download_npy(url: str, destination: Path) -> np.ndarray:
    if destination.exists():
        logger.info("Cache hit — skipping download: %s", destination)
        return np.load(destination, allow_pickle=True)

    logger.info("Downloading from %s", url)
    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(
            f"Download failed with HTTP {response.status_code} for URL: {url}"
        )

    destination.write_bytes(response.content)
    logger.info("Saved to %s (%d bytes)", destination, len(response.content))
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

    train_data = download_npy(
        SMAP_TRAIN_URL.format(channel=CHANNEL), train_raw_path
    )
    test_data = download_npy(
        SMAP_TEST_URL.format(channel=CHANNEL), test_raw_path
    )

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
