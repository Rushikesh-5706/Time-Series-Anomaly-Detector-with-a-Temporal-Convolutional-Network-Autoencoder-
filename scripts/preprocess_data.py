"""
Preprocessing pipeline for the NASA SMAP dataset.

Downloads channel P-1 telemetry data via git sparse-checkout, applies MinMax
normalization, creates overlapping sliding windows, and serializes all
artifacts to disk. Falls back to seeded synthetic data if the download fails.
"""

import os
import sys
import logging
import subprocess
import shutil
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", 100))
RAW_DIR = Path(os.getenv("DATASET_RAW_DIR", "data/raw"))
PROCESSED_DIR = Path(os.getenv("DATASET_PROCESSED_DIR", "data/processed"))
CHANNEL = os.getenv("DATASET_CHANNEL", "P-1")
SYNTHETIC_SEED = 42


def create_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Directories verified: %s, %s", RAW_DIR, PROCESSED_DIR)


def fetch_via_git_sparse(channel: str, split: str, destination: Path) -> bool:
    """
    Attempts to download a single .npy file from the telemanom repository
    using git sparse-checkout, which correctly resolves Git LFS pointers.
    Returns True on success, False on any failure.
    """
    clone_dir = RAW_DIR / "_telemanom_clone"
    try:
        if clone_dir.exists():
            shutil.rmtree(clone_dir)

        subprocess.run(
            [
                "git", "clone",
                "--no-checkout",
                "--filter=blob:none",
                "--depth=1",
                "https://github.com/khundman/telemanom.git",
                str(clone_dir),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )

        target_path = f"data/{split}/{channel}.npy"
        subprocess.run(
            ["git", "sparse-checkout", "set", target_path],
            cwd=str(clone_dir),
            check=True,
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            ["git", "checkout"],
            cwd=str(clone_dir),
            check=True,
            capture_output=True,
            timeout=60,
        )

        source = clone_dir / target_path
        if not source.exists():
            logger.warning("Sparse checkout did not produce the expected file at %s", source)
            return False

        shutil.copy2(source, destination)
        shutil.rmtree(clone_dir)
        logger.info("Downloaded %s via git sparse-checkout to %s", target_path, destination)
        return True

    except Exception as exc:
        logger.warning("Git sparse-checkout failed: %s", exc)
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
        return False


def generate_synthetic_smap(channel: str, is_train: bool) -> np.ndarray:
    """
    Generates realistic synthetic sensor data as a fallback when the upstream
    repository is unreachable. The random state is seeded for reproducibility.

    The signal structure mirrors the SMAP P-1 channel: 25 features with
    oscillatory dynamics and injected anomaly in the test set.
    """
    rng = np.random.RandomState(SYNTHETIC_SEED + (0 if is_train else 1))
    num_samples = 4000 if is_train else 1500
    num_features = 25

    time_axis = np.linspace(0, 100, num_samples)
    data = np.zeros((num_samples, num_features))

    for i in range(num_features):
        freq = rng.uniform(0.1, 2.0)
        phase = rng.uniform(0, 2 * np.pi)
        amplitude = rng.uniform(0.5, 1.5)
        data[:, i] = amplitude * np.sin(time_axis * freq + phase) + rng.normal(0, 0.15, num_samples)

    if not is_train:
        anomaly_start = int(num_samples * 0.70)
        anomaly_end = anomaly_start + 60
        data[anomaly_start:anomaly_end, 0] += rng.normal(3.5, 0.8, 60)

    return data


def load_or_generate(channel: str, is_train: bool, destination: Path) -> np.ndarray:
    if destination.exists():
        logger.info("Cache hit: %s", destination)
        return np.load(destination, allow_pickle=True)

    split = "train" if is_train else "test"
    success = fetch_via_git_sparse(channel, split, destination)

    if success:
        data = np.load(destination, allow_pickle=True)
        logger.info("Real SMAP data loaded: %s %s", channel, split)
    else:
        logger.warning(
            "Real data unavailable. Using seeded synthetic fallback for %s %s.", channel, split
        )
        data = generate_synthetic_smap(channel, is_train)
        np.save(destination, data)

    return data


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

    train_data = load_or_generate(CHANNEL, is_train=True, destination=train_raw_path)
    test_data = load_or_generate(CHANNEL, is_train=False, destination=test_raw_path)

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
    logger.info("Unwindowed test data saved to data/processed/test_raw.npy")

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
