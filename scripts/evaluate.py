"""
Evaluation pipeline for the trained TCN Autoencoder.

Generates reconstruction errors, applies smoothing, and produces
anomaly detection results using both Percentile and Peak-Over-Threshold
methods. All outputs are written to the results/ directory.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from scipy.stats import genpareto

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(os.getenv("DATASET_PROCESSED_DIR", "data/processed"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))
PERCENTILE_QUANTILE = float(os.getenv("PERCENTILE_QUANTILE", 0.99))
POT_INITIAL_QUANTILE = float(os.getenv("POT_INITIAL_QUANTILE", 0.95))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))


# Import model architecture from train module
sys.path.insert(0, str(Path(__file__).parent))
from train import TCNAutoencoder


def load_model(metadata: dict, device: torch.device) -> TCNAutoencoder:
    model = TCNAutoencoder(
        num_features=metadata["num_features"],
        hidden_channels=metadata["hidden_channels"],
        latent_dim=metadata["latent_dim"],
        num_layers=metadata["tcn_layers"],
        kernel_size=metadata["tcn_kernel_size"],
    ).to(device)
    state_path = MODEL_DIR / "tcn_autoencoder.pth"
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()
    logger.info("Model loaded from %s", state_path)
    return model


def reconstruct(
    model: TCNAutoencoder, windows: np.ndarray, device: torch.device
) -> np.ndarray:
    all_reconstructions = []
    with torch.no_grad():
        for start in range(0, len(windows), BATCH_SIZE):
            batch = torch.tensor(
                windows[start : start + BATCH_SIZE], dtype=torch.float32
            ).to(device)
            recon = model(batch).cpu().numpy()
            all_reconstructions.append(recon)
    return np.concatenate(all_reconstructions, axis=0)


def compute_errors(
    originals: np.ndarray, reconstructions: np.ndarray
) -> np.ndarray:
    """Per-window MSE averaged over all timesteps and features."""
    return np.mean((originals - reconstructions) ** 2, axis=(1, 2))


def exponential_moving_average(values: np.ndarray, span: int = 20) -> np.ndarray:
    series = pd.Series(values)
    return series.ewm(span=span, adjust=False).mean().values


def apply_percentile_threshold(
    smoothed_errors: np.ndarray, quantile: float
) -> tuple[float, np.ndarray]:
    threshold = np.quantile(smoothed_errors, quantile)
    anomaly_indices = np.where(smoothed_errors > threshold)[0]
    return threshold, anomaly_indices


def apply_pot_threshold(
    smoothed_errors: np.ndarray, initial_quantile: float
) -> tuple[float, np.ndarray]:
    """
    Peak-Over-Threshold using Extreme Value Theory.

    Selects the tail of the error distribution above the initial_quantile,
    fits a Generalized Pareto Distribution to that tail, and derives a
    statistically grounded threshold corresponding to a low exceedance probability.
    """
    initial_threshold = np.quantile(smoothed_errors, initial_quantile)
    exceedances = smoothed_errors[smoothed_errors > initial_threshold] - initial_threshold

    if len(exceedances) < 10:
        logger.warning(
            "Insufficient tail observations (%d) for GPD fit. "
            "Falling back to 99.5th percentile.",
            len(exceedances),
        )
        threshold = np.quantile(smoothed_errors, 0.995)
        anomaly_indices = np.where(smoothed_errors > threshold)[0]
        return threshold, anomaly_indices

    try:
        shape, loc, scale = genpareto.fit(exceedances, floc=0)
        # Target exceedance probability: 1 - 0.9999
        target_probability = 1e-4
        if abs(shape) < 1e-6:
            # Exponential case
            gpd_threshold = initial_threshold - scale * np.log(target_probability)
        else:
            gpd_threshold = (
                initial_threshold
                + (scale / shape) * (target_probability ** (-shape) - 1)
            )
        threshold = gpd_threshold
    except Exception as exc:
        logger.warning("GPD fit failed (%s). Using 99.5th percentile fallback.", exc)
        threshold = np.quantile(smoothed_errors, 0.995)

    anomaly_indices = np.where(smoothed_errors > threshold)[0]
    return threshold, anomaly_indices


def main() -> None:
    logger.info("=== Evaluation pipeline started ===")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = MODEL_DIR / "training_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Training metadata not found at {metadata_path}. "
            "Run scripts/train.py first."
        )
    with open(metadata_path) as f:
        metadata = json.load(f)

    test_path = PROCESSED_DIR / "test.npy"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {test_path}. "
            "Run scripts/preprocess_data.py first."
        )
    test_windows = np.load(test_path)
    logger.info("Test windows loaded: shape %s", test_windows.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(metadata, device)

    logger.info("Generating reconstructions...")
    reconstructions = reconstruct(model, test_windows, device)
    logger.info("Reconstructions shape: %s", reconstructions.shape)

    np.save(RESULTS_DIR / "reconstructions.npy", reconstructions)
    logger.info("Reconstructions saved to results/reconstructions.npy")

    raw_errors = compute_errors(test_windows, reconstructions)
    smoothed_errors = exponential_moving_average(raw_errors, span=20)
    logger.info(
        "Error stats — min: %.6f, max: %.6f, mean: %.6f",
        raw_errors.min(),
        raw_errors.max(),
        raw_errors.mean(),
    )

    timestamps = np.arange(len(raw_errors))

    scores_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "raw_error": raw_errors,
            "smoothed_error": smoothed_errors,
        }
    )
    scores_df.to_csv(RESULTS_DIR / "anomaly_scores.csv", index=False)
    logger.info("Anomaly scores saved to results/anomaly_scores.csv")

    percentile_threshold, percentile_indices = apply_percentile_threshold(
        smoothed_errors, PERCENTILE_QUANTILE
    )
    logger.info(
        "Percentile threshold (%.2f): %.6f — %d anomalies detected",
        PERCENTILE_QUANTILE,
        percentile_threshold,
        len(percentile_indices),
    )
    percentile_df = pd.DataFrame(
        {
            "timestamp": timestamps[percentile_indices],
            "anomaly_score": smoothed_errors[percentile_indices],
        }
    )
    percentile_df.to_csv(RESULTS_DIR / "anomalies_percentile.csv", index=False)
    logger.info("Percentile anomalies saved to results/anomalies_percentile.csv")

    pot_threshold, pot_indices = apply_pot_threshold(
        smoothed_errors, POT_INITIAL_QUANTILE
    )
    logger.info(
        "POT threshold: %.6f — %d anomalies detected",
        pot_threshold,
        len(pot_indices),
    )
    pot_df = pd.DataFrame(
        {
            "timestamp": timestamps[pot_indices],
            "anomaly_score": smoothed_errors[pot_indices],
        }
    )
    pot_df.to_csv(RESULTS_DIR / "anomalies_pot.csv", index=False)
    logger.info("POT anomalies saved to results/anomalies_pot.csv")

    thresholds = {
        "percentile_threshold": float(percentile_threshold),
        "pot_threshold": float(pot_threshold),
        "percentile_quantile": PERCENTILE_QUANTILE,
        "pot_initial_quantile": POT_INITIAL_QUANTILE,
    }
    with open(RESULTS_DIR / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    logger.info("Thresholds saved to results/thresholds.json")

    logger.info("=== Evaluation pipeline complete ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        sys.exit(1)
