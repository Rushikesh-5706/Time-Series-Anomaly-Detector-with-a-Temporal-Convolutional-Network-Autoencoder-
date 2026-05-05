"""
Training pipeline for the TCN Autoencoder.

Builds the model, trains on windowed normal data, and serializes the
trained weights along with training metadata to the models/ directory.
"""

import os
import sys
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

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
LATENT_DIM = int(os.getenv("LATENT_DIM", 32))
TCN_LAYERS = int(os.getenv("TCN_LAYERS", 4))
TCN_KERNEL_SIZE = int(os.getenv("TCN_KERNEL_SIZE", 3))
TCN_CHANNELS = int(os.getenv("TCN_CHANNELS", 64))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
EPOCHS = int(os.getenv("EPOCHS", 50))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))
PROCESSED_DIR = Path(os.getenv("DATASET_PROCESSED_DIR", "data/processed"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class CausalConv1d(nn.Module):
    """
    Causal 1D convolution that ensures no future information leaks into
    the current timestep. Achieved by left-padding the sequence by
    (kernel_size - 1) * dilation positions before applying the convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNResidualBlock(nn.Module):
    """
    A single TCN residual block consisting of two causal dilated convolutions,
    layer normalization, ReLU activations, dropout, and a residual connection.
    The dilation factor doubles with each layer to grow the receptive field
    exponentially without increasing the number of parameters linearly.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm2 = nn.LayerNorm(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        # LayerNorm expects (batch, length, channels) — permute before and after
        out = self.norm1(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.norm2(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.relu2(out)
        out = self.drop2(out)

        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        return out + residual


class TCNEncoder(nn.Module):
    """
    Stacks TCN residual blocks with exponentially increasing dilation factors
    (1, 2, 4, 8, ...) to capture both short-term and long-range temporal
    dependencies. Outputs a compressed latent representation.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_dim: int,
        num_layers: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        layers = []
        current_channels = in_channels
        for i in range(num_layers):
            dilation = 2 ** i
            out_ch = hidden_channels if i < num_layers - 1 else latent_dim
            layers.append(
                TCNResidualBlock(current_channels, out_ch, kernel_size, dilation)
            )
            current_channels = out_ch
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCNDecoder(nn.Module):
    """
    Mirror of the encoder. Uses transposed convolutions to upsample the
    latent representation back to the original sequence shape. The dilation
    factors decrease to mirror the encoder's increasing pattern.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        layers = []
        current_channels = latent_dim
        for i in range(num_layers):
            dilation = 2 ** (num_layers - 1 - i)
            out_ch = hidden_channels if i < num_layers - 1 else out_channels
            layers.append(
                TCNResidualBlock(current_channels, out_ch, kernel_size, dilation)
            )
            current_channels = out_ch
        self.network = nn.Sequential(*layers)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        return self.output_activation(out)


class TCNAutoencoder(nn.Module):
    """
    Full TCN Autoencoder for unsupervised time-series anomaly detection.

    Trained exclusively on normal data to learn a compressed representation
    of nominal behavior. Anomalies are detected as high reconstruction error
    at inference time, since the model cannot faithfully reconstruct patterns
    outside its training distribution.

    Input shape:  (batch_size, sequence_length, num_features)
    Internal:     (batch_size, num_features, sequence_length)  [channels-first for Conv1d]
    Output shape: (batch_size, sequence_length, num_features)
    """

    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 64,
        latent_dim: int = 32,
        num_layers: int = 4,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = TCNEncoder(
            in_channels=num_features,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
        )
        self.decoder = TCNDecoder(
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            out_channels=num_features,
            num_layers=num_layers,
            kernel_size=kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) -> (batch, features, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        # Return (batch, seq_len, features)
        return reconstruction.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(
    model: TCNAutoencoder,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()
    model.train()
    loss_history = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            reconstruction = model(x)
            loss = criterion(reconstruction, x)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(x)

        avg_loss = epoch_loss / len(loader.dataset)
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Epoch %03d/%03d | Loss: %.6f | LR: %.6f",
                epoch,
                epochs,
                avg_loss,
                optimizer.param_groups[0]["lr"],
            )

    return loss_history


def main() -> None:
    logger.info("=== Training pipeline started ===")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DIR / "train.npy"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Preprocessed training data not found at {train_path}. "
            "Run scripts/preprocess_data.py first."
        )

    train_windows = np.load(train_path)
    logger.info("Loaded training data: shape %s", train_windows.shape)

    num_features = train_windows.shape[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    train_tensor = torch.tensor(train_windows, dtype=torch.float32)
    dataset = TensorDataset(train_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = TCNAutoencoder(
        num_features=num_features,
        hidden_channels=TCN_CHANNELS,
        latent_dim=LATENT_DIM,
        num_layers=TCN_LAYERS,
        kernel_size=TCN_KERNEL_SIZE,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model initialized — trainable parameters: %d", total_params)

    start_time = time.time()
    loss_history = train_model(model, loader, EPOCHS, LEARNING_RATE, device)
    elapsed = time.time() - start_time

    model_path = MODEL_DIR / "tcn_autoencoder.pth"
    torch.save(model.state_dict(), model_path)
    logger.info("Model saved to %s", model_path)

    metadata = {
        "dataset": "NASA_SMAP",
        "channel": "P-1",
        "num_features": num_features,
        "window_size": WINDOW_SIZE,
        "hidden_channels": TCN_CHANNELS,
        "latent_dim": LATENT_DIM,
        "tcn_layers": TCN_LAYERS,
        "tcn_kernel_size": TCN_KERNEL_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "final_loss": round(loss_history[-1], 8),
        "training_time_seconds": round(elapsed, 2),
        "loss_history": [round(v, 8) for v in loss_history],
    }

    metadata_path = MODEL_DIR / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Training metadata saved to %s", metadata_path)
    logger.info(
        "Training complete — final loss: %.6f, elapsed: %.1fs", loss_history[-1], elapsed
    )
    logger.info("=== Training pipeline complete ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        sys.exit(1)
