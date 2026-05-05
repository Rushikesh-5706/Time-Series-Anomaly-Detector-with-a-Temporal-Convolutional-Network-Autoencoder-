"""Unit tests for the TCN Autoencoder architecture."""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def test_causal_conv_no_future_leakage():
    """
    Causality test: a spike at position 5 must not affect the output at
    position 0. The correct approach computes outputs with and without the
    spike and asserts they are identical at the past position.
    """
    from train import CausalConv1d
    torch.manual_seed(0)
    conv = CausalConv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=1)
    conv.eval()

    x_clean = torch.zeros(1, 1, 10)
    x_spiked = torch.zeros(1, 1, 10)
    x_spiked[0, 0, 5] = 1.0

    with torch.no_grad():
        out_clean = conv(x_clean)
        out_spiked = conv(x_spiked)

    # Position 0 output must be identical regardless of spike at position 5
    assert out_clean[0, 0, 0].item() == pytest.approx(out_spiked[0, 0, 0].item(), abs=1e-6)
    # Position 6 output must differ because it is downstream of the spike
    assert out_clean[0, 0, 6].item() != pytest.approx(out_spiked[0, 0, 6].item(), abs=1e-6)


def test_autoencoder_output_shape():
    from train import TCNAutoencoder
    model = TCNAutoencoder(num_features=5, hidden_channels=16, latent_dim=8, num_layers=2, kernel_size=3)
    x = torch.rand(8, 100, 5)  # batch=8, seq=100, features=5
    out = model(x)
    assert out.shape == x.shape


def test_residual_block_channel_mismatch():
    from train import TCNResidualBlock
    block = TCNResidualBlock(in_channels=4, out_channels=8, kernel_size=3, dilation=1)
    x = torch.rand(2, 4, 50)
    out = block(x)
    assert out.shape == (2, 8, 50)


def test_residual_block_same_channels():
    from train import TCNResidualBlock
    block = TCNResidualBlock(in_channels=16, out_channels=16, kernel_size=3, dilation=2)
    x = torch.rand(4, 16, 100)
    out = block(x)
    assert out.shape == x.shape
