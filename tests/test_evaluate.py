"""Unit tests for evaluation utilities."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def test_compute_errors_shape():
    from evaluate import compute_errors
    originals = np.random.rand(100, 50, 5)
    reconstructions = np.random.rand(100, 50, 5)
    errors = compute_errors(originals, reconstructions)
    assert errors.shape == (100,)


def test_compute_errors_zero():
    from evaluate import compute_errors
    data = np.random.rand(10, 20, 3)
    errors = compute_errors(data, data)
    np.testing.assert_allclose(errors, 0.0, atol=1e-10)


def test_ema_length():
    from evaluate import exponential_moving_average
    values = np.random.rand(500)
    smoothed = exponential_moving_average(values, span=20)
    assert len(smoothed) == len(values)


def test_percentile_threshold():
    from evaluate import apply_percentile_threshold
    errors = np.linspace(0, 1, 1000)
    threshold, indices = apply_percentile_threshold(errors, quantile=0.95)
    assert threshold == pytest.approx(0.95, abs=0.01)
    assert len(indices) == pytest.approx(50, abs=5)
