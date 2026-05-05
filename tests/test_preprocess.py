"""Unit tests for preprocessing utilities."""

import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def test_create_windows_basic():
    from preprocess_data import create_windows
    data = np.random.rand(200, 5)
    windows = create_windows(data, window_size=50)
    assert windows.shape == (151, 50, 5)


def test_create_windows_exact_fit():
    from preprocess_data import create_windows
    data = np.random.rand(100, 3)
    windows = create_windows(data, window_size=100)
    assert windows.shape == (1, 100, 3)


def test_create_windows_too_short():
    from preprocess_data import create_windows
    data = np.random.rand(10, 2)
    with pytest.raises(ValueError):
        create_windows(data, window_size=50)


def test_normalize_range():
    from preprocess_data import normalize
    train = np.random.rand(500, 4) * 100
    test = np.random.rand(100, 4) * 80
    train_scaled, test_scaled, scaler = normalize(train, test)
    assert train_scaled.min() >= 0.0 - 1e-6
    assert train_scaled.max() <= 1.0 + 1e-6
