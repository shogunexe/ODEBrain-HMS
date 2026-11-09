"""Edge padding is cropped consistently; interior long gaps stay invalid."""

import numpy as np

from src.features.eeg import CHANNELS, extract_features, feature_names


def _signal(seconds=50, rate=200):
    t = np.arange(seconds * rate) / rate
    return np.stack([np.sin(2 * np.pi * (4 + i % 8) * t + i * 0.1)
                     for i in range(len(CHANNELS))], axis=1)


def test_common_edge_pad_uses_shared_span_without_interpolation():
    signal = _signal()
    signal[:400] = np.nan
    repaired = extract_features(signal, CHANNELS)
    reference = extract_features(signal[400:], CHANNELS)
    assert repaired["quality_ok"]
    assert repaired["valid_channel_count"] == 19
    assert repaired["missing_fraction"] == 0.04
    assert repaired["duration_seconds"] == 48.0
    for name in feature_names():
        if name != "missing_fraction":
            assert np.isclose(repaired[name], reference[name], atol=1e-9, rtol=1e-9), name


def test_long_interior_gap_is_not_silently_filled():
    signal = _signal()
    signal[3000:3400] = np.nan
    result = extract_features(signal, CHANNELS)
    assert not result["quality_ok"]
    assert result["valid_channel_count"] == 0
    assert result["duration_seconds"] == 50.0
    assert result["missing_fraction"] == 0.04


def test_single_edge_bad_channel_does_not_trim_other_channels():
    signal = _signal()
    signal[:400, 0] = np.nan
    result = extract_features(signal, CHANNELS)
    assert result["quality_ok"]
    assert result["valid_channel_count"] == 18
    assert result["duration_seconds"] == 50.0
