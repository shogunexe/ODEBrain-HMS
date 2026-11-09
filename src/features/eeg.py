"""Window-level EEG features. All time axes are samples at ``sample_rate`` Hz."""

from __future__ import annotations

import numpy as np


CHANNELS = (
    "Fp1", "F3", "C3", "P3", "O1", "F7", "T3", "T5",
    "Fp2", "F4", "C4", "P4", "O2", "F8", "T4", "T6",
    "Fz", "Cz", "Pz",
)
LEFT_RIGHT = (("Fp1", "Fp2"), ("F3", "F4"), ("C3", "C4"),
              ("P3", "P4"), ("O1", "O2"), ("F7", "F8"),
              ("T3", "T4"), ("T5", "T6"))
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0),
         "alpha": (8.0, 13.0), "beta": (13.0, 30.0),
         "gamma": (30.0, 45.0)}
SCHEMA_VERSION = "eeg_features_v1"


def feature_names():
    basic = ("std", "ptp", "line_length", "hjorth_activity",
             "hjorth_mobility", "hjorth_complexity", "spectral_entropy")
    names = ["%s_%s" % (x, s) for x in basic for s in ("mean", "sd")]
    for band in BANDS:
        for kind in ("abs", "rel"):
            names += ["%s_%s_%s" % (band, kind, s) for s in ("mean", "sd")]
        names.append("%s_lr_absdiff" % band)
    names += ["corr_mean", "corr_sd", "corr_abs_mean", "graph_corr_abs_mean",
              "graph_corr_abs_sd", "valid_channel_fraction", "missing_fraction"]
    return names


def _summary(values):
    good = np.asarray(values, dtype=np.float64)
    good = good[np.isfinite(good)]
    return (float(good.mean()), float(good.std())) if good.size else (0.0, 0.0)


def _fill_short_gaps(x, max_gap):
    """Interpolate only short interior gaps; report original missingness."""
    finite = np.isfinite(x)
    if finite.all():
        return x, 0.0, True
    missing = float((~finite).mean())
    if finite.sum() < 2:
        return x, missing, False
    bad = np.flatnonzero(~finite)
    groups = np.split(bad, np.where(np.diff(bad) != 1)[0] + 1)
    if any(len(g) > max_gap or g[0] == 0 or g[-1] == len(x) - 1 for g in groups):
        return x, missing, False
    index = np.arange(len(x))
    out = np.interp(index, index[finite], x[finite])
    return out, missing, True


def extract_features(samples, channel_names, sample_rate=200, frame_seconds=1.0):
    """Return scalar features and quality for one 50 s window.

    ``samples`` has shape (time, channels). A channel with a long NaN gap or
    constant signal is masked; no large missing region is silently zero-filled.
    Spectral estimates use the whole 50 s context and are for offline labels.
    """
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 3 or sample_rate <= 0:
        raise ValueError("expected (samples, channels) EEG and positive rate")
    lookup = {str(name).lower(): i for i, name in enumerate(channel_names)}
    duration = data.shape[0] / float(sample_rate)
    present = [lookup[name.lower()] for name in CHANNELS if name.lower() in lookup]
    original_missing = (float((~np.isfinite(data[:, present])).mean())
                        if present else 1.0)
    # Some HMS windows contain a common 0.75–2 s NaN pad at the start/end.
    # Only consider trimming when at least 12 electrodes share an edge defect;
    # otherwise existing valid channels retain their full time span. This
    # threshold also means previously quality-valid rows remain unchanged.
    # Drop that *edge* span, never interpolate it or connect samples across it.
    # Interior gaps retain the conservative short-gap rule below.
    if present and np.count_nonzero(~np.isfinite(data[0, present]) |
                                    ~np.isfinite(data[-1, present])) >= 12:
        complete = np.isfinite(data[:, present]).all(axis=1)
        good = np.flatnonzero(complete)
        if good.size:
            first, last = int(good[0]), int(good[-1])
            limit = int(round(0.05 * data.shape[0]))
            if first <= limit and data.shape[0] - 1 - last <= limit:
                data = data[first:last + 1]
    usable_duration = data.shape[0] / float(sample_rate)
    per = {k: [] for k in ("std", "ptp", "line_length", "hjorth_activity",
                              "hjorth_mobility", "hjorth_complexity", "spectral_entropy")}
    band_abs = {b: [] for b in BANDS}
    band_rel = {b: [] for b in BANDS}
    channel_band = {}
    clean = []
    valid_names = []
    missing = []
    max_gap = max(1, int(round(sample_rate * 0.25)))
    eps = 1e-12
    for name in CHANNELS:
        idx = lookup.get(name.lower())
        if idx is None:
            continue
        x, frac, ok = _fill_short_gaps(data[:, idx], max_gap)
        missing.append(frac)
        if not ok or np.std(x) < eps:
            continue
        x = x - np.mean(x)
        d1 = np.diff(x)
        d2 = np.diff(d1)
        v0 = np.var(x)
        v1 = np.var(d1)
        v2 = np.var(d2)
        mob = np.sqrt(v1 / max(v0, eps))
        comp = np.sqrt(v2 / max(v1, eps)) / max(mob, eps)
        vals = (np.std(x), np.ptp(x), np.mean(np.abs(d1)), v0, mob, comp)
        for key, value in zip(tuple(per)[:6], vals):
            per[key].append(float(value))
        # Parseval-normalized one-sided power. Ratios and between-channel
        # comparisons remain deterministic under identical sampling rates.
        freq = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
        power = np.abs(np.fft.rfft(x)) ** 2 / len(x) ** 2
        if len(power) > 2:
            power[1:-1] *= 2
        selected = (freq >= 0.5) & (freq < 45)
        total = max(float(power[selected].sum()), eps)
        p = power[selected] / total
        entropy = -np.sum(p[p > 0] * np.log(p[p > 0])) / max(np.log(max(p.size, 2)), eps)
        per["spectral_entropy"].append(float(entropy))
        channel_band[name] = {}
        for band, (lo, hi) in BANDS.items():
            val = float(power[(freq >= lo) & (freq < hi)].sum())
            band_abs[band].append(val)
            band_rel[band].append(val / total)
            channel_band[name][band] = val
        clean.append(x)
        valid_names.append(name)
    result = {}
    for key, values in per.items():
        result[key + "_mean"], result[key + "_sd"] = _summary(values)
    for band in BANDS:
        result[band + "_abs_mean"], result[band + "_abs_sd"] = _summary(band_abs[band])
        result[band + "_rel_mean"], result[band + "_rel_sd"] = _summary(band_rel[band])
        diffs = [abs(channel_band[a][band] - channel_band[b][band])
                 for a, b in LEFT_RIGHT if a in channel_band and b in channel_band]
        result[band + "_lr_absdiff"] = _summary(diffs)[0]
    corr_values = []
    frame_values = []
    if len(clean) >= 2:
        matrix = np.asarray(clean)
        tri = np.triu_indices(len(clean), 1)
        corr_values = np.nan_to_num(np.corrcoef(matrix), nan=0.0)[tri]
        frame_len = max(3, int(round(frame_seconds * sample_rate)))
        for start in range(0, matrix.shape[1] - frame_len + 1, frame_len):
            frame = matrix[:, start:start + frame_len]
            stable = np.std(frame, axis=1) > eps
            if stable.sum() >= 2:
                fc = np.nan_to_num(np.corrcoef(frame[stable]), nan=0.0)
                fv = fc[np.triu_indices(stable.sum(), 1)]
                frame_values.append(float(np.mean(np.abs(fv))))
    result["corr_mean"], result["corr_sd"] = _summary(corr_values)
    result["corr_abs_mean"] = _summary(np.abs(corr_values))[0]
    result["graph_corr_abs_mean"], result["graph_corr_abs_sd"] = _summary(frame_values)
    result["valid_channel_fraction"] = len(clean) / len(CHANNELS)
    result["missing_fraction"] = original_missing
    result["valid_channel_count"] = len(clean)
    result["duration_seconds"] = usable_duration
    result["quality_ok"] = (len(clean) >= 12 and result["missing_fraction"] <= 0.1
                            and usable_duration >= 0.9 * duration)
    return result
