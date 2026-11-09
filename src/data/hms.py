"""HMS metadata and EEG windows. Source: Kaggle HMS data description.

The 50 s sample starts at eeg_label_offset_seconds. Only its central 10 s
(20..30 s) has the row's expert label. EKG is never an EEG graph node.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import hashlib
from functools import lru_cache
import json
import sys

import numpy as np
import pandas as pd
from scipy import signal


CLASSES = ("seizure", "lpd", "gpd", "lrda", "grda", "other")
VOTE_COLUMNS = tuple(name + "_vote" for name in CLASSES)
ELECTRODES = ("Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz",
              "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2")
SAMPLE_RATE = 200
WINDOW_SECONDS = 50
LABEL_START_SECONDS = 20
LABEL_END_SECONDS = 30


def _id_string(value):
    if pd.isna(value):
        raise ValueError("missing ID")
    return str(int(value)) if isinstance(value, (int, np.integer, float, np.floating)) and float(value).is_integer() else str(value)


def validate_metadata(table):
    required = {"eeg_id", "patient_id", "eeg_label_offset_seconds", "expert_consensus"} | set(VOTE_COLUMNS)
    missing = required - set(table.columns)
    if missing:
        raise ValueError("Missing HMS columns: " + ", ".join(sorted(missing)))
    table = table.copy()
    for name in CLASSES:
        table[name + "_vote"] = pd.to_numeric(table[name + "_vote"], errors="coerce")
    votes = table[list(VOTE_COLUMNS)].to_numpy(dtype=np.float64)
    valid_vote = np.isfinite(votes).all(axis=1) & (votes >= 0).all(axis=1) & (votes.sum(axis=1) > 0)
    valid_class = table.expert_consensus.astype(str).str.lower().isin(CLASSES).to_numpy()
    offset = pd.to_numeric(table.eeg_label_offset_seconds, errors="coerce").to_numpy()
    valid_offset = np.isfinite(offset) & (offset >= 0)
    table["audit_reason"] = np.where(~valid_vote, "invalid_votes", np.where(~valid_class, "invalid_consensus", np.where(~valid_offset, "invalid_offset", "ok")))
    table["class_id"] = table.expert_consensus.astype(str).str.lower().map({v: i for i, v in enumerate(CLASSES)})
    for idx, name in enumerate(CLASSES):
        table["target_" + name] = np.where(valid_vote, votes[:, idx] / np.maximum(votes.sum(axis=1), 1), np.nan)
    return table


def intervals_union_seconds(intervals):
    """Union length of half-open intervals, avoiding overlap double count."""
    sorted_intervals = sorted((float(a), float(b)) for a, b in intervals if b > a)
    if not sorted_intervals:
        return 0.0
    start, end = sorted_intervals[0]
    total = 0.0
    for a, b in sorted_intervals[1:]:
        if a > end:
            total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    return total + end - start


def manifest_from_csv(metadata_csv, raw_dir, output_csv=None):
    """Create a per-annotation manifest without assigning split membership."""
    table = validate_metadata(pd.read_csv(metadata_csv))
    raw_dir = Path(raw_dir)
    table["eeg_path"] = table.eeg_id.map(lambda x: str((raw_dir / (_id_string(x) + ".parquet")).resolve()))
    exists = {path: Path(path).is_file() for path in table.eeg_path.unique()}
    table["file_exists"] = table.eeg_path.map(exists)
    table.loc[(table.audit_reason == "ok") & ~table.file_exists, "audit_reason"] = "missing_eeg_file"
    table["window_id"] = table.index.map(lambda i: "row_" + str(i))
    table["sample_rate"] = SAMPLE_RATE
    table["label_start_in_window_seconds"] = LABEL_START_SECONDS
    table["label_end_in_window_seconds"] = LABEL_END_SECONDS
    table["spectrogram_offset_field"] = "spectogram_label_offset_seconds" if "spectogram_label_offset_seconds" in table else ("spectrogram_label_offset_seconds" if "spectrogram_label_offset_seconds" in table else "absent")
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_csv, index=False)
    return table


def audit_manifest(table):
    ok = table[table.audit_reason == "ok"]
    grouped = ok.groupby("eeg_id")
    unique_seconds = sum(intervals_union_seconds((float(v), float(v) + WINDOW_SECONDS) for v in group.eeg_label_offset_seconds) for _, group in grouped)
    paths = sorted(set(table.loc[table.file_exists, "eeg_path"]))
    report = {
        "schema_version": "hms_manifest_v1", "metadata_rows": int(len(table)),
        "patient_count": int(ok.patient_id.nunique()),
        "metadata_patient_count": int(table.patient_id.nunique()),
        "raw_eeg_files_found": len(paths),
        "raw_eeg_bytes_found": int(sum(Path(p).stat().st_size for p in paths)),
        "valid_windows": int(len(ok)), "unique_annotated_context_seconds": unique_seconds,
        "audit_reason_counts": {str(k): int(v) for k, v in table.audit_reason.value_counts().items()},
        "spectrogram_offset_field": str(table.spectrogram_offset_field.iloc[0]) if len(table) else "absent",
    }
    if "raw_recording_seconds" in table:
        report["unique_raw_recording_seconds"] = float(table.groupby("eeg_path").raw_recording_seconds.first().dropna().sum())
    if "missing_fraction" in table:
        report["mean_window_missing_fraction"] = float(table.missing_fraction.dropna().mean()) if table.missing_fraction.notna().any() else None
        report["quality_checked_windows"] = int(table.missing_fraction.notna().sum())
    return report


def _audit_recording(item):
    path, offsets, max_missing_fraction = item
    try:
        frame = pd.read_parquet(path)
        if any(name not in frame for name in ELECTRODES):
            return [(idx, "missing_electrodes", np.nan, np.nan, np.nan) for idx, _ in offsets]
        data = frame[list(ELECTRODES)].to_numpy(dtype=np.float32).T
        recording_seconds = data.shape[1] / SAMPLE_RATE
    except Exception:
        return [(idx, "unreadable_eeg_file", np.nan, np.nan, np.nan) for idx, _ in offsets]
    result = []
    for idx, offset in offsets:
        start = int(round(float(offset) * SAMPLE_RATE))
        stop = start + WINDOW_SECONDS*SAMPLE_RATE
        if start < 0 or stop > data.shape[1]:
            result.append((idx, "window_out_of_bounds", np.nan, np.nan, recording_seconds))
            continue
        window = data[:, start:stop]
        observed = np.isfinite(window)
        missing = float((~observed).mean())
        amplitude = float(np.nanmax(np.abs(window))) if observed.any() else np.nan
        reason = "ok"
        try:
            clean_eeg(window, observed, max_missing_fraction=max_missing_fraction)
        except ValueError as exc:
            reason = str(exc)
        result.append((idx, reason, missing, amplitude, recording_seconds))
    return result


def audit_window_quality(table, max_missing_fraction=0.05, workers=1):
    """Read each recording once and mark invalid offsets/quality before training.

    This is intentionally a local reference. Spark's matching quality pass
    handles the full dataset at scale. It never replaces the row's label with
    a whole-recording label.
    """
    table = table.copy()
    table["missing_fraction"] = np.nan
    table["max_abs_amplitude"] = np.nan
    table["raw_recording_seconds"] = np.nan
    if workers < 1:
        raise ValueError("workers must be positive")
    groups = table[table.audit_reason == "ok"].groupby("eeg_path").groups
    tasks = ((path, [(idx, table.at[idx, "eeg_label_offset_seconds"]) for idx in indices],
              max_missing_fraction) for path, indices in groups.items())
    updates = []
    if workers == 1:
        for task in tasks:
            updates.extend(_audit_recording(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for group_number, rows in enumerate(executor.map(_audit_recording, tasks), 1):
                updates.extend(rows)
                if group_number % 1000 == 0:
                    print("Quality-audited %d / %d EEG recordings" % (group_number, len(groups)),
                          file=sys.stderr, flush=True)
    if updates:
        frame = pd.DataFrame(updates, columns=["index", "audit_reason", "missing_fraction",
                                                "max_abs_amplitude", "raw_recording_seconds"]).set_index("index")
        for column in frame:
            table.loc[frame.index, column] = frame[column]
    return table


@lru_cache(maxsize=8)
def _cached_eeg_frame(path):
    """Keep a few immutable recordings per data-loader process for grouped windows."""
    return pd.read_parquet(path)


def read_eeg_window(path, offset_seconds, electrodes=ELECTRODES, sample_rate=SAMPLE_RATE):
    """Return [channels, 10000 samples] and observed-value mask."""
    start = int(round(float(offset_seconds) * sample_rate))
    stop = start + WINDOW_SECONDS * sample_rate
    if start < 0:
        raise ValueError("negative EEG offset")
    frame = _cached_eeg_frame(str(path))
    absent = [c for c in electrodes if c not in frame]
    if absent:
        raise ValueError("Missing electrode columns: " + ", ".join(absent))
    if stop > len(frame):
        raise ValueError("EEG window exceeds recording: %s [%d:%d] of %d" % (path, start, stop, len(frame)))
    values = frame.iloc[start:stop][list(electrodes)].to_numpy(dtype=np.float32).T
    observed = np.isfinite(values)
    return values, observed


def clean_eeg(values, observed, max_missing_fraction=0.05, max_gap_samples=20):
    """Interpolate short interior gaps only; keep long gaps masked and zero filled."""
    values = np.asarray(values, dtype=np.float32).copy()
    observed = np.asarray(observed, dtype=bool)
    if values.shape != observed.shape or values.ndim != 2:
        raise ValueError("EEG and mask must both be [electrodes, samples]")
    if (~observed).mean() > max_missing_fraction:
        raise ValueError("too_many_missing_samples")
    valid = observed.copy()
    for channel in range(values.shape[0]):
        row = values[channel]
        finite = observed[channel]
        gaps = np.flatnonzero(np.diff(np.r_[False, ~finite, False]))
        for a, b in gaps.reshape(-1, 2):
            if 0 < a and b < row.size and b - a <= max_gap_samples:
                row[a:b] = np.interp(np.arange(a, b), [a - 1, b], [row[a - 1], row[b]])
                valid[channel, a:b] = True
        row[~valid[channel]] = 0.0
    if np.any(np.nanstd(values, axis=1) < 1e-8):
        raise ValueError("constant_channel")
    return values, valid


def eeg_spectrogram(values, sample_rate=SAMPLE_RATE, nperseg=256, hop=64):
    """Short-context spectrogram derived from exactly the provided EEG."""
    _, _, z = signal.stft(values, fs=sample_rate, nperseg=nperseg, noverlap=nperseg-hop, axis=-1, boundary=None, padded=False)
    return np.log1p(np.abs(z).astype(np.float32))


def graph_frames(values, valid, sample_rate=SAMPLE_RATE, seconds_per_frame=1):
    """Past-only frame features [time,electrode,5] and correlation proxy edges."""
    width = int(sample_rate * seconds_per_frame)
    count = values.shape[1] // width
    bands = ((0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45))
    nodes, edges, masks = [], [], []
    for i in range(count):
        x = values[:, i*width:(i+1)*width]
        m = valid[:, i*width:(i+1)*width]
        good = (m.mean(axis=1) >= 0.95) & (np.std(x, axis=1) > 1e-8)
        freqs, power = signal.welch(x, fs=sample_rate, nperseg=min(width, 200), axis=-1)
        feats = np.stack([np.log1p(power[:, (freqs >= lo) & (freqs < hi)].sum(axis=-1)) for lo, hi in bands], axis=-1).astype(np.float32)
        feats[~good] = 0
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(x)
        corr = np.nan_to_num(corr, nan=0.0).astype(np.float32)
        corr[~good, :] = 0
        corr[:, ~good] = 0
        np.fill_diagonal(corr, 0)
        nodes.append(feats); edges.append(corr); masks.append(good)
    return np.stack(nodes), np.stack(edges), np.stack(masks)


def row_target(row):
    return np.array([row["target_" + name] for name in CLASSES], dtype=np.float32)


def manifest_hash(table):
    fields = ["window_id", "eeg_id", "patient_id", "eeg_label_offset_seconds", "split"]
    canonical = table[fields].sort_values("window_id").to_csv(index=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
