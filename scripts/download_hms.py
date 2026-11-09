"""Download authorized Kaggle HMS files without recording the API secret.

The account holder must accept the competition rules in Kaggle first.
Use --balanced-smoke to fetch a patient-diverse real subset; --all-eegs
fetches every EEG named by train.csv. Resume skips existing nonempty files.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request
import urllib.error
import zipfile
import time

import pandas as pd


COMPETITION = "hms-harmful-brain-activity-classification"
CLASSES = ("Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other")


class RateLimited(RuntimeError):
    """Kaggle asked us to stop making requests for now."""


class AuthSafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        if result is not None and urllib.parse.urlparse(newurl).hostname != urllib.parse.urlparse(req.full_url).hostname:
            result.remove_header("Authorization")
        return result


def auth_header():
    path = Path.home() / ".kaggle" / "kaggle.json"
    if not path.is_file():
        raise FileNotFoundError("Configure ~/.kaggle/kaggle.json after accepting Kaggle HMS rules")
    credentials = json.loads(path.read_text())
    token = (credentials["username"] + ":" + credentials["key"]).encode()
    return "Basic " + base64.b64encode(token).decode()


def _download_one(name, root, authorization):
    target = Path(root) / name
    if target.is_file() and target.stat().st_size > 0:
        return (name, "exists", target.stat().st_size)
    url = "https://www.kaggle.com/api/v1/competitions/data/download/" + COMPETITION + "/" + urllib.parse.quote(name, safe="")
    opener = urllib.request.build_opener(AuthSafeRedirect())
    req = urllib.request.Request(url, headers={"Authorization": authorization, "User-Agent": "ODEBrain-HMS-research/0.1"})
    with opener.open(req, timeout=120) as response:
        content = response.read()
    if zipfile.is_zipfile(io.BytesIO(content)):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if len(members) != 1 or Path(members[0].filename).name != Path(name).name:
                raise ValueError("Unexpected Kaggle archive members for " + name)
            payload = archive.read(members[0])
    elif name.endswith(".parquet") and content.startswith(b"PAR1") and content.endswith(b"PAR1"):
        payload = content
    else:
        raise ValueError("Unexpected Kaggle response format for " + name)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".download")
    with temporary.open("wb") as dest:
        dest.write(payload)
    os.replace(temporary, target)
    return (name, "downloaded", target.stat().st_size)


def download_one(name, root, authorization):
    for attempt in range(6):
        try:
            return _download_one(name, root, authorization)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise RateLimited("Kaggle returned HTTP 429; stop and retry after the rate limit clears") from exc
            if exc.code in (400, 401, 403, 404):
                raise
            error = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, zipfile.BadZipFile) as exc:
            error = exc
        if attempt < 5:
            time.sleep(min(30, 2 ** attempt))
    raise error


def balanced_smoke_ids(csv_path, per_class_per_split=2, seed=42):
    """Choose files from a fixed patient split, without inspecting signals."""
    import numpy as np
    table = pd.read_csv(csv_path)
    patients = np.array(sorted(table.patient_id.astype(str).unique()))
    np.random.RandomState(seed).shuffle(patients)
    n_test = max(1, round(len(patients)*.2))
    n_val = max(1, round((len(patients)-n_test)*.2))
    sets = {"test": set(patients[:n_test]), "val": set(patients[n_test:n_test+n_val]),
            "train": set(patients[n_test+n_val:])}
    selected = []
    used_patients = set()
    for split in ("train", "val", "test"):
        for label in CLASSES:
            choices = table[(table.patient_id.astype(str).isin(sets[split])) & (table.expert_consensus == label)]
            choices = choices.sort_values(["patient_id", "eeg_id", "eeg_label_offset_seconds"])
            count = 0
            for row in choices.itertuples():
                if row.patient_id in used_patients:
                    continue
                selected.append(str(row.eeg_id))
                used_patients.add(row.patient_id)
                count += 1
                if count >= per_class_per_split:
                    break
            if count < per_class_per_split:
                raise ValueError("Not enough distinct patients for %s/%s" % (split, label))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--file", action="append", default=[], help="Relative Kaggle file path, repeatable")
    parser.add_argument("--metadata", action="store_true")
    parser.add_argument("--balanced-smoke", action="store_true")
    parser.add_argument("--per-class-per-split", type=int, default=2)
    parser.add_argument("--all-eegs", action="store_true")
    parser.add_argument("--all-spectrograms", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        parser.error("workers must be 1..16")
    names = list(args.file)
    if args.metadata:
        names.append("train.csv")
    csv_path = Path(args.root) / "train.csv"
    if args.balanced_smoke or args.all_eegs or args.all_spectrograms:
        if not csv_path.is_file():
            parser.error("download train.csv with --metadata first")
        if args.balanced_smoke or args.all_eegs:
            ids = balanced_smoke_ids(csv_path, args.per_class_per_split) if args.balanced_smoke else sorted(set(pd.read_csv(csv_path).eeg_id.astype(str)))
            names.extend("train_eegs/" + eeg_id + ".parquet" for eeg_id in ids)
        if args.all_spectrograms:
            ids = sorted(set(pd.read_csv(csv_path, usecols=["spectrogram_id"]).spectrogram_id.astype(str)))
            names.extend("train_spectrograms/" + spec_id + ".parquet" for spec_id in ids)
    names = list(dict.fromkeys(names))
    if not names:
        parser.error("choose --metadata, --balanced-smoke, --all-eegs, or --file")
    auth = auth_header()
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # Keep a small queue: a 429 must not leave thousands of pending requests.
        for start in range(0, len(names), args.workers):
            batch = names[start:start + args.workers]
            futures = {pool.submit(download_one, name, args.root, auth): name for name in batch}
            rate_limited = False
            for future in as_completed(futures):
                name = futures[future]
                try:
                    print(*future.result())
                except RateLimited as exc:
                    rate_limited = True
                    print(name, "STOPPED", str(exc))
                except Exception as exc:
                    errors.append((name, type(exc).__name__, str(exc)))
                    print(name, "FAILED", type(exc).__name__, str(exc))
            if rate_limited:
                raise SystemExit("Kaggle rate limit reached; download stopped safely, rerun later to resume")
    if errors:
        raise SystemExit("%d files failed; rerun to resume" % len(errors))


if __name__ == "__main__":
    main()
