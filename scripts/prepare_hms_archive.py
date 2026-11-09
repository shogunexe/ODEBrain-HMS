"""Validate and extract only HMS training files from the Kaggle competition ZIP.

The ZIP must first pass `unzip -t`; this command checks paths and resumes
extraction by skipping existing files of the expected uncompressed size.
"""
import argparse
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile


def safe_members(archive):
    seen = set()
    for info in archive.infolist():
        name = info.filename
        path = PurePosixPath(name)
        if not name or name.startswith("/") or "\\" in name or ".." in path.parts:
            raise ValueError("Unsafe ZIP member path")
        if name in seen:
            raise ValueError("Duplicate ZIP member path")
        seen.add(name)
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError("ZIP symlinks are not accepted")
        if info.is_dir():
            continue
        if name == "train.csv" or (len(path.parts) == 2 and path.parts[0] in
                                    ("train_eegs", "train_spectrograms") and path.suffix == ".parquet"):
            yield info


def prepare(archive_path, output_dir):
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = list(safe_members(archive))
        eeg_count = sum(info.filename.startswith("train_eegs/") for info in members)
        spec_count = sum(info.filename.startswith("train_spectrograms/") for info in members)
        if not any(info.filename == "train.csv" for info in members) or not eeg_count:
            raise ValueError("Archive lacks HMS training metadata or EEG files")
        print("Training archive: %d EEG, %d spectrogram parquet files" % (eeg_count, spec_count), flush=True)
        written = skipped = 0
        for info in members:
            destination = (root / info.filename).resolve()
            destination.relative_to(root)
            if destination.is_file() and destination.stat().st_size == info.file_size:
                skipped += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".part.%d" % os.getpid())
            try:
                with archive.open(info) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, 8 * 1024 * 1024)
                if temporary.stat().st_size != info.file_size:
                    raise IOError("Incomplete extracted member: " + info.filename)
                os.replace(temporary, destination)
                written += 1
            finally:
                if temporary.exists():
                    temporary.unlink()
            if written and written % 1000 == 0:
                print("extracted %d, skipped %d" % (written, skipped), flush=True)
        print("Extraction complete: %d written, %d already present" % (written, skipped))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/raw/hms-harmful-brain-activity-classification.zip")
    parser.add_argument("--output", default="data/raw")
    args = parser.parse_args()
    prepare(args.archive, args.output)
