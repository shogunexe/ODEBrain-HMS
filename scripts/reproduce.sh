#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -d /export/usr/python/3.8.9/lib ]]; then
  export LD_LIBRARY_PATH="/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-}"
fi
PYTHON="${PYTHON:-$PWD/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing project Python environment: $PYTHON" >&2
  exit 2
fi
export PYSPARK_PYTHON="$PYTHON"
run_training() {
  local config_path="$1" run_path="$2"
  shift 2
  if [[ -f "$run_path/completed.json" && -f "$run_path/best.pt" ]]; then
    if "$PYTHON" - "$config_path" "$run_path" <<'PY'
import json
import sys
from pathlib import Path
from src.cli import file_sha256
config_path, run_path = sys.argv[1:]
run = Path(run_path)
marker = json.loads((run / "completed.json").read_text(encoding="utf-8"))
assert marker["source_config_sha256"] == file_sha256(config_path)
assert marker["manifest_sha256"] == file_sha256("data/curated/split_manifest.csv")
assert marker["best_checkpoint_sha256"] == file_sha256(run / "best.pt")
saved = json.loads((run / "config.json").read_text(encoding="utf-8"))
if saved.get("init_checkpoint"):
    assert saved["init_checkpoint_sha256"] == file_sha256(saved["init_checkpoint"])
PY
    then
      echo "Verified completed training: $run_path"
      return 0
    fi
  fi
  local -a train_args=(--config "$config_path" --run-dir "$run_path" --workers 4)
  train_args+=("$@")
  if [[ -f "$run_path/latest.pt" ]]; then
    train_args+=(--resume "$run_path/latest.pt")
  fi
  "$PYTHON" -m src.cli train "${train_args[@]}"
}
mode="${1:-smoke}"
"$PYTHON" -m src.cli doctor
"$PYTHON" -m pytest -q tests/
if [[ "$mode" == smoke ]]; then
  if [[ -f data/raw/train.csv && -d data/raw/train_eegs ]]; then
    "$PYTHON" -m src.cli audit --quality
    "$PYTHON" -m src.cli split
    "$PYTHON" -m src.cli train --config configs/models/tcn.yaml --smoke
  else
    echo "Synthetic interface smoke passed; real HMS smoke awaits Kaggle data."
  fi
elif [[ "$mode" == full ]]; then
  test -f data/raw/train.csv || { echo "Missing HMS train.csv" >&2; exit 3; }
  if [[ -f data/curated/split_manifest.csv ]] && "$PYTHON" - <<'PY'
from src.cli import require_full_data
require_full_data('data/curated/split_manifest.csv')
PY
  then
    echo "Reusing complete quality-audited patient split"
  else
    "$PYTHON" -m src.cli audit --quality --workers 8
    "$PYTHON" -m src.cli split
  fi
  "$PYTHON" - <<'PY'
from src.cli import require_full_data
require_full_data('data/curated/split_manifest.csv')
print('Complete quality-audited HMS training data confirmed')
PY
  "$PYTHON" -m src.data.graph_cache build --manifest data/curated/split_manifest.csv --output data/training/graph_classification --task classification --workers 8
  "$PYTHON" -m src.data.graph_cache build --manifest data/curated/split_manifest.csv --output data/training/graph_forecast_v2 --task forecast --cut 5 --horizons 1,3,5 --workers 4
  if [[ -f data/features/eeg_features_v1_summary.json ]] && "$PYTHON" - <<'PY'
import json, pandas as pd
from pathlib import Path
summary = json.loads(Path('data/features/eeg_features_v1_summary.json').read_text())
manifest = pd.read_csv('data/curated/split_manifest.csv')
counts = manifest.loc[manifest.audit_reason == 'ok'].groupby('split').size().to_dict()
assert summary['experiment_status'] == 'formal'
assert Path(summary['output']).resolve() == Path('data/features/eeg_features_v1').resolve()
assert all(summary['by_split'][name]['windows'] == count and
           summary['by_split'][name]['valid'] == count for name, count in counts.items())
PY
  then
    echo "Verified complete Spark feature table"
  else
    "$PWD/.venv/bin/spark-submit" scripts/build_features.py --config configs/spark.yaml --overwrite --experiment-status formal
  fi
  if [[ -f results/spark_baselines/metrics.json ]] && "$PYTHON" - <<'PY'
import json, pandas as pd
from pathlib import Path
result = json.loads(Path('results/spark_baselines/metrics.json').read_text())
manifest = pd.read_csv('data/curated/split_manifest.csv')
counts = manifest.loc[manifest.audit_reason == 'ok'].groupby('split').size().to_dict()
assert result['experiment_status'] == 'formal' and result['target_source'] == 'expert_votes'
assert result['counts'] == counts
assert all(name in result['models'] and all(
    split in result['models'][name]['splits'] for split in counts) for name in ('lr', 'rf'))
PY
  then
    echo "Verified complete Spark LR/RF baselines"
  else
    "$PWD/.venv/bin/spark-submit" scripts/train_spark_baselines.py --features data/features/eeg_features_v1 --output results/spark_baselines --master 'local[4]' --experiment-status formal --overwrite
  fi
  for name in tcn spectrogram_cnn fusion; do
    run_training "configs/models/$name.yaml" "artifacts/runs/$name"
    "$PYTHON" -m src.cli evaluate --run-dir "artifacts/runs/$name" --split val --workers 4
    "$PYTHON" -m src.cli evaluate --run-dir "artifacts/runs/$name" --split test --workers 4
  done
  run_training configs/models/odebrain_pretrain.yaml artifacts/runs/ode_pretrain
  run_training configs/models/odebrain_inspired_hms.yaml artifacts/runs/ode_finetune --init-checkpoint artifacts/runs/ode_pretrain/best.pt
  "$PYTHON" -m src.cli evaluate --run-dir artifacts/runs/ode_finetune --split val --workers 4
  "$PYTHON" -m src.cli evaluate --run-dir artifacts/runs/ode_finetune --split test --workers 4
  run_training configs/models/graph_gru_pretrain.yaml artifacts/runs/graph_gru_pretrain
  run_training configs/models/graph_gru.yaml artifacts/runs/graph_gru --init-checkpoint artifacts/runs/graph_gru_pretrain/best.pt
  "$PYTHON" -m src.cli evaluate --run-dir artifacts/runs/graph_gru --split val --workers 4
  "$PYTHON" -m src.cli evaluate --run-dir artifacts/runs/graph_gru --split test --workers 4
  "$PYTHON" scripts/evaluate_forecast.py --ode-run-dir artifacts/runs/ode_pretrain --gru-run-dir artifacts/runs/graph_gru_pretrain --manifest data/curated/split_manifest.csv --graph-cache-dir data/training/graph_forecast_v2 --output results/forecast_comparison.json --device cuda --workers 4
  "$PYTHON" scripts/write_model_cards.py --run tcn=artifacts/runs/tcn --run spectrogram_cnn=artifacts/runs/spectrogram_cnn --run fusion=artifacts/runs/fusion --run graph_gru=artifacts/runs/graph_gru --run odebrain_inspired_hms=artifacts/runs/ode_finetune
  "$PYTHON" scripts/benchmark_spark.py --manifest data/curated/split_manifest.csv --output results/spark_benchmark --experiment-status formal --methods serial,multiprocessing,spark --workers 1 4 --repeats 3
  "$PYTHON" scripts/export_dynamics.py --checkpoint artifacts/runs/ode_finetune/best.pt --manifest data/curated/split_manifest.csv --output results/dynamics
  scripts/reproduce_final_local.sh --run-seeds "${FULL_SEED_GPU:-0}"
else
  echo "Use: scripts/reproduce.sh [smoke|full]" >&2
  exit 2
fi
