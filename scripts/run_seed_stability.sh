#!/usr/bin/env bash
# Repeat the frozen train+val refit with a new initialization seed.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: scripts/run_seed_stability.sh SEED CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi
seed="$1"
device="$2"
if [[ ! "$seed" =~ ^[0-9]+$ ]] || [[ "$seed" == 42 ]] || [[ ! "$device" =~ ^[0-9]+$ ]]; then
  echo "SEED must differ from 42; GPU index must be nonnegative" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH="/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="$device"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
python_bin="$PWD/.venv/bin/python"
root="$PWD/artifacts/seed_stability/seed${seed}"
results="$PWD/results/seed_stability/seed${seed}"
mkdir -p "$root" "$results"

run_one() {
  local source="$1" name="$2" init="${3:-}"
  local run="$root/$name"
  local -a args=(train --source-run "$source" --run-dir "$run"
                 --seed "$seed" --device cuda --workers 4)
  if [[ -n "$init" ]]; then args+=(--init-refit-run "$root/$init"); fi
  if [[ -f "$run/completed.json" && -f "$run/final.pt" ]]; then
    "$python_bin" - "$run" "$seed" <<'PY'
import json, sys
from pathlib import Path
from scripts.refit_final import _check_final_run
run = _check_final_run(Path(sys.argv[1]))
assert run["config"]["refit"]["seed"] == int(sys.argv[2])
PY
  else
    if [[ -f "$run/latest.pt" ]]; then args+=(--resume); fi
    "$python_bin" scripts/refit_final.py "${args[@]}"
  fi
}

for name in tcn spectrogram_cnn fusion; do
  run_one "artifacts/runs/$name" "$name"
  "$python_bin" scripts/refit_final.py evaluate --run-dir "$root/$name" \
    --output-dir "$results/$name" --device cuda --workers 4
done
run_one artifacts/runs/ode_pretrain ode_pretrain
run_one artifacts/runs/graph_gru_pretrain graph_gru_pretrain
run_one artifacts/runs/ode_finetune odebrain_inspired_hms ode_pretrain
"$python_bin" scripts/refit_final.py evaluate \
  --run-dir "$root/odebrain_inspired_hms" \
  --output-dir "$results/odebrain_inspired_hms" --device cuda --workers 4
run_one artifacts/runs/graph_gru graph_gru graph_gru_pretrain
"$python_bin" scripts/refit_final.py evaluate --run-dir "$root/graph_gru" \
  --output-dir "$results/graph_gru" --device cuda --workers 4

"$python_bin" scripts/refit_final.py summarize \
  --run "tcn=$root/tcn" \
  --run "spectrogram_cnn=$root/spectrogram_cnn" \
  --run "fusion=$root/fusion" \
  --run "graph_gru=$root/graph_gru" \
  --run "odebrain_inspired_hms=$root/odebrain_inspired_hms" \
  --results-root "$results" --output "$results/final_comparison.json"
