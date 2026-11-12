#!/usr/bin/env bash
# Reproduce the local final experiment and preview from completed initial runs.
# Publication and URL verification are separate, because they require a public release.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ -d /export/usr/python/3.8.9/lib ]]; then
  export LD_LIBRARY_PATH="/export/usr/python/3.8.9/lib:${LD_LIBRARY_PATH:-}"
fi
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

python_bin="${PYTHON:-$project_dir/.venv/bin/python}"
device="${FINAL_DEVICE:-cuda}"
workers="${FINAL_WORKERS:-4}"
seed_gpu=""
if [[ $# -eq 2 && "$1" == "--run-seeds" && "$2" =~ ^[0-9]+$ ]]; then
  seed_gpu="$2"
elif [[ $# -ne 0 ]]; then
  echo "Usage: scripts/reproduce_final_local.sh [--run-seeds GPU_INDEX]" >&2
  exit 2
fi
[[ -x "$python_bin" ]] || { echo "Missing project Python: $python_bin" >&2; exit 2; }
[[ "$device" == cuda || "$device" == cpu ]] || { echo "FINAL_DEVICE must be cuda or cpu" >&2; exit 2; }
[[ "$workers" =~ ^[0-9]+$ ]] || { echo "FINAL_WORKERS must be nonnegative" >&2; exit 2; }

for required in data/raw/train.csv data/curated/split_manifest.csv \
                results/spark_baselines/metrics.json \
                results/spark_benchmark/runs.json results/spark_benchmark/summary.json; do
  [[ -f "$required" ]] || { echo "Initial full experiment missing: $required" >&2; exit 3; }
done

run_refit() {
  local source="$1" output="$2"
  shift 2
  if [[ -f "$output/completed.json" && -f "$output/final.pt" ]]; then
    "$python_bin" - "$output" <<'PY'
import sys
from pathlib import Path
from scripts.refit_final import _check_final_run
_check_final_run(Path(sys.argv[1]))
PY
    echo "Verified completed refit: $output"
    return
  fi
  local -a args=(train --source-run "$source" --run-dir "$output"
                 --device "$device" --workers "$workers")
  args+=("$@")
  if [[ -f "$output/latest.pt" ]]; then args+=(--resume); fi
  "$python_bin" scripts/refit_final.py "${args[@]}"
}

# Refit budgets and architecture come from the initial validation-selected runs.
run_refit artifacts/runs/ode_pretrain artifacts/refit/ode_pretrain
run_refit artifacts/runs/graph_gru_pretrain artifacts/refit/graph_gru_pretrain
for model in tcn spectrogram_cnn fusion; do
  run_refit "artifacts/runs/$model" "artifacts/refit/$model"
done
run_refit artifacts/runs/ode_finetune artifacts/refit/odebrain_inspired_hms \
  --init-refit-run artifacts/refit/ode_pretrain
run_refit artifacts/runs/graph_gru artifacts/refit/graph_gru \
  --init-refit-run artifacts/refit/graph_gru_pretrain

for model in tcn spectrogram_cnn fusion odebrain_inspired_hms graph_gru; do
  "$python_bin" scripts/refit_final.py evaluate --run-dir "artifacts/refit/$model" \
    --device "$device" --workers "$workers"
done
"$python_bin" scripts/refit_final.py summarize \
  --run tcn=artifacts/refit/tcn \
  --run spectrogram_cnn=artifacts/refit/spectrogram_cnn \
  --run fusion=artifacts/refit/fusion \
  --run graph_gru=artifacts/refit/graph_gru \
  --run odebrain_inspired_hms=artifacts/refit/odebrain_inspired_hms \
  --output results/refit/final_comparison.json

# These are repetitions of fixed development-set choices, not test-set selection.
if [[ -n "$seed_gpu" ]]; then
  for seed in 44 45; do scripts/run_seed_stability.sh "$seed" "$seed_gpu"; done
fi
for seed in 44 45; do
  [[ -f "results/seed_stability/seed${seed}/final_comparison.json" ]] || {
    echo "Missing seed $seed repeat; run with --run-seeds GPU_INDEX or run scripts/run_seed_stability.sh $seed GPU_INDEX" >&2
    exit 3
  }
done
"$python_bin" scripts/aggregate_seed_stability.py --seeds 42 44 45

"$python_bin" scripts/run_ablations.py train --device "$device" --workers "$workers"
"$python_bin" scripts/run_ablations.py test --device "$device" --workers "$workers"
"$python_bin" scripts/supplement_ablation_metrics.py
"$python_bin" scripts/supplement_spark_record_metrics.py
"$python_bin" scripts/supplement_classification_metrics.py
"$python_bin" scripts/paired_refit_ci.py --n-boot 5000 --seed 42
"$python_bin" scripts/analyze_disagreement.py

"$python_bin" scripts/evaluate_forecast.py \
  --ode-run-dir artifacts/runs/ode_pretrain \
  --gru-run-dir artifacts/runs/graph_gru_pretrain \
  --graph-cache-dir data/training/graph_forecast_v2 \
  --output results/forecast_comparison.json \
  --device "$device" --workers "$workers"
"$python_bin" scripts/evaluate_forecast.py --protocol refit \
  --ode-run-dir artifacts/refit/ode_pretrain \
  --gru-run-dir artifacts/refit/graph_gru_pretrain \
  --graph-cache-dir data/training/graph_forecast_v2 \
  --output results/refit/forecast_comparison.json \
  --device "$device" --workers "$workers"

"$python_bin" scripts/write_model_cards.py --protocol refit \
  --run tcn=artifacts/refit/tcn \
  --run spectrogram_cnn=artifacts/refit/spectrogram_cnn \
  --run fusion=artifacts/refit/fusion \
  --run graph_gru=artifacts/refit/graph_gru \
  --run odebrain_inspired_hms=artifacts/refit/odebrain_inspired_hms
"$python_bin" scripts/export_dynamics.py \
  --checkpoint artifacts/refit/odebrain_inspired_hms/final.pt \
  --manifest data/curated/split_manifest.csv \
  --output results/dynamics_refit --device "$device"

# The preview uses real formal experiment evidence with a publication-pending mark.
"$python_bin" scripts/capture_ui.py --page 'ODE 动力学' \
  --formal-inputs results/formal_delivery_inputs.json \
  --url "${FINAL_UI_URL:-http://127.0.0.1:8502}" --launch-app --timeout 120
"$python_bin" scripts/prepare_formal_slides.py \
  --inputs results/formal_delivery_inputs.json --preview
"$python_bin" scripts/build_slides.py --preview
scripts/export_slides_pdf.sh \
  "$project_dir/slides/ODEBrain_LOCAL_PREVIEW_pending_publication.pptx" \
  "$project_dir/slides"
echo "Local preview complete; inspect the rendered PDF contact sheet before publication."
