"""Export real ODE cases: python scripts/export_dynamics.py --help."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dynamics.export import export_dynamics


def main():
    parser = argparse.ArgumentParser(description="Export train-only PCA and validation/test ODE cases from real HMS data")
    parser.add_argument("--checkpoint", required=True, help="Trusted trained ODEBrainInspiredHMS checkpoint")
    parser.add_argument("--manifest", required=True, help="Audited, patient-disjoint split manifest CSV")
    parser.add_argument("--output", default="results/dynamics")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-train-projection", type=int, default=256)
    parser.add_argument("--max-per-group", type=int, default=1)
    parser.add_argument("--graph-normalizer", help="Train-fitted graph normalizer NPZ containing mean and std")
    parser.add_argument("--noise-fraction", type=float, default=0.02)
    parser.add_argument("--feature-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rk4-step-seconds", type=float)
    args = parser.parse_args()
    if args.noise_fraction < 0 or args.feature_fraction < 0 or args.max_per_group < 1:
        parser.error("perturbation fractions must be nonnegative and max-per-group positive")
    index = export_dynamics(args.checkpoint, args.manifest, args.output,
                            args.device, args.max_train_projection, args.max_per_group,
                            args.graph_normalizer, args.noise_fraction,
                            args.feature_fraction, args.seed, args.rk4_step_seconds)
    print("Exported %d real cases to %s" % (len(index["cases"]), args.output))


if __name__ == "__main__":
    main()
