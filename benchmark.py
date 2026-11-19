"""Reproducible multi-seed benchmark for the TinyGuard AI research report.

Run: python benchmark.py --seeds 3 --output results/benchmark.csv
Metrics are measured on synthetic holdout data, NOT real industrial data.
"""
import argparse
from pathlib import Path
import pandas as pd
from tinyguard.core import simulate, split_normal_train_test, fit_detector, evaluate


def run_benchmark(seeds=3, samples=1200):
    if seeds < 1 or samples < 100:
        raise ValueError("seeds must be positive and samples >= 100")
    rows = []
    for seed in range(seeds):
        df = simulate(n=samples, seed=seed)
        x_train, x_test, y_test = split_normal_train_test(df)
        for kind in ("Isolation Forest", "Tiny Autoencoder"):
            detector = fit_detector(x_train, kind=kind, seed=seed)
            report = evaluate(detector, x_test, y_test)
            rows.append({
                "seed": seed,
                "model": kind,
                "train_normal_count": len(x_train),
                "test_count": len(x_test),
                "threshold": detector.threshold,
                "precision": report["precision"],
                "recall": report["recall"],
                "f1": report["f1"],
                "roc_auc": report["roc_auc"],
                "tn": report["confusion"][0][0],
                "fp": report["confusion"][0][1],
                "fn": report["confusion"][1][0],
                "tp": report["confusion"][1][1],
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--output", default="results/benchmark.csv")
    args = parser.parse_args()
    result = run_benchmark(args.seeds, args.samples)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    print(result.groupby("model")[["precision", "recall", "f1", "roc_auc"]].agg(["mean", "std"]).to_string())
    print(f"Saved {len(result)} experimental rows to {destination}")


if __name__ == "__main__":
    main()
