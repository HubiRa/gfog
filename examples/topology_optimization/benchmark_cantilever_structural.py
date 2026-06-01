import argparse
from types import SimpleNamespace

import numpy as np
from rich.console import Console
from rich.table import Table

from cantilever_structural import build_parser, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GFog structural cantilever")
    parser.add_argument(
        "--curiosity_values",
        nargs="+",
        type=float,
        default=[0.0, 20.0, 100.0, 200.0],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])

    base_parser = build_parser()
    known_args, remaining = parser.parse_known_args()
    base_args = base_parser.parse_args(remaining)

    console = Console()
    results = []
    for curiosity in known_args.curiosity_values:
        for seed in known_args.seeds:
            args = SimpleNamespace(**vars(base_args))
            args.curiosity = curiosity
            args.seed = seed
            args.output_dir = (
                base_args.output_dir / f"curiosity_{curiosity:g}" / f"seed_{seed}"
            )
            console.rule(f"backend={args.backend} curiosity={curiosity} seed={seed}")
            results.append(run_experiment(args))

    summary = Table(title="Structural Cantilever Benchmark")
    summary.add_column("backend")
    summary.add_column("curiosity")
    summary.add_column("seeds")
    summary.add_column("median best compliance", justify="right")
    summary.add_column("median mean top-k compliance", justify="right")
    summary.add_column("median pairwise L2", justify="right")
    summary.add_column("median pairwise Hamming", justify="right")

    for curiosity in known_args.curiosity_values:
        rows = [row for row in results if row["curiosity"] == curiosity]
        summary.add_row(
            base_args.backend,
            f"{curiosity:.2f}",
            str(len(rows)),
            f"{np.median([row['best_compliance'] for row in rows]):.4f}",
            f"{np.median([row['mean_compliance_topk'] for row in rows]):.4f}",
            f"{np.median([row['mean_l2'] for row in rows]):.4f}",
            f"{np.median([row['mean_hamming'] for row in rows]):.4f}",
        )

    console.print()
    console.print(summary)


if __name__ == "__main__":
    main()
