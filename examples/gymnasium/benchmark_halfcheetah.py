import argparse
from types import SimpleNamespace

import numpy as np
from rich.console import Console
from rich.table import Table

from halfcheetah import build_parser, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark GFog optimizers on HalfCheetah"
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["default", "hinge", "wgangp"],
        choices=["default", "hinge", "wgan", "wgangp"],
    )
    parser.add_argument(
        "--curiosity_values",
        nargs="+",
        type=float,
        default=[2.0],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--show_tables", action="store_true")

    base_parser = build_parser()
    known_args, remaining = parser.parse_known_args()
    base_args = base_parser.parse_args(remaining)

    console = Console()
    results = []
    for optimizer in known_args.optimizers:
        for curiosity in known_args.curiosity_values:
            for seed in known_args.seeds:
                args = SimpleNamespace(**vars(base_args))
                args.optimizer = optimizer
                args.curiosity = curiosity
                args.seed = seed
                console.rule(f"optimizer={optimizer} curiosity={curiosity} seed={seed}")
                result = run_experiment(args, show_tables=known_args.show_tables)
                results.append(result)

    summary = Table(title="HalfCheetah Optimizer Benchmark")
    summary.add_column("optimizer")
    summary.add_column("curiosity")
    summary.add_column("seeds")
    summary.add_column("median final best", justify="right")
    summary.add_column("median improvement", justify="right")
    summary.add_column("median best generalization", justify="right")

    for optimizer in known_args.optimizers:
        for curiosity in known_args.curiosity_values:
            rows = [
                row
                for row in results
                if row["optimizer"] == optimizer and row["curiosity"] == curiosity
            ]
            final_best = np.median([row["final_best"] for row in rows])
            improvement = np.median([row["improvement"] for row in rows])
            generalization = np.median([row["generalization_best"] for row in rows])
            summary.add_row(
                optimizer,
                f"{curiosity:.2f}",
                str(len(rows)),
                f"{final_best:.4f}",
                f"{improvement:.4f}",
                f"{generalization:.4f}",
            )

    console.print()
    console.print(summary)


if __name__ == "__main__":
    main()
