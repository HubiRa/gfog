"""Summarize TSP argsort benchmark result archives."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


def scalar(data: np.lib.npyio.NpzFile, key: str, default: Any = "") -> Any:
    if key not in data.files:
        return default
    value = data[key]
    if value.ndim == 0:
        return value.item()
    if value.size == 1:
        return value.reshape(-1)[0].item()
    return ",".join(str(item) for item in value.tolist())


def load_row(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    history = data["history"] if "history" in data.files else np.empty((0, 0))
    final_mean = float(history[-1, 3]) if history.size else float("nan")
    final_median = float(history[-1, 4]) if history.size else float("nan")
    best_length = float(scalar(data, "best_length", float("inf")))
    best_raw_length = float(scalar(data, "best_raw_length", best_length))
    nearest_neighbor = float(scalar(data, "nearest_neighbor_length", float("nan")))
    two_opt_nn = float(scalar(data, "two_opt_nearest_neighbor_length", float("nan")))
    return {
        "path": str(path),
        "best_length": best_length,
        "best_raw_length": best_raw_length,
        "final_mean": final_mean,
        "final_median": final_median,
        "nn_ratio": best_length / nearest_neighbor
        if nearest_neighbor > 0
        else float("nan"),
        "two_opt_nn_ratio": best_length / two_opt_nn
        if two_opt_nn > 0
        else float("nan"),
        "optimizer": scalar(data, "optimizer"),
        "n_cities": scalar(data, "n_cities"),
        "city_seed": scalar(data, "city_seed"),
        "seed": scalar(data, "seed"),
        "n_iter": scalar(data, "n_iter"),
        "batch_size": scalar(data, "batch_size"),
        "buffer_multiplier": scalar(data, "buffer_multiplier"),
        "generator_type": scalar(data, "generator_type", "mlp"),
        "discriminator_type": scalar(data, "discriminator_type", "mlp"),
        "generator_hidden_dims": scalar(data, "generator_hidden_dims"),
        "discriminator_hidden_dims": scalar(data, "discriminator_hidden_dims"),
        "set_generator_dim": scalar(data, "set_generator_dim", ""),
        "set_generator_depth": scalar(data, "set_generator_depth", ""),
        "set_generator_heads": scalar(data, "set_generator_heads", ""),
        "set_discriminator_dim": scalar(data, "set_discriminator_dim", ""),
        "set_discriminator_depth": scalar(data, "set_discriminator_depth", ""),
        "set_discriminator_heads": scalar(data, "set_discriminator_heads", ""),
        "generator_output_norm": scalar(data, "generator_output_norm"),
        "g_optimizer": scalar(data, "g_optimizer"),
        "d_optimizer": scalar(data, "d_optimizer"),
        "g_lr": scalar(data, "g_lr", float("nan")),
        "d_lr": scalar(data, "d_lr", float("nan")),
        "curiosity": scalar(data, "curiosity", float("nan")),
        "curiosity_reference": scalar(data, "curiosity_reference"),
        "curiosity_t": scalar(data, "curiosity_t"),
        "ranker_tau": scalar(data, "ranker_tau"),
        "route_prior": scalar(data, "route_prior", "none"),
        "route_prior_alpha": scalar(data, "route_prior_alpha", 1.0),
        "objective_two_opt_passes": scalar(data, "objective_two_opt_passes", 0),
        "ranker_list_size": scalar(data, "ranker_list_size"),
        "ranker_sample_pool_size": scalar(data, "ranker_sample_pool_size"),
        "random_best": scalar(data, "random_best"),
        "nearest_neighbor_length": nearest_neighbor,
        "two_opt_nearest_neighbor_length": two_opt_nn,
        "two_opt_random_length": scalar(data, "two_opt_random_length"),
    }


def format_number(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir", type=Path, default=Path("results/tsp_argsort_sweep")
    )
    parser.add_argument("--pattern", default="*.npz")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [load_row(path) for path in sorted(args.results_dir.glob(args.pattern))]
    rows.sort(key=lambda row: row["best_length"])
    if not rows:
        print(f"No result files found in {args.results_dir} matching {args.pattern}")
        return

    columns = list(rows[0].keys())
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    print(f"files={len(rows)}")
    for idx, row in enumerate(rows[: args.top], start=1):
        print(
            f"{idx:02d} best={row['best_length']:.6f} mean={row['final_mean']:.6f} "
            f"raw={row['best_raw_length']:.6f} f2opt={row['objective_two_opt_passes']} "
            f"opt={row['optimizer']} gopt={row['g_optimizer']} dopt={row['d_optimizer']} "
            f"glr={format_number(row['g_lr'])} dlr={format_number(row['d_lr'])} "
            f"curio={format_number(row['curiosity'])} "
            f"norm={row['generator_output_norm']} gtype={row['generator_type']} "
            f"prior={row['route_prior']} alpha={format_number(row['route_prior_alpha'])} "
            f"dtype={row['discriminator_type']} gh={row['generator_hidden_dims']} "
            f"dh={row['discriminator_hidden_dims']} sg={row['set_generator_dim']} "
            f"sd={row['set_discriminator_dim']} tau={format_number(row['ranker_tau'])} "
            f"path={row['path']}"
        )


if __name__ == "__main__":
    main()
