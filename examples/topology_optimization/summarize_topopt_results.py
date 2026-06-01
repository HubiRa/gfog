import argparse
from pathlib import Path
from typing import Any

import numpy as np


def scalar(data: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in data:
        return default
    value = data[key]
    if value.size == 0:
        return default
    item = value.reshape(-1)[0]
    if isinstance(item, np.generic):
        return item.item()
    return item


def last_history_value(
    data: np.lib.npyio.NpzFile,
    column: str,
    default: float = float("nan"),
) -> float:
    if "history" not in data or "history_columns" not in data:
        return default
    columns = [str(name) for name in data["history_columns"]]
    if column not in columns:
        return default
    history = data["history"]
    if history.size == 0:
        return default
    return float(history[-1, columns.index(column)])


def first_history_value(
    data: np.lib.npyio.NpzFile,
    column: str,
    default: float = float("nan"),
) -> float:
    if "history" not in data or "history_columns" not in data:
        return default
    columns = [str(name) for name in data["history_columns"]]
    if column not in columns:
        return default
    history = data["history"]
    if history.size == 0:
        return default
    return float(history[0, columns.index(column)])


def summarize_file(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {
            "path": path,
            "best_feasible": float(
                scalar(data, "best_feasible_compliance", float("nan"))
            ),
            "archive_best": float(
                scalar(data, "archive_best_compliance", float("nan"))
            ),
            "best_any": float(scalar(data, "best_any_compliance", float("nan"))),
            "feasible_rate": float(scalar(data, "feasible_rate", float("nan"))),
            "history_feasible_rate": last_history_value(data, "feasible_rate"),
            "history_best": last_history_value(data, "best_last"),
            "history_best_feasible": last_history_value(data, "best_feasible_last"),
            "history_mean": last_history_value(data, "mean_last"),
            "history_mean_initial": first_history_value(data, "mean_last"),
            "optimizer": str(scalar(data, "optimizer_type", "?")),
            "encoding": str(scalar(data, "encoding", "?")),
            "generator": str(scalar(data, "generator_type", "?")),
            "generator_output_norm": str(scalar(data, "generator_output_norm", "none")),
            "discriminator": str(scalar(data, "discriminator_type", "?")),
            "g_lr": float(scalar(data, "g_lr", float("nan"))),
            "d_lr": float(scalar(data, "d_lr", float("nan"))),
            "g_opt": str(scalar(data, "g_torch_optimizer", "?")),
            "d_opt": str(scalar(data, "d_torch_optimizer", "?")),
            "ranker_tau": float(scalar(data, "ranker_tau", float("nan"))),
            "ranker_pool": int(scalar(data, "ranker_sample_pool_size", -1)),
            "buffer_diversity": float(
                scalar(data, "buffer_diversity_min_hamming", 0.0)
            ),
            "mean_hamming": float(scalar(data, "mean_hamming", float("nan"))),
            "curiosity": float(scalar(data, "curiosity", float("nan"))),
            "curiosity_space": str(scalar(data, "curiosity_space", "?")),
            "plummer_power": float(scalar(data, "plummer_power", float("nan"))),
            "plummer_eps": float(scalar(data, "plummer_eps", float("nan"))),
            "plummer_normalize": str(scalar(data, "plummer_normalize", "?")),
            "plummer_terms": str(scalar(data, "plummer_terms", "?")),
            "load_case": str(scalar(data, "load_case", "?")),
            "seed": int(scalar(data, "seed", -1)),
            "n_iter": int(round(float(last_history_value(data, "iteration", 0.0)))),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize topology optimization runs")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    files: list[Path] = []
    for path in args.paths:
        if path.is_file():
            files.append(path)
        else:
            files.extend(path.rglob("top_designs_*.npz"))

    rows = [summarize_file(path) for path in files]
    rows.sort(
        key=lambda row: (
            np.inf if np.isnan(row["best_feasible"]) else row["best_feasible"],
            np.inf if np.isnan(row["archive_best"]) else row["archive_best"],
        )
    )

    for idx, row in enumerate(rows[: args.limit], start=1):
        print(
            f"{idx:02d} best_feasible={row['best_feasible']:.6g} "
            f"archive_best={row['archive_best']:.6g} best_any={row['best_any']:.6g} "
            f"feasible_rate={row['feasible_rate']:.3f} "
            f"hist_feasible={row['history_feasible_rate']:.3f} "
            f"hist_best={row['history_best']:.6g} "
            f"hist_best_feas={row['history_best_feasible']:.6g} "
            f"hist_mean={row['history_mean']:.6g} "
            f"hist_mean0={row['history_mean_initial']:.6g} "
            f"iter={row['n_iter']} opt={row['optimizer']} enc={row['encoding']} "
            f"G={row['generator']} gnorm={row['generator_output_norm']} "
            f"D={row['discriminator']} "
            f"g={row['g_opt']}:{row['g_lr']:g} d={row['d_opt']}:{row['d_lr']:g} "
            f"tau={row['ranker_tau']:g} pool={row['ranker_pool']} "
            f"bufdiv={row['buffer_diversity']:g} ham={row['mean_hamming']:.4f} "
            f"load={row['load_case']} "
            f"curio={row['curiosity']:g}/{row['curiosity_space']} "
            f"plummer={row['plummer_power']:g},{row['plummer_eps']:g},{row['plummer_normalize']},{row['plummer_terms']} "
            f"seed={row['seed']} "
            f"path={row['path']}"
        )


if __name__ == "__main__":
    main()
