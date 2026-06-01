import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


def pairwise_l2_mean(designs: np.ndarray) -> float:
    flat = designs.reshape(designs.shape[0], -1)
    if flat.shape[0] < 2:
        return 0.0
    diffs = flat[:, None, :] - flat[None, :, :]
    dists = np.sqrt((diffs**2).sum(axis=-1))
    triu = np.triu_indices(flat.shape[0], k=1)
    return float(dists[triu].mean())


def pairwise_hamming_mean(designs: np.ndarray, threshold: float = 0.5) -> float:
    binary = (designs >= threshold).reshape(designs.shape[0], -1)
    if binary.shape[0] < 2:
        return 0.0
    diffs = binary[:, None, :] != binary[None, :, :]
    dists = diffs.mean(axis=-1)
    triu = np.triu_indices(binary.shape[0], k=1)
    return float(dists[triu].mean())


def summarize_run(path: Path) -> dict[str, float | np.ndarray]:
    data = np.load(path)
    designs = data["designs"]
    values = data["values"]
    return {
        "designs": designs,
        "values": values,
        "curiosity": float(data["curiosity"][0]),
        "seed": int(data["seed"][0]),
        "best_objective": float(values[0, 2]),
        "mean_objective_topk": float(values[:, 2].mean()),
        "mean_l2": pairwise_l2_mean(designs),
        "mean_hamming": pairwise_hamming_mean(designs),
    }


def make_side_by_side(
    left: dict[str, float | np.ndarray],
    right: dict[str, float | np.ndarray],
    output_path: Path,
) -> None:
    left_designs = left["designs"]
    right_designs = right["designs"]
    assert isinstance(left_designs, np.ndarray)
    assert isinstance(right_designs, np.ndarray)

    rows = max(left_designs.shape[0], right_designs.shape[0])
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(8, 3 * rows))
    axes = np.atleast_2d(axes)

    titles = [
        (
            f"curiosity={left['curiosity']:.0f}\n"
            f"best obj={left['best_objective']:.4f}\n"
            f"L2={left['mean_l2']:.4f}, ham={left['mean_hamming']:.4f}"
        ),
        (
            f"curiosity={right['curiosity']:.0f}\n"
            f"best obj={right['best_objective']:.4f}\n"
            f"L2={right['mean_l2']:.4f}, ham={right['mean_hamming']:.4f}"
        ),
    ]

    for row in range(rows):
        for col, run in enumerate([left, right]):
            ax = axes[row, col]
            designs = run["designs"]
            values = run["values"]
            assert isinstance(designs, np.ndarray)
            assert isinstance(values, np.ndarray)
            if row >= designs.shape[0]:
                ax.axis("off")
                continue
            ax.imshow(designs[row], cmap="gray_r", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"#{row + 1}: obj={values[row, 2]:.4f}\nvol={values[row, 0]:.3f} rough={values[row, 1]:.3f}",
                fontsize=9,
            )

    fig.text(0.25, 0.995, titles[0], ha="center", va="top", fontsize=11)
    fig.text(0.75, 0.995, titles[1], ha="center", va="top", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two toy topopt runs")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/toy_topopt/comparison.png"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    left = summarize_run(args.left)
    right = summarize_run(args.right)

    logger.info(
        f"Left run: curiosity={left['curiosity']:.0f} best_obj={left['best_objective']:.4f} "
        f"mean_l2={left['mean_l2']:.4f} mean_hamming={left['mean_hamming']:.4f}"
    )
    logger.info(
        f"Right run: curiosity={right['curiosity']:.0f} best_obj={right['best_objective']:.4f} "
        f"mean_l2={right['mean_l2']:.4f} mean_hamming={right['mean_hamming']:.4f}"
    )

    make_side_by_side(left, right, args.output)
    logger.info(f"Saved comparison figure to {args.output}")


if __name__ == "__main__":
    main()
