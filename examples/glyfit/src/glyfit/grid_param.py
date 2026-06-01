"""Grid-based parameterization for arbitrary shape representation.

Instead of deforming a font's control points, this uses a dense grid of
bezier curves that can represent arbitrary shapes with many more parameters.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GridTopology:
    """Grid-based topology with many control points.

    Attributes:
        grid_points: Base grid points, shape (N, 2).
        n_rows: Number of rows in the grid.
        n_cols: Number of columns in the grid.
        n_curves: Number of bezier curves.
    """

    grid_points: np.ndarray  # shape (N, 2), float32
    n_rows: int
    n_cols: int
    n_curves: int

    @property
    def n_points(self) -> int:
        return self.grid_points.shape[0]

    @property
    def dim(self) -> int:
        return 2 * self.n_points


def build_grid_topology(
    n_rows: int = 16,
    n_cols: int = 16,
    width: float = 1.0,
    height: float = 1.0,
    center: tuple[float, float] = (0.5, 0.5),
) -> GridTopology:
    """Build a grid-based topology.

    Args:
        n_rows: Number of rows in the grid.
        n_cols: Number of columns in the grid.
        width: Grid width (normalized).
        height: Grid height (normalized).
        center: Center of the grid (normalized).

    Returns:
        GridTopology with grid points.
    """
    # Create grid points
    x = np.linspace(center[0] - width / 2, center[0] + width / 2, n_cols)
    y = np.linspace(center[1] - height / 2, center[1] + height / 2, n_rows)

    # Create meshgrid
    xx, yy = np.meshgrid(x, y)
    grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1).astype(np.float32)

    # Number of bezier curves (horizontal + vertical edges)
    n_curves = (n_rows - 1) * n_cols + n_rows * (n_cols - 1)

    return GridTopology(
        grid_points=grid_points,
        n_rows=n_rows,
        n_cols=n_cols,
        n_curves=n_curves,
    )


def build_radial_topology(
    n_rings: int = 8,
    n_spokes: int = 16,
    inner_radius: float = 0.1,
    outer_radius: float = 0.45,
    center: tuple[float, float] = (0.5, 0.5),
) -> GridTopology:
    """Build a radial/polar grid topology.

    Better for letter-like shapes that have a central hole (like A, O, etc.)

    Args:
        n_rings: Number of concentric rings.
        n_spokes: Number of radial spokes.
        inner_radius: Inner radius (normalized).
        outer_radius: Outer radius (normalized).
        center: Center point (normalized).

    Returns:
        GridTopology with radial grid points.
    """
    points = []

    # Add center point
    points.append([center[0], center[1]])

    # Add rings
    radii = np.linspace(inner_radius, outer_radius, n_rings)
    angles = np.linspace(0, 2 * np.pi, n_spokes, endpoint=False)

    for r in radii:
        for theta in angles:
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            points.append([x, y])

    grid_points = np.array(points, dtype=np.float32)
    n_curves = n_rings * n_spokes + (n_rings - 1) * n_spokes

    return GridTopology(
        grid_points=grid_points,
        n_rows=n_rings,
        n_cols=n_spokes,
        n_curves=n_curves,
    )


def build_letter_template(
    letter: str = "A",
    n_outline_points: int = 64,
    n_interior_points: int = 32,
) -> GridTopology:
    """Build a template with points arranged in a letter-like pattern.

    Args:
        letter: Letter to use as template shape.
        n_outline_points: Number of points on the outline.
        n_interior_points: Number of interior fill points.

    Returns:
        GridTopology with letter-shaped point distribution.
    """
    points = []

    if letter.upper() == "A":
        # Outer triangle
        t = np.linspace(0, 1, n_outline_points // 3, endpoint=False)

        # Left edge
        for ti in t:
            points.append([0.5 - 0.4 * ti, 0.1 + 0.8 * ti])

        # Top to right
        for ti in t:
            points.append([0.1 + 0.4 * ti, 0.9 - 0.8 * ti])

        # Bottom edge
        for ti in t:
            points.append([0.5 + 0.4 * (1 - ti), 0.1])

        # Inner triangle (hole)
        t_inner = np.linspace(0, 1, n_outline_points // 6, endpoint=False)
        for ti in t_inner:
            points.append([0.5 - 0.15 * ti, 0.25 + 0.3 * ti])
        for ti in t_inner:
            points.append([0.35 + 0.15 * ti, 0.55 - 0.3 * ti])
        for ti in t_inner:
            points.append([0.5 + 0.15 * (1 - ti), 0.25])

        # Crossbar points
        for i in range(n_interior_points // 4):
            x = 0.25 + 0.5 * i / (n_interior_points // 4)
            points.append([x, 0.4])
            points.append([x, 0.45])

        # Fill points
        for i in range(n_interior_points // 2):
            x = 0.2 + 0.6 * np.random.random()
            y = 0.15 + 0.7 * np.random.random()
            # Check if inside the A shape (rough approximation)
            if y < 0.9 - 1.6 * abs(x - 0.5) and (y < 0.25 or y > 0.55 or abs(x - 0.5) > 0.15):
                points.append([x, y])

    else:
        # Generic: use radial pattern
        return build_radial_topology(n_rings=8, n_spokes=n_outline_points // 8)

    grid_points = np.array(points, dtype=np.float32)

    return GridTopology(
        grid_points=grid_points,
        n_rows=1,
        n_cols=len(points),
        n_curves=len(points),
    )


def apply_grid_vector(
    topology: GridTopology,
    V: torch.Tensor,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Apply offset vector to grid points and return pixel coordinates.

    Args:
        topology: GridTopology with base points (normalized 0-1).
        V: Offset tensor of shape (2*N,) in normalized coordinates.
        image_width: Output image width.
        image_height: Output image height.

    Returns:
        Deformed points as numpy array (N, 2) in pixel coordinates.
    """
    n_points = topology.n_points

    if V.numel() != 2 * n_points:
        raise ValueError(f"V has {V.numel()} elements, expected {2 * n_points}")

    # Reshape V into (N, 2) offsets
    V_np = V.detach().cpu().numpy().reshape(n_points, 2)

    # Add offsets to base points (all in normalized 0-1 coordinates)
    deformed = topology.grid_points + V_np

    # Scale to pixel coordinates
    deformed[:, 0] *= image_width
    deformed[:, 1] *= image_height

    return deformed
