"""Parameter space for glyph deformation."""

import torch

from .topology import GlyphTopology


def apply_vector(
    topology: GlyphTopology, V: torch.Tensor
) -> list[tuple[str, list[tuple[float, float]]]]:
    """Apply offset vector V to deform glyph control points.

    Args:
        topology: GlyphTopology with base_points (N, 2).
        V: Offset tensor of shape (2*N,) where each point k gets (dx_k, dy_k).

    Returns:
        List of (command_name, updated_points) for rasterization.

    Raises:
        ValueError: If V has wrong dimension.
    """
    n_points = topology.base_points.shape[0]
    expected_dim = 2 * n_points

    if V.numel() != expected_dim:
        raise ValueError(
            f"V has {V.numel()} elements, expected {expected_dim} (2 * {n_points} points)"
        )

    # Reshape V into (N, 2) offsets
    V_np = V.detach().cpu().numpy().reshape(n_points, 2)

    # Add offsets to base points
    deformed_points = topology.base_points + V_np

    # Rebuild command list with deformed points
    commands: list[tuple[str, list[tuple[float, float]]]] = []

    for cmd_name, indices in zip(topology.commands, topology.cmd_point_indices):
        points = [(float(deformed_points[i, 0]), float(deformed_points[i, 1])) for i in indices]
        commands.append((cmd_name, points))

    return commands
