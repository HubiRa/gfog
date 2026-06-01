"""Glyph topology representation - separates structure from coordinates."""

from dataclasses import dataclass

import numpy as np


@dataclass
class GlyphTopology:
    """Represents glyph outline topology with flattened control points.

    Attributes:
        commands: List of command names (e.g. "moveTo", "lineTo", "curveTo").
        cmd_point_indices: For each command, list of indices into base_points.
        base_points: Flattened array of control points, shape (N, 2).
    """

    commands: list[str]
    cmd_point_indices: list[list[int]]
    base_points: np.ndarray  # shape (N, 2), float32


def build_topology(
    commands: list[tuple[str, list[tuple[float, float]]]],
    subdivisions: int = 1,
) -> GlyphTopology:
    """Convert glyph commands into a flat topology representation.

    Args:
        commands: List of (cmd_name, point_list) from load_glyph_commands.
        subdivisions: Number of subdivisions per curve segment. 1 = no subdivision.
                      Use higher values (e.g. 200) for more control points.

    Returns:
        GlyphTopology with base_points and command-to-point index mapping.
    """
    if subdivisions < 1:
        subdivisions = 1

    if subdivisions == 1:
        # Original behavior - no subdivision
        return _build_topology_simple(commands)
    else:
        # Subdivide curves for more control points
        return _build_topology_subdivided(commands, subdivisions)


def _build_topology_simple(commands: list[tuple[str, list[tuple[float, float]]]]) -> GlyphTopology:
    """Build topology without subdivision (original behavior)."""
    cmd_names: list[str] = []
    cmd_point_indices: list[list[int]] = []
    all_points: list[tuple[float, float]] = []

    for cmd_name, points in commands:
        cmd_names.append(cmd_name)

        # Track indices for this command's points
        indices: list[int] = []
        for pt in points:
            indices.append(len(all_points))
            all_points.append(pt)

        cmd_point_indices.append(indices)

    # Convert to numpy array
    if all_points:
        base_points = np.array(all_points, dtype=np.float32)
    else:
        base_points = np.zeros((0, 2), dtype=np.float32)

    return GlyphTopology(
        commands=cmd_names,
        cmd_point_indices=cmd_point_indices,
        base_points=base_points,
    )


def _build_topology_subdivided(
    commands: list[tuple[str, list[tuple[float, float]]]],
    subdivisions: int,
) -> GlyphTopology:
    """Build topology with subdivided curves for more control points."""
    cmd_names: list[str] = []
    cmd_point_indices: list[list[int]] = []
    all_points: list[tuple[float, float]] = []

    current_point: tuple[float, float] | None = None

    for cmd_name, points in commands:
        if cmd_name == "moveTo":
            # moveTo: just store the point, no subdivision
            cmd_names.append("moveTo")
            indices = [len(all_points)]
            all_points.append(points[0])
            cmd_point_indices.append(indices)
            current_point = points[0]

        elif cmd_name == "lineTo":
            # Subdivide line into multiple lineTo commands
            if current_point is None:
                current_point = (0.0, 0.0)

            end_point = points[0]
            sub_points = _subdivide_line(current_point, end_point, subdivisions)

            for pt in sub_points:
                cmd_names.append("lineTo")
                indices = [len(all_points)]
                all_points.append(pt)
                cmd_point_indices.append(indices)

            current_point = end_point

        elif cmd_name == "qCurveTo":
            # Subdivide quadratic bezier
            if current_point is None:
                current_point = (0.0, 0.0)

            # qCurveTo can have multiple control points with implied on-curve points
            sub_points = _subdivide_quadratic_chain(current_point, points, subdivisions)

            for pt in sub_points:
                cmd_names.append("lineTo")  # Subdivided curves become line segments
                indices = [len(all_points)]
                all_points.append(pt)
                cmd_point_indices.append(indices)

            current_point = points[-1]

        elif cmd_name == "curveTo":
            # Subdivide cubic bezier
            if current_point is None:
                current_point = (0.0, 0.0)

            # curveTo has 3 points: cp1, cp2, endpoint
            sub_points = _subdivide_cubic(
                current_point, points[0], points[1], points[2], subdivisions
            )

            for pt in sub_points:
                cmd_names.append("lineTo")
                indices = [len(all_points)]
                all_points.append(pt)
                cmd_point_indices.append(indices)

            current_point = points[2]

        elif cmd_name == "closePath":
            cmd_names.append("closePath")
            cmd_point_indices.append([])

        elif cmd_name == "endPath":
            cmd_names.append("endPath")
            cmd_point_indices.append([])

        else:
            # Unknown command - pass through
            cmd_names.append(cmd_name)
            indices = []
            for pt in points:
                indices.append(len(all_points))
                all_points.append(pt)
            cmd_point_indices.append(indices)

    if all_points:
        base_points = np.array(all_points, dtype=np.float32)
    else:
        base_points = np.zeros((0, 2), dtype=np.float32)

    return GlyphTopology(
        commands=cmd_names,
        cmd_point_indices=cmd_point_indices,
        base_points=base_points,
    )


def _subdivide_line(
    p0: tuple[float, float],
    p1: tuple[float, float],
    n: int,
) -> list[tuple[float, float]]:
    """Subdivide a line segment into n points (excluding start)."""
    points = []
    for i in range(1, n + 1):
        t = i / n
        x = p0[0] + t * (p1[0] - p0[0])
        y = p0[1] + t * (p1[1] - p0[1])
        points.append((x, y))
    return points


def _subdivide_quadratic(
    p0: tuple[float, float],
    p1: tuple[float, float],  # control point
    p2: tuple[float, float],  # endpoint
    n: int,
) -> list[tuple[float, float]]:
    """Subdivide a quadratic bezier into n points (excluding start)."""
    points = []
    for i in range(1, n + 1):
        t = i / n
        # Quadratic bezier: B(t) = (1-t)^2*P0 + 2*(1-t)*t*P1 + t^2*P2
        mt = 1 - t
        x = mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0]
        y = mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1]
        points.append((x, y))
    return points


def _subdivide_quadratic_chain(
    start: tuple[float, float],
    points: list[tuple[float, float]],
    n: int,
) -> list[tuple[float, float]]:
    """Subdivide a chain of quadratic beziers (TrueType style).

    In TrueType, qCurveTo can have multiple off-curve points with implied
    on-curve points between consecutive off-curve points.
    """
    if len(points) < 2:
        # Single point - treat as line
        return _subdivide_line(start, points[0], n)

    result = []
    current = start
    off_curve = points[:-1]
    endpoint = points[-1]

    for i, cp in enumerate(off_curve):
        if i == len(off_curve) - 1:
            # Last control point before endpoint
            result.extend(_subdivide_quadratic(current, cp, endpoint, n))
        else:
            # Implied on-curve point is midpoint between consecutive control points
            next_cp = off_curve[i + 1]
            implied = ((cp[0] + next_cp[0]) / 2, (cp[1] + next_cp[1]) / 2)
            result.extend(_subdivide_quadratic(current, cp, implied, n))
            current = implied

    return result


def _subdivide_cubic(
    p0: tuple[float, float],
    p1: tuple[float, float],  # control point 1
    p2: tuple[float, float],  # control point 2
    p3: tuple[float, float],  # endpoint
    n: int,
) -> list[tuple[float, float]]:
    """Subdivide a cubic bezier into n points (excluding start)."""
    points = []
    for i in range(1, n + 1):
        t = i / n
        # Cubic bezier: B(t) = (1-t)^3*P0 + 3*(1-t)^2*t*P1 + 3*(1-t)*t^2*P2 + t^3*P3
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt
        t2 = t * t
        t3 = t2 * t
        x = mt3 * p0[0] + 3 * mt2 * t * p1[0] + 3 * mt * t2 * p2[0] + t3 * p3[0]
        y = mt3 * p0[1] + 3 * mt2 * t * p1[1] + 3 * mt * t2 * p2[1] + t3 * p3[1]
        points.append((x, y))
    return points
