"""Cairo-based rasterization of glyph outlines."""

import cairo
import numpy as np


def rasterize(
    commands: list[tuple[str, list[tuple[float, float]]]],
    width: int = 128,
    height: int = 128,
    scale: float = 1.0,
    padding: float = 0.1,
) -> np.ndarray:
    """Rasterize glyph commands to a grayscale image.

    Args:
        commands: List of (cmd_name, point_list) in font units.
        width: Output image width in pixels.
        height: Output image height in pixels.
        scale: Additional scaling factor for font units to pixels.
        padding: Fraction of image to use as padding (0.1 = 10% on each side).

    Returns:
        Grayscale image as numpy array of shape (height, width), float32 in [0, 1].
    """
    # Create alpha-only surface for grayscale output
    surface = cairo.ImageSurface(cairo.FORMAT_A8, width, height)
    ctx = cairo.Context(surface)

    # Clear to black (transparent)
    ctx.set_source_rgba(0, 0, 0, 0)
    ctx.paint()

    # Compute bounding box of all points
    all_points = []
    for _, points in commands:
        all_points.extend(points)

    if not all_points:
        # No points - return empty image
        return np.zeros((height, width), dtype=np.float32)

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    glyph_width = max_x - min_x
    glyph_height = max_y - min_y

    # Handle degenerate cases
    if glyph_width < 1e-6:
        glyph_width = 1.0
    if glyph_height < 1e-6:
        glyph_height = 1.0

    # Compute scale to fit glyph in image with padding
    usable_width = width * (1 - 2 * padding)
    usable_height = height * (1 - 2 * padding)

    fit_scale = min(usable_width / glyph_width, usable_height / glyph_height)
    total_scale = fit_scale * scale

    # Center of glyph in font units
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Transform: translate to center of image, flip y, scale
    ctx.translate(width / 2, height / 2)
    ctx.scale(total_scale, -total_scale)  # Flip y-axis
    ctx.translate(-center_x, -center_y)

    # Draw path
    for cmd_name, points in commands:
        if cmd_name == "moveTo":
            if points:
                ctx.move_to(points[0][0], points[0][1])
        elif cmd_name == "lineTo":
            if points:
                ctx.line_to(points[0][0], points[0][1])
        elif cmd_name == "qCurveTo":
            # Quadratic Bezier: convert to cubic
            # For TrueType fonts, qCurveTo can have multiple points
            # The last point is the endpoint, intermediate points are on-curve implied
            if len(points) >= 2:
                _draw_quadratic_curves(ctx, points)
            elif len(points) == 1:
                # Single point - treat as line
                ctx.line_to(points[0][0], points[0][1])
        elif cmd_name == "curveTo":
            # Cubic Bezier: (cp1, cp2, endpoint)
            if len(points) == 3:
                ctx.curve_to(
                    points[0][0],
                    points[0][1],
                    points[1][0],
                    points[1][1],
                    points[2][0],
                    points[2][1],
                )
        elif cmd_name == "closePath":
            ctx.close_path()
        elif cmd_name == "endPath":
            # Open path - don't close
            pass

    # Fill path with white (opaque)
    ctx.set_source_rgba(1, 1, 1, 1)
    ctx.fill()

    # Extract image data
    surface.flush()
    data = surface.get_data()
    image = np.frombuffer(data, dtype=np.uint8).reshape(height, width)

    # Convert to float32 in [0, 1]
    return image.astype(np.float32) / 255.0


def _draw_quadratic_curves(ctx: cairo.Context, points: list[tuple[float, float]]) -> None:
    """Draw quadratic bezier curves, converting to cubic for Cairo.

    TrueType qCurveTo can have multiple off-curve points with implied on-curve
    points between them. The last point is always on-curve.
    """
    if len(points) < 2:
        return

    # Get current point
    current = ctx.get_current_point()

    # Process points: all except last are off-curve control points
    # Last point is on-curve endpoint
    off_curve = points[:-1]
    endpoint = points[-1]

    if len(off_curve) == 1:
        # Simple quadratic: one control point
        _quad_to_cubic(ctx, current, off_curve[0], endpoint)
    else:
        # Multiple off-curve points: implied on-curve points between them
        for i, cp in enumerate(off_curve):
            if i == len(off_curve) - 1:
                # Last control point before endpoint
                _quad_to_cubic(ctx, current, cp, endpoint)
            else:
                # Implied on-curve point is midpoint between consecutive control points
                next_cp = off_curve[i + 1]
                implied = ((cp[0] + next_cp[0]) / 2, (cp[1] + next_cp[1]) / 2)
                _quad_to_cubic(ctx, current, cp, implied)
                current = implied


def _quad_to_cubic(
    ctx: cairo.Context,
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> None:
    """Convert quadratic bezier to cubic and draw with Cairo.

    Quadratic: P0, P1 (control), P2 (endpoint)
    Cubic equivalent: P0, CP1, CP2, P2
    where CP1 = P0 + 2/3 * (P1 - P0)
          CP2 = P2 + 2/3 * (P1 - P2)
    """
    cp1 = (
        p0[0] + 2 / 3 * (p1[0] - p0[0]),
        p0[1] + 2 / 3 * (p1[1] - p0[1]),
    )
    cp2 = (
        p2[0] + 2 / 3 * (p1[0] - p2[0]),
        p2[1] + 2 / 3 * (p1[1] - p2[1]),
    )
    ctx.curve_to(cp1[0], cp1[1], cp2[0], cp2[1], p2[0], p2[1])
