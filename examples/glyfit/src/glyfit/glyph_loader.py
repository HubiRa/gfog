"""Load glyph outlines from TTF/OTF font files."""

from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont


def load_glyph_commands(
    ttf_path: str, char: str, font_number: int = 0
) -> list[tuple[str, list[tuple[float, float]]]]:
    """Load glyph for the given character from ttf_path.

    Args:
        ttf_path: Path to TTF/OTF font file (or .ttc collection).
        char: Single character to load (e.g. "A").
        font_number: For .ttc font collections, which font to use (default 0).

    Returns:
        List of (command_name, points) pairs.
        command_name: "moveTo", "lineTo", "qCurveTo", "curveTo", "closePath", etc.
        points: list of (x, y) tuples in font units.

    Raises:
        ValueError: If character is not found in font.
    """
    # Handle TrueType Collections (.ttc files)
    try:
        font = TTFont(ttf_path)
    except Exception as e:
        if "font number" in str(e).lower() or "collection" in str(e).lower():
            font = TTFont(ttf_path, fontNumber=font_number)
        else:
            raise

    # Get cmap table to map character to glyph name
    cmap = font.getBestCmap()
    if cmap is None:
        raise ValueError(f"No cmap table found in font: {ttf_path}")

    codepoint = ord(char)
    if codepoint not in cmap:
        raise ValueError(f"Character '{char}' (U+{codepoint:04X}) not found in font: {ttf_path}")

    glyph_name = cmap[codepoint]

    # Get glyph set and draw to recording pen
    glyph_set = font.getGlyphSet()
    if glyph_name not in glyph_set:
        raise ValueError(f"Glyph '{glyph_name}' not found in glyph set")

    pen = RecordingPen()
    glyph_set[glyph_name].draw(pen)

    # Convert RecordingPen output to our format
    # RecordingPen.value is list of (method_name, args) tuples
    commands: list[tuple[str, list[tuple[float, float]]]] = []
    for method_name, args in pen.value:
        if method_name == "moveTo":
            # args is ((x, y),)
            commands.append(("moveTo", [args[0]]))
        elif method_name == "lineTo":
            # args is ((x, y),)
            commands.append(("lineTo", [args[0]]))
        elif method_name == "qCurveTo":
            # args is ((x1, y1), (x2, y2), ...) - quadratic curve points
            commands.append(("qCurveTo", list(args)))
        elif method_name == "curveTo":
            # args is ((x1, y1), (x2, y2), (x3, y3)) - cubic bezier
            commands.append(("curveTo", list(args)))
        elif method_name == "closePath":
            commands.append(("closePath", []))
        elif method_name == "endPath":
            # Open paths end with endPath instead of closePath
            commands.append(("endPath", []))
        else:
            # Handle any other commands
            commands.append((method_name, list(args) if args else []))

    font.close()
    return commands
