#!/usr/bin/env python3
"""Demo script to verify glyfit infrastructure works correctly."""

import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from glyfit import GlyphEnv, apply_vector, build_topology, load_glyph_commands, rasterize


def find_system_font() -> str:
    """Find a suitable TTF font on the system."""
    # Common font paths on different systems
    font_paths = [
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Times.ttc",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/times.ttf",
    ]

    for path in font_paths:
        if Path(path).exists():
            return path

    raise FileNotFoundError("No suitable font found. Please provide a TTF font path as argument.")


def main():
    """Run demo to verify the glyfit pipeline works."""
    # Parse font path from args or find system font
    if len(sys.argv) > 1:
        ttf_path = sys.argv[1]
    else:
        ttf_path = find_system_font()

    char = "A"
    width, height = 128, 128
    batch_size = 8

    print(f"Using font: {ttf_path}")
    print(f"Character: '{char}'")
    print(f"Resolution: {width}x{height}")
    print()

    # 1) Load glyph and build topology
    print("Step 1: Loading glyph...")
    commands = load_glyph_commands(ttf_path, char)
    print(f"  Loaded {len(commands)} commands")

    topology = build_topology(commands)
    print(f"  Topology: {topology.base_points.shape[0]} control points")
    print(f"  Parameter dimension: {2 * topology.base_points.shape[0]}")
    print()

    # 2) Create target by rasterizing unmodified glyph
    print("Step 2: Creating target image (unmodified glyph)...")
    V_zero = torch.zeros(2 * topology.base_points.shape[0])
    base_commands = apply_vector(topology, V_zero)
    target_np = rasterize(base_commands, width, height)
    print(f"  Target shape: {target_np.shape}")
    print(f"  Target range: [{target_np.min():.3f}, {target_np.max():.3f}]")
    print()

    # 3) Construct environment
    print("Step 3: Creating GlyphEnv...")
    env = GlyphEnv(
        ttf_path=ttf_path,
        char=char,
        target_image_np=target_np,
        width=width,
        height=height,
    )
    print(f"  env.dim = {env.dim}")
    print(f"  env.num_points = {env.num_points}")
    print()

    # 4) Test with zero offsets (should give near-zero loss)
    print("Step 4: Evaluating zero offsets...")
    V_batch_zero = torch.zeros(1, env.dim)
    loss_zero = env.evaluate_batch(V_batch_zero)
    print(f"  Loss with zero offsets: {loss_zero.item():.6f}")
    assert loss_zero.item() < 1e-6, "Zero offset should give near-zero loss"
    print()

    # 5) Sample random candidates and evaluate
    print(f"Step 5: Evaluating {batch_size} random candidates...")
    # Use small perturbations relative to glyph scale
    perturbation_scale = 20.0  # font units
    V_batch = torch.randn(batch_size, env.dim) * perturbation_scale
    losses = env.evaluate_batch(V_batch)

    print(f"  Losses shape: {losses.shape}")
    print(f"  Losses: {losses.tolist()}")
    print(f"  Min loss: {losses.min().item():.6f}")
    print(f"  Max loss: {losses.max().item():.6f}")
    print(f"  Mean loss: {losses.mean().item():.6f}")
    print()

    # Verify output
    assert losses.shape == (batch_size,), f"Expected shape ({batch_size},), got {losses.shape}"
    assert torch.isfinite(losses).all(), "All losses should be finite"
    assert (losses >= 0).all(), "All losses should be non-negative"

    print("All checks passed! Pipeline is working correctly.")

    # Optional: save visualization
    try:
        from PIL import Image

        # Save target
        target_img = Image.fromarray((target_np * 255).astype(np.uint8), mode="L")
        target_img.save("demo_target.png")
        print("\nSaved target image to demo_target.png")

        # Save best and worst candidates
        best_idx = losses.argmin().item()
        worst_idx = losses.argmax().item()

        best_img_np = env.render(V_batch[best_idx])
        worst_img_np = env.render(V_batch[worst_idx])

        Image.fromarray((best_img_np * 255).astype(np.uint8), mode="L").save("demo_best.png")
        Image.fromarray((worst_img_np * 255).astype(np.uint8), mode="L").save("demo_worst.png")
        print(f"Saved best candidate (loss={losses[best_idx]:.4f}) to demo_best.png")
        print(f"Saved worst candidate (loss={losses[worst_idx]:.4f}) to demo_worst.png")
    except ImportError:
        print("\nNote: Install pillow to save visualization images")


if __name__ == "__main__":
    main()
