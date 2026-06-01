"""
INSTRUCTIONS FOR AGENT (PSEUDOCODE-LEVEL, NOT DETAILED IMPLEMENTATION)

Context:
- User is using `uv` for dependency and environment management.
- The optimizer is a custom gradient-free optimizer that can handle high dimensions.
- The goal is to set up the *infrastructure* for a glyph-fitting experiment:
    - Vector glyph representation
    - High-dimensional parameterization
    - Rasterization
    - A simple metric (MSE for now)
    - A clean PyTorch-based interface for batched evaluation
- Code should be structured cleanly from the beginning (proper package layout, pyproject, dev tooling).

The agent should follow the steps below.
"""


# =============================================================================
# 0. PROJECT SETUP WITH UV
# =============================================================================
# Task:
#   Initialize a proper Python project using `uv`, with a clean structure and
#   reproducible dependencies.
#
# Agent actions:
#
#   1. Initialize the project (if not already done):
#
#       uv init glyfit
#       cd glyfit
#
#   2. Edit `pyproject.toml` to:
#       - Set project name: "glyfit"
#       - Require Python >= 3.11 (or whatever the user uses).
#       - Add runtime dependencies:
#           - torch (CPU-only is fine for now)
#           - fonttools
#           - cairo (pycairo)
#           - numpy
#           - pillow
#       - Add dev dependencies (optional but recommended):
#           - ruff (lint)
#           - pytest (tests)
#           - mypy (optional type checking)
#
#     PSEUDOCODE for pyproject sections:
#
#       [project]
#       name = "glyfit"
#       version = "0.1.0"
#       requires-python = ">=3.11"
#       dependencies = [
#           "torch",
#           "fonttools",
#           "pycairo",
#           "numpy",
#           "pillow",
#       ]
#
#       [project.optional-dependencies]
#       dev = [
#           "ruff",
#           "pytest",
#           "mypy",
#       ]
#
#   3. Install dependencies using uv:
#
#       uv add torch fonttools pycairo numpy pillow
#       uv add --group dev ruff pytest mypy
#
#   4. Ensure `glyfit/` is a package (has `__init__.py`).
#
# Directory structure (goal):
#
#   glyfit/
#     pyproject.toml
#     src/
#       glyfit/
#         __init__.py
#         glyph_loader.py
#         topology.py
#         param_space.py
#         rasterizer.py
#         metric.py
#         env.py
#     tests/
#       test_env.py
#     scripts/
#       demo_env.py
#
# Use `src/` layout (recommended with uv).


# =============================================================================
# 1. SOFTWARE STACK CHOICES (DESIGN, NOT CODE)
# =============================================================================
# The agent should stick to this stack:
#
#   - Numerical / tensors: PyTorch (torch)
#   - Font outlines: fonttools
#   - Rasterization: cairo via pycairo
#   - Images: numpy + pillow (PIL)
#   - Metric: PyTorch-based (start with MSE)
#
# Design principles:
#
#   - Keep the *optimization* logic out of this project.
#     This project should provide:
#         - a high-dimensional parameter space
#         - a callable `evaluate_batch(V_batch)` in PyTorch
#   - Keep rasterization CPU-only for now (Cairo is CPU anyway).
#   - Keep the API simple so it can later be reimplemented in Rust or C++.


# =============================================================================
# 2. GLYPH LOADER (src/glyfit/glyph_loader.py)
# =============================================================================
# Purpose:
#   Load a glyph (e.g. "A") from a TTF/OTF and return a list of path commands.
#
# Pseudocode API:
#
#   def load_glyph_commands(ttf_path: str, char: str) -> list[tuple[str, list[tuple[float, float]]]]:
#       """
#       Load glyph for the given character from ttf_path and return
#       a list of (command_name, points) pairs.
#       command_name: "moveTo", "lineTo", "qCurveTo", "curveTo", "closePath", etc.
#       points: list of (x, y) tuples in font units.
#       """
#       # use fonttools.TTFont + RecordingPen
#       # handle missing glyph with a clear error
#       ...
#
# Responsibilities:
#   - Use `fontTools.ttLib.TTFont`.
#   - Use `fontTools.pens.recordingPen.RecordingPen` to capture commands.
#   - Map the input char to glyph via cmap.
#   - No PyTorch here; just Python and numpy-compatible types.


# =============================================================================
# 3. TOPOLOGY (src/glyfit/topology.py)
# =============================================================================
# Purpose:
#   Represent the glyph outline as:
#     - A list of command names (strings)
#     - A flattened array of control points (base_points)
#     - A mapping from commands to indices in base_points
#
# Pseudocode structures:
#
#   @dataclass
#   class GlyphTopology:
#       commands: list[str]
#       cmd_point_indices: list[list[int]]
#       base_points: np.ndarray  # shape (N, 2), float32
#
#   def build_topology(commands: list[tuple[str, list[tuple[float, float]]]]) -> GlyphTopology:
#       """
#       Convert RecordingPen-like output into a flat topology representation.
#       - commands: list of (cmd_name, point_list)
#       - returns GlyphTopology with:
#           base_points: flattened coordinates
#           cmd_point_indices: which points each command refers to
#       """
#       # iterate commands
#       # accumulate points into a single list
#       # track indices for each command
#       ...
#
# Design reasoning:
#   - This isolates the *topology* (which points belong to which command)
#     from the actual coordinate values.
#   - It makes it easy to define a high-dimensional parameter vector V that
#     just adds offsets to base_points.


# =============================================================================
# 4. PARAMETER SPACE (src/glyfit/param_space.py)
# =============================================================================
# Purpose:
#   Define how a candidate vector V (from the optimizer) deforms the glyph.
#
# Initial simple parameterization:
#   - High dimensional:
#       V has size 2 * N where N = number of points in base_points.
#       Each point k gets (dx_k, dy_k) offsets.
#
# Later extensions (but not required now):
#   - Add global parameters (scale, rotation, shear).
#   - Add structured modes (e.g. shape PCA, semantic modes).
#
# Pseudocode API:
#
#   def apply_vector(topology: GlyphTopology, V: torch.Tensor) -> list[tuple[str, list[tuple[float, float]]]]:
#       """
#       Given:
#         - topology with base_points (N,2)
#         - V of length 2N (dx, dy per point)
#
#       Returns:
#         - new list of (command_name, updated_points) for rasterization.
#       """
#       # reshape V into (N,2)
#       # add to base_points
#       # rebuild per-command points using cmd_point_indices
#       ...
#
# Note:
#   - This is the bridge between PyTorch (V tensor) and Cairo (path commands).
#   - Keep all heavy math in PyTorch where convenient, but final path data
#     can be standard Python tuples/lists for Cairo.


# =============================================================================
# 5. RASTERIZER (src/glyfit/rasterizer.py)
# =============================================================================
# Purpose:
#   Convert a candidate glyph (as command list) into a grayscale raster image.
#
# Pseudocode API:
#
#   def rasterize(commands, width: int = 128, height: int = 128, scale: float = 1.0) -> np.ndarray:
#       """
#       - commands: list of (cmd_name, point_list) using font units, already transformed.
#       - width, height: output resolution
#       - scale: overall scaling factor for mapping font space to pixels
#       Returns:
#         - image: np.ndarray of shape (height, width), float32 in [0,1]
#       """
#       # create cairo ImageSurface(FORMAT_A8)
#       # create cairo.Context
#       # center coordinate system, flip y-axis if needed
#       # iterate over commands:
#       #   - moveTo, lineTo: ctx.move_to / ctx.line_to
#       #   - qCurveTo: approximate quadratic with cubic
#       #   - curveTo: use ctx.curve_to directly
#       #   - closePath: ctx.close_path()
#       # fill the path with white
#       # extract buffer to numpy
#       ...
#
# Notes:
#   - This will run on CPU and be the main performance bottleneck initially.
#   - For now, correctness and simplicity are more important than speed.
#   - Later we can:
#       - reduce resolution during experimentation
#       - parallelize across candidates with multiprocessing
#       - or move rasterization to Rust.


# =============================================================================
# 6. METRIC (src/glyfit/metric.py)
# =============================================================================
# Purpose:
#   Define the measure M(R_i, I) between rendered candidate R_i and target I.
#
# For now:
#   - Use simple MSE (mean squared error) in PyTorch.
#   - This is sufficient for a first test, even if not perceptually ideal.
#
# Pseudocode API:
#
#   def mse_loss(R: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
#       """
#       R, I: tensors of shape (B, 1, H, W) or (1, 1, H, W)
#       Return: scalar loss per sample or a tensor of shape (B,)
#       """
#       # compute elementwise squared difference and mean
#       ...
#
# Future improvements (do NOT implement yet, but document):
#   - Edge-aware metrics:
#       - Apply a Sobel/Laplacian filter to both R and I and compare those.
#   - Multi-scale metrics:
#       - Compare downsampled versions at multiple resolutions.
#   - Perceptual metrics:
#       - SSIM or variants.
#   - Combination:
#       - MSE on raw pixels + MSE on edge maps.
#
# For now, implement only MSE and keep the rest as documented TODO.


# =============================================================================
# 7. ENVIRONMENT WRAPPER (src/glyfit/env.py)
# =============================================================================
# Purpose:
#   Provide a clean PyTorch-based interface for the optimizer:
#
#       env = GlyphEnv(...)
#       losses = env.evaluate_batch(V_batch)
#
#   so that the optimizer only needs to:
#       - propose V_batch (B, D)
#       - call env.evaluate_batch(V_batch)
#
# Pseudocode design:
#
#   class GlyphEnv:
#       def __init__(
#           self,
#           ttf_path: str,
#           char: str,
#           target_image_np: np.ndarray,
#           width: int = 128,
#           height: int = 128,
#           scale: float = 1.0,
#           device: str = "cpu",
#       ):
#           """
#           - Load and store topology for the base glyph.
#           - Convert target_image_np (H, W) to torch tensor (1,1,H,W) on `device`.
#           - Determine dimension D = 2 * number_of_points.
#           """
#           # load commands via glyph_loader.load_glyph_commands
#           # build topology via topology.build_topology
#           # store width, height, scale
#           # convert target_image_np → self.target (torch tensor on device)
#           # compute self.num_points, self.dim
#           ...
#
#       def evaluate_batch(self, V_batch: torch.Tensor) -> torch.Tensor:
#           """
#           - V_batch: (B, D) PyTorch tensor (device can be CPU or GPU)
#           - For each candidate V:
#               - Apply V → new commands (param_space.apply_vector)
#               - Rasterize commands → R_np (rasterizer.rasterize)
#               - Convert R_np → R_t (torch tensor, shape (1,1,H,W), on device)
#               - Compute loss with metric.mse_loss
#           - Return tensor of shape (B,) on `device`.
#           """
#           # ensure V_batch is available on CPU for Cairo
#           # loop over candidates
#           # accumulate losses into a list
#           # stack into a tensor and return
#           ...
#
# Notes:
#   - This class should NOT know anything about the optimizer;
#     it is just a black-box environment that maps V_batch → loss_batch.
#   - Keep the API stable so the optimizer can be swapped easily.


# =============================================================================
# 8. DEMO SCRIPT (scripts/demo_env.py)
# =============================================================================
# Purpose:
#   Provide a minimal sanity check for the infra, without doing real optimization.
#
# Pseudocode:
#
#   def main():
#       # choose a font and letter, e.g.:
#       #   ttf_path = "path/to/PlayfairDisplay-Regular.ttf"
#       #   char = "A"
#
#       # 1) Load base glyph, build topology
#       # 2) Build a "target" image by rasterizing the UNMODIFIED glyph:
#       #    - create zero offsets V_zero (dimension 2*N)
#       #    - apply_vector(topology, V_zero) → base_commands
#       #    - rasterize(base_commands) → target_np
#
#       # 3) Construct env with target_np
#       #    env = GlyphEnv(ttf_path, char, target_np, width=..., height=..., scale=...)
#
#       # 4) Sample a small random batch of candidates:
#       #    V_batch = torch.randn(B, env.dim) * SOME_SCALE
#
#       # 5) Call env.evaluate_batch(V_batch) and print losses
#
#       # Optional: visualize the best/worst candidate with matplotlib for sanity.
#       ...
#
#   if __name__ == "__main__":
#       main()
#
# This script ensures:
#   - The whole pipeline runs without errors.
#   - Losses are finite tensors.
#   - Dimensionality (env.dim) matches expectation.
#
# It is NOT meant to do serious optimization; that’s delegated to the external
# gradient-free optimizer.


# =============================================================================
# 9. DEV TOOLING (OPTIONAL BUT RECOMMENDED)
# =============================================================================
# With uv, the agent can also:
#
#   - Configure ruff:
#       - Add a basic ruff config in pyproject.toml (line length, ignores, etc.).
#
#   - Add minimal tests in tests/test_env.py to check:
#       - env.dim > 0
#       - evaluate_batch on zero and random V does not crash
#       - output shape is (B,)
#
#   - Optionally enable mypy in dev group and add type hints in the core modules.
#
# These are not required for the first experiment, but they keep the project
# clean and future-proof from the beginning.
#
# =============================================================================
# END OF INSTRUCTIONS
# =============================================================================
