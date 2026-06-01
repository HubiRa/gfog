"""Basic tests for GlyphEnv."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from glyfit import GlyphEnv, apply_vector, build_topology, load_glyph_commands, rasterize


def find_system_font() -> str:
    """Find a suitable TTF font on the system for testing."""
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in font_paths:
        if Path(path).exists():
            return path
    pytest.skip("No suitable font found for testing")


@pytest.fixture
def font_path():
    """Fixture providing a system font path."""
    return find_system_font()


@pytest.fixture
def env(font_path):
    """Fixture providing a GlyphEnv instance."""
    # Create target from unmodified glyph
    commands = load_glyph_commands(font_path, "A")
    topology = build_topology(commands)
    V_zero = torch.zeros(2 * topology.base_points.shape[0])
    base_commands = apply_vector(topology, V_zero)
    target_np = rasterize(base_commands, 64, 64)

    return GlyphEnv(
        ttf_path=font_path,
        char="A",
        target_image_np=target_np,
        width=64,
        height=64,
    )


class TestGlyphLoader:
    """Tests for glyph loading functionality."""

    def test_load_glyph_commands(self, font_path):
        """Test that glyph commands can be loaded."""
        commands = load_glyph_commands(font_path, "A")
        assert len(commands) > 0
        assert all(isinstance(cmd, tuple) and len(cmd) == 2 for cmd in commands)

    def test_load_glyph_missing_char(self, font_path):
        """Test error handling for missing character."""
        # Most fonts don't have this obscure Unicode character
        with pytest.raises(ValueError):
            load_glyph_commands(font_path, "\U0001f9ff")  # Some rare emoji


class TestTopology:
    """Tests for topology building."""

    def test_build_topology(self, font_path):
        """Test topology building from commands."""
        commands = load_glyph_commands(font_path, "A")
        topology = build_topology(commands)

        assert len(topology.commands) == len(commands)
        assert topology.base_points.shape[0] > 0
        assert topology.base_points.shape[1] == 2
        assert topology.base_points.dtype == np.float32


class TestParamSpace:
    """Tests for parameter space application."""

    def test_apply_vector_zero(self, font_path):
        """Test that zero vector gives original commands."""
        commands = load_glyph_commands(font_path, "A")
        topology = build_topology(commands)

        V_zero = torch.zeros(2 * topology.base_points.shape[0])
        new_commands = apply_vector(topology, V_zero)

        assert len(new_commands) == len(commands)

    def test_apply_vector_wrong_dim(self, font_path):
        """Test error handling for wrong dimension."""
        commands = load_glyph_commands(font_path, "A")
        topology = build_topology(commands)

        V_wrong = torch.zeros(10)  # Wrong dimension
        with pytest.raises(ValueError):
            apply_vector(topology, V_wrong)


class TestRasterizer:
    """Tests for rasterization."""

    def test_rasterize_output_shape(self, font_path):
        """Test rasterizer output shape and range."""
        commands = load_glyph_commands(font_path, "A")
        topology = build_topology(commands)
        V_zero = torch.zeros(2 * topology.base_points.shape[0])
        new_commands = apply_vector(topology, V_zero)

        image = rasterize(new_commands, width=64, height=64)

        assert image.shape == (64, 64)
        assert image.dtype == np.float32
        assert image.min() >= 0.0
        assert image.max() <= 1.0


class TestGlyphEnv:
    """Tests for GlyphEnv."""

    def test_env_dim_positive(self, env):
        """Test that env.dim > 0."""
        assert env.dim > 0
        assert env.num_points > 0
        assert env.dim == 2 * env.num_points

    def test_evaluate_batch_zero(self, env):
        """Test evaluation with zero offsets."""
        V_batch = torch.zeros(1, env.dim)
        losses = env.evaluate_batch(V_batch)

        assert losses.shape == (1,)
        assert losses.item() < 1e-5  # Should be near zero

    def test_evaluate_batch_random(self, env):
        """Test evaluation with random offsets."""
        batch_size = 4
        V_batch = torch.randn(batch_size, env.dim) * 10.0
        losses = env.evaluate_batch(V_batch)

        assert losses.shape == (batch_size,)
        assert torch.isfinite(losses).all()
        assert (losses >= 0).all()

    def test_evaluate_batch_output_shape(self, env):
        """Test that output shape matches batch size."""
        for batch_size in [1, 5, 10]:
            V_batch = torch.randn(batch_size, env.dim)
            losses = env.evaluate_batch(V_batch)
            assert losses.shape == (batch_size,)

    def test_render(self, env):
        """Test single candidate rendering."""
        V = torch.randn(env.dim) * 5.0
        image = env.render(V)

        assert image.shape == (env.height, env.width)
        assert image.dtype == np.float32
