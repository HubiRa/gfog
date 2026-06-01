"""GlyphEnv: PyTorch-based environment for glyph optimization."""

from typing import Callable, Literal

import numpy as np
import torch

from .glyph_loader import load_glyph_commands
from .metric import combined_loss, mse_loss, sdt_loss, smoothness_loss
from .param_space import apply_vector
from .rasterizer import rasterize
from .topology import GlyphTopology, build_topology

LossType = Literal["mse", "sdt", "combined"]


class GlyphEnv:
    """Environment for evaluating glyph deformations against a target image.

    This class provides a clean interface for gradient-free optimizers:
        env = GlyphEnv(...)
        losses = env.evaluate_batch(V_batch)

    The optimizer proposes V_batch (B, D) and receives loss values.
    """

    def __init__(
        self,
        ttf_path: str,
        char: str,
        target_image_np: np.ndarray,
        width: int = 128,
        height: int = 128,
        scale: float = 1.0,
        device: str = "cpu",
        subdivisions: int = 1,
        loss_type: LossType = "mse",
        mse_weight: float = 1.0,
        sdt_weight: float = 1.0,
        smoothness_weight: float = 0.0,
    ):
        """Initialize the glyph environment.

        Args:
            ttf_path: Path to TTF/OTF font file.
            char: Character to optimize (e.g. "A").
            target_image_np: Target image as numpy array (H, W), float32 in [0, 1].
            width: Rasterization width in pixels.
            height: Rasterization height in pixels.
            scale: Additional scaling factor for rasterization.
            device: PyTorch device ("cpu" or "cuda").
            subdivisions: Number of subdivisions per curve segment.
            loss_type: Loss function type ("mse", "sdt", or "combined").
            mse_weight: Weight for MSE loss (when using "combined").
            sdt_weight: Weight for SDT loss (when using "combined").
            smoothness_weight: Weight for smoothness regularization (0 = disabled).
        """
        # Load glyph and build topology
        commands = load_glyph_commands(ttf_path, char)
        self.topology: GlyphTopology = build_topology(commands, subdivisions=subdivisions)

        # Setup loss function
        self.loss_type = loss_type
        self.smoothness_weight = smoothness_weight

        if loss_type == "mse":
            self.loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = mse_loss
        elif loss_type == "sdt":
            self.loss_fn = sdt_loss
        elif loss_type == "combined":
            self.loss_fn = lambda rendered, target: combined_loss(
                rendered, target, mse_weight, sdt_weight
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Store rendering parameters
        self.width = width
        self.height = height
        self.scale = scale
        self.device = device

        # Compute parameter space dimension
        self.num_points = self.topology.base_points.shape[0]
        self.dim = 2 * self.num_points

        # Convert target image to torch tensor (1, 1, H, W)
        if target_image_np.ndim == 2:
            target_image_np = target_image_np[np.newaxis, np.newaxis, :, :]
        elif target_image_np.ndim == 3:
            target_image_np = target_image_np[np.newaxis, :, :, :]

        self.target = torch.from_numpy(target_image_np).float().to(device)

    def evaluate_batch(self, V_batch: torch.Tensor) -> torch.Tensor:
        """Evaluate a batch of candidate deformation vectors.

        Args:
            V_batch: Tensor of shape (B, D) where D = 2 * num_points.

        Returns:
            Tensor of shape (B,) with loss for each candidate.
        """
        batch_size = V_batch.shape[0]
        losses = []

        # Move to CPU for Cairo rasterization
        V_batch_cpu = V_batch.detach().cpu()

        for i in range(batch_size):
            V = V_batch_cpu[i]

            # Apply deformation to get new commands
            commands = apply_vector(self.topology, V)

            # Rasterize to image
            R_np = rasterize(commands, self.width, self.height, self.scale)

            # Convert to tensor (1, 1, H, W)
            R_t = torch.from_numpy(R_np[np.newaxis, np.newaxis, :, :]).float()
            R_t = R_t.to(self.device)

            # Compute image loss
            loss = self.loss_fn(R_t, self.target)
            losses.append(loss)

        # Stack losses into (B,) tensor
        image_loss = torch.cat(losses, dim=0).to(self.device)

        # Add smoothness regularization if enabled
        if self.smoothness_weight > 0:
            smooth_loss = smoothness_loss(V_batch, self.num_points)
            image_loss = image_loss + self.smoothness_weight * smooth_loss

        return image_loss

    def render(self, V: torch.Tensor) -> np.ndarray:
        """Render a single candidate for visualization.

        Args:
            V: Deformation vector of shape (D,) or (1, D).

        Returns:
            Rendered image as numpy array (H, W), float32 in [0, 1].
        """
        if V.dim() == 2:
            V = V.squeeze(0)

        V_cpu = V.detach().cpu()
        commands = apply_vector(self.topology, V_cpu)
        return rasterize(commands, self.width, self.height, self.scale)

    def get_target_np(self) -> np.ndarray:
        """Get target image as numpy array for visualization."""
        return self.target.squeeze().cpu().numpy()
