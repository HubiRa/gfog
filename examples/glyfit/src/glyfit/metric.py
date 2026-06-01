"""Loss metrics for comparing rendered glyphs to targets."""

import numpy as np
import torch
from scipy import ndimage


def mse_loss(R: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean squared error between rendered and target images.

    Args:
        R: Rendered image tensor of shape (B, 1, H, W) or (1, 1, H, W).
        target: Target image tensor of shape (B, 1, H, W) or (1, 1, H, W).

    Returns:
        Tensor of shape (B,) with MSE loss per sample.
    """
    # Ensure same shape via broadcasting
    diff = R - target

    # Compute MSE per sample: mean over (C, H, W) dimensions
    # Shape: (B,)
    return diff.pow(2).mean(dim=(1, 2, 3))


def mse_loss_scalar(R: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute scalar mean squared error (averaged over batch).

    Args:
        R: Rendered image tensor of shape (B, 1, H, W).
        target: Target image tensor of shape (B, 1, H, W).

    Returns:
        Scalar tensor with mean MSE across all samples.
    """
    return mse_loss(R, target).mean()


def compute_sdt(binary_image: np.ndarray) -> np.ndarray:
    """Compute Signed Distance Transform of a binary image.

    Args:
        binary_image: Binary image (H, W), values in {0, 1} or [0, 1].

    Returns:
        SDT array (H, W) where:
        - Positive values = distance to nearest edge (inside shape)
        - Negative values = distance to nearest edge (outside shape)
    """
    # Threshold to binary
    binary = (binary_image > 0.5).astype(np.float32)

    # Distance transform for inside (foreground)
    dist_inside = ndimage.distance_transform_edt(binary)

    # Distance transform for outside (background)
    dist_outside = ndimage.distance_transform_edt(1 - binary)

    # Signed distance: positive inside, negative outside
    sdt = dist_inside - dist_outside

    return sdt.astype(np.float32)


def sdt_loss(R: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss on Signed Distance Transforms.

    This compares the distance fields rather than raw pixels,
    which better captures shape similarity.

    Args:
        R: Rendered image tensor of shape (B, 1, H, W) or (1, 1, H, W).
        target: Target image tensor of shape (B, 1, H, W) or (1, 1, H, W).

    Returns:
        Tensor of shape (B,) with SDT MSE loss per sample.
    """
    batch_size = R.shape[0]
    losses = []

    R_np = R.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    for i in range(batch_size):
        # Get single images (H, W)
        r_img = R_np[i, 0]
        i_img = target_np[i, 0]

        # Compute SDTs
        r_sdt = compute_sdt(r_img)
        i_sdt = compute_sdt(i_img)

        # MSE on SDTs
        loss = np.mean((r_sdt - i_sdt) ** 2)
        losses.append(loss)

    return torch.tensor(losses, dtype=R.dtype, device=R.device)


def smoothness_loss(V: torch.Tensor, n_points: int) -> torch.Tensor:
    """Compute Laplacian smoothness loss on offset vectors.

    Penalizes differences between neighboring points along the outline,
    encouraging smooth deformations.

    Args:
        V: Offset tensor of shape (B, 2*N) or (2*N,).
        n_points: Number of control points N.

    Returns:
        Scalar or (B,) tensor with smoothness loss.
    """
    if V.dim() == 1:
        V = V.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = V.shape[0]

    # Reshape to (B, N, 2) - each point has (dx, dy)
    V_points = V.view(batch_size, n_points, 2)

    # Compute differences to neighbors (circular - outline is closed)
    # diff_prev[i] = V[i] - V[i-1]
    # diff_next[i] = V[i] - V[i+1]
    diff_prev = V_points - torch.roll(V_points, 1, dims=1)
    diff_next = V_points - torch.roll(V_points, -1, dims=1)

    # Laplacian: how much does each point deviate from its neighbors' average
    # laplacian[i] = V[i] - (V[i-1] + V[i+1]) / 2 = (diff_prev + diff_next) / 2
    laplacian = (diff_prev + diff_next) / 2

    # Loss is mean squared Laplacian
    loss = laplacian.pow(2).mean(dim=(1, 2))

    if squeeze_output:
        return loss.squeeze(0)
    return loss


def combined_loss(
    R: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float = 1.0,
    sdt_weight: float = 1.0,
) -> torch.Tensor:
    """Combine MSE and SDT losses.

    Args:
        R: Rendered image tensor of shape (B, 1, H, W).
        target: Target image tensor of shape (B, 1, H, W).
        mse_weight: Weight for pixel MSE loss.
        sdt_weight: Weight for SDT loss.

    Returns:
        Tensor of shape (B,) with combined loss per sample.
    """
    loss = torch.zeros(R.shape[0], dtype=R.dtype, device=R.device)

    if mse_weight > 0:
        loss = loss + mse_weight * mse_loss(R, target)

    if sdt_weight > 0:
        loss = loss + sdt_weight * sdt_loss(R, target)

    return loss
