import torch

from .base import BaseOpt


class GANOptMixin(BaseOpt):
    """Shared helpers for GAN-style optimizers."""

    def _critic_output(self, x: torch.Tensor) -> torch.Tensor:
        return self.gan.D(x).reshape(-1)

    def _select_elite_batch(self) -> torch.Tensor:
        current_len = len(self.buffer.B)
        batch_size = min(self.components.batch_size, current_len)
        pool_size = self.components.elite_pool_size or min(current_len, batch_size * 4)
        pool_size = max(batch_size, min(pool_size, current_len))

        if self.components.elite_sampling == "random_top_k" and pool_size > batch_size:
            elite = self.buffer.B.get_random_batch_from_top_k(
                k=pool_size,
                batch_size=batch_size,
            )
        else:
            elite = self.buffer.B.get_top_k(k=batch_size)
        return elite.to(self.gan.device, self.gan.dtype)

    def _sample_generator_output(self) -> torch.Tensor:
        z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
        return self.gan.G(z)

    def _curiosity_loss(self, x: torch.Tensor) -> torch.Tensor:
        loss_curiosity = torch.zeros((), device=self.gan.device, dtype=self.gan.dtype)
        if self.gan.curiosity_loss is not None:
            loss_curiosity = self.gan.curiosity_loss(x)
        return loss_curiosity

    def _apply_weight_clipping(self) -> None:
        if self.components.weight_clip is None:
            return
        clip_value = self.components.weight_clip
        for parameter in self.gan.D.parameters():
            parameter.data.clamp_(-clip_value, clip_value)

    def evaluate(self, proposals: torch.Tensor) -> None:
        values = self.fn.f(proposals.detach().to(self.fn.device, self.fn.dtype))
        self.buffer.B.insert_many(values=list(values), tensors=list(proposals.detach()))
