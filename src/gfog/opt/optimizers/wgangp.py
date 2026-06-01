import torch
from torch import autograd

from .common import GANOptMixin


class WGANGPOpt(GANOptMixin):
    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        batch_size = real.size(0)
        alpha_shape = (batch_size,) + (1,) * (real.dim() - 1)
        alpha = torch.rand(alpha_shape, device=real.device, dtype=real.dtype)
        interpolated = alpha * real + (1.0 - alpha) * fake
        interpolated.requires_grad_(True)

        critic_interpolated = self.gan.D(interpolated)
        grad_outputs = torch.ones_like(critic_interpolated)
        gradients = autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.reshape(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        return ((grad_norm - 1.0) ** 2).mean()

    def _train_discriminator_step(self) -> None:
        self.gan.optimizerD.zero_grad()

        elite = self._select_elite_batch()
        out_real = self._critic_output(elite)

        with torch.no_grad():
            fake = self._sample_generator_output()
        out_fake = self._critic_output(fake.detach())

        gp = self._gradient_penalty(elite, fake.detach())
        loss = (
            out_fake.mean()
            - out_real.mean()
            + self.components.gradient_penalty_weight * gp
        )
        loss.backward()
        self.gan.optimizerD.step()

    def propose(self) -> torch.Tensor:
        self.gan.optimizerG.zero_grad()

        for _ in range(self.components.discriminator_steps):
            self._train_discriminator_step()

        x = self._sample_generator_output()
        loss_curiosity = self._curiosity_loss(x)
        loss_g = -self._critic_output(x).mean()
        loss = loss_g + loss_curiosity
        loss.backward()
        self.gan.optimizerG.step()
        return x
