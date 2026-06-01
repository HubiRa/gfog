import torch

from .common import GANOptMixin


class WGANOpt(GANOptMixin):
    def _train_discriminator_step(self) -> None:
        self.gan.optimizerD.zero_grad()

        elite = self._select_elite_batch()
        out_real = self._critic_output(elite)

        with torch.no_grad():
            fake = self._sample_generator_output()
        out_fake = self._critic_output(fake.detach())

        loss = out_fake.mean() - out_real.mean()
        loss.backward()
        self.gan.optimizerD.step()
        self._apply_weight_clipping()

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
