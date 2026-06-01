import torch
import torch.nn.functional as F

from .common import GANOptMixin


class HingeGANOpt(GANOptMixin):
    def _train_discriminator_step(self) -> None:
        self.gan.optimizerD.zero_grad()

        elite = self._select_elite_batch()
        out_real = self.gan.D(elite)

        with torch.no_grad():
            fake = self._sample_generator_output()
        out_fake = self.gan.D(fake.detach())

        loss_real = F.relu(1.0 - out_real).mean()
        loss_fake = F.relu(1.0 + out_fake).mean()
        loss = loss_real + loss_fake
        loss.backward()
        self.gan.optimizerD.step()

    def propose(self) -> torch.Tensor:
        self.gan.optimizerG.zero_grad()

        for _ in range(self.components.discriminator_steps):
            self._train_discriminator_step()

        x = self._sample_generator_output()
        loss_curiosity = self._curiosity_loss(x)
        loss_g = -self.gan.D(x).mean()
        loss = loss_g + loss_curiosity
        loss.backward()
        self.gan.optimizerG.step()
        return x
