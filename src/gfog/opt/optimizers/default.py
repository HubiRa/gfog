import torch
from .base import BaseOpt


class DefaultOpt(BaseOpt):
    def propose(self) -> torch.Tensor:
        # zero grad
        self.gan.optimizerG.zero_grad()
        self.gan.optimizerD.zero_grad()

        elite = self.buffer.B.get_top_k(k=self.components.batch_size).to(
            self.gan.device
        )

        out_buffer = self.gan.D(elite)
        loss_buffer = self.gan.loss(out_buffer, torch.ones_like(out_buffer))

        x = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)

        with torch.no_grad():
            gen_samples = self.gan.G(x)

        out_model = self.gan.D(gen_samples.detach())
        loss_model = self.gan.loss(out_model, torch.zeros_like(out_model))

        loss = loss_buffer + loss_model
        loss.backward()
        self.gan.optimizerD.step()

        # train generator
        x = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)

        # forward pass
        x = self.gan.G(x)

        # panalize lack of curiousity
        loss_curiosity = 0.0
        if self.gan.curiosity_loss:
            loss_curiosity = self.gan.curiosity_loss(x)

        # get disciminator output
        outG = self.gan.D(x)

        # calculate loss
        lossG = self.gan.loss(outG, torch.ones_like(outG))

        # total loss
        loss = lossG + loss_curiosity

        # backward pass
        loss.backward()

        # update generator
        self.gan.optimizerG.step()

        return x

    def evaluate(self, proposals: torch.Tensor) -> None:
        # function evaluation
        values = self.fn.f(proposals.detach().to(self.fn.device, self.fn.dtype))

        # add values to buffer
        self.buffer.B.insert_many(values=list(values), tensors=list(proposals.detach()))
