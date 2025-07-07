import torch
from tqdm import tqdm
from .base import BaseOpt


class DefaultOpt(BaseOpt):
    _x = None

    def step(self) -> torch.Tensor:
        # zero grad
        self.gan.optimizerG.zero_grad()
        self.gan.optimizerD.zero_grad()

        elite = torch.stack(self.buffer.B.get_top_k(k=self.components.batch_size)).to(
            self.gan.device
        )

        out_buffer = self.gan.D(elite)
        loss_buffer = self.gan.loss(out_buffer, torch.ones_like(out_buffer))

        # TODO: replace with x = self.gan.latent_sampler()
        x = (
            torch.rand(self.components.batch_size, self.gan.latent_dim).to(
                self.gan.device
            )
            * 2
            - 1
        )
        x = x.to(self.gan.device)

        with torch.no_grad():
            gen_samples = self.gan.G(x)

        out_model = self.gan.D(gen_samples.detach())
        loss_model = self.gan.loss(out_model, torch.zeros_like(out_model))

        loss = loss_buffer + loss_model
        loss.backward()
        self.gan.optimizerD.step()

        # train generator
        # TODO: replace with x = self.gan.latent_sampler()
        x = (
            torch.rand(self.components.batch_size, self.gan.latent_dim).to(
                self.gan.device
            )
            * 2
            - 1
        )
        x = x.to(self.gan.device)
        # x = DefaultOpt._x

        # forward pass
        x = self.gan.G(x)

        # panalize lack of curiousity
        loss_curiosity = self.gan.curiosity(x)

        # get disciminator output
        outG = self.gan.G(x)

        # calculate loss
        lossG = self.gan.loss(outG, torch.ones_like(outG))

        # total loss
        loss = lossG + loss_curiosity

        # backward pass
        loss.backward()

        # update generator
        self.gan.optimizerG.step()

        # function evaluation
        values = self.f.f(x.detach().to(self.f.device, self.f.dtype))

        # add values to buffer
        self.buffer.B.insert_many(list(values), list(x.detach()))

    def optimize(self, n_iter: int, termination_eps: float = None) -> torch.Tensor:
        for _ in tqdm(range(n_iter)):
            self.step()
            if termination_eps is not None:
                if self.buffer.B.values[0] < termination_eps:
                    break
        return self.buffer.get_best_value()
