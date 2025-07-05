import torch
from tqdm import tqdm
from .base import OptimGanBase


class DefaultOptimGan(OptimGanBase):
    _x = None

    def step(self) -> torch.Tensor:
        # zero grad
        self.optimizerG.zero_grad()
        self.optimizerD.zero_grad()

        # train discriminator on old buffer
        # print(f'debug:::device = {self.device}')
        good_samples = torch.stack(
            self.buffer.get_random_batch_from_top_p(p=0.5, batch_size=self.batch_size)
        ).to(self.device)
        out_buffer = self.discriminator(good_samples)
        loss_buffer = self.loss_fn(out_buffer, torch.ones_like(out_buffer))

        if DefaultOptimGan._x is None:
            x = torch.rand(self.batch_size, self.latent_dim).to(self.device) * 2 - 1
            x = x.to(self.device)
            DefaultOptimGan._x = x
        else:
            x = DefaultOptimGan._x

        with torch.no_grad():
            gen_samples = self.generator(x)
        out_model = self.discriminator(gen_samples.detach())
        loss_model = self.loss_fn(out_model, torch.zeros_like(out_model))

        loss = loss_buffer + loss_model
        loss.backward()
        self.optimizerD.step()

        # train generator
        # x = torch.rand(self.batch_size, self.latent_dim).to(self.device) * 2 - 1
        # x = x.to(self.device)
        x = DefaultOptimGan._x

        # forward pass
        x = self.generator(x)

        # panalize lack of curiousity
        loss_curiosity = self.lack_of_curiosity(x)

        # get disciminator output
        outG = self.discriminator(x)

        # calculate loss
        lossG = self.loss_fn(outG, torch.ones_like(outG))

        # logger.debug(f'loggG = {lossG.item()}, loss_curiosity = {loss_curiosity.item()}')

        # total loss
        loss = lossG + self.curiosty * loss_curiosity

        # backward pass
        loss.backward()

        # update generator
        self.optimizerG.step()

        # function evaluation
        values = self.f(x.detach().to(self.f_device))

        # add values to buffer
        self.buffer.insert_many(list(values), list(x.detach()))

    def optimize(self, n_iter: int, termination_eps: float = None) -> torch.Tensor:
        for _ in tqdm(range(n_iter)):
            self.step()
            if termination_eps is not None:
                if self.buffer.values[0] < termination_eps:
                    break
        return self.buffer.get_best_value()
