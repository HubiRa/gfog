import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from tqdm import tqdm
from loguru import logger
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Dict, Any, Callable, Optional

from optimgan.buffer import Buffer

class OptimGan(ABC):
    def __init__(
        self, 
        f: Callable[[mx.array], mx.array],
        f_input_dim: int,
        buffer_size: int,
        generator: nn.Module = None, 
        latent_dim: int = None,
        discriminator: nn.Module = None, 
        optimizerG: optim.Optimizer = None,
        optimizerD: optim.Optimizer = None,
        batch_size: int = None,
        curiosity: float = 1.,
        init_sampler_func: Callable[[int, int], mx.array] = None,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.f = f
        self.f_input_dim = f_input_dim
        self.buffer_size = buffer_size
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.curiosit_loss = nn.losses.cross_entropy
        assert not curiosity < 0
        self.curiosty = curiosity
        self.loss_fn = nn.losses.binary_cross_entropy
        self.buffer = Buffer(buffer_size, f_input_dim, device)
        self.init_buffer()


    @abstractmethod
    def optimize(self, x_intial: mx.array) -> mx.array:
        pass

    def init_sampler(self) -> mx.array:
        # print(f'debbug:::self.batch_size = {self.batch_size}')
        return mx.randn(self.batch_size, self.latent_dim)

    def init_buffer(self) -> None:
        n_iter = self.buffer.buffer_size // self.batch_size
        logger.info(f"Filling buffer of size {self.buffer_size} with {n_iter + 1} iterations")
        for i in range(n_iter + 1):
            logger.info(f"Filling buffer iteration [{i+1}/{n_iter + 1}]")
            if self.generator is not None:
                x = self.init_sampler().to(self.device)
                with mx.no_grad():
                    x = self.generator(x)
                    # print(f'debbug:::x.shape = {x.shape}')
            else:
                x = mx.randn(self.batch_size, self.f_input_dim).to(self.device)
            values = self.f(x.to(self.f_device))
            self.buffer.insert_many(list(values), list(x.detach()))
            # print(f'# debbug: self.buffer.buffer[:15] = {self.buffer.buffer[15]}')
            

    def lack_of_curiosity(self, x: mx.array, beta=10.) -> mx.array:
        bs = x.shape[0]
        buffer = torch.stack(self.buffer.get_random_batch(bs)).to(self.device)
        x = x / x.norm(dim=-1, keepdim=True)
        buffer = buffer / buffer.norm(dim=-1, keepdim=True)
        exploration = (beta * x @ buffer.T).softmax(dim=-1)
        internal_curiosity = (beta * x @ x.T).softmax(dim=-1)
        # exploitation_loss = self.curiosit_loss(exploration, labels)
        exploitation_loss = (exploration - 1./bs).abs().mean()
        internal_curiosity_loss = (internal_curiosity.diag() - 1.).abs().mean()
        return exploitation_loss + internal_curiosity_loss

    @abstractmethod
    def step(self) -> mx.array:
        pass

    def optimize(self, n_iter: int, termination_eps: float = None) -> mx.array:
        pass



class RandomOpt(OptimGan):
    def step(self) -> mx.array:
        x = torch.rand(self.batch_size, self.buffer.buffer_dim)
        # function evaluation
        values = self.f(x.detach().to(self.f_device))
        # add values to buffer
        self.buffer.insert_many(list(values), list(x.detach()))
        


class SimpleOptimGan(OptimGan):
    def step(self) -> mx.array:
        # zero grad
        self.optimizerG.zero_grad()
        self.optimizerD.zero_grad()

        # train discriminator on old buffer
        # print(f'debug:::device = {self.device}')
        good_samples = torch.stack(
            self.buffer.get_random_batch_from_top_p(
                p=0.5, batch_size=self.batch_size
            )
        ).to(self.device)
        out_buffer = self.discriminator(good_samples)
        loss_buffer = self.loss_fn(out_buffer, torch.ones_like(out_buffer))
        
        x = torch.rand(self.batch_size, self.latent_dim).to(self.device)*2 -1
        x = x.to(self.device)
        with torch.no_grad():
            gen_samples = self.generator(x)
        out_model = self.discriminator(gen_samples.detach())
        loss_model = self.loss_fn(out_model, torch.zeros_like(out_model))

        loss = loss_buffer + loss_model
        loss.backward()
        self.optimizerD.step()
        

        # train generator
        x = torch.rand(self.batch_size, self.latent_dim).to(self.device)*2 -1
        x = x.to(self.device)

        # forward pass
        x = self.generator(x)

        # panalize lack of curiousity
        loss_curiosity = self.lack_of_curiosity(x)

        # get disciminator output
        outG = self.discriminator(x)

        # calculate loss
        lossG = self.loss_fn(outG, torch.ones_like(outG))

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
    
    def optimize(self, n_iter: int, termination_eps: float = None) -> mx.array:
        for _ in tqdm(range(n_iter)):
            self.step()
            if termination_eps is not None:
                if self.buffer.values[0] < termination_eps:
                    break
        return self.buffer.get_best_value()
    


class WassersteinOptimGan(OptimGan):
    def step(self) -> mx.array:
        # zero grad
        self.optimizerG.zero_grad()
        self.optimizerD.zero_grad()

        # train discriminator on old buffer
        good_samples = torch.stack(
            self.buffer.get_random_batch_from_top_p(
                p=0.75, batch_size=self.batch_size
            )
        ).to(self.device)
        out_buffer = self.discriminator(good_samples)
        # loss_buffer = self.loss_fn(out_buffer, torch.ones_like(out_buffer))
        
        x = torch.randn(self.batch_size, self.latent_dim).to(self.device)
        x = x.to(self.device)
        with torch.no_grad():
            gen_samples = self.generator(x)
        out_model = self.discriminator(gen_samples.detach())
        # loss_model = self.loss_fn(out_model, torch.zeros_like(out_model))

        loss = out_model - out_buffer
        loss.backward()
        self.optimizerD.step()
        

        # train generator
        x = torch.randn(self.batch_size, self.latent_dim).to(self.device)
        x = x.to(self.device)

        # forward pass
        x = self.generator(x)

        # panalize lack of curiousity
        loss_curiosity = self.lack_of_curiosity(x)

        # get disciminator output
        outG = self.discriminator(x)

        # calculate loss
        lossG = self.loss_fn(outG, torch.ones_like(outG))

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
    
    def optimize(self, n_iter: int, termination_eps: float = None) -> mx.array:
        for _ in tqdm(range(n_iter)):
            self.step()
            if termination_eps is not None:
                if self.buffer.values[0] < termination_eps:
                    break
        return self.buffer.get_best_value()