from dataclasses import dataclass

import torch

from gfog.buffer import Buffer
from gfog.models import MLP
from gfog.opt import BaseOpt, OptimizerProtocol, components
from gfog.opt.latents_sampler import LatentSamplerLambda


class _ToyOpt(BaseOpt):
    def propose(self) -> torch.Tensor:
        z = self.gan.latent_sampler().to(self.gan.device, self.gan.dtype)
        return self.gan.G(z)

    def evaluate(self, proposals: torch.Tensor) -> None:
        values = self.fn.f(proposals.detach().to(self.fn.device, self.fn.dtype))
        self.buffer.B.insert_many(tensors=list(proposals.detach()), values=list(values))


@dataclass
class _ProtocolOnlyOpt:
    steps: int = 0

    def step(self) -> None:
        self.steps += 1

    def optimize(
        self,
        n_iter: int,
        termination_eps: float | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del termination_eps, verbose, kwargs
        for _ in range(n_iter):
            self.step()
        return torch.tensor([self.steps], dtype=torch.float32)


def _build_components() -> components.OptComponents:
    batch_size = 4
    latent_dim = 3
    input_dim = 2
    device = torch.device("cpu")

    fn = components.Fn(
        f=lambda x: (x**2).sum(dim=-1),
        input_dim=input_dim,
        device=device,
        dtype=torch.float32,
    )
    buffer = components.BufferComp(B=Buffer(buffer_size=8))
    g = MLP(input_dim=latent_dim, output_dim=input_dim, hidden_dims=[8]).to(device)
    d = MLP(input_dim=input_dim, output_dim=1, hidden_dims=[8]).to(device)
    gan = components.GAN(
        G=g,
        D=d,
        loss=torch.nn.BCEWithLogitsLoss(),
        curiosity_loss=None,
        latent_dim=latent_dim,
        optimizerG=torch.optim.Adam(g.parameters(), lr=1e-2),
        optimizerD=torch.optim.Adam(d.parameters(), lr=1e-2),
        latent_sampler=LatentSamplerLambda(
            lambda b, d: torch.randn(b, d), b=batch_size, d=latent_dim
        ),
        device=device,
        dtype=torch.float32,
    )
    return components.OptComponents(
        fn=fn, gan=gan, batch_size=batch_size, buffer=buffer
    )


def test_baseopt_is_public_subclass_extension_point() -> None:
    opt = _ToyOpt(_build_components())
    opt.step()
    assert len(opt.buffer.B) > 0


def test_optimizer_protocol_runtime_checkable() -> None:
    opt = _ProtocolOnlyOpt()
    assert isinstance(opt, OptimizerProtocol)
    assert torch.equal(opt.optimize(3), torch.tensor([3.0]))
