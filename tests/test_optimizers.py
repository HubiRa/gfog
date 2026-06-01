import torch

from gfog.buffer import Buffer
from gfog.models import MLP
from gfog.opt import (
    DefaultOpt,
    HingeGANOpt,
    HybridContextualUtilityRankerOpt,
    LSGANOpt,
    QuantileRankedDefaultOpt,
    WGANOpt,
    WGANGPOpt,
    components,
)
from gfog.opt.latents_sampler import LatentSamplerLambda


def _build_components(
    *,
    weight_clip: float | None = None,
    gradient_penalty_weight: float = 10.0,
) -> components.OptComponents:
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
            lambda b, d: torch.randn(b, d),
            b=batch_size,
            d=latent_dim,
        ),
        device=device,
        dtype=torch.float32,
    )
    return components.OptComponents(
        fn=fn,
        gan=gan,
        batch_size=batch_size,
        buffer=buffer,
        discriminator_steps=2,
        elite_sampling="random_top_k",
        elite_pool_size=8,
        weight_clip=weight_clip,
        gradient_penalty_weight=gradient_penalty_weight,
    )


def _run_smoke_test(
    opt_cls: type,
    *,
    weight_clip: float | None = None,
    gradient_penalty_weight: float = 10.0,
    **opt_kwargs,
) -> None:
    opt = opt_cls(
        _build_components(
            weight_clip=weight_clip,
            gradient_penalty_weight=gradient_penalty_weight,
        ),
        **opt_kwargs,
    )
    params_before = [p.detach().clone() for p in opt.gan.G.parameters()]
    opt.step()

    assert len(opt.buffer.B) > 0
    best = opt.buffer.B.get_top_k(1)
    assert torch.isfinite(best).all()
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(params_before, opt.gan.G.parameters(), strict=False)
    )


def test_default_opt_smoke() -> None:
    _run_smoke_test(DefaultOpt)


def test_hinge_gan_opt_smoke() -> None:
    _run_smoke_test(HingeGANOpt)


def test_lsgan_opt_smoke() -> None:
    _run_smoke_test(LSGANOpt)


def test_wgan_opt_smoke() -> None:
    _run_smoke_test(WGANOpt, weight_clip=0.01)


def test_wgangp_opt_smoke() -> None:
    _run_smoke_test(WGANGPOpt, gradient_penalty_weight=10.0)


def test_quantile_ranked_default_opt_smoke() -> None:
    _run_smoke_test(
        QuantileRankedDefaultOpt,
        ranker_list_size=4,
        ranker_sample_pool_size=8,
        ranker_target_curve="exp",
        ranker_tau=4.0,
    )


def test_hybrid_contextual_utility_ranker_opt_smoke() -> None:
    _run_smoke_test(
        HybridContextualUtilityRankerOpt,
        ranker_list_size=4,
        ranker_sample_pool_size=8,
        d_score_center_weight=0.01,
        d_score_scale_weight=0.01,
    )


def test_wgan_weight_clipping_is_applied() -> None:
    opt = WGANOpt(_build_components(weight_clip=0.01))
    for parameter in opt.gan.D.parameters():
        parameter.data.fill_(0.5)
    opt._apply_weight_clipping()
    assert all(
        torch.all(parameter.detach() <= 0.01) and torch.all(parameter.detach() >= -0.01)
        for parameter in opt.gan.D.parameters()
    )


def test_wgangp_gradient_penalty_is_finite_scalar() -> None:
    opt = WGANGPOpt(_build_components(gradient_penalty_weight=10.0))
    real = torch.randn(4, 2)
    fake = torch.randn(4, 2)
    gp = opt._gradient_penalty(real, fake)
    assert gp.ndim == 0
    assert torch.isfinite(gp)
