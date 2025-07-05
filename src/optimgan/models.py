import torch
from torch import nn

from typing import List


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: nn.Module = nn.ReLU(),
        output_layernorm=False,
        zero_one_normalization=False,
        spectral_norm=False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.spectral_norm = nn.utils.spectral_norm if spectral_norm else lambda x: x

        self.layers = nn.ModuleList()
        self.layers.append(self.spectral_norm(nn.Linear(input_dim, hidden_dims[0])))
        for i in range(len(hidden_dims) - 1):
            self.layers.append(
                self.spectral_norm(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            )
        self.layers.append(self.spectral_norm(nn.Linear(hidden_dims[-1], output_dim)))
        self.layernorm = nn.LayerNorm(output_dim) if output_layernorm else None
        self.zero_one_normalization = zero_one_normalization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            if layer != self.layers[-1]:
                x = self.activation(x)

        if self.layernorm is not None:
            x = self.layernorm(x)
        elif self.zero_one_normalization:
            min_vals = x.min(dim=1, keepdim=True).values
            max_vals = x.max(dim=1, keepdim=True).values
            x = (x - min_vals) / ((max_vals - min_vals) + 1e-5)
        return x


# DC GAN from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# Generator Code


class DCGenerator(nn.Module):
    def __init__(self, ngf, nz, nc):
        super(DCGenerator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            # nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)


class DCDiscriminator(nn.Module):
    def __init__(self, ndf, nz, nc):
        super(DCDiscriminator, self).__init__()

        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


if __name__ == "__main__":
    # Number of channels in the training images. For color images this is 3
    nc = 1

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    ngf = 64

    # Size of feature maps in discriminator
    ndf = 64

    G = DCGenerator(ngf=ngf, nz=nz, nc=nc)
    D = DCDiscriminator(ndf=ndf, nz=nz, nc=nc)

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1)

    out = G(fixed_noise)
    # print(out.shape)
    y = D(out)
    # print(y.shape)
