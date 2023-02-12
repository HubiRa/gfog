import torch
from optimgan.optimgan import MLP, SimpleOptimGan, RandomOpt
from testfunctions import (
    GoldsteinPriceFunction,
    Rosenbrock2DFunction, 
    BealeFunction, 
    AckleyFunction,
    HolderTableFunction,
    HimmelblauFunction, 
    ThreeHumpCamelFunction,
    EasomFunction,
    CrossInTrayFunction, # results in infs: FIXME
    McCormickFunction,
    BukinFunction
)



LATENT_DIM = 10
BATCH_SIZE = 32
DEVICE = torch.device("mps")
F_DEVICE = torch.device("cpu")

TEST_FUNCTION = BukinFunction()
F_DIM = TEST_FUNCTION.input_dim()

X_RANGE = torch.linspace(TEST_FUNCTION.domain.get_lower()[0].item(), TEST_FUNCTION.domain.get_upper()[0].item(), 100)
Y_RANGE = torch.linspace(TEST_FUNCTION.domain.get_lower()[1].item(), TEST_FUNCTION.domain.get_upper()[1].item(), 100) 


D = MLP(input_dim=F_DIM, output_dim=1, hidden_dims=[32, 32]).to(DEVICE)
G = MLP(input_dim=LATENT_DIM, output_dim=F_DIM, hidden_dims=[32, 32]).to(DEVICE)


optimizer = SimpleOptimGan(
    generator=G,
    discriminator=D,
    device=DEVICE,
    f=TEST_FUNCTION,
    f_input_dim=F_DIM,
    f_device=F_DEVICE,
    buffer_size=1000,
    optimizerG=torch.optim.Adam(lr=0.0001, params=G.parameters()),
    optimizerD=torch.optim.Adam(lr=0.0001, params=D.parameters()),
    batch_size=BATCH_SIZE,
    latent_dim=LATENT_DIM,
    curiosity=1.,
)


random_optimizer = RandomOpt(
    f=TEST_FUNCTION,
    f_input_dim=F_DIM,
    f_device=F_DEVICE,
    device=F_DEVICE,
    buffer_size=1000,
    batch_size=100,
    latent_dim=LATENT_DIM,
)


def main() -> None:
    N_ITER = 1000
    N_LOG = 10

    best_samples = []
    best_values = []

    for i in range(N_ITER):
        optimizer.step()
        best_samples += optimizer.buffer.get_top_k(1)
        best_values.append(optimizer.buffer.values[0])
        if i % N_LOG == 0:
            print(f"Iteration {i}: best value {best_values[-1]}")

    print(f'found minimum at {optimizer.buffer.get_top_k(1)} with value {optimizer.buffer.values[0]}')

    # plot best samples in buffer over time
    xy = torch.stack(best_samples).to('cpu')
    xyz = torch.cat([xy,torch.tensor(best_values).view(-1,1)], dim=-1) 

    TEST_FUNCTION.plot3D(
        x_range=X_RANGE,
        y_range=Y_RANGE,
        sample_path=xyz
    )

    # plot buffer
    sorted_buffer = optimizer.buffer.get_top_k(k=optimizer.buffer.buffer_size)
    xy = torch.stack(sorted_buffer).to('cpu')
    xyz = torch.cat([xy,torch.tensor(optimizer.buffer.values).view(-1,1)], dim=-1)

    TEST_FUNCTION.plot3D(
        x_range=X_RANGE,
        y_range=Y_RANGE,
        sample_path=xyz,
        path_style='k.'
    )


if __name__ == "__main__":
    main()