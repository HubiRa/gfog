# Topology Optimization Examples

This folder contains experimental topology-optimization-style examples for GFog.

## Current validated FEM setting

After fixing the FEM objective sign convention, the current most promising direct FEM setting is roughly:

- `grid_width=40`
- `grid_height=20`
- `batch_size=64`
- `buffer_multiplier=2` (buffer size `128`)
- `latent_dim=64`
- `density_filter_radius=1`
- `projection_beta=1`
- `curiosity=40`
- `n_iter=500`

This is currently the best corrected large-scale direct-density benchmark regime tested in this repo.

## Isolated JAX environment

The structural cantilever example supports an optional JAX backend. To avoid polluting the main GFog environment, create a dedicated virtual environment inside this folder.

From the repository root:

```bash
python -m venv examples/topology_optimization/.venv
source examples/topology_optimization/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r examples/topology_optimization/requirements-jax.txt
```

If you want to run GFog examples inside this isolated environment, also build and install the Rust buffer extension into it:

```bash
cd src/gfog/buffer/buffer_core
/Users/hubi/Work/gfog/examples/topology_optimization/.venv/bin/python -m pip install maturin
/Users/hubi/Work/gfog/examples/topology_optimization/.venv/bin/python -m maturin build --release -i /Users/hubi/Work/gfog/examples/topology_optimization/.venv/bin/python
/Users/hubi/Work/gfog/examples/topology_optimization/.venv/bin/python -m pip install --force-reinstall target/wheels/buffer_core-*.whl
cd /Users/hubi/Work/gfog
```

This keeps JAX local to the example environment.

If `pip` is missing in the venv on your system, bootstrap it with:

```bash
python -m ensurepip --upgrade
```

## Running the structural cantilever example

### NumPy backend

```bash
source examples/topology_optimization/.venv/bin/activate
python examples/topology_optimization/cantilever_structural.py \
  --backend numpy \
  --grid_width 32 \
  --grid_height 16 \
  --n_iter 120 \
  --batch_size 12 \
  --curiosity 20
```

### JAX backend

```bash
source examples/topology_optimization/.venv/bin/activate
python examples/topology_optimization/cantilever_structural.py \
  --backend jax \
  --backend_device cpu \
  --jit \
  --grid_width 32 \
  --grid_height 16 \
  --n_iter 120 \
  --batch_size 12 \
  --curiosity 20
```

If you have a working JAX GPU installation, you can try:

```bash
python examples/topology_optimization/cantilever_structural.py \
  --backend jax \
  --backend_device gpu \
  --jit
```

## Why the split backend architecture?

GFog only needs objective values from the structural evaluator. That means the optimizer and the structural mechanics backend can stay strictly separated and communicate only via arrays:

- GFog / generator / discriminator: Torch
- structural evaluator: NumPy or JAX
- bridge: detached arrays

This makes it possible to test CPU and GPU structural backends without changing GFog internals.

## Comparing saved runs

The structural example saves raw `.npz` artifacts. You can compare two runs side by side:

```bash
python examples/topology_optimization/compare_cantilever_runs.py \
  results/structural_cantilever/run_a/top_designs_backend_numpy_curiosity_0_seed_0.npz \
  results/structural_cantilever/run_b/top_designs_backend_numpy_curiosity_20_seed_0.npz \
  --output results/structural_cantilever/comparison.png
```

## Benchmark script

You can benchmark the structural cantilever example over seeds and curiosity values:

```bash
python examples/topology_optimization/benchmark_cantilever_structural.py \
  --backend numpy \
  --curiosity_values 0 20 100 200 \
  --seeds 0 1 2
```

## FEM cantilever example

The FEM example is the more realistic current direction. It uses:

- raw generator logits
- black-box density decoding
- density filter
- optional projection
- sparse 2D linear-elasticity FEM

Single run example:

```bash
python examples/topology_optimization/cantilever_fem.py \
  --grid_width 40 \
  --grid_height 20 \
  --n_iter 500 \
  --batch_size 64 \
  --buffer_multiplier 2 \
  --latent_dim 64 \
  --density_filter_radius 1 \
  --projection_beta 1 \
  --curiosity 40 \
  --seed 0 \
  --output_dir results/fem_cantilever_batch64_buf128_iter500_seed0_signfix
```

Benchmark example:

```bash
python examples/topology_optimization/benchmark_cantilever_fem.py \
  --grid_width 40 \
  --grid_height 20 \
  --n_iter 500 \
  --batch_size 64 \
  --buffer_multiplier 2 \
  --latent_dim 64 \
  --density_filter_radius 1 \
  --projection_beta 1 \
  --curiosity_values 40 \
  --seeds 0 1 2 \
  --output_dir results/fem_cantilever_batch64_buf128_iter500_benchmark_signfix
```

Note: older pre-sign-fix FEM artifacts were removed because they used the wrong compliance sign relative to the archive semantics and are not meaningful.

Or try the JAX backend in the isolated environment:

```bash
examples/topology_optimization/.venv/bin/python \
  examples/topology_optimization/benchmark_cantilever_structural.py \
  --backend jax \
  --backend_device cpu \
  --jit \
  --curiosity_values 0 20 \
  --seeds 0 1
```
