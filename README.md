<div align="center">
  <img src="assets/gfog.png" alt="Gfog Logo" width="250">
</div>

# Gfog - Gradient-Free Optimization via Gradients

**Gfog** is a gradient-free optimization library using GANs to find solutions to black-box optimization problems — potentially discovering multiple optima.
It builds upon and improves the method proposed in the paper [A GAN based solver of black-box inverse problems](https://openreview.net/pdf?id=rJeNnm25US) (OptimGan)

![Optimization Animation](./assets/examples.gif)
**Gfog** finding all four minima of the Himmelblau function

---

## Quick Start

To run the example shown in [at the top](#Gfog) run the following from the Gfog base directory:

```python
# create and activate venv
uv venv
source .venv/bin/activate

# Install dependencies
uv sync

# install gfog in editable mode
uv pip install -e .

# Run example
python examples/testfunctions/example.py
```

## Code Example

TODO

## Improvements to OptimGan

**Gfog** introduces two improvements over **OptimGan**:

1. Curiosity Loss

   **OptimGan** often stalled before reaching a solution. This is counter intuitive — one might expect a GAN to explore and even discover multiple solutions.
   **Gfog** adds a curiosity loss that encourages exploration and often leads to discovering multiple solutions.

2. Hierarchically Sorted Buffer

   **Gfog** supports multiple objectives using a hierarchically sorted buffer.
   For example, in a constrained optimization problem, each constraint can define a hierarchy level.
   This avoids the need to merge objectives of potentially vastly different magnitudes into a single loss

## How it works

TODO: more detail

- **Generator**: Proposes samples
- **Samples** are evaluated via the function to be optimized
- **Buffer**: Maintains best solutions found so far
- **Discriminator**: discriminates between generated samples and samples in the buffer
- **Curiosity Loss**: Encourages exploration of unexplored regions
