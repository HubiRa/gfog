<div align="center">
  <img src="assets/gfog.png" alt="GFog Logo" width="250">
</div>

# GFog - Gradient-Free Optimization via Gradients

**GFog** is a gradient-free optimization library using GANs to find solutions to black-box optimization problems — potentially discovering multiple optima.
It builds upon and improves the method proposed in the paper [A GAN based solver of black-box inverse problems](https://openreview.net/pdf?id=rJeNnm25US) (OptimGan)

## Quick Start

To run the example shown [below](#Example) run the following from the GFog base directory:

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

## Improvements to OptimGan

**GFog** introduces two improvements over **OptimGan**:

1. Curiosity Loss

   **OptimGan** often stalled before reaching a solution. This is counter intuitive — one might expect a GAN to explore and even discover multiple solutions.
   **GFog** adds a curiosity loss that encourages exploration and often leads to discovering multiple solutions.

2. Hierarchically Sorted Buffer

   **GFog** supports multiple objectives using a hierarchically sorted buffer.
   For example, in a constrained optimization problem, each constraint can define a hierarchy level.
   This avoids the need to merge objectives of potentially vastly different magnitudes into a single loss

## Example: [Himmelblau's function](https://en.wikipedia.org/wiki/Himmelblau%27s_function)

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="./assets/example.gif" alt="With Curiosity" width="400" />
        <br>
        <em>With Curiosity Loss</em>
      </td>
      <td align="center">
        <img src="./assets/example_not_curious.gif" alt="Without Curiosity" width="400" />
        <br>
        <em>Without Curiosity Loss</em>
      </td>
    </tr>
  </table>
</div>

<p align="center"><em>Comparison showing how curiosity loss helps discover all four minima but also taking more iterations</em></p>

---

## Code Example

TODO

## How it works

TODO: more detail

- **Generator**: Proposes samples
- **Samples** are evaluated via the function to be optimized
- **Buffer**: Maintains best solutions found so far
- **Discriminator**: discriminates between generated samples and samples in the buffer
- **Curiosity Loss**: Encourages exploration of unexplored regions
