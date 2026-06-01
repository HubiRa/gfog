# Optimizer & Extensibility Plan

## Goal

Make GFog extensible in two complementary ways:

1. **Built-in optimizer variants** for different GAN training rules
2. **Custom external optimizers** via a small public abstraction layer

---

## Phase 1 — Public optimizer extension API

### 1. Expose `BaseOpt` as public API

**Change**
- Export `BaseOpt` from `src/gfog/opt/__init__.py`

**Check**
- `from gfog.opt import BaseOpt` works

---

### 2. Add an `OptimizerProtocol`

**Change**
Create:
- `src/gfog/opt/protocols.py`

With a minimal protocol for:
- `step() -> None`
- `optimize(...) -> torch.Tensor`

Optional later:
- `propose()`
- `evaluate(...)`

**Check**
- An external class can satisfy the protocol without subclassing `BaseOpt`
- Static typing with mypy/pyright is possible

---

### 3. Document `BaseOpt` as the primary subclassing point

**Change**
Add docstrings and README/docs section describing:
- when to use built-in optimizers
- when to subclass `BaseOpt`
- when a protocol-only external optimizer is enough

**Check**
- README contains a “Custom optimizers” section
- Includes a minimal subclass example

---

## Phase 2 — Add built-in GAN optimizer variants

### 4. Add `HingeGANOpt`

**Change**
Create:
- `src/gfog/opt/optimizers/hinge.py`

Losses:
- `D`: `relu(1 - D(real)) + relu(1 + D(fake))`
- `G`: `-D(fake).mean()`

**Check**
- `from gfog.opt import HingeGANOpt` works
- A smoke-test optimizer step runs with existing `MLP` models
- A test verifies finite loss and buffer updates

---

### 5. Add `LSGANOpt`

**Change**
Create:
- `src/gfog/opt/optimizers/lsgan.py`

Losses:
- `D`: least-squares real/fake loss
- `G`: least-squares toward real target

**Check**
- Import works
- One optimization step runs
- Test passes with a simple toy function

---

### 6. Add `WGANOpt`

**Change**
Create:
- `src/gfog/opt/optimizers/wgan.py`

Behavior:
- Critic objective instead of BCE
- Weight clipping after critic step

Need config for:
- `weight_clip`
- likely `discriminator_steps`

**Check**
- Import works
- Parameters get clipped after D step
- Step runs without requiring BCE loss module semantics

---

### 7. Add `WGANGPOpt`

**Change**
Create:
- `src/gfog/opt/optimizers/wgangp.py`

Behavior:
- Critic loss + gradient penalty
- Interpolation between real/fake samples
- Configurable `gradient_penalty_weight`

**Check**
- Import works
- Gradient penalty computes without error
- Step runs and backprop works
- Test verifies finite gradients

---

## Phase 3 — Config cleanup for optimizer-specific behavior

### 8. Add optimizer config dataclasses

**Change**
Either:
- `src/gfog/opt/configs.py`

or per-optimizer config dataclasses inside the optimizer files.

Examples:
- `HingeGANOptConfig`
- `LSGANOptConfig`
- `WGANOptConfig`
- `WGANGPOptConfig`

**Check**
- Optimizer-specific constants are explicit
- No hidden magic numbers in optimizer code

---

### 9. Keep backward compatibility for `DefaultOpt`

**Change**
- `DefaultOpt` remains the vanilla GAN optimizer
- Existing examples continue to run

**Check**
- `example_himmelblau.py` still runs unchanged or with only minimal explicit config
- Current tests still pass

---

## Phase 4 — External/custom optimizer pathway

### 10. Add minimal custom optimizer example outside core models

**Change**
Create an example showing:
- custom `G` and `D` defined outside `src/gfog/models`
- custom optimizer subclassing `BaseOpt`

Suggested path:
- `examples/custom_optimizers/example_custom_opt.py`

**Check**
- Example imports `BaseOpt` and components only
- Custom models live in the example folder, not `src/gfog/models`
- Example runs successfully

---

### 11. Add protocol-only example

**Change**
Create an example where the optimizer:
- does **not** subclass `BaseOpt`
- only satisfies `OptimizerProtocol`

**Check**
- Example demonstrates out-of-tree integration
- Interface is clear enough for users to copy

---

## Phase 5 — Tests

### 12. Add tests per optimizer

**Change**
Create:
- `tests/test_optimizers.py`

For each optimizer:
- instantiate tiny `G`, `D`
- run `step()`
- assert:
  - no error
  - buffer length stays valid
  - outputs are finite
  - parameters receive gradients / update

**Check**
- All optimizers have at least one smoke test

---

### 13. Add WGAN-specific tests

**Change**
Tests for:
- weight clipping is actually applied
- no BCE-style labels are required
- critic output can be unrestricted real values

**Check**
- Explicit assertions on clipped parameter ranges

---

### 14. Add WGAN-GP-specific tests

**Change**
Tests for:
- gradient penalty is finite
- works on CPU
- handles batch shapes properly

**Check**
- GP term is scalar and finite

---

### 15. Add custom optimizer subclass test

**Change**
Add a tiny fake optimizer subclassing `BaseOpt`

**Check**
- Proves the extension API works

---

## Phase 6 — Docs

### 16. Update README optimizer section

**Change**
Document available built-in optimizers:
- `DefaultOpt`
- `HingeGANOpt`
- `LSGANOpt`
- `WGANOpt`
- `WGANGPOpt`

**Check**
- README contains usage snippet for at least one non-default optimizer

---

### 17. Add “Extending GFog” doc

**Change**
Create:
- `docs/extending.md`

Cover:
- custom models outside `src`
- subclassing `BaseOpt`
- protocol-only optimizers
- expected tensor contracts

**Check**
- Doc exists
- Includes minimal examples

---

## Acceptance criteria

We are done when all of these are true:

### API
- [x] `BaseOpt` is public
- [x] `OptimizerProtocol` exists
- [x] custom external optimizers are documented

### Built-in optimizers
- [x] `DefaultOpt` works
- [x] `HingeGANOpt` exists and runs
- [x] `LSGANOpt` exists and runs
- [x] `WGANOpt` exists and runs
- [x] `WGANGPOpt` exists and runs

### Extensibility
- [ ] custom models outside `src/gfog/models` work
- [x] subclassing `BaseOpt` works
- [ ] protocol-only optimizer example exists

### Quality
- [ ] tests pass with `pytest -q`
- [ ] examples run
- [x] README reflects current architecture

---

## Suggested implementation order

1. public `BaseOpt` + `OptimizerProtocol`
2. `HingeGANOpt`
3. `LSGANOpt`
4. `WGANOpt`
5. `WGANGPOpt`
6. tests
7. docs/examples
