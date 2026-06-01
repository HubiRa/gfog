# Symbolic Regression

```bash
python examples/symbolic_regression/symbolic_regression.py
```

GFog emits continuous genomes that are hard-decoded into fixed-depth expression
trees. Internal nodes choose one operator by argmax from:

```text
const, add, sub, mul, sin, cos
```

Leaves are affine terminals:

```text
a * x + b
```

The black-box objective evaluates the decoded expression on sampled target
points and returns MSE plus a small complexity penalty. `G` and `D` are
differentiable, but `f` contains discrete operator choices.

The grammar also supports a hard `const` operator that returns zero and ignores
its children. This gives regularization a way to prune subtrees instead of only
penalizing them.

Example:

```bash
TMPDIR=/private/tmp python examples/symbolic_regression/symbolic_regression.py \
  --optimizer ranked \
  --n_iter 1000 \
  --batch_size 128 \
  --buffer_multiplier 4 \
  --ranker_list_size 64 \
  --ranker_sample_pool_size 128 \
  --depth 3 \
  --target nguyen1
```

Seed-0 results:

```text
nguyen1 = x^3 + x^2 + x

depth=2, 1000 iters:
  n_params = 23
  best_mse = 0.016706

depth=3, 1000 iters:
  n_params = 51
  best_mse = 0.000748

sin_poly = sin(x) + x^2

depth=2, 1000 iters:
  n_params = 23
  best_mse = 0.000629

nguyen4 = x^6 + x^5 + x^4 + x^3 + x^2 + x

depth=3, 2000 iters:
  n_params = 51
  best_mse = 0.037594

depth=4, 2000 iters:
  n_params = 107
  best_mse = 0.001741
```

This task is much more forgiving than the black-box RNN examples. The current
decoder finds accurate fits quickly, but expressions are not simplified and are
not yet optimized for human readability. Good next steps are expression
simplification, train/validation splits for extrapolation, and a grammar with
constants/operators that more closely matches standard symbolic-regression
benchmarks.

## Regularization

Available knobs:

```text
--complexity_weight       penalty for binary ops: add/sub/mul
--unary_weight            penalty for unary ops: sin/cos
--coefficient_l1_weight   L1 penalty on affine leaf coefficients
--roughness_weight        penalty on output second finite differences
--lexicographic_simplicity_weight
                          legacy scalarized tie-breaker for active op count
--ordering lexicographic
                          true multi-level buffer ordering
--mse_bucket_size         bucket size for near-tie MSE lexicographic sorting
```

After adding `const`, the depth-3 genome has 58 parameters. Seed-0 `nguyen1`
regularization sweep at 1000 iterations:

```text
baseline-ish:
  complexity_weight=0.0001, unary_weight=0, coefficient_l1_weight=0.000001
  best_mse = 0.023951
  binary_ops = 7, unary_ops = 0

medium:
  complexity_weight=0.001, unary_weight=0.001, coefficient_l1_weight=0.0001
  best_mse = 0.014691
  binary_ops = 6, unary_ops = 1

strong:
  complexity_weight=0.01, unary_weight=0.01, coefficient_l1_weight=0.001
  best_mse = 0.019657
  binary_ops = 4, unary_ops = 2
```

Regularization can prune structure, but there is a clear tradeoff: the strong
setting gives a shorter expression and uses `const`, but loses accuracy. The
best next improvement is likely post-hoc simplification/constant refitting of
the best decoded tree, not just stronger penalties during search.

Lexicographic sorting:

```text
--ordering lexicographic
```

The buffer supports multi-level values, so this uses true lexicographic ordering
rather than scalarization. The inserted value vector is:

```text
(mse_key, active_ops, coefficient_l1, raw_mse)
```

With `--mse_bucket_size 0`, `mse_key = raw_mse`, so simplicity only breaks exact
MSE ties. With a positive bucket size, simplicity breaks near-ties inside each
MSE bucket.

Seed-0 `nguyen1`, depth 3, 1000 iterations:

```text
exact lexicographic, mse_bucket_size=0:
  best_mse = 0.009955
  active_ops = 7, const_ops = 0

bucketed lexicographic, mse_bucket_size=0.001:
  best_mse = 0.024380
  active_ops = 7, const_ops = 0

bucketed lexicographic, mse_bucket_size=0.01:
  best_mse = 0.027059
  active_ops = 6, const_ops = 1
```

Exact lexicographic sorting is technically correct, but exact MSE ties are too
rare to help simplicity. Bucketed lexicographic sorting is the practical version:
it can prune subtrees by allowing simplicity to decide among similarly accurate
expressions. The tradeoff is explicit: larger buckets simplify more aggressively
but can accept worse raw MSE.

Cascade ordering:

```text
--ordering cascade
```

This uses the official `Levels.ladder(...)` mechanism. The objective returns raw
values:

```text
(mse, active_ops, coefficient_l1)
```

and the buffer expands them into interleaved rung scores:

```text
mse#1, active_ops#1, coefficient_l1#1,
mse#2, active_ops#2, coefficient_l1#2, ...
mse#open
```

Example:

```bash
TMPDIR=/private/tmp python examples/symbolic_regression/symbolic_regression.py \
  --target nguyen1 \
  --depth 3 \
  --ordering cascade \
  --cascade_mse_thresholds 0.1,0.03,0.01,0.003,0.001 \
  --cascade_active_thresholds 7,6,5,4 \
  --cascade_l1_thresholds 1.0,0.75,0.5
```

Seed-0 `nguyen1`, depth 3, 1000 iterations:

```text
loose cascade:
  best_mse = 0.022356
  active_ops = 7, const_ops = 0

strict cascade:
  mse thresholds: 0.05,0.02,0.01,0.005
  active thresholds: 6,5,4,3
  l1 thresholds: 0.8,0.6,0.4
  best_mse = 0.046084
  active_ops = 6, const_ops = 1
```

The strict cascade is the first fully non-scalarized version that clearly uses
the intended mechanism: once it reaches the current MSE rung, it prefers simpler
expressions and prunes a subtree. As expected, this can trade away raw MSE.

Additional cascade sweep:

```text
depth=3, mid cascade, 2000 iters:
  mse thresholds: 0.08,0.04,0.02,0.01,0.005
  active thresholds: 6,5,4
  best_mse = 0.027528
  active_ops = 6, const_ops = 1

depth=3, active5 cascade, 3000 iters:
  mse thresholds: 0.08,0.04,0.02,0.01,0.005
  active thresholds: 5,4,3
  best_mse = 0.019819
  active_ops = 6, const_ops = 1

depth=3, active4 cascade, 3000 iters:
  mse thresholds: 0.1,0.05,0.025,0.0125
  active thresholds: 4,3,2
  best_mse = 0.024885
  active_ops = 6, const_ops = 1

depth=2 cascade, 3000 iters:
  best_mse = 0.023966
  active_ops = 3, const_ops = 0
```

The best cascade Pareto point so far is probably the depth-2 run: MSE is close
to the pruned depth-3 runs, but the expression is much smaller. If raw accuracy
matters more, depth-3 active5 is better.

## Harder Targets

The current harder benchmark targets are:

```text
nguyen4 = x^6 + x^5 + x^4 + x^3 + x^2 + x
mixed_trig_poly = x^4 - 0.5*x^2 + sin(3*x) + 0.5*x*cos(2*x)
```

Seed-0 MLP-G/MLP-D ranked runs, 20k iterations, batch size 128:

```text
nguyen4, depth=4, scalar MSE:
  n_params = 122
  best_mse = 0.003664
  active_ops = 15, const_ops = 0

nguyen4, depth=4, cascade:
  n_params = 122
  best_mse = 0.047664
  active_ops = 9, const_ops = 6

nguyen4, depth=5, scalar MSE:
  n_params = 250
  best_mse = 0.021546
  active_ops = 27, const_ops = 4

nguyen4, depth=5, cascade:
  n_params = 250
  best_mse = 0.017353
  active_ops = 25, const_ops = 6

mixed_trig_poly, depth=4, scalar MSE:
  n_params = 122
  best_mse = 0.021851
  active_ops = 14, const_ops = 1

mixed_trig_poly, depth=4, cascade:
  n_params = 122
  best_mse = 0.013078
  active_ops = 7, const_ops = 8

nguyen4, depth=4, fine cascade:
  mse thresholds: 0.2,0.15,0.1,0.075,0.05,0.035,0.025,0.02,0.015,0.0125,0.01,0.0075,0.005,0.003,0.002,0.001
  active thresholds: 14,12,10,9,8,7,6,5,4
  l1 thresholds: 1.2,1.0,0.85,0.7,0.6,0.5,0.4,0.3,0.2
  best_mse = 0.017750
  active_ops = 11, const_ops = 4

mixed_trig_poly, depth=4, fine cascade:
  same thresholds as above
  best_mse = 0.021862
  active_ops = 8, const_ops = 7
```

Depth alone is not a free win. For `nguyen4`, depth 5 expanded the genome to
250 parameters and performed worse than depth 4 at the same iteration budget.
For the mixed polynomial/trig target, cascade was better than scalar MSE in this
seed and produced a much smaller expression, so the ladder can occasionally help
optimization rather than only simplify after accuracy has already been found.
The fine cascade improved `nguyen4` over the coarse cascade, but hurt the mixed
target; denser rungs are therefore a tuning knob, not a strict improvement.

## Local Coefficient Refit

The symbolic-regression objective can optionally do local search inside `f`:

```bash
--refit_steps 20 --refit_lr 0.05
```

For each candidate genome, hard operator choices are frozen with `argmax`, then
only the affine leaf coefficients are optimized with Adam before scoring. In the
ranked optimizer, the refit genome is inserted into the buffer, so `D` learns
from the actual candidate that received the score.

Seed-0 MLP-G/MLP-D ranked runs with local refit:

```text
nguyen4, depth=4, scalar MSE, 2000 iters:
  refit_steps = 20
  best_mse = 0.004474
  active_ops = 14, const_ops = 1

mixed_trig_poly, depth=4, cascade, 2000 iters:
  refit_steps = 20
  best_mse = 0.003921
  active_ops = 6, const_ops = 9
```

This is the strongest symbolic-regression result so far. `nguyen4` with refit
gets close to the previous 20k scalar no-refit result in only 2k iterations, and
`mixed_trig_poly` beats the previous 20k cascade result by a large margin. The
main cost is runtime: 2k refit iterations took roughly 6 minutes on CPU because
each proposal batch runs inner coefficient optimization.

Longer MLP-G/MLP-D cascade runs:

```text
depth=2 cascade, 20000 iters:
  best_mse = 0.023939
  active_ops = 2, const_ops = 1
  expression = 0 + ((1.222*x + -0.000) * (0.817*x + 1.318))

depth=3 active5 cascade, 20000 iters:
  best_mse = 0.009070
  active_ops = 5, const_ops = 2

depth=3 active5 cascade, seed=1, 50000 iters:
  best_mse = 0.015366
  active_ops = 5, const_ops = 2

depth=3 scalar MSE, 20000 iters:
  best_mse = 0.000358
  active_ops = 7, const_ops = 0

depth=4 cascade, 20000 iters:
  best_mse = 0.023944
  active_ops = 10, const_ops = 5
```

The clean Pareto story remains stable: depth-2 gives the simplest readable
model around MSE `0.024`; depth-3 cascade improves to MSE `0.009` while pruning
two subtrees; scalar MSE gets much lower error but uses the full tree.

Uniformity:

```text
--uniformity_weight
--uniformity_reference buffer
```

By default symbolic regression uses no uniformity (`--uniformity_weight 0`).
With `--uniformity_reference buffer`, Wang-Isola uniformity is applied to the
generated batch plus the current top buffer elites, so generated proposals are
repelled from each other and from elite genomes. This is added to the generator
loss only.

Depth-3 active5 cascade, 20k, seed 0:

```text
uniformity_weight = 0:
  best_mse = 0.009070
  active_ops = 5, const_ops = 2

uniformity_weight = 0.01:
  best_mse = 0.023949
  active_ops = 4, const_ops = 3

uniformity_weight = 0.1:
  best_mse = 0.023947
  active_ops = 4, const_ops = 3

uniformity_weight = 1.0:
  best_mse = 0.031561
  active_ops = 4, const_ops = 3
```

Elite-buffer uniformity does what we wanted structurally: it pushes the generator
away from the elite cluster and finds simpler/different expressions. The cost is
lower raw accuracy. For this task, `0.01` or `0.1` is a reasonable exploration
setting; `1.0` is too strong.

Longer 50k runs confirm the same tradeoff:

```text
uniformity_weight = 0:
  best_mse = 0.009070
  active_ops = 5, const_ops = 2

uniformity_weight = 0.01:
  best_mse = 0.023938
  active_ops = 4, const_ops = 3

uniformity_weight = 0.1:
  best_mse = 0.023947
  active_ops = 4, const_ops = 3
```

Uniformity does not merely slow convergence here; it consistently steers the
search into a simpler quadratic-like basin. That is useful for diversity or a
simpler expression, but worse for best raw MSE.

## Set G/D

The script supports batch-context set models:

```text
--generator_type set
--discriminator_type set
```

These use transformer encoder blocks over the candidate/proposal set. On the
depth-2 cascade `nguyen1` setup with 3000 iterations:

```text
MLP-G + MLP-D baseline:
  best_mse = 0.023966
  active_ops = 3

set-G + set-D, set_dim=64, depth=1:
  best_mse = 0.077167
  active_ops = 3

set-G + set-D, set_dim=128, depth=2:
  best_mse = 0.096462
  active_ops = 3

set-G + MLP-D, set_dim=64, depth=1:
  best_mse = 0.031827
  active_ops = 3

MLP-G + set-D, set_dim=64, depth=1:
  best_mse = 0.072415
  active_ops = 3
```

Set models did not help here. The same pattern as in earlier RNN tests appears:
contextual `D` makes the reward signal worse. `set-G` alone is less damaging,
but still below the simple MLP baseline for this tiny genome.

Longer set-model runs did not close the gap:

```text
depth=2 cascade, set-G + set-D, 20000 iters:
  best_mse = 0.077167
  active_ops = 3

depth=3 active5 cascade, set-G + set-D, 20000 iters:
  best_mse = 0.043042
  active_ops = 5, const_ops = 2

depth=3 active5 cascade, set-G + MLP-D, 20000 iters:
  best_mse = 0.027746
  active_ops = 5, const_ops = 2
```

Compared with MLP-G/MLP-D at the same 20k budget (`0.023939` for depth 2,
`0.009070` for depth 3), set models remain worse. The depth-2 set/set run was
effectively stuck at the same result from 3k to 20k iterations.
