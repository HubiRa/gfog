# Discrete Sequence Design Plan

This is probably the cleanest external benchmark direction for GFog so far.
The goal is not low-budget sample efficiency. The goal is high-dimensional,
massively batched, gradient-free optimization where `f` is fast enough that we
can scale evaluations like a deep-learning workload.

## External Context

`poli` is a library of discrete objective functions for benchmarking optimizers.
Its interface is close to what we need: inputs are numpy arrays of strings, and
outputs are numpy arrays of floats. It includes toy problems, small-molecule
objectives, and protein objectives. It also provides `poli-baselines`, including
random mutations, LaMBO2, Bounce, ProbRep, CMA-ES, SAASBO, Turbo, and related
baselines.

Relevant docs:

- `poli`: https://machinelearninglifescience.github.io/poli-docs/
- objective list: https://machinelearninglifescience.github.io/poli-docs/using_poli/objective_repository/all_objectives.html
- HDBO benchmark: https://machinelearninglifescience.github.io/hdbo_benchmark/docs/hdbo/introduction/
- HDBO benchmark tasks/results: https://machinelearninglifescience.github.io/hdbo_benchmark/benchmarks/

The HDBO benchmark targets high-dimensional discrete sequence optimization in
chemistry and biology. The documented tasks include PMO over SELFIES
representations, red fluorescent protein stability with RaSP, and Ehrlich
closed-form motif objectives. Their default benchmark setup is small-budget
BO-style evaluation: batch size 1, 10 initialization points, budget 300, and 3
seeds. GFog should deliberately test a different scaling regime.

## Why This Fits GFog

Discrete sequence design has the properties we actually want:

- High-dimensional categorical inputs, e.g. length 128-1024 with alphabet size
  8-32.
- `f` can be non-differentiable and may include parsing, validity checks,
  simulators, docking proxies, motif logic, or pretrained black-box predictors.
- Good solutions should contain reusable motifs and local structure, so a
  learned generator can exploit patterns across elites.
- Evaluation can often be batched, especially for synthetic/motif objectives
  and predictor-based objectives.
- Standard BO becomes awkward in high dimensions; random mutation baselines are
  strong but simple and fair.

The core claim to test:

```text
With many cheap batched evaluations, GFog learns a proposal distribution over
high-dimensional discrete sequences that improves over random mutation,
genetic algorithms, and BO-style sequence optimizers.
```

## Representation

Start with categorical sequences:

```text
sequence length L
alphabet size A
genome shape = L x A logits or one-hot tokens
f input = argmax-token string
D input = one-hot tokens or logits
buffer stores = the representation used to train D, plus the decoded sequence
                that was actually evaluated
```

Important design choice: do not disconnect `D` from the `G` update. If `G`
receives gradients through logits or soft token probabilities, then `D` must be
trained on that same representation, not only on detached hard argmax tokens.
The score still belongs to the decoded hard sequence, but the buffer should keep
enough information to train `D` in the space where `G` is optimized.

Practical first representation:

```text
G(z) -> logits
hard = argmax(logits)
f evaluates hard sequence
buffer stores logits and score
D is trained on logits
G update uses D(logits)
```

There are three viable variants:

- `D(soft)`: train `D` on softmax/logit-derived representations. This keeps
  gradients clean but risks scoring non-discrete mixtures.
- `D(straight_through_hard)`: forward pass is hard one-hot, backward pass uses a
  soft estimator. This aligns `D` with evaluated sequences more closely, but the
  gradient is an estimator.
- `D(logits)`: train `D` directly on raw generator logits while `f` evaluates
  `argmax(logits)`. This keeps the `D/G` path fully differentiable without a
  straight-through estimator, but `D` may learn logit-scale artifacts that do not
  affect the decoded sequence.

The current implementation uses `D(logits)`.

Initial `G` variants:

- MLP-G: latent vector to `L x A` logits. Simple baseline, probably weak for
  long sequences.
- Conv1D-G: latent channels to sequence logits. Better local motif bias.
- Transformer-G: latent tokens to sequence logits. More expressive, but harder
  to tune.
- Edit-G: condition on an elite sequence and emit mutation masks plus replacement
  tokens. This is closest to random mutation/genetic baselines and probably the
  strongest first serious variant.

Initial `D` variants:

- MLP-D on flattened one-hot. Simple baseline.
- Conv1D-D. Good motif detector and cheap.
- Transformer-D. Useful if long-range motif interactions matter.

## Objective Order

### 1. Internal Motif Task

Implement a local synthetic objective before installing more dependencies:

```text
L = 256 or 512
A = 8 or 20
target contains hidden motifs at unknown positions
score = motif matches + pairwise motif interactions - invalidity penalties
```

Why first:

- Fully controlled and very fast.
- Lets us debug representation, batching, buffer semantics, and scaling.
- Easy to make harder by adding epistasis, distractor motifs, variable motif
  positions, and deceptive local optima.

Baselines:

- random search
- random mutation from elites
- simple genetic algorithm with crossover
- cross-entropy method over per-position categorical probabilities

First budgets:

```text
L=256, A=8, batch=4096, buffer=65536, evals=1M
L=512, A=8, batch=4096, buffer=65536, evals=1M
L=256, A=20, batch=4096, buffer=65536, evals=1M
```

Success criterion:

```text
GFog beats random mutation and CEM at equal evaluations across 5 seeds,
and improves with larger batch/hidden size.
```

### 2. poli Ehrlich Functions

Use `poli`'s Ehrlich objective next. The docs describe it as a closed-form
discrete sequence objective maximized when motifs are fulfilled, and it runs out
of the box with no special prerequisites.

Suggested config from docs:

```text
sequence_length = 256
motif_length = 8
n_motifs = 4
quantization = 8
```

GFog setup:

```text
alphabet size = quantization
batch = 4096
buffer = 65536 or 131072
eval budget = 1M, 5M
models = Conv1D-G + Conv1D-D first
```

Baselines:

- poli random mutations
- genetic algorithm / directed evolution
- CEM categorical baseline
- Bounce or ProbRep if practical to run

Metric:

```text
best score vs evaluations
mean/std over 5-10 seeds
wall-clock throughput
```

### 3. PMO SELFIES

This is chemically more meaningful, but dependency and validity handling are
more annoying. Use after the categorical pipeline works.

Start with one cheap objective:

```text
rdkit_qed or rdkit_logp
SELFIES representation
fixed max length with padding/mask
```

Questions to settle before running:

- How to handle invalid strings.
- Whether we optimize fixed-length SELFIES directly or use an existing
  tokenizer/alphabet from poli.
- Whether higher score is better and whether GFog buffer should minimize
  `-score`.

Baselines:

- random mutation
- genetic algorithm
- poli-baselines if installed
- maybe CMA in latent/embedding space only as a sanity check

### 4. Protein Predictor Task

Only after the above:

```text
RaSP red fluorescent protein stability
possibly protein stability/SASA if dependencies are manageable
```

This is attractive because it is high-dimensional and biologically meaningful,
but it may be less batchable and more dependency-heavy than Ehrlich/PMO.

## GFog Experiment Matrix

Keep the first matrix small:

```text
G architecture: Conv1D-G, Edit-G
D architecture: Conv1D-D
batch: 1024, 4096, 16384
buffer: 16x batch
ranker pool: 4x batch or top 8192
optimizer: Muon if available, Adam otherwise
curiosity/uniformity: off first
```

Do not tune all knobs at once. First establish:

```text
does larger batch help?
does Conv1D beat MLP?
does Edit-G beat direct generation?
does GFog beat CEM/random mutation at equal evals?
```

## Evaluation Protocol

Report both:

```text
best score vs evaluations
best score vs wall-clock
```

For maximization objectives, store `value = -score` in the GFog buffer but print
the true score. Every run should save:

```text
best_sequence
best_score
best_value
score_curve
eval_count
wall_clock
seed
model config
```

Minimum acceptable comparison:

```text
5 seeds
equal evaluation budget
same initial alphabet/sequence constraints
random search
elite random mutation
CEM categorical baseline
GFog Conv1D
GFog Edit-G
```

## First Implementation Steps

1. Add `examples/discrete_sequence_design/motif_task.py`. Done.
2. Implement vectorized synthetic motif objective over integer token tensors. Done.
3. Implement sequence adapters:
   `tokens -> one-hot`, `logits -> argmax tokens`, `tokens -> strings`. Partly done.
4. Implement baselines: random search, elite mutation, CEM. Done.
5. Add GFog runner with Conv1D-G/Conv1D-D. Done, plus MLP-G/MLP-D.
6. Run `L=256, A=8, budget=1M, batch=4096`.
7. If GFog beats mutation/CEM, install/test `poli-core[ehrlich]`.
8. Wrap `EhrlichHoloBlackBox`.

## First Local Results

Implemented:

```text
examples/discrete_sequence_design/motif_task.py
```

The first version supports:

- fixed-position motif objective
- anywhere-position motif objective
- raw-logit `G(z)` outputs; `f` evaluates `argmax(logits)`
- MLP-G/MLP-D
- Conv1D-G/Conv1D-D
- MLP Edit-G conditioned on sampled elite logits
- MLP Token-Edit-G conditioned on sampled elite hard tokens
- Transformer Token-Edit-G: attention reads elite tokens and outputs edit gates
  plus replacement-token actions
- random search baseline
- elite mutation baseline
- categorical CEM baseline

Initial seed-0 results:

```text
fixed motifs, L=64, A=8, 4 motifs, 53k evals
  GFog Conv = 1.50
  GFog MLP = 5.54
  Random = 2.26
  Mutation = 10.07
  CEM = 10.07

fixed motifs, L=256, A=8, 8 motifs, 520k evals
  GFog MLP = 2.64
  Random = 2.89
  Mutation = 22.04
  CEM = 22.08

anywhere motifs, L=256, A=8, 8 motifs, 520k evals
  GFog MLP = 5.27
  Random = 5.51
  Mutation = 22.04
  CEM = 13.53

anywhere motifs, L=256, A=8, 8 motifs, 65k evals, Muon/Muon, batch 128
  GFog MLP logits = 6.52
  Random = 5.26
  Mutation = 15.67
  CEM = 7.42

anywhere motifs, L=256, A=8, 8 motifs, 64k evals, Muon/Muon, batch 128,
buffer 2x, tau 4
  GFog direct MLP logits = 6.15
  GFog Edit-MLP scale 2.0 = 5.89
  GFog Edit-MLP scale 0.5 = 5.76
  GFog Edit-MLP scale 0.25 = 5.52
  GFog Token-Edit-MLP mutation_bias -4 = 5.39
  GFog Token-Edit-MLP mutation_bias -2 = 5.15
  GFog Token-Edit-MLP mutation_bias -1 = 5.65
  GFog Transformer-Token-Edit depth 1, hidden 128, mutation_bias -1 = 5.02
  GFog direct MLP logits, no fake loss = 5.14
  GFog Token-Edit-MLP mutation_bias -1, no fake loss = 5.26
  GFog direct MLP logits, mixed rank update = 5.90
  GFog Token-Edit-MLP mutation_bias -1, mixed rank update = 5.89
  Random = 5.26
  Mutation = 17.92-19.91
  CEM = 9.41-13.66
```

The first attempt is not competitive. This is useful: the fixed-position motif
objective is too separable by position, so CEM and elite mutation are exactly
the right baselines and solve it quickly. Direct unconditional GFog is weak for
this representation. MLP-G/MLP-D works better than Conv on small fixed-position
tasks; the first Conv-D accidentally used global average pooling, which erased
absolute positions and was fixed, but Conv remains CPU-heavy.

Edit-G correction attempted:

```text
Edit-G conditioned on elites:
  input = elite one-hot sequence + noise
  output = mutation mask + replacement-token logits
  f evaluates hard edited sequence
  D trains on the same straight-through edited representation
```

The implemented first version uses elite logits rather than one-hot tokens:

```text
proposal = elite_logits + edit_scale * tanh(MLP([elite_logits, z]))
D sees proposal logits
f evaluates argmax(proposal logits)
```

It did not improve the 500-iteration result. Direct MLP-G was still better than
Edit-G, and both were far behind random elite mutation. The likely issue is that
this edit representation is still too indirect: changing argmax tokens requires
moving logits across category boundaries, while mutation directly flips tokens.

More discrete token-edit correction attempted:

```text
input = elite hard one-hot/tokens + noise
G outputs per-position mutation probability and replacement-token logits
proposal for D = differentiable soft edit representation
proposal for f = sampled/argmax hard edit
```

The implemented version:

```text
elite logits -> hard tokens -> one-hot
G([elite_one_hot, z]) -> mutation probability + replacement-token distribution
proposal = (1 - p_mut) * elite_one_hot + p_mut * replacement_probs
D sees proposal soft tokens
f evaluates argmax(proposal)
```

This also did not beat direct MLP-G or CEM at 500 iterations. Current diagnosis:
the objective is too easy for explicit mutation, and the learned editor still
does not get a clean enough credit signal for which positions/tokens should
change. If we continue this benchmark, the next step should be objective design
rather than more optimizer tuning: add deceptive motifs, long-range epistasis,
or validity constraints where uniform random mutation is no longer so strong.

Transformer edit-policy was also tested:

```text
Transformer reads elite one-hot tokens + position embedding + noise context
attention is internal only
heads output per-position edit gate and replacement-token distribution
proposal = gated copy/replacement soft tokens
```

This is the right conceptual shape, but it was worse on the current toy task and
much slower on CPU. Result at `L=256`, batch 128, 500 iters: `5.02`, below
random in that run. Do not sweep Transformer depth/width until the objective is
changed to require learned long-range/nonlocal edits.

The ranked `D` fake loss was also ablated:

```text
default D step:
  D(buffer ranked samples) -> exp(-rank/tau)
  D(G samples) -> 0

no-fake D step:
  D(buffer ranked samples) -> exp(-rank/tau)
```

On this sequence task, removing fake loss hurt. Direct MLP dropped from `6.15`
to `5.14`, and Token-Edit-G dropped to roughly random (`5.26`). This does not
prove fake loss is conceptually necessary, but in the current implementation it
appears to regularize `D` enough to provide a usable gradient to `G`.

Mixed rank update was then added:

```text
1. generate G proposals
2. evaluate G proposals with f
3. take top ranked buffer samples
4. train D on the true-score ranking of buffer samples + evaluated G proposals
5. insert evaluated G proposals into the buffer
6. update G to maximize D(G(z))
```

This avoids the incorrect `D(G)->0` label while still training `D` on the
current support of `G`. It helped compared with no-fake-only: direct MLP reached
`5.90` instead of `5.14`, and Token-Edit-G reached `5.89` instead of `5.26`.
However, it still did not beat the old fake-loss direct MLP result (`6.15`) or
the meaningful baselines. The formulation is cleaner, but the toy objective
remains poorly matched to learned global latent search.

## Risks

- If `f` is too cheap, wall-clock may be dominated by `G/D` training rather than
  evaluation. That is still useful; it tells us where GFog overhead becomes the
  bottleneck.
- If random mutation is very strong, direct unconditional `G(z)` may be weak.
  Edit-G conditioned on elites is the right response.
- If the objective is mostly additive per position, CEM may dominate. We need
  motif interactions and epistasis to justify a neural proposal model.
- If molecule/protein dependencies dominate setup, stay on synthetic/Ehrlich
  until the optimizer story is clear.
