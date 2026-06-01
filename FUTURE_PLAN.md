# GFog Future Plan

## Current state

We now have a meaningful non-toy topology-optimization direction:

- strict **GFog/Torch ↔ black-box evaluator** separation
- regular-grid **cantilever FEM** benchmark
- black-box design decoding pipeline:
  - generator outputs **raw logits**
  - evaluator applies decode/filter/projection
  - evaluator solves mechanics and returns values only
- corrected FEM objective sign convention so the archive now minimizes actual compliance correctly
- evidence that:
  - **curiosity helps** on the FEM task
  - **batch size matters a lot**
  - **buffer size matters**, but once batch size is large it does not need to be huge
  - mild black-box decoding helps more than aggressive projection

This is already a credible prototype, and is now beyond the tiny-budget toy regime, but it is still not yet fully aligned with colleague-grade benchmark settings.

---

## Main conclusions so far

### 1. The black box should own design decoding
The generator should not be forced to emit valid physical densities directly.

Preferred structure:

```text
z -> G -> raw code/logits -> D
                    \
                     -> black-box decode -> repair/projection -> physics -> values
```

This preserves better gradient flow in the GAN while allowing the evaluator to enforce whatever physical representation it wants.

### 2. Differentiability is not required inside the black box
Because GFog is gradient free with respect to the objective, the evaluator can contain:

- sigmoid or other nonlinear decoding
- density filtering
- Heaviside projection
- hard binarization
- repair heuristics
- connectivity cleanup
- exact/approximate constraint projection
- procedural or learned decoders

The only real constraint is runtime.

### 3. Throughput is a core hyperparameter
The number of fresh candidates generated per iteration matters a lot. Increasing batch size from tiny values to much larger values improved quality substantially.

### 4. Buffer size matters, but scales differently once batch size is large
Small buffers were too restrictive in the small-batch regime. In the larger-batch regime, moderate buffers such as roughly `2x` batch size can already work well.

### 5. The FEM sign bug was important and is now fixed
Earlier FEM experiments used the wrong objective sign relative to the archive semantics. Those results should be treated as obsolete. Current valid FEM conclusions come from the corrected sign-fix runs.

### 6. We are now entering a medium-compute regime
Current best experiments are no longer tiny:

- grid up to `40x20`
- batch size up to `64`
- buffer size up to `512`
- iteration counts up to `500`

There is still headroom from scaling compute further, but we are now in a regime where the benchmark is informative rather than purely toy-scale.

---

## Near-term plan: strengthen the direct FEM benchmark

### A. Increase optimization budget
Move beyond quick runs.

Planned changes:
- more iterations, e.g. `200`, `500`, `1000+`
- larger batch sizes where stable
- moderate-to-large buffers matched to batch size
- larger latent dimension
- larger generator and discriminator MLPs, but only with retuning if needed

Goal:
- determine whether GFog keeps improving with more compute
- understand whether curiosity remains beneficial at scale
- identify convergence, drift, and archive-quality regimes more clearly

### B. Increase problem size
Current grids are still modest.

Planned changes:
- move from `20x10` toward `30x15`, `40x20`, and beyond

Goal:
- get closer to a more serious topology-optimization regime
- test whether GFog remains competitive in higher-dimensional settings

### C. Improve benchmark realism
Align more closely with colleague/reference cantilever setups.

Potential changes:
- exact load placement and support conventions
- exact aspect ratio / domain conventions
- passive solid / passive void zones
- improved normalization/reporting
- refined projection/filter conventions

Goal:
- move from a credible prototype toward a genuinely comparable benchmark family

---

## Medium-term plan: use much more compute

We should explicitly test a higher-compute regime.

### Scaling directions
- larger `G` / `D`
- deeper/wider MLPs
- larger latent spaces
- larger buffers
- more iterations
- larger grids
- possibly different batch sizes / discriminator schedules

### Why this matters
Right now we are mainly testing whether the setup works.
A larger-compute regime is where GFog may better express its strengths in:

- high-dimensional search
- multimodal search
- diversity maintenance
- learning useful proposal distributions

---

## Strategic next direction: black-box decoders / generative priors

The most interesting longer-term direction is to stop thinking of the evaluator as only “physics”.
Instead, treat it as:

```text
black box = decoder + repair + physics + metrics
```

### Core idea
GFog does not have to search directly in raw density space.
It can search over a **structured control space**, with the evaluator decoding that control into a physical design.

General form:

```text
z -> G -> latent/control code c -> black-box decoder(c) -> design -> repair -> FEM -> values
```

### Why this is promising
A decoder can impose useful priors such as:
- smoothness
- connectivity
- minimum feature size
- plausible topology families
- manufacturability
- domain knowledge

This can reduce junk directions in the search space and make exploration more meaningful.

---

## Candidate decoder-based directions

### 1. Learned autoencoder / VAE decoder
Use a pretrained decoder that maps latent codes to density fields or structural designs.
GFog then searches over decoder input space rather than raw pixels.

Motivation:
- lower-dimensional structured search
- learned prior over plausible designs
- natural fit for GFog because end-to-end differentiability is unnecessary

### 2. Coarse-to-fine decoder
Use a smaller latent code to generate a coarse structure, then upsample/filter/project to a full design.

Motivation:
- cheap first step before a full learned prior
- introduces structure without needing a dataset immediately

### 3. Decoder + repair pipeline
Even a mediocre decoder can be useful if the black box repairs outputs before evaluation.

Possible repair stages:
- volume normalization
- connectivity repair
- support-connected filtering
- passive-zone enforcement
- hard binarization

### 4. Hybrid global-local encoding
Search over:
- a global latent code for large-scale structure
- plus a local correction field for refinement

Motivation:
- balance strong prior with flexibility

---

## Why GFog is especially suitable for this

Gradient-based methods usually want the entire pipeline to be smooth and differentiable.
GFog does not.

This means we can use:
- frozen decoders
- discrete decoders
- heuristic repairs
- hard thresholding
- morphology-style cleanup
- learned priors without end-to-end backprop through physics

This may become one of the strongest application stories for GFog.

---

## Suggested execution order

### Phase 1: solidify the current direct FEM benchmark
1. run bigger-buffer FEM benchmarks
2. increase iterations further
3. increase model sizes and latent dim
4. increase grid size
5. align benchmark conventions with colleague/reference setup

### Phase 2: introduce structured decoder benchmarks
1. start with a simple coarse-to-fine decoder
2. compare direct-density GFog vs decoder-based GFog
3. then test pretrained autoencoder/VAE-style decoders

### Phase 3: explore richer black-box pipelines
1. repair-heavy evaluators
2. connectivity-aware cleanup
3. exact/approximate constraint projection
4. discrete or component-based decoders

---

## Key hypothesis to test

### Direct benchmark hypothesis
With enough compute and sufficiently large buffers, GFog should perform much better than early small-budget runs suggested.

### Decoder hypothesis
GFog may become significantly more powerful when it searches over a structured latent/control space rather than raw density space.

That is, the most compelling long-term setup may be:
- **GFog over learned or procedural design priors**
- with a **nondifferentiable black-box evaluator**
- for **high-dimensional constrained structural optimization**

---

## Current corrected best direct FEM configuration

The strongest corrected direct FEM configuration tested so far is approximately:

- `grid = 40x20`
- `n_iter = 500`
- `batch_size = 64`
- `buffer_size = 128` (via `buffer_multiplier = 2`)
- `latent_dim = 64`
- `G/D hidden dims = 128, 128`
- `density_filter_radius = 1`
- `projection_beta = 1`
- `curiosity = 40`

Representative corrected benchmark result:
- median best relative compliance around `8.6`
- high Hamming diversity around `0.45`
- stable multi-seed behavior in the corrected sign-fix regime

## Immediate next actions

1. Continue corrected-sign FEM benchmarks in the high-throughput regime.
2. Scale to larger grids while keeping large batch size.
3. Revisit the encoding/parameterization rather than only scaling raw density logits.
4. Start a first decoder-based topology benchmark prototype.

Potential first decoder prototype:
- latent code
- small decoder network producing coarse density map
- upsample + filter + projection in black box
- FEM evaluation

This is likely the cleanest bridge from the current direct setup toward a richer learned-prior optimization framework.
