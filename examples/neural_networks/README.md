# Non-Differentiable Neural Network Training

This folder contains GFog examples where the optimized object is a neural
network, but the task objective is black-box or gradient-hostile.

## Quantized Hard-Threshold MLP

```bash
python examples/neural_networks/nondiff_mlp_training.py
```

The generator emits flattened weights for a small classifier. The black-box
objective then:

- decodes the vector into MLP parameters,
- optionally quantizes the weights,
- applies hard thresholding to predictions,
- returns 0/1 classification error plus a small L2 penalty.

There is no useful gradient through this objective. GFog trains its generator
through the discriminator/elite buffer instead.

Short smoke run:

```bash
python examples/neural_networks/nondiff_mlp_training.py \
  --n_iter 100 \
  --batch_size 64 \
  --output_dir results/nondiff_mlp_training_smoke
```

Initial smoke result on seed 0:

```text
best_value    = 0.105478
best_accuracy = 0.8945
```

## Vanilla RNN Delayed XOR

```bash
python examples/neural_networks/nondiff_rnn_delayed_xor.py
```

The generator emits flattened parameters for a vanilla tanh RNN. The black-box
objective then:

- decodes the vector into RNN parameters,
- runs the RNN over delayed-XOR sequences,
- applies a hard threshold to the final output,
- returns 0/1 classification error plus a small L2 penalty.

This avoids BPTT entirely. It is meant to stress the regime where vanilla RNNs
are gradient-hostile because credit assignment spans many recurrent steps.
Use `--quantization_levels 0` for the clean continuous-weight comparison.
Weight quantization is only an optional extra non-differentiable stressor.

Short smoke run:

```bash
python examples/neural_networks/nondiff_rnn_delayed_xor.py \
  --n_iter 500 \
  --batch_size 64 \
  --rnn_hidden_dim 8 \
  --seq_len 20 \
  --n_samples 256 \
  --quantization_levels 0 \
  --output_dir results/nondiff_rnn_delayed_xor_500
```

Initial smoke result on seed 0:

```text
best_value    = 0.394540
best_accuracy = 0.6055
```

This is intentionally much harder than the MLP example. Treat it as a starting
point for optimizer/architecture experiments, not as a solved benchmark.

Gradient-based BPTT baseline:

```bash
python examples/neural_networks/nondiff_rnn_delayed_xor.py \
  --bptt_baseline \
  --bptt_steps 500 \
  --bptt_lr 0.01 \
  --rnn_hidden_dim 8 \
  --seq_len 20 \
  --n_samples 256
```

Seed 0 continuous-weight comparison (`--quantization_levels 0` for GFog):

```text
seq_len=20 ranked GFog 500: best_accuracy = 0.6133
seq_len=20 BPTT 500 steps:  bptt_accuracy = 0.7578

seq_len=50 GFog 500 iters:  best_accuracy = 0.6016  # old quantized smoke run
seq_len=50 BPTT 500 steps:  bptt_accuracy = 0.6680

seq_len=100 ranked GFog 500: best_accuracy = 0.6133
seq_len=100 ranked GFog 500, batch=256/buffer=2: best_accuracy = 0.6406
seq_len=100 BPTT 500 steps: bptt_accuracy = 0.6133
seq_len=100 ranked GFog 1000, batch=256/buffer=2: best_accuracy = 0.6523
seq_len=100 BPTT 1000 steps: bptt_accuracy = 0.5625

seq_len=200 ranked GFog 500: best_accuracy = 0.5859
seq_len=200 BPTT 500 steps: bptt_accuracy = 0.7734
```

BPTT is still stronger on this small setup. It degrades on some longer
sequences (`seq_len=100`) but the result is not monotonic for a single seed
(`seq_len=200` recovers strongly). The GFog path is most relevant when the task
objective itself is non-differentiable, externally evaluated, or otherwise
unavailable to BPTT.

The ranked GFog variant can be enabled with:

```bash
python examples/neural_networks/nondiff_rnn_delayed_xor.py \
  --optimizer ranked \
  --ranker_list_size 64 \
  --ranker_sample_pool_size 128 \
  --ranker_tau 4
```

It uses the ordered elite buffer as a scalar reward model target instead of only
training D as binary real/fake. On seed 0 it slightly beats BPTT at `seq_len=100`
but not at `seq_len=20` or `seq_len=200`.

Larger generated batches improved the ranked run:

```bash
python examples/neural_networks/nondiff_rnn_delayed_xor.py \
  --optimizer ranked \
  --batch_size 256 \
  --buffer_multiplier 2 \
  --ranker_list_size 128 \
  --ranker_sample_pool_size 256 \
  --quantization_levels 0 \
  --seq_len 100
```

This reached `0.6406` accuracy at 500 iterations on seed 0.
At 1000 iterations, the same setting reached `0.6523`; the matching 1000-step
BPTT run reached `0.5625` on the same seed.

Increasing the fixed evaluation set from `n_samples=256` to `1024` did not help
the current ranked GFog setup:

```text
seq_len=100 ranked GFog 500, n_samples=256:  best_accuracy = 0.6406
seq_len=100 ranked GFog 500, n_samples=1024: best_accuracy = 0.5674
seq_len=100 BPTT 500, n_samples=1024:        bptt_accuracy = 0.5967
```

The cleaner score is also a harder generalization target. A better variant may
be to evaluate each candidate on fresh/random minibatches and maintain a small
reevaluation archive for elites, rather than using one larger fixed dataset for
every candidate.

The RNN objective is vectorized over candidate networks and sequence batches:

```text
candidates x sequences x hidden
```

This matters for random scoring. A smoke run with `batch_size=256`,
`eval_batch_size=1024`, `seq_len=100`, and `n_iter=20` completed in about 13s on
CPU after vectorization.

First random-scoring runs did not improve validation accuracy:

```text
seq_len=100 ranked GFog 500, random eval_batch=1024: validation accuracy = 0.5234
seq_len=100 ranked GFog 500, random eval_batch=256:  validation accuracy = 0.5225
seq_len=100 ranked GFog 500, random eval_batch=256 x4 repeats: validation accuracy = 0.5234
```

The noisy training score can look decent, but the best candidates do not
generalize to the fixed validation set. This likely needs elite reevaluation or
averaged scores before insertion into the buffer.

`--score_repeats` averages multiple independent random score batches before
buffer insertion. In the first run, `--eval_batch_size 256 --score_repeats 4`
matched the random-1024 result but did not improve validation. The next
mechanism should be explicit elite validation/refresh rather than only averaging
every proposal score.

Nondifferentiable black-box weight decodes can be selected with `--weight_mode`.
These keep `G` and `D` differentiable, but change `f(g)` by converting the
generated continuous genome into a harder RNN before evaluating 0/1 delayed-XOR
error:

```text
continuous:  weight_scale * tanh(g), optionally quantized
sign:        +/- weight_scale
ternary:     {-weight_scale, 0, +weight_scale}
sparse_topk: keep only the largest-magnitude fraction of continuous weights
```

Seed-0 ranked `seq_len=100`, `batch_size=256`, `buffer_multiplier=2`,
`ranker_list_size=128`, `ranker_sample_pool_size=256`, 500 iterations:

```text
continuous weights:      best_accuracy = 0.6406
sign weights:            best_accuracy = 0.6367
ternary threshold 0.25:  best_accuracy = 0.6133
sparse_topk fraction .25 best_accuracy = 0.6055
```

Sign-binarized weights are surprisingly close to continuous in this first run.
Ternary and sparse-top-k make the black-box landscape much coarser and were
clearly worse at the same budget.

## Binary-Weight Template Digits

```bash
python examples/neural_networks/nondiff_binary_digits.py
```

This is a larger, image-like black-box classifier experiment without external
dataset dependencies. It generates noisy 8x8 digit templates for 10 classes.
The genome is a flattened MLP classifier:

```text
64 pixels -> hidden -> 10 classes
```

The default hidden size is 64, giving 4810 generated classifier parameters.
`f(g)` decodes the continuous genome into classifier weights, optionally
sign-binarizes or ternarizes them, runs hard argmax classification, and returns
0/1 error plus a small L2 term. `G` and `D` remain differentiable.

The script also supports a compressed deep binary architecture:

```text
--task_arch circulant_residual
--task_arch binary_conv
```

This uses dense input/output heads, but the hidden stack is a residual chain of
binary circulant layers. Each hidden layer is represented by one length-`width`
vector and expanded by circular convolution inside `f`, so depth costs `O(width)`
parameters per layer instead of `O(width^2)`.

The binary-conv path uses shared 3x3 sign-binarized convolutional kernels with
global-average pooling. This is the most natural architecture for the 8x8 image
task, but the current evaluator is too slow for sweeps because it uses a generic
per-candidate `unfold + einsum` implementation. A 5-step smoke run passed
(`conv_channels=8`, `conv_depth=3`, `n_params=1338`, `best_accuracy=0.2031`),
but larger/deeper runs were stopped.

The smallest practical conv run completed:

```text
binary conv, channels=4, depth=1, easy data, 500 iters:
  n_params = 90, best_accuracy = 0.2969, runtime ~= 4m13s

dense MLP, H=4, easy data, 500 iters:
  n_params = 310, best_accuracy = 0.3203, runtime ~= 2s
```

So the current conv evaluator is not worth scaling yet: even the tiny conv is
slower and worse than the dense baseline. Before revisiting this path, replace
the evaluator with grouped convolution or a specialized small-kernel
implementation.

Default ranked binary run:

```bash
python examples/neural_networks/nondiff_binary_digits.py \
  --optimizer ranked \
  --n_iter 500 \
  --batch_size 256 \
  --buffer_multiplier 2 \
  --ranker_list_size 128 \
  --ranker_sample_pool_size 256 \
  --task_hidden_dim 64 \
  --n_samples 1024 \
  --noise 0.20 \
  --dropout 0.05 \
  --weight_mode sign
```

Seed-0 results:

```text
sign, 4810 params, noisy data, 500 iters:       best_accuracy = 0.2324
sign, 4810 params, noisy data, 2000 iters:      best_accuracy = 0.2949
continuous, 4810 params, noisy data, 500 iters: best_accuracy = 0.3291

sign, 4810 params, easier data, 1000 iters:      best_accuracy = 0.3340
continuous, 4810 params, easier data, 1000 iters: best_accuracy = 0.3779

sign, 2410 params, easier data, 1000 iters: best_accuracy = 0.3115
sign, 2410 params, noisy data, 1000 iters:  best_accuracy = 0.2910
```

This scales the black-box from tiny/RNN examples into a few-thousand-parameter
10-class image classifier. It is learnable above random chance, but currently
not strong. Binary weights remain close enough to continuous to be interesting,
but the reward landscape is much harsher than the small two-class MLP example.

First circulant-residual runs used the faster small setting
`batch_size=128`, `buffer_multiplier=2`, `ranker_list_size=64`,
`ranker_sample_pool_size=128`, `n_samples=512`:

```text
dense MLP, H=32, easy data, 500 iters:
  n_params = 2410, best_accuracy = 0.3477

circulant residual, width=32, depth=16, easy data, 500 iters:
  n_params = 3434, best_accuracy = 0.3750

circulant residual, width=32, depth=16, noisy data, 500 iters:
  n_params = 3434, best_accuracy = 0.2285

circulant residual, width=32, depth=32, residual_scale=1, easy data, 500 iters:
  n_params = 4458, best_accuracy = 0.3105

circulant residual, width=32, depth=32, residual_scale=4, easy data, 500 iters:
  n_params = 4458, best_accuracy = 0.2910
```

The width-32/depth-16 circulant net beat the matched dense binary baseline on
the easy split, which supports the "cheap deep binary information filter" idea.
Going deeper to 32 layers did not help without more tuning. Larger width-64
circulant runs were stopped because the current FFT-per-layer evaluator is too
slow for the quick experiment loop.

Possible next angle: nondifferentiable memory access. A clean toy version would
make `f` evaluate a model with hard nearest-neighbor or argmax memory reads,
where the generated genome controls keys/values/readout but the environment
uses discrete lookup during scoring.

Recurrent-matrix parameterization:

```bash
python examples/neural_networks/nondiff_rnn_delayed_xor.py \
  --optimizer ranked \
  --seq_len 100 \
  --recurrent_param spectral_radius \
  --recurrent_radius 1.0
```

By default, all decoded weights are simply bounded. The `spectral_radius` option
rescales `W_hh` to a target spectral radius after decoding. First seed-0 ranked
GFog results at `seq_len=100` used the earlier quantized stress setting:

```text
bounded W_hh:          best_accuracy = 0.6172
spectral radius 0.95:  best_accuracy = 0.5938
spectral radius 1.0:   best_accuracy = 0.6133
spectral radius 1.1:   best_accuracy = 0.5938
```

Simple spectral-radius normalization did not improve this run, but the hook is
useful for testing more structured recurrent parameterizations.

Model/representation ablations on the clean ranked `seq_len=100` setup:

```text
baseline G/D 128, H=8, 500 iters:     best_accuracy = 0.6406
bigger G/D 256, latent=64, H=8, 500:  best_accuracy = 0.6406
bigger G/D 256, latent=64, H=16, 500: best_accuracy = 0.6367
fixed orthogonal W_hh, H=16, 500:     best_accuracy = 0.6445
fixed orthogonal W_hh, H=16, 1000:    best_accuracy = 0.6523
baseline H=8, 1000:                   best_accuracy = 0.6523
```

`--recurrent_param fixed_orthogonal` fixes the recurrent core to an orthogonal
matrix and leaves GFog to optimize the rest of the parameter vector. It helps at
500 iterations but ties the baseline at 1000 on seed 0.

## Vanilla RNN Shakespeare

```bash
python examples/neural_networks/rnn_shakespeare.py
```

This moves the RNN experiments from synthetic delayed XOR to a character-level
next-token task. The script accepts `--text_path` for a real corpus and includes
a small Shakespeare excerpt fallback so it runs without downloads. GFog emits
flattened vanilla-RNN parameters; `f` evaluates fixed text windows and returns
either cross entropy (`--score_mode ce`) or hard next-character error
(`--score_mode accuracy`).

Example:

```bash
TMPDIR=/private/tmp python examples/neural_networks/rnn_shakespeare.py \
  --optimizer ranked \
  --n_iter 1000 \
  --batch_size 128 \
  --buffer_multiplier 2 \
  --ranker_list_size 64 \
  --ranker_sample_pool_size 128 \
  --rnn_hidden_dim 16 \
  --seq_len 32 \
  --n_sequences 256 \
  --score_mode accuracy
```

Seed-0 fallback-excerpt results:

```text
vocab_size = 44
uniform random next-char accuracy ~= 0.0227
majority-space next-char accuracy ~= 0.1371

h=16, score_mode=ce, 1000 iters:
  n_params = 1724, best_ce = 3.5036, best_accuracy = 0.0938

h=16, score_mode=accuracy, 1000 iters:
  n_params = 1724, best_ce = 3.7282, best_accuracy = 0.2216

h=32, score_mode=accuracy, 1000 iters:
  n_params = 3916, best_ce = 3.7579, best_accuracy = 0.2230

h=16, score_mode=accuracy, 3000 iters:
  n_params = 1724, best_ce = 3.7184, best_accuracy = 0.2313
```

For this black-box RNN setup, directly ranking by hard next-character accuracy
worked better than ranking by CE if the target metric is accuracy. More hidden
capacity did not help much at 1k iterations; more iterations helped slightly.

Reservoir mode:

```text
--rnn_param reservoir
```

This fixes `W_hh` to an orthogonal reservoir and removes it from the generated
genome. GFog optimizes only input weights, hidden bias, output weights, and
output bias. Results on the same fallback-excerpt setup:

```text
h=16, reservoir_radius=1.0, 1000 iters:
  n_params = 1468, best_accuracy = 0.2042

h=32, reservoir_radius=1.0, 1000 iters:
  n_params = 2892, best_accuracy = 0.1328

h=16, reservoir_radius=0.5, 1000 iters:
  n_params = 1468, best_accuracy = 0.2179

h=32, reservoir_radius=0.5, 1000 iters:
  n_params = 2892, best_accuracy = 0.1752
```

The reservoir parameterization did not beat the full h=16 RNN baseline at 1k
iterations (`0.2216`), but radius mattered a lot. Smaller radius made the fixed
reservoir less unstable, especially for h=32.

Readout-only reservoir mode:

```text
--rnn_param readout
```

This fixes both `W_ih` and `W_hh`; GFog only optimizes `W_out` and `b_out`.
Results with `reservoir_radius=0.5`:

```text
h=32, readout-only, 1000 iters:
  n_params = 1452, best_accuracy = 0.2181

h=64, readout-only, 1000 iters:
  n_params = 2860, best_accuracy = 0.1675
```

The h=32 readout-only reservoir is surprisingly close to the full h=16 1k
baseline (`0.2216`) with fewer trainable/generated parameters. h=64 was worse,
so simply widening the random reservoir is not enough.

Set-based G/D ablations on the same clean ranked `seq_len=100`,
`batch_size=256`, `buffer_multiplier=2`, `ranker_list_size=128`,
`ranker_sample_pool_size=256`, `latent_dim=64` setup:

```bash
python examples/neural_networks/nondiff_rnn_delayed_xor.py \
  --optimizer ranked \
  --n_iter 500 \
  --batch_size 256 \
  --buffer_multiplier 2 \
  --ranker_list_size 128 \
  --ranker_sample_pool_size 256 \
  --generator_type set \
  --discriminator_type set \
  --latent_dim 64 \
  --set_dim 128 \
  --set_depth 2 \
  --set_heads 4 \
  --set_mlp_ratio 2 \
  --rnn_hidden_dim 8 \
  --seq_len 100 \
  --n_samples 256 \
  --quantization_levels 0
```

```text
MLP-G + MLP-D baseline: best_accuracy = 0.6406
set-G + set-D:          best_accuracy = 0.6367
MLP-G + set-D:          best_accuracy = 0.6094
set-G + MLP-D:          best_accuracy = 0.6211
```

The current transformer-set variants do not beat the MLP baseline. The combined
set/set version is close, but set-D alone degrades the ranking signal. If this
direction is revisited, the likely next fixes are smaller set models, residual
MLP scoring heads, or explicit permutation-invariant summary features instead
of contextualizing every candidate score through self-attention.

Cross-attention variants use the ranked elite-buffer subset as context:

- `cross-G`: latent proposal queries attend to elite parameter vectors before
  emitting candidate RNN parameters.
- `cross-D`: candidate parameter vectors attend to elite context before getting
  per-candidate scores.

Same seed-0 setup as above:

```text
cross-G + cross-D: best_accuracy = 0.6289
cross-G + MLP-D:   best_accuracy = 0.6328
MLP-G + cross-D:   best_accuracy = 0.6211
```

This did not improve over the MLP baseline either. The main negative signal is
again contextual D: once the discriminator score depends on the elite set, the
generator seems to get a noisier/more moving reward target. Cross-G alone is
closer, but still below the plain MLP-G.
