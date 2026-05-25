# GFog Understanding: Crypto-Like Controls

This directory is intentionally diagnostic. It is not a serious cryptographic
attack attempt and should not be presented as one.

The goal is to understand when GFog can learn useful proposal distributions in
high-dimensional discrete black-box search. Hash-like objectives are useful
negative controls because full SHA-256 should behave like a random oracle, while
deliberately weak objectives can test whether the GFog machinery exploits
structure when structure exists.

## Current Benchmark

Main script:

```bash
python examples/crypto_pow/sha_pow.py
```

Candidate encodings:

- `raw_bytes`: `G` emits continuous byte coordinates; `f` rounds and wraps them
  modulo 256.
- `binhead_bits`: `G` emits continuous bit scores; `f` applies the BinHead
  positive-vector-to-binary projection and packs bits into bytes.

Objectives:

- `sha256`: double-SHA256 over `prefix || candidate`; negative control.
- `toy_arx`: deliberately weak ARX-style mixer; still often too discontinuous.
- `byte_linear`: deliberately structured positive control.

## Important Modeling Notes

- `D` sees the continuous proposal vector, not decoded bytes.
- BinHead decoding happens only inside `f`.
- There is no straight-through estimator between `G` and `D`.
- The default `G`/`D` sizes for this diagnostic are now 256 hidden units and
  256 latent dimensions.
- The arbitrary old `1024 * G(z)` output scale is disabled by default.

## Best Diagnostic Setting So Far

This is the most useful setting for checking whether GFog can exploit a
structured binary objective:

```bash
python examples/crypto_pow/sha_pow.py \
  --candidate_encoding binhead_bits \
  --generator_output_transform softplus_l2 \
  --generator_output_temperature 10 \
  --generator_output_bias_init -2 \
  --curiosity 0.3 \
  --curiosity_reference batch \
  --hash_mode byte_linear \
  --toy_rounds 0 \
  --n_iter 1000 \
  --batch_size 512 \
  --buffer_multiplier 2 \
  --ranker_list_size 128 \
  --ranker_sample_pool_size 256
```

Five-seed result on `byte_linear`:

```text
wins vs random: 5/5
median GFog/random best ratio: 0.460
median unique candidates: about 155k / 513k
```

The same representation did not reliably help `toy_arx r1`, and `sha256`
remained too noisy to interpret as structure.

## Scaling Question

The useful open question is whether larger `G`/`D`, larger batches, and larger
buffers improve the GFog/random ratio on structured objectives while leaving
random-oracle objectives near 1.0.

Suggested scaling grid:

```text
batch_size: 512, 2048, 8192
hidden_dim: 256, 1024
latent_dim: 256, 1024
buffer_multiplier: 2, 4
```

Primary metric:

```text
median(gfog_best / random_best) over seeds
```

Expected behavior:

```text
byte_linear: ratio should improve with scale
toy_arx: likely noisy or worse unless structure survives the decoder
sha256: should converge toward random with enough seeds/evaluations
```

Use `run_understanding_gfog_crypto_sweep.sh` to launch explicit diagnostic
sweeps.
