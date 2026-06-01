# FIX.md — GFog Issues & Improvements

## Critical Bugs

- [x] **Operator precedence bug in `base.py:75`** — fixed by separating retrieval of `print_table_every_n_steps` from the `> 0` check.
- [x] **NaN panic in Rust buffer** — fixed by replacing `partial_cmp(...).unwrap()` with explicit total ordering that treats NaNs as worst.
- [x] **Dead attribute `next_free_slot`** — removed from `Buffer.clear()`.

## Math / Algorithmic Issues

- [x] **CLIP-style cross-similarity loss assumes false 1-to-1 pairing** — replaced with a set-level repulsion loss instead of index-based pairing.
- [x] **Self-similarity diagonal is trivial** — fixed by masking the diagonal in `self_similarity_loss` and `self_siglip`.
- [x] **Elite selection lacks stochasticity** — addressed by adding configurable stochastic elite sampling (`random_top_k`) and pool size controls.
- [x] **D/G training ratio is fixed at 1:1** — addressed by adding configurable `discriminator_steps`.
- [x] **Scheduler `step()` doesn't clamp beyond `total_steps`** — fixed by clamping the effective step before evaluating the schedule.
- [x] **`warmup_cosine_annealing` division by zero** — fixed by guarding degenerate cycle lengths and clamping schedule progress.

## Code Quality

- [x] **No tests** — added source tests covering buffer behavior, ladder transforms, curiosity losses, scheduler edge cases, and Rust-backed buffer behavior via the Python wrapper.
- [x] **Typo: `"panalize lack of curiousity"`** — fixed.
- [x] **Typo: `"verbous"`** — fixed to `verbose` while preserving backward compatibility for callers still using `verbous`.
- [x] **`loss_curiosity` type is `float | Tensor`** — fixed by initializing it as a zero tensor on the GAN device/dtype.
- [x] **`insert_many` column-major layout is confusing** — improved by accepting row-major and column-major layouts explicitly and documenting the behavior.
- [x] **Rust `update_sorted_indices` is O(n log n) per insert** — addressed by maintaining sorted indices incrementally via binary search insertion.
- [x] **Missing `__len__` / `__getitem__` on `Buffer`** — added.
- [x] **No input validation on `OptComponents`** — added `__post_init__` validation to config dataclasses.
- [~] **Fragile device handling** — improved in the curiosity-loss paths by explicitly moving buffer elites to the active tensor device/dtype, but not fully centralized yet.
- [x] **`CuriosityLossConfig` "disabled" semantics are implicit** — documented and normalized in config handling (`<= 0` disables the term).
