# Training-divergence investigation — decision log (2026-09)

Retrospective on debugging why training runs (mainly on Kaggle, 2×T4, mixed precision)
were diverging to `NaN`, and what actually got fixed vs. what was considered and ruled
out. Written in the order the investigation actually happened, including the theories
that turned out wrong — that's part of the record, not just the final fixes.

## Starting point

Local training (Mac, single device, fp32, milder synthetic degradation) converged
normally — e.g. `residual_unet` reached ~36.8 dB val PSNR over ~13 epochs. Moving to
Kaggle to use free GPU time, and separately escalating the degradation severity and
dataset size (10k → 50k crops) for a more serious data-scale comparison, is when
training started diverging: `val_psnr` / `val_loss` going to `nan`, sometimes
`train_loss` too, usually within the first epoch.

## 1. Mixed precision (fp16) and the loss function

Enabled `+trainer.precision=16-mixed` for speed on Kaggle's T4s — measured ~2.7x
throughput vs. fp32 on the same workload. Two things checked directly rather than
assumed:

- **Does `16-mixed` already scale the loss?** Yes — confirmed from Lightning's
  `MixedPrecision` plugin source: it instantiates `torch.amp.GradScaler` automatically
  when precision is `"16-mixed"`. No manual scaler setup needed.
- **Does our loss function need special handling under autocast?** We use `MSELoss`
  (every model config: `loss: mse`). Checked PyTorch's own AMP docs: `mse_loss` (and
  `l1_loss`, our other option) are explicitly on the "ops forced to float32 under
  autocast" list — handled automatically. The real gotcha here is `BCELoss` (not
  `BCEWithLogitsLoss`), which PyTorch documents as unsafe/erroring under autocast —
  doesn't apply to us at all, we don't use it.

**Verdict: precision and loss function were not the problem.**

## 2. Gradient clipping — norm vs. value

Added `gradient_clip_val=1.0` early as a general anti-NaN mitigation. It didn't stop
the divergence, and inspecting *why* turned into the most concrete finding of the
whole investigation:

- Lightning defaults `gradient_clip_algorithm` to `"norm"` when left unset (confirmed
  from source) — a **global** L2 norm computed across every parameter's gradient
  combined, then every parameter rescaled by the same factor.
- Directly loaded a failed checkpoint's `state_dict` and checked every tensor for
  `NaN`/`Inf`: **64 of 71 tensors were `NaN` simultaneously** — plain `Conv2d` weights
  and biases included, not just BatchNorm buffers. This disproved an earlier,
  narrower theory that only BatchNorm's forward-pass statistics were poisoned.
- The mechanism: `sqrt(sum of squares)` where even one term is `NaN` is itself `NaN`.
  One bad gradient anywhere → global norm `NaN` → rescale factor `NaN` → **every**
  parameter's gradient multiplied by `NaN` in that one step. That's exactly the
  all-at-once, everywhere pattern the checkpoint showed.

**Fix**: switched to `gradient_clip_algorithm=value` (`clip_grad_value_`) — clips each
parameter independently, so one bad gradient can't contaminate the rest of the model.
Doesn't prevent a `NaN` from occurring, but keeps it localized — meaning if it
happens again, the resulting checkpoint will actually show *where* it started instead
of corrupting everything uniformly and erasing that information.

## 3. Ruling out the data pipeline

Suspected candidates: the escalated degradation severity (blur radius up to 2.5, JPEG
quality down to 5, noise σ up to 0.06 @ 70% probability), or a corrupt crop in the
larger 50k-image set.

- Scanned all 50,000 crops directly through the real degradation pipeline — zero
  anomalies (no non-finite values, no degenerate/blank images, no exceptions).
- Traced the actual arithmetic in `transforms.py`: `GaussianNoise` clips to `[0,1]`
  *before* the `uint8` cast (correct order — no overflow), `JPEGNoise`/`GaussianBlur`
  use standard, bounded-parameter PIL codecs. No path can produce non-finite or
  out-of-range pixel data.

**Verdict: the data pipeline was cleanly ruled out**, via direct evidence, not
assumption.

Found one real, unrelated bug along the way: `apply_deterministic`'s seed used
Python's builtin `hash(name)`, which is randomized per-process — verified directly
(4 separate process launches gave 4 different seeds for the same filename). This
undermined the "same validation set, same noise" comparison between separate
`denoising-train` process launches (e.g. the 10k vs. 50k comparison notebook), not
training stability. Fixed with a stable `hashlib.md5`-based seed.

## 4. Architecture — BatchNorm placement and ResidualUNet's output bound

A parallel multi-agent code audit (config / architecture / data / training-loop, each
scoped to verify facts, not theorize) turned up two real architectural anomalies:

**`DoubleConv`'s layer order** was `Conv → ReLU → Conv → ReLU → BatchNorm` — a single
trailing BatchNorm after *both* convolutions, with zero normalization between them,
rather than the canonical `Conv → BN → ReLU` applied per conv. Researched whether this
is actually non-standard before changing anything: it's a genuine two-sided design
choice in the literature (DnCNN itself uses fully unconstrained linear residual
output), not a clear-cut violation — but the canonical per-conv ordering is
well-justified (BatchNorm sees each conv's own pre-activation output, not an
already-twice-rectified mix of both), so changed it anyway. Verified: smoke tests
pass, a real short training run completes cleanly.

**`ResidualUNet`'s `noise_head`** was a fully unconstrained linear `Conv2d`, feeding
`(x - noise).clamp(0, 1)`. Empirically verified `torch.clamp`'s gradient is exactly
zero in its saturated region — so any pixel where the raw prediction pushed the
result outside `[0,1]` got zero learning signal for that step (not `NaN`, a milder
but real training-efficiency issue). Derived that the *true* noise value is provably
bounded to `[-1,1]` (difference of two `[0,1]`-bounded pixels), and bounded
`noise_head`'s output with `tanh` to match. Researched first here too — this doesn't
eliminate the dead-gradient zone entirely (`(x - noise)` can still exceed `[0,1]` at
the extremes even with `noise ∈ [-1,1]`), just narrows it from "anything an untrained
conv layer might output" to "the actual edges of the physically meaningful range."

## 5. Multi-GPU (DDP) setup

Discovered `trainer.devices=1` was a leftover single-GPU-Mac-oriented default — Kaggle
gives 2 GPUs, only one was ever being used. Setting up real 2-GPU training required
three things together, not independently:

- `trainer.devices=2`
- `+trainer.strategy=ddp_notebook` — plain `ddp` assumes it can relaunch the script as
  a fresh process, which breaks inside a live Jupyter kernel.
- `+trainer.sync_batchnorm=true` — **required alongside multi-GPU**, and skipping it
  was a real, confirmed bug: DDP's gradient all-reduce syncs gradients, not buffers,
  so each GPU's BatchNorm running stats drift independently without this flag. This
  directly explained an earlier symptom (`val_psnr` oscillating incoherently while
  `train_loss` looked fine) in an initial multi-GPU attempt.

Also fixed an adjacent rank-safety bug found at the same time: `LogPredictionsCallback`
was running on *every* DDP process, not just rank 0, likely conflicting when multiple
processes tried to push images to the same W&B run simultaneously. Guarded with
`trainer.is_global_zero`.

## 6. W&B + DDP interaction (prediction images silently missing)

After finally getting a full, clean 1-epoch 2-GPU run (`val_psnr = 27.76 dB`, `NaN`
nowhere — confirmed directly via the W&B API, not by eyeballing a terminal paste),
noticed the 4 logged prediction images showed "No matching media" in the W&B UI.

- Queried the W&B API directly for the run's server-side file list: the metadata
  *reference* to the images existed, but the actual image files were never uploaded.
- Found the exact cause in the run's own captured log: wandb's own documented warning
  — "setting step in multiprocessing can result in data loss." `LogPredictionsCallback`
  was passing an explicit `step=trainer.global_step` under a DDP (multi-process) run,
  exactly the pattern wandb warns about.

**Fix**: dropped the explicit `step=` argument, letting wandb auto-assign it, per
wandb's own recommended remediation. Not yet re-verified against a fresh live run.

## 7. Logging/progress-bar visibility (tooling, not divergence — but blocked diagnosis)

Lightning's default `RichProgressBar` (auto-selected because `rich` is installed)
produces a wall of thousands of near-duplicate lines when captured by Kaggle's
non-interactive notebook kernel — its `Live` display redraws on a wall-clock timer via
a background thread, independent of batch count, and each redraw gets captured as a
separate transcript entry. Switched to `TQDMProgressBar`, verified locally (30
batches → ~5 lines instead of thousands). Also restored `log_every_n_steps` to a
usable value after it had been bumped to 1000 — a separate, unrelated setting that
only controls CSV/W&B metric flush cadence, not progress-bar output at all.

## Where this landed

A clean, fully verified 1-epoch run on 2 GPUs, `residual_unet`, `16-mixed` precision,
every fix above combined: `val_psnr = 27.76 dB`, no `NaN` anywhere, confirmed directly
via the W&B API. First fully clean result in the entire investigation.

**Considered and declined**: Kaggle's TPU v5e-8 option. Far more raw compute than
2×T4, but would require re-deriving most of the multi-device setup above on a
different backend (`torch_xla`, `bf16` instead of `fp16`, XLA's own cross-core
BatchNorm sync instead of PyTorch DDP's) for a 7.7M-parameter model that isn't
compute-starved on GPU in the first place. Not worth the re-verification cost given
how much of section 5 was hard-won GPU-specific debugging.

## Methodological note

Several early theories in this investigation were proposed with real-sounding
reasoning and then directly contradicted by the next piece of evidence — most
notably, an initial "BatchNorm's forward-pass statistics overflow fp16" theory was
disproven by directly inspecting a failed checkpoint's tensors. The throughline that
actually worked was insisting on direct evidence at each step (checkpoint inspection,
W&B API queries, empirical hash/gradient tests, local reproduction attempts) over
plausible-sounding narratives — worth preserving as the actual record rather than a
cleaned-up version that only shows the fixes that stuck.
