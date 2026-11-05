# Edge-compute feasibility analysis: 1–5 mammal speed optimizer

No new optimizer cases were run for this analysis. Memory numbers are measured directly
(profiling the actual `handover` import/load stack on the dev machine, not a GA run). Runtime
numbers are averaged from the per-seed timings already logged by today's real C1–C5 runs
(pop=32, n_gen=200, 5 seeds each, peak-band mode, post-operators-fix). CPU extrapolation to edge
hardware is an estimate, clearly flagged as such — no edge device was available to measure
directly.

## 1. Memory footprint (measured)

Profiled on the dev machine (13th Gen Intel i7-13700H, 20 cores, 15GB RAM), importing exactly
what `run.py` needs, no more:

| Stage | Resident memory | Marginal |
|---|---|---|
| Bare Python interpreter | 8 MB | — |
| + numpy, scipy, pymoo, pyproj | 42 MB | +35 MB |
| + this package's own modules (objective/ga_problem/tl_model, not yet loaded) | 43 MB | +0 MB |
| + light TL model loaded into memory | **283 MB** | **+240 MB** |

**The TL model dominates memory, and it's a one-time, flat cost — not a function of mammal
count.** The model is a cached singleton (`tl_model.load_tl_model`); every mammal in a case
shares the same loaded model object, no per-mammal duplication. GA population arrays
(`pop=32 × n_wp=30` floats, ×however many mammals) are on the order of a few KB even at N=5 —
immaterial next to the 240MB model load.

**So memory does not scale with the number of mammals (1 through 5): expect ~280–300MB resident
regardless of N.** This is well within a 1GB-class device, and comfortable on anything 2GB+
(including the project's existing $12/mo DigitalOcean droplet, which is already known to run the
production app at 2GB).

Caveat: this is the **light** TL model only. If `full` mode is ever used instead (29-band + PCA,
1.3GB on disk), expect a correspondingly larger resident footprint — not quantified here since
`full` isn't the validated/production path.

## 2. Disk footprint (measured)

| Component | Size |
|---|---|
| numpy | 39 MB |
| scipy | 108 MB |
| pymoo | 23 MB |
| pyproj | 19 MB |
| **Core runtime deps subtotal** | **189 MB** |
| Light TL model file (`model.pkl`) | 171 MB |
| This package's own code + shipped test data | ~2 MB |
| **Total (inference-only, no matplotlib/sklearn)** | **~360 MB** |

`matplotlib` (viz/figures) and `scikit-learn` (only needed for `full` TL mode) are not required
for a pure optimizer-inference deployment — drop them from an edge image and save ~115MB
(matplotlib) if figures aren't generated on-device.

## 3. Runtime (measured on dev hardware, extrapolated to edge)

Per-seed wall time, averaged across every real run logged today (multiple w_time values, both
constraint conventions — GA search cost doesn't depend much on those, only on mammal count):

| N mammals | avg per-seed (dev laptop) | samples | per-mammal cost |
|---|---|---|---|
| 1 | 36.2s | 40 | 36.2s |
| 2 | 62.2s | 10 | 31.1s |
| 3 | 74.8s | 10 | 24.9s |
| 5 | 309.5s | 8 | **61.9s** |

**Not linear.** Per-mammal cost actually *improves* from N=1→3 (economies of batched TL queries),
then jumps sharply at N=5 — 2.5× the per-mammal cost of N=3. This matches a finding already on
record (`SCIENTIFIC_FINDINGS.md`: "GA runtime is superlinear in mammal count... unexplained").
With today's numbers, the likely mechanism is clearer: per-generation compute should scale close
to linearly (TL queries batch as `M×pop` points, cheap to vectorize) — so the real driver is
probably the GA needing more effort to find a feasible region at all once 5 simultaneous hard
per-mammal constraints are active, not raw arithmetic cost. Not confirmed, flagged as the
working explanation.

**Full case cost (5 seeds, sequential, as `run.py` does it)**, dev laptop:

| N | Full case (5 seeds) |
|---|---|
| 1 | ~3.0 min |
| 2 | ~5.2 min |
| 3 | ~6.2 min |
| 5 | ~25.8 min |

### Extrapolation to edge devices (estimates, not measured)

The project already has one documented single-core slowdown estimate for exactly this reason
(`sandbox/optimizer/LOG.md`, 2026-10-01): a DigitalOcean shared-vCPU Basic droplet was estimated
at **2–4× slower per-core** than this same dev laptop for sustained numpy/scipy compute. Extending
that same style of estimate to other edge classes, using published single-thread benchmark
ballparks for mixed floating-point/vectorized workloads (not validated against this specific
workload):

| Device class | Est. single-core slowdown vs. dev laptop | Basis |
|---|---|---|
| DigitalOcean 1vCPU droplet (current prod) | 2–4× | Prior project estimate, not re-measured here |
| Raspberry Pi 4/5 (ARM Cortex-A72/A76, ~1.5–2.4GHz) | ~4–7× | General ARM-vs-modern-x86 single-thread ratio for numpy/OpenBLAS workloads |
| Jetson Nano (ARM Cortex-A57, 4×1.43GHz, 4GB shared) | ~6–10× | Older/weaker ARM core, no AVX-class SIMD |
| Jetson Orin Nano (ARM Cortex-A78AE, 6×~1.5GHz, 4–8GB) | ~3–5× | Newer ARM core, closer to modern laptop IPC but lower clock |

Applying the **conservative (slower) end** of each range to the N=5 full-case dev-laptop time
(25.8 min):

| Device | Estimated N=5 full case (5 seeds) |
|---|---|
| DigitalOcean droplet (current prod) | ~1.7–3.4 hours |
| Raspberry Pi 4/5 | ~1.7–3.0 hours |
| Jetson Nano | ~2.6–4.3 hours |
| Jetson Orin Nano | ~1.3–2.2 hours |

For N=1 (3.0 min dev-laptop baseline), the same multipliers give roughly **6–30 minutes**
depending on device — far more tractable than N=5.

**pymoo itself may not parallelize across cores by default in this codebase's configuration**
(single-threaded GA loop as written) — these estimates assume single-core execution throughout,
matching how `run.py` actually runs today. A multi-core edge device (Pi 4 has 4 cores, Orin Nano
has 6) is not automatically faster unless the GA population evaluation is explicitly parallelized,
which it isn't here.

## 4. Bottom line

- **Memory is not the constraint anywhere down to ~1GB**, regardless of mammal count — confirmed
  by direct measurement, not estimate.
- **CPU wall-clock time is the real constraint**, and it's the superlinear N=5 jump that matters
  most: an N=5 case that takes ~26 minutes on this dev laptop could plausibly take **1.5–4+ hours**
  on genuinely edge-class hardware (Pi/Jetson Nano), which may not be acceptable for anything
  needing a timely replan.
- Options worth considering for an edge deployment, roughly in order of impact:
  1. **Fewer seeds.** `n_seeds=5` is a multi-start robustness measure, not strictly required —
     dropping to 1–2 seeds cuts wall time proportionally (~5× or ~2.5×) at some risk of landing in
     a worse local optimum (the exact risk this multi-start was designed to guard against — see
     `SCIENTIFIC_FINDINGS.md`'s seed-variability finding).
  2. **Smaller `pop_size`/`n_gen`** for N≥4 specifically, if a case-specific sensitivity study
     confirms the single-mammal pop=32/n_gen=200 finding doesn't hold at higher N (it hasn't been
     separately validated there — also already flagged as an open item).
  3. **Investigate the N=5 feasibility-search cost directly** (why it's 2.5× per-mammal, not just
     accept it) — could reveal a cheap fix (e.g. a better-than-min-speed seed shape at high N)
     rather than just buying less search budget.
  4. Offload the GA to a non-edge server and ship only results to the edge device, if connectivity
     allows — sidesteps the hardware question entirely for the compute-heavy part.
