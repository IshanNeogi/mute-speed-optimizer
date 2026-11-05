# Scientific findings behind this package

Everything here comes from `sandbox/optimizer/LOG.md` and `sandbox/optimizer/poster_results/
SUMMARY.md` — the full validation record. This file is the condensed, action-oriented version:
what's fixed in this package, what's a validated contribution, and what's still open.

## Fixed in this package (not in `../MUTE`/production — needs separate sign-off there)

1. **Adaptive mutation/crossover rate annealing was dead code.** `route_pymoo_config.py` /
   `speed_pymoo_config.py` construct a fresh `PolynomialMutation(prob=rate, ...)` / `SBX(prob=
   rate, ...)` each generation and call `._do()` directly — but pymoo's `prob` is only ever read
   inside `.do()`, never `._do()`. The annealed rate was computed and thrown away; only `eta`
   (held fixed) could ever affect anything. Fixed in `operators/operators.py` by replicating the
   Bernoulli-masking `.do()` performs, inline. Regression-tested in
   `operators/tests/test_operators.py`.
2. **`jomopans_model_matrix`'s `-inf` guard zeroed the whole population, not just the offending
   candidate.** `if np.isneginf(...).any(): array[:] = 0.0` has no axis — one degenerate `V<=0`
   speed anywhere in a (pop, n_waypoints, n_bands) array silently zeroed every candidate's source
   level for that generation. Fixed in `source_model/jomopans_model.py` with a per-(pop,waypoint)
   mask. Regression-tested in `source_model/tests/test_source_model.py`.

## The core validated contribution

The unconstrained weighted-sum formulation (`noise_cost = sel_max_weight*SEL_max +
(1-sel_max_weight)*SEL_mean`, `cost = noise_cost + w_time*T_hr`) is **bimodal and seed-unreliable**:
across 115 tested seed/config combinations, 52–100% landed in a "bad basin" (worse than every
fixed-speed heuristic), depending on `w_time`.

Adding a **hard per-mammal non-regression constraint** — every mammal's SEL under the candidate
must be ≤ its SEL under constant mid-speed — eliminates this failure mode entirely: **0/35
bad-basin** across every constrained trial tested, 69% also beat the real recorded voyage outright.
This is `ga_problem.MultiMammalSpeedGAProblem`'s `baseline_sel_per_mammal` constraint, on by
default (`optimizer.use_hard_mammal_constraint: true`).

Quantitatively (7 completed trial configs, 1–5 mammals): median ~67% less sound energy vs. the
real recorded voyage, at a median cost of tens of minutes of extra voyage time.

## GA methodology findings baked into this package's defaults

- **`pop_size=32, n_gen=200`** is the single-mammal-validated sweet spot (captures effectively all
  improvement available vs. pop=64/n_gen=400+, which is bit-identical beyond it; converges by
  ~generation 99 consistently). **Not separately validated for multi-mammal** — see "Open, not
  yet acted on" below.
- **Min-speed seeding, not baseline-shaped seeding**, is the validated standard
  (`optimizer.seed_population_shape: "min_speed"`). Two seed *shapes* at identical hyperparameters
  landed in optima 4+dB apart for a 3-mammal case — min-speed seeding won outright once adopted.
- **"No free waypoint" mechanism**: a waypoint where the current candidate's received level is
  already 0 still matters — its speed sets arrival time at every later waypoint, which determines
  where a *moving* mammal is when the ship reaches its real closest-approach point. This is why a
  small nonzero `w_time` can actually *improve* SEL (not just trade it for speed) — it gives the
  GA a reason to move efficiently through indifferent stretches instead of drifting.
- **Simultaneous multi-mammal scoring, not superimposition.** An earlier approach (optimize each
  mammal separately, combine via pointwise-min) was tried and abandoned: re-scoring the combined
  profile against each mammal's real objective showed large optimality loss (one mammal regressed
  +8.2dB, nearly back to its unoptimized baseline) because combining changes early-route timing,
  which shifts which part of a mammal's real track the ship later overlaps with. `objective.py`
  scores every candidate against every mammal simultaneously, every generation, for this reason.

## TL model characterization (relevant if you touch `tl_model/`)

- Light model is **9.9–12.2× faster** per query than full, agreement MAE 0.23–0.73dB, correlation
  1.0000 — confirmed empirically across 3 source locations, not a single-point extrapolation.
- The light model's real spatial support is much narrower than the full study region — a fixed
  "sentinel" (~740dB placeholder) corridor exists in the same stretch of the mid-Strait-of-Georgia
  regardless of source position. Don't trust TL output there.
- **RBF extrapolation risk past ~90-100km from a source**: querying well outside where the model
  has ever been validated can return a physically-backward result (received level not decaying
  monotonically with distance). `tl_model.query_tl`'s `max_trusted_radius_km` parameter guards
  against this — use it for anything spanning a large area (e.g. a full-domain map or animation).
- No training/retraining pipeline exists anywhere for either TL model — validation can only ever
  be behavioral (sanity queries, monotonicity, cross-model agreement), never reference-recomputed.

## Open, not yet acted on in this package

- **Multi-mammal GA convergence reliability is not established** the way the single-mammal case
  is. Two different seed shapes landed 4+dB apart in one tested 3-mammal case. The pop=32/n_gen=200
  recommendation is carried over from the single-mammal sensitivity study, not independently
  re-validated for multi-mammal timing-coupled landscapes.
- **GA runtime is superlinear in mammal count** (14.9s/mammal at N=2 → 26.7s/mammal at N=5 in one
  measurement), not linear — unexplained, relevant to any cost estimate for a real multi-mammal
  voyage on constrained hardware.
- `sel_max_weight=0.95` (production's actual value, vs. this package's validated default of 0.5)
  has not been tested together with the hard constraint — results could shift materially (0.95
  weights the worst-off mammal almost exclusively).
- Mutation/crossover RATE sensitivity (as opposed to whether the annealing fires at all, which is
  now fixed) has never been properly swept — the one attempt used the same seed for every variant,
  fully confounding the comparison. No evidence to deviate from the inherited defaults, but no
  evidence they're optimal either.
