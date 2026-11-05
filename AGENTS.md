# Instructions for AI coding tools (Claude, Copilot, Cursor, etc.)

This file governs `sandbox/handover/` specifically — a self-contained, tested port of the
validated ship-speed optimizer, meant to become the basis for the public MUTE-DSS repo. If you
are an AI agent making changes here, read this file before editing anything, the same way a human
contributor would read `README.md` and `SCIENTIFIC_FINDINGS.md` first.

## Hard rule: scientific core is frozen, not legacy code to improve

The following are **validated scientific logic** — correct, tested against known values, and
load-bearing for every number this package produces. Do not modify the math/formulas in these
files for refactoring, style, or "improvement" reasons. If a change here ever seems necessary,
stop, name exactly what and why, and get explicit human sign-off before touching it:

- `source_model/jomopans_model.py` — the JOMOPANS-ECHO source-level formulas (`_compute_baseline`,
  `jomopans_model`, `jomopans_model_matrix`). The one intentional deviation from `../MUTE` (the
  V≤0 masking fix) is already made and tested — don't "fix" it again or revert it.
- `tl_model/rbf_light.py`, `tl_model/rbf_full.py` — verbatim ports of `../MUTE`'s trained-model
  interpolator classes. Don't change kernel/neighbor/epsilon parameters; they must match the
  pickled models these classes load.
- `objective/objective.py`'s `evaluate_population_multi` — the SEL + time cost function. The
  formula (`noise_cost = sel_max_weight*SEL_max + (1-sel_max_weight)*SEL_mean`,
  `cost = noise_cost + w_time*T_hr`) and the half-segment duration weighting are validated;
  changing the formula shape is a scientific decision, not a code-quality one.
  `route_utils/route_utils.py`'s kinematics it depends on are equally frozen.
  `operators/operators.py`'s bug fix (the `.do()`/`_do()` distinction) is settled — don't
  "simplify" it back to calling `_do()` directly, that reintroduces the dead-code bug.
- `ga_problem/ga_problem.py`'s constraint logic (acceleration + the hard per-mammal
  non-regression constraint) — this is the core validated contribution (`SCIENTIFIC_FINDINGS.md`
  §2). Don't change what the constraint checks or how `build_seeded_population` shapes the GA's
  initial population without sign-off.
- `mammal_model/constants.py`, `mammal_model/motion.py` — the Ornstein-Uhlenbeck motion model and
  coastal-step logic (Lin, Dowd & Joy 2025), ported verbatim from `backend/app/mammal/`. Same
  frozen status as the optimizer physics above — don't touch the O-U parameters (`ALPHA`, `SIGMA`),
  the direction-rule blend, or the coastal-step candidate-selection logic. `mammal_model/generate.py`
  (the heading/speed→velocity conversion, the land-start check, the I/O-shaping wrapper) is NOT
  frozen — it's a thin convenience layer, safe to extend (e.g. new input conventions) as long as
  it keeps calling `motion.simulate_paths` unchanged underneath.

**What counts as "the math" vs. safe refactoring**: renaming a variable, adding a type hint,
splitting a long function into named helpers that compute the exact same numbers — fine. Changing
an exponent, a weighting scheme, a clip bound, a default hyperparameter, or which quantity feeds
into which formula — not fine without sign-off, even if it looks like it "should" be more correct.

## What's safe to change freely

- `config/` (YAML files) — the whole point of the config system is that retuning a run never
  needs a code change. Add new example configs under `config/examples/` freely.
- `viz/` — figure styling, new chart types, new export formats. Follow the dataviz skill's rules
  already baked into `viz/palette.py`/`viz/style.py` (fixed categorical color order, run
  `validate_palette.js` before changing any hex value, real LaTeX text via `text.usetex`).
- `reconstruct/` — as long as `reconstruct_full_order`'s output stays a pure function of
  (manifest, optimized_speed, data files) with no new randomness, you can add columns/detail.
- `data/` — adding new named route/mammal CSVs (same schema) is fine. Do not overwrite
  `baseline_route.csv` or `mammal_paths.csv` in place — every number in every test and every
  `SCIENTIFIC_FINDINGS.md` claim is pinned to these exact files. Add a new file with a new name
  and point a config at it instead. `mammal_model/io.write_mammal_csv` already enforces this
  (always writes a new path, raises rather than silently overwriting a duplicate `mammal_id`
  unless `overwrite_id=True` is passed explicitly) — don't build a path around that guard.
- Tests, docs, examples, CLI ergonomics (`run.py`, `viz/make_figures.py`, `generate_mammal.py`) —
  improve freely.

## Before any change, run the tests; after any change, they must still pass

```bash
cd handover/
python -m source_model.tests.test_source_model
python -m tl_model.tests.test_tl_model
python -m route_utils.tests.test_route_utils
python -m objective.tests.test_objective
python -m operators.tests.test_operators
python -m ga_problem.tests.test_ga_problem
python -m baselines.tests.test_baselines
python -m reconstruct.tests.test_reconstruct
python -m viz.tests.test_viz
python -m mammal_model.tests.test_motion
python -m mammal_model.tests.test_generate
python -m mammal_model.tests.test_io
python -m tests.test_end_to_end
```

Tests needing the external TL model skip cleanly (print `SKIP ...`) if `MUTE_LIGHT_MODEL_DIR`
isn't set — if you're working in an environment without it, you will not see those tests actually
exercise the code path you changed. Say so explicitly rather than reporting "tests pass" when a
relevant test silently skipped.

If you fix a bug, add a regression test for it in the same module's `tests/` directory, following
the existing pattern: a test that would have failed on the old code and passes on the fix (see
`operators/tests/test_operators.py` and `source_model/tests/test_source_model.py` for the
template — both exist specifically because this session found and fixed real bugs this way).

## Documentation discipline

- `SCIENTIFIC_FINDINGS.md` is the validation record. If you discover something surprising (a bug,
  an unexpected result, a convergence failure) while working here, add it there — don't just fix
  it silently or leave it undocumented.
- Every module file's docstring explains *why*, not just what — provenance (ported from where),
  what's validated, what's a known caveat. Match that style in new code; don't strip comments to
  "clean up" a file.
- `config/default_config.yaml` is the single documented reference for every tunable — if you add
  a new config field, document it there inline, in the same style as its neighbors.

## Determinism

`objective.evaluate_population_multi` → `ga_problem` → the GA's output (`optimized_speed`) is the
only non-reproducible step in the whole pipeline (stochastic search). Everything downstream
(`reconstruct/`, `viz/`) must stay a **pure function** of (manifest config, optimized_speed, the
fixed `data/` CSVs) — this is what lets `reconstruct_from_results_dir` regenerate a full result
without rerunning the GA. Don't introduce hidden randomness, wall-clock dependence, or external
state into any function downstream of the GA's output.

## Scope discipline

This package was built by porting and cleaning up `sandbox/optimizer/`'s exploratory work (see
`SCIENTIFIC_FINDINGS.md`), not by inventing new science. Don't add new scientific capabilities
(new noise models, new constraint types, new objective terms) speculatively — if a task seems to
call for one, flag it and ask, the same way the sandbox work always surfaced a proposed direction
before building it out.
