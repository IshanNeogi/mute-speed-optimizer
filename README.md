# MUTE speed optimizer — handover package

Given a route and one or more mammal tracks, finds a per-waypoint ship speed profile that reduces
sound exposure level (SEL) without making any mammal worse off than the ship going constant
medium speed. A clean, modular, tested port of the validated work in `sandbox/optimizer/` — see
`SCIENTIFIC_FINDINGS.md` for the full validation record and `AGENTS.md` if an AI coding tool is
driving changes here. See `EDGE_COMPUTE_ANALYSIS.md` for measured memory/disk footprint and
estimated runtime on edge hardware (Jetson/Pi-class devices).

## 1. Install

```bash
pip install -r requirements.txt
sudo apt-get install -y cm-super dvipng   # one-time, only needed for the figures (real LaTeX text)
```

## 2. Point at the transmission-loss (TL) model

Not shipped in this repo — it's a 171MB trained pickle, not ours to redistribute. Set an
environment variable once, pointing at wherever you have a copy (see `data/README.md` for exactly
which file and where `../MUTE` keeps it):

```bash
export MUTE_LIGHT_MODEL_DIR=/path/to/model_single_freq
```

Every example config below reads this automatically — you never have to edit YAML just to point
at your local model file.

## 3. Run a preloaded case

```bash
python run.py --config config/examples/quickstart.yaml
```

Finishes in a couple of seconds (deliberately small GA settings, one mammal) — just confirms your
setup works end to end. You'll see per-mammal SEL before/after and where the results landed
(`results/quickstart/`).

Once that works, run one of the actual validated cases (a few minutes, real settings):

```bash
python run.py --config config/examples/validated_3mammal.yaml   # mammals 3, 12, 8
python run.py --config config/examples/validated_5mammal.yaml   # mammals 0, 4, 8, 12, 16
```

Then generate publication-ready figures (PDF + EPS, real LaTeX text) from any of these:

```bash
python -m viz.make_figures --results results/validated_3mammal --figures route,sel,timeline,convergence --formats pdf,eps
```

## 4. Set up your own case

Copy an example config and change what you need — every field is documented inline in
`config/default_config.yaml`:

```bash
cp config/examples/validated_3mammal.yaml config/examples/my_case.yaml
```

What you'll typically change:
- **`mammal_ids`** — which rows of `data/mammal_paths.csv` to run against (IDs 0–19; see
  `data/README.md` — these are synthetic test tracks, not validated against real telemetry). To
  use your own mammal tracks, replace `data/mammal_paths.csv` with the same schema
  (`mammal_id, minute, lat, lon`) and point `data.mammal_paths_csv` at it.
- **`ship.*`** — vessel class/length/speed bounds, if modeling a different ship than Case_3/OKIANA.
- **`optimizer.w_time` / `sel_max_weight`** — the cost-function weights; see
  `SCIENTIFIC_FINDINGS.md` for what's actually been validated at which values before changing these.
- **`optimizer.use_hard_mammal_constraint`** — leave this `true` unless you specifically want to
  reproduce the old, unconstrained, seed-unreliable formulation for comparison (see
  `SCIENTIFIC_FINDINGS.md` §3 for why it's `true` by default).
- **`optimizer.constraint_voyage_time_hr`** — the "no mammal worse off than this" reference speed,
  given as a voyage TIME in hours (the equivalent constant speed is computed automatically).
  `null` (default) = the echo_ais recorded voyage's own time on this route.
- **`data.baseline_route_csv`** — to run a different route, same schema
  (`lat, lon, speed_kn_estimated`, one row per waypoint).

### Adding a new mammal (instead of using the pre-loaded test tracks)

`data/mammal_paths.csv`'s 20 tracks are pre-generated and pinned (every test in this repo is tied
to their exact contents — see `data/README.md`). To analyze a mammal of your own, generate a track
from a starting location/heading/speed and write it to a **new** file:

```bash
python generate_mammal.py --lat 48.6 --lon -123.2 --heading-deg 270 --speed-kn 4 \
    --horizon-min 900 --seed 7 --base-csv data/mammal_paths.csv --out data/my_mammals.csv
```

This never modifies `data/mammal_paths.csv` — it writes a new file containing the 20 existing
tracks plus your new one (under the next free `mammal_id`, printed at the end). Omit `--base-csv`
for a file with only your new mammal. Then point a config at it:

```yaml
data:
  mammal_paths_csv: "data/my_mammals.csv"
mammal_ids: [20]   # whatever id generate_mammal.py printed
```

Uses the same validated Lin, Dowd & Joy (2025) motion model as production (`mammal_model/` —
ported from `backend/app/mammal/`, physics unchanged, land/shallow-water-avoiding). Not ported:
production's Watson-prior random-spawn sampling (a different, separate thing — picking a *random*
start location from historical sighting density) — this only covers "I know where/how fast/which
way a mammal is moving, project it forward," same as production's own `predict.py`.

Then:

```bash
python run.py --config config/examples/my_case.yaml
```

## Running the tests

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

Tests that need the external TL model print `SKIP ...` and pass cleanly if `MUTE_LIGHT_MODEL_DIR`
isn't set / doesn't exist, rather than failing — so `operators`/`route_utils`/`source_model` always
run fully, and everything else runs fully once you've pointed at a real model file.

## Reproducing a full-order result without rerunning the GA

`run.py` writes `results/<case>/run_manifest.json` (the exact config used) alongside
`seed_results.json` (each seed's `optimized_speed` control-point array). The GA search is the only
non-reproducible step in a run; everything downstream is a pure function of those two files plus
the fixed `data/` CSVs. To materialize the full 583-waypoint per-mammal trace (lat/lon/time/speed/
distance/received level/cumulative SEL) for a specific case, on demand, without storing it every
run:

```python
from pathlib import Path
from reconstruct.reconstruct import reconstruct_from_results_dir

reconstruct_from_results_dir(Path("results/validated_3mammal"), Path("."))   # -> full_order_seed<N>.csv
```

`viz.make_figures` calls this automatically for any figure that needs it — you don't need to run
it yourself unless you want the raw CSV.

## Module map

| module | what it does | depends on |
|---|---|---|
| `source_model/` | JOMOPANS-ECHO ship source-level model | — |
| `tl_model/` | transmission-loss model (light + full RBF), config-driven, cached | — |
| `route_utils/` | route downsample/upsample, segment distance/time/weighting | — |
| `objective/` | multi-mammal SEL+time cost function | `source_model`, `tl_model`, `route_utils` |
| `operators/` | adaptive mutation/crossover GA operators (bug-fixed, see below) | — (pymoo only) |
| `ga_problem/` | the pymoo Problem: acceleration + hard per-mammal constraint | `objective`, `route_utils` |
| `baselines/` | fixed-speed heuristics + the multi-seed selection rule | `objective` |
| `reconstruct/` | rebuild the full-order (583-waypoint) result from a run's saved output, no re-optimization | `route_utils`, `source_model`, `tl_model` |
| `viz/` | publication/poster figures — PDF + EPS, real LaTeX, CVD-validated colors | `reconstruct` |
| `mammal_model/` | generate a new mammal track from location/heading/speed (Lin et al. 2025 model) | — (uses `data/*_mask.npz`) |
| `run.py` | CLI entrypoint wiring everything together | all of the above |
| `generate_mammal.py` | CLI: generate a mammal track, write it as a new `mammal_paths.csv`-schema file | `mammal_model` |

Each module directory is self-contained enough to lift out and reuse elsewhere on its own (e.g.
`source_model/` has no dependency on anything else in this repo).

## Figures — details

`python -m viz.make_figures --results <dir> --figures route,sel,timeline,convergence --formats pdf,eps`
generates: a speed-colored route map (optimized vs. mammal tracks), a per-mammal SEL comparison
(echo_ais vs. optimized, dB + \%-energy labels), received-level and cumulative-SEL timelines, and a
GA convergence plot. `--figures`/`--formats` are comma-separated subsets; `--context poster`
(default, large text) or `--context journal` (compact, single-column sizing).

Colors follow a CVD-validated fixed categorical order (never cycled/reassigned per figure) — see
`viz/palette.py`; re-run `node .../dataviz/scripts/validate_palette.js "<hex,...>" --mode light`
from the dataviz skill if you ever change any of them. Text renders through real LaTeX
(`text.usetex=True`) — needs `cm-super`/`dvipng` installed (step 1 above); without it,
`viz.style.apply_style()` fails loudly rather than silently falling back to a different renderer.

## What's different from `../MUTE` / production, on purpose

1. **A hard per-mammal non-regression constraint** (`optimizer.use_hard_mammal_constraint: true`):
   every mammal's SEL under the GA's candidate must not exceed its SEL under constant mid-speed.
   This is the core validated finding — see `SCIENTIFIC_FINDINGS.md` §3. Toggle it off to
   reproduce the old, unconstrained, seed-unreliable formulation for comparison.
2. **Two bugs found in `../MUTE` / production, fixed here** (not in `../MUTE` itself — that
   requires separate sign-off, this is a fresh codebase):
   - `operators.py`: the adaptive mutation/crossover rate annealing was dead code (called an
     operator's `_do()` directly instead of pymoo's `.do()`, the only place `prob` is read) — see
     the module docstring and `operators/tests/test_operators.py`'s regression tests.
   - `source_model/jomopans_model.py`'s `jomopans_model_matrix`: a single degenerate `V<=0`
     candidate anywhere in a population used to zero the ENTIRE population's source-level output
     for that generation (`.any()` with no axis) — now masked per-candidate. See
     `source_model/tests/test_source_model.py`'s regression test.
3. **Multi-seed + a selection rule** (`baselines.select_best_seed`), not production's unseeded
   single-shot run.

See `SCIENTIFIC_FINDINGS.md` for the full list of findings (including several not yet acted on
here — e.g. GA runtime being superlinear in mammal count, multi-mammal convergence not being
pop/gen-validated the way the single-mammal case was).
