# Data

**`baseline_route.csv`** — Case_3/OKIANA's real Project Echo transit, resampled onto a shared
124.6 NM baseline route (583 points, lat/lon/speed_kn_estimated). Copied from
`sandbox/optimizer/source_model/data/uniform_case3_okiana.csv` — see that sandbox's `LOG.md`
(2026-09-29 entries) for how it was built.

**`mammal_paths.csv` / `mammal_metadata.csv`** — 20 synthetic mammal tracks (1000 min each,
1-minute resolution), generated for optimizer testing only, not a validation artifact themselves
(see `sandbox/whale_motion/optimizer_test_paths/generate_test_paths.py`) — though they do use the
same motion-model physics now shipped in `mammal_model/` (same lineage, pre-port). Each is fully
reproducible from its metadata row (seed, start lat/lon/velocity/heading).

**`land_mask.npz` / `blocked_mask.npz`** — the Salish Sea shoreline mask and the land+shallow-water
(<2m, GEBCO 2023)+safety-buffer mask `mammal_model/` keeps new mammal tracks out of. Copied from
`backend/app/mammal/data/` — see `mammal_model/land_mask.py` for the schema. Small (~270KB total,
both are compressed 4000×4000 bool arrays) and included directly, unlike the TL model below.

## Generating a new mammal track

`generate_mammal.py` (see the top-level README) uses `mammal_model/` — a port of the validated
Lin, Dowd & Joy (2025) motion model (`backend/app/mammal/`'s production physics, unchanged) — to
project a track forward from a location/heading/speed you provide. It never edits
`mammal_paths.csv` in place; it always writes a new file.

## TL model files — NOT included

The transmission-loss RBF models are large trained pickles that live in `../MUTE`, not this repo:

| | path (relative to `../MUTE`) | size |
|---|---|---|
| light (single-freq) | `v2/src/noise_propagation_model/noise_propagation_model/models/RBF/model_single_freq/model.pkl` | ~171 MB |
| full (29-band + PCA) | `v2/src/noise_propagation_model/noise_propagation_model/models/RBF/model/{interpolators,ipca,yobs_mean}.pkl` | ~1.3 GB |

Point `tl_model.light_model_dir` / `tl_model.full_model_dir` in your config at wherever you have
these checked out (see `config/default_config.yaml`). There is no training pipeline anywhere to
regenerate them — see `sandbox/optimizer/LOG.md` (2026-09-29) — so you need a copy of one of these
exact files, not just the right format.
