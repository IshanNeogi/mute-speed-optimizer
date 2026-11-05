"""Physical constants for the mammal-motion model.

Ported verbatim from backend/app/mammal/constants.py (itself verbatim from ../MUTE
mammalMotion/layer1_visualizer.py and watson_prior.py -- Lin 2023 O-U motion model parameters,
Table 4.1). Salish-Sea-specific, not generalized to other regions. FROZEN scientific core -- see
AGENTS.md.
"""

# O-U (Ornstein-Uhlenbeck) motion model parameters (Lin 2023, Table 4.1)
ALPHA = 2.48   # mean-reversion rate  [h^-1]
SIGMA = 26.77  # diffusion coefficient [km h^-3/2]

# Study area bounds (Case 3 route extent +/- 1 deg buffer)
LAT_MIN, LAT_MAX = 47.408, 50.318
LON_MIN, LON_MAX = -125.360, -122.151

# Map projection: UTM zone 10N (EPSG:32610), as in Lin, Dowd & Joy (2025). The whole study area lies
# inside zone 10 (126W-120W); UTM's scale error there is below 0.05%.
UTM_EPSG = 32610

# path_projection.py (P_init -> forward path) projection timestep
DT_PROJECTION_HOURS = 1.0 / 60.0  # 1-minute timestep

# Direction rule (Lin, Dowd & Joy 2025, Algorithm 1 fallback branch) written in physical time,
# independent of the projection step; validated in the production app's VALIDATION.md (Stage 2b).
TENDENCY_WINDOW_MIN = 9.0   # [min] window of the directional tendency (paper: 3 steps of 3 min)
PULL_HALF_MIN = 3.0         # [min] pull weight on the tendency per step = 1 - 2**(-dt_min / PULL_HALF_MIN)

# Coastal step (paper appendix): a move that would touch land/shallow water is replaced by the best
# of N candidate headings, each travelling the intended distance or stopping at the boundary;
# top two-thirds by distance kept, closest to the directional tendency chosen.
COAST_N_CANDIDATES = 40          # from ../MUTE / Lin 2023 thesis; the 2025 paper does not state the count
COAST_KEEP_FRACTION = 2.0 / 3.0  # paper appendix
COAST_SAMPLE_SPACING_KM = 0.025  # [km] spacing of mask look-ups along a move (about half a mask cell)
