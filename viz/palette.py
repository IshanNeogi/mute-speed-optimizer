"""Color constants -- from the dataviz skill's validated reference palette
(references/palette.md), not eyeballed. Categorical order is CVD-validated and fixed; never
reassign a slot based on which series happens to be "interesting" in a given figure.
"""

# Fixed-order categorical slots (passes every adjacent-pair CVD gate, light mode) -- used for
# per-mammal identity when there are multiple mammals in one figure. Mammal 1 always gets slot 1,
# mammal 2 always slot 2, etc., regardless of which mammal "matters more" in that run.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#6250d6",  # 7 violet
    "#e34948",  # 8 red
]
# All-pairs-validated subset (small multiples / scatter-style use where every pair can appear
# together, not just adjacent ones) -- cap at 3 series before folding into "Other" or faceting.
CATEGORICAL_ALL_PAIRS_SAFE = CATEGORICAL[:3]

# Fixed roles, NOT part of the rotating categorical order -- these represent a condition
# (recorded vs. optimized), not an entity identity, so they stay constant across every figure
# regardless of how many mammals are in CATEGORICAL use at the same time.
ECHO_AIS_COLOR = "#eb6834"      # orange -- same role color used throughout sandbox/optimizer
OPTIMIZED_COLOR = "#2a78d6"     # blue

# Sequential (magnitude: speed, received level) -- one hue, light->dark. Use the 100-700 range
# for continuous fields (heatmaps/contours); clip to >=250 if used as a discrete ordinal ramp.
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

# Text/ink roles -- never use a series color for text; text stays in these.
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#6b6a66"
GRID = "#e5e4de"
AXIS = "#c9c8c0"
SURFACE = "#fcfcfb"


def sequential_cmap(name="viz_sequential_blue"):
    """matplotlib LinearSegmentedColormap built from SEQUENTIAL_BLUE, for contourf/LineCollection
    magnitude encodings (route speed, received level maps) -- the skill's "one hue, light->dark"
    rule, not a multi-hue perceptual ramp like viridis/plasma (which earlier sandbox figures used)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, SEQUENTIAL_BLUE)
