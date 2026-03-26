"""
Experimental parameters and animal lists
"""

# Animal lists by stimulation location
VIS_STIM_ANIMALS = [
    'mouse598181', 'mouse598183', 'mouse622179', 'mouse622180',
    'mouse631538', 'mouse638329', 'mouse638330', 'mouse655957', 'mouse657902'
]

SS_STIM_ANIMALS = [
    'mouse577578', 'mouse634572', 'mouse657904', 'mouse657905'
]

MO_STIM_ANIMALS = ['mouse598180']

ALL_ANIMALS = VIS_STIM_ANIMALS + SS_STIM_ANIMALS + MO_STIM_ANIMALS

# Default analysis uses VIS and SS stim animals
DEFAULT_ANIMAL_LIST = VIS_STIM_ANIMALS + SS_STIM_ANIMALS

# Stimulation parameters
FREQUENCIES = [8, 28, 140]  # Hz
AMPLITUDES = [1, 5]  # µA

# Figure 2 specific parameter combinations (amplitude, frequency)
FIGURE2_PARAMS = [
    (1, 8),    # Row 1
    (5, 8),    # Row 2
    (5, 28),   # Row 3
    (5, 140)   # Row 4
]

# Spatial decay panels (row 5)
FIGURE2_SPATIAL_PARAMS = [
    (5, 8),
    (5, 28),
    (5, 140)
]

# Cell-class colours for cluster visualisation
cty_colors_ = ['#9932CC', '#228B22', '#CD5C5C', '#4169E1', '#FFD700', '#8B008B']

# Quality thresholds
n_spike_thresh = 51
error_threshold = 0.1
