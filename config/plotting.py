"""
Plotting configuration following Nature publication standards
"""
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Nature figure specifications (in points and mm)
SINGLE_COLUMN_WIDTH = 89  # mm
DOUBLE_COLUMN_WIDTH = 183  # mm
MAX_HEIGHT = 247  # mm

def mm_to_inch(mm):
    """Convert millimeters to inches for matplotlib"""
    return mm / 25.4

# Nature style matplotlib rcParams
NATURE_RC_PARAMS = {
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'lines.linewidth': 0.75,
    'patch.linewidth': 0.5,
}

def apply_nature_style():
    """Apply Nature publication style to matplotlib"""
    rcParams.update(NATURE_RC_PARAMS)
    
def remove_top_right_spines(ax):
    """Remove top and right spines from axis"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Frequency-specific colors
FREQUENCY_COLORS = {
    8: 'limegreen',
    28: 'darkorange',
    140: 'fuchsia'
}

# Paired comparison plot parameters
PAIRED_COMPARISON_PARAMS = {
    'bar_width': 0.9,
    'delta': 0.35,  # Spacing between pre and stim
    'y_max': 0.95,
    'y_sig_offset': 0.05,
    'pair_line_alpha': 0.25,
    'pair_line_width': 0.5,
    'trend_line_width': 1.5,
    'background_alpha': 0.1,
    'marker_size': 3,  # In points for Nature style
}

# Significance markers
SIGNIFICANCE_THRESHOLDS = {
    'ns': 1.0,
    'low': 0.01,
    'high': 0.001
}

SIGNIFICANCE_MARKER_PARAMS = {
    'marker': 'o',
    'size': 3,  # In points
    'alpha': 1.0,
    'linewidth': 0.5
}

# Spatial decay plot parameters
SPATIAL_DECAY_PARAMS = {
    'scatter_alpha': 0.1,
    'scatter_size': 2,  # In points
    'fit_line_width': 1.5,
    'fit_line_style': '--',
    'marker_size_top': 3,
    'marker_alpha_dim': 0.25,
    'label_fontsize': 6,
    'label_offset': (2, 2),  # In points
    'xrange': [0.25, 4.25],
    'yticks_delta': [0.0, 0.25, 0.5, 0.75],
    'yticks_vl': [0.0, 0.1, 0.2, 0.3],
    'xticks': range(0, 5)
}

# Statistical test parameters
STATS_PARAMS = {
    'test_type': 'trend_swarm',
    'min_units_per_area': 8,
    'entrainment_thresh_low': 0.01,
    'entrainment_thresh_high': 0.001
}

# Spatial decay parameters
SPATIAL_DECAY_OPTIONS = {
    'use_absolute_difference': True,  # Set to False for signed difference
}
