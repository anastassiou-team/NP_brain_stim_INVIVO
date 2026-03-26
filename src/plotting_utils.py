"""
Plotting utilities with Nature publication style
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from config.plotting import (
    NATURE_RC_PARAMS, apply_nature_style, remove_top_right_spines,
    FREQUENCY_COLORS, PAIRED_COMPARISON_PARAMS, SIGNIFICANCE_MARKER_PARAMS,
    SPATIAL_DECAY_PARAMS, mm_to_inch, DOUBLE_COLUMN_WIDTH
)
from src.statistics import get_significance_level


def setup_nature_figure():
    """Initialize matplotlib with Nature style"""
    apply_nature_style()


def plot_paired_comparison_row(
    brain_areas,
    comparison_data,
    amplitude,
    frequency,
    show_area_labels=False,
    ax=None,
    # ylabel='vector length'
    ylabel='VL'
):
    """
    Create one row of paired pre vs stim comparison across brain areas
    
    Parameters
    ----------
    brain_areas : list
        Brain areas sorted by distance
    comparison_data : dict
        Keys are brain areas, values are dicts with 'pre', 'stim', 'stats'
    amplitude : int
        Stimulation amplitude (µA)
    frequency : int
        Stimulation frequency (Hz)
    show_area_labels : bool
        Whether to show area names on plot
    ax : matplotlib axis, optional
    
    Returns
    -------
    ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(mm_to_inch(DOUBLE_COLUMN_WIDTH), mm_to_inch(30))
        )
    
    color = FREQUENCY_COLORS[frequency]
    params = PAIRED_COMPARISON_PARAMS
    
    for i, area in enumerate(brain_areas):
        if area not in comparison_data:
            continue
        
        data = comparison_data[area]
        pre_vl = data['pre']
        stim_vl = data['stim']
        stats = data['stats']
        
        # Background bar
        ax.bar(
            i, params['y_max'],
            width=params['bar_width'],
            color='lightgrey',
            alpha=params['background_alpha'],
            linewidth=0,
            zorder=0
        )
        
        # Area labels
        if show_area_labels:
            ax.text(
                i,  # Center at position i
                params['y_max'] + 0.02,  # Slightly above the plot
                area,
                rotation=45,
                fontsize=6,
                color='k',
                ha='left',  # Changed from 'right' to 'left'
                va='bottom'
            )
        
        # Paired lines
        ax.plot(
            [i - params['delta'], i + params['delta']],
            [pre_vl, stim_vl],
            color='k',
            linewidth=params['pair_line_width'],
            alpha=params['pair_line_alpha'],
            zorder=1
        )
        
        # Trend line
        y_start = stats['median_pre']
        y_end = stats['median_stim']
        ax.plot(
            [i - params['delta'], i + params['delta']],
            [y_start, y_end],
            color=color,
            linewidth=params['trend_line_width'],
            alpha=1.0,
            zorder=2
        )
        
        # Significance markers
        pval_to_use = stats.get('pval_corrected', stats['pval'])
        add_significance_markers(ax, i, pval_to_use, color, params)
    
    # Format axis
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_xlim(-0.2 - params['delta'], len(brain_areas))
    ax.set_ylim(0, params['y_max'])
    ax.set_yticks(np.linspace(0, 0.8, 5))
    ax.set_xticks([])
    ax.set_xticklabels([])
    remove_top_right_spines(ax)
    
    return ax


def add_significance_markers(ax, x_pos, pval, color, params):
    """
    Add significance markers above a comparison
    
    Parameters
    ----------
    ax : matplotlib axis
    x_pos : float
        X position
    pval : float
        P-value
    color : str
        Marker color
    params : dict
        Plotting parameters
    """
    n_markers = get_significance_level(pval)
    
    if n_markers == 0:
        return
    
    y_pos = params['y_max'] - params['y_sig_offset']
    marker_params = SIGNIFICANCE_MARKER_PARAMS
    
    # Calculate x positions for markers
    if n_markers == 1:
        x_positions = [x_pos]
    else:
        spacing = 0.2 * params['bar_width'] 
        half_width = (n_markers - 1) * spacing / 2
        x_positions = [x_pos - half_width + i * spacing for i in range(n_markers)]
    
    ax.scatter(
        x_positions,
        [y_pos] * n_markers,
        color=color,
        marker=marker_params['marker'],
        s=marker_params['size']**2,  # Convert to area
        alpha=marker_params['alpha'],
        linewidth=marker_params['linewidth'],
        zorder=3
    )


def plot_spatial_decay_panel(
    distances,
    delta_vl,
    fit_results,
    entrained_areas,
    area_distances,
    area_metrics,
    frequency,
    ax_scatter=None,
    use_absolute=True,
    ax_low=None,
    ax_high=None,
    ax_summary=None
):
    """
    Create spatial decay panel with significance indicators and area summary
    
    Parameters
    ----------
    distances : array
        Unit-level distances
    delta_vl : array
        Unit-level ΔVL
    fit_results : dict
        Curve fit results
    entrained_areas : dict
        'low' and 'high' threshold entrained areas
    area_distances : array
        Per-area distances
    area_metrics : array
        Per-area summary metrics
    frequency : int
        Stimulation frequency (Hz)
    ax_scatter : matplotlib axis
        Main scatter plot axis
    ax_low : matplotlib axis
        Low threshold indicator axis
    ax_high : matplotlib axis
        High threshold indicator axis
    ax_summary : matplotlib axis
        Area summary axis
    
    Returns
    -------
    tuple
        (ax_scatter, ax_low, ax_high, ax_summary)
    """
    if ax_scatter is None:
        fig = plt.figure(figsize=(mm_to_inch(60), mm_to_inch(60)))
        gs = GridSpec(3, 1, height_ratios=[0.25, 0.25, 3], hspace=0.0)
        ax_high = fig.add_subplot(gs[0])
        ax_low = fig.add_subplot(gs[1])
        ax_scatter = fig.add_subplot(gs[2])
    
    color = FREQUENCY_COLORS[frequency]
    params = SPATIAL_DECAY_PARAMS
    marker_size = params['scatter_size']**2  # Convert to area
    
    # Top panels: significance indicators
    plot_significance_indicators(
        ax_high, ax_low, area_distances,
        entrained_areas, color, params
    )
    
    # Main scatter plot
    ax_scatter.plot(
        distances, delta_vl,
        'o', color='k',
        markersize=params['scatter_size'],
        alpha=params['scatter_alpha'],
        zorder=1
    )
    
    # Fit line
    ax_scatter.plot(
        fit_results['x_pred'],
        fit_results['y_pred'],
        params['fit_line_style'],
        color=color,
        linewidth=params['fit_line_width'],
        zorder=2
    )
    
    # Format
    ax_scatter.set_xlabel('Distance (mm)', fontsize=7)
    if not use_absolute:
        ax_scatter.set_ylabel('ΔVL', fontsize=7)
    elif use_absolute:
        ax_scatter.set_ylabel('|ΔVL|', fontsize=7)
    ax_scatter.set_xlim(params['xrange'])
    ax_scatter.set_ylim([-0.05, 0.9])
    ax_scatter.set_xticks(params['xticks'])
    ax_scatter.set_yticks(params['yticks_delta'])
    remove_top_right_spines(ax_scatter)
    
    return ax_scatter, ax_low, ax_high


def plot_significance_indicators(
    ax_high, ax_low, distances, entrained_areas, color, params
):
    """
    Plot significance indicator rows
    
    Parameters
    ----------
    ax_high : matplotlib axis
        High threshold (p<0.001) indicator
    ax_low : matplotlib axis
        Low threshold (p<0.01) indicator
    distances : array
        Distances for all areas
    entrained_areas : dict
        Contains 'high' and 'low' lists of distances
    color : str
        Marker color
    params : dict
        Plotting parameters
    """
    marker_size = params['marker_size_top']**2
    
    # High threshold (p<0.001)
    ax_high.scatter(
        distances, np.ones_like(distances),
        color='k', marker='o', s=marker_size,
        alpha=params['marker_alpha_dim']
    )
    ax_high.scatter(
        entrained_areas['high'], np.ones(len(entrained_areas['high'])),
        facecolor=color, edgecolors=color,
        marker='o', s=marker_size,
        linewidth=0.5, alpha=1.0
    )
    format_indicator_axis(ax_high, '0.001', params)
    
    # Low threshold (p<0.01)
    ax_low.scatter(
        distances, np.ones_like(distances),
        color='k', marker='o', s=marker_size,
        alpha=params['marker_alpha_dim']
    )
    ax_low.scatter(
        entrained_areas['low'], np.ones(len(entrained_areas['low'])),
        color=color, marker='o', s=marker_size,
        alpha=1.0
    )
    format_indicator_axis(ax_low, '0.01', params)


def format_indicator_axis(ax, label, params):
    """Format significance indicator axis"""
    ax.set_xlim(params['xrange'])
    ax.set_xticks([])
    ax.set_yticks([1])
    ax.set_yticklabels([label], fontsize=6)
    ax.tick_params(axis='y', which='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
