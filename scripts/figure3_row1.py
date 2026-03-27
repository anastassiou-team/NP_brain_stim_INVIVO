#!/usr/bin/env python3
"""
Figure 3 Row 1: Panels a + b
  Panel a: 2x3 grid of VL scatter plots (VISp, CA1, RSPd / MOp, MOs, CL)
  Panel b: Bubble chart of clustering significance vs distance
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from statsmodels.stats.multitest import multipletests

from config.paths import FIGURES_OUTPUT
from config.plotting import mm_to_inch, apply_nature_style, remove_top_right_spines
from config.experiments import cty_colors_
from fig3_clustering import Figure3ClusteringAnalysis


def plot_scatter_panel(results, ax, show_xlabel=True, show_ylabel=True):
    """Plot a single VL scatter panel matching the paper exactly."""
    area = results['area']
    area_data = results['area_data']
    labels = results['cluster_labels']
    n_clusters = results['n_clusters']
    p_val = results['cluster_test_pval']

    vl_pre = area_data['VL_pre'].values
    vl_stim = area_data['VL_stimOn'].values

    # Cluster colours matching the paper (purple, green, pink/red)
    colors = [cty_colors_[i % len(cty_colors_)] for i in range(n_clusters)]

    for cid in range(n_clusters):
        mask = labels == cid
        ax.scatter(vl_pre[mask], vl_stim[mask],
                   c=[colors[cid]], alpha=0.6, s=10, edgecolors='none',
                   zorder=2, rasterized=True)

    # Axis limits – symmetric, based on data
    lo = min(vl_pre.min(), vl_stim.min()) * 0.95
    hi = max(vl_pre.max(), vl_stim.max()) * 1.05
    lo = max(lo, 0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # Diagonal identity line + offset guides
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, lw=0.5, zorder=1)
    offset = 0.1
    ax.plot([lo, hi - offset], [lo + offset, hi], '--', color='gray',
            alpha=0.3, lw=0.5, zorder=1)

    # Matched ticks
    ax.locator_params(axis='both', nbins=4)
    yticks = ax.get_yticks()
    valid = yticks[(yticks >= lo) & (yticks <= hi)]
    if len(valid) > 0:
        ax.set_xticks(valid)
        ax.set_yticks(valid)

    ax.set_aspect('equal', adjustable='box')

    # Title: area name on line 1, n and p on line 2
    ax.set_title(f'{area}\nn={len(area_data)}, p={p_val:.1e}', fontsize=8)

    if show_xlabel:
        ax.set_xlabel('VL no stim', fontsize=7)
    if show_ylabel:
        ax.set_ylabel('VL sES', fontsize=7)

    remove_top_right_spines(ax)
    ax.tick_params(labelsize=6)


def plot_bubble_panel(all_results, data, target_condition, ax):
    """Panel b: significance vs distance bubble chart."""
    freq, amp = target_condition
    target_data = data[
        (data['stim_freq'] == freq) &
        (data['stim_current'] == amp)
    ]

    areas, distances, p_values, n_units_list = [], [], [], []
    for area, res in all_results.items():
        ad = target_data[target_data['area_main'] == area]
        if len(ad) > 0:
            areas.append(area)
            distances.append(ad['distance_peakch_stim_tip'].mean())
            p_values.append(res['cluster_test_pval'])
            n_units_list.append(res['n_units'])

    distances = np.array(distances)
    p_values = np.array(p_values)
    n_units = np.array(n_units_list)

    # FDR correction
    _, p_corr, _, _ = multipletests(p_values, method='fdr_bh')

    neg_log_p = -np.log10(p_corr)
    sig_mask = p_corr < 0.01

    # Bubble size proportional to unit count (scale for visibility)
    sizes = n_units / 3

    # Significant = orange, non-significant = gray
    ax.scatter(distances[sig_mask], neg_log_p[sig_mask],
               s=sizes[sig_mask], c='darkorange', alpha=0.8,
               edgecolors='black', linewidth=0.3, zorder=3)
    ax.scatter(distances[~sig_mask], neg_log_p[~sig_mask],
               s=sizes[~sig_mask], c='lightgray', alpha=0.7,
               edgecolors='black', linewidth=0.3, zorder=2)

    # Significance threshold lines
    ax.axhline(-np.log10(0.01), color='red', ls='--', alpha=0.6, lw=0.75)
    ax.axhline(-np.log10(0.001), color='red', ls='--', alpha=0.6, lw=0.75)

    # Label significant areas
    for i in range(len(areas)):
        if p_corr[i] < 0.01:
            ax.annotate(areas[i], (distances[i], neg_log_p[i]),
                        xytext=(3, 3), textcoords='offset points',
                        fontsize=6, ha='left', va='bottom')

    ax.set_xlabel('distance from sES / mm', fontsize=7)
    ax.set_ylabel(r'$-\log_{10}$(p-value)', fontsize=7)
    remove_top_right_spines(ax)
    ax.tick_params(labelsize=6)


def main():
    apply_nature_style()

    # Run analysis
    analyzer = Figure3ClusteringAnalysis()
    analyzer.load_and_filter_data()
    all_results = analyzer.analyze_all_areas_clustering()

    # --- Build Row 1 figure ---
    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(85)))

    # Panel a (left ~65%) : 2x3 scatter grid
    # Panel b (right ~35%): bubble chart
    gs = GridSpec(1, 2, width_ratios=[2.2, 1], wspace=0.35,
                  left=0.07, right=0.97, top=0.88, bottom=0.13)

    # Panel a: 2x3 sub-grid
    gs_a = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0],
                                   hspace=0.55, wspace=0.35)

    panel_a_areas = ['VISp', 'CA1', 'RSPd', 'MOp', 'MOs', 'CL']
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for idx, area in enumerate(panel_a_areas):
        row, col = positions[idx]
        ax = fig.add_subplot(gs_a[row, col])
        if area in all_results:
            plot_scatter_panel(
                all_results[area], ax,
                show_xlabel=(row == 1),
                show_ylabel=(col == 0),
            )
        else:
            ax.text(0.5, 0.5, f'{area}\n(no data)', ha='center',
                    va='center', fontsize=8, transform=ax.transAxes)
            ax.axis('off')

    # Panel a label
    fig.text(0.01, 0.94, 'a', fontsize=12, fontweight='bold')

    # Panel b: bubble chart
    ax_b = fig.add_subplot(gs[1])
    plot_bubble_panel(all_results, analyzer.data,
                      analyzer.target_condition, ax_b)
    fig.text(0.62, 0.94, 'b', fontsize=12, fontweight='bold')

    # Save
    for ext in ['pdf', 'png']:
        out = os.path.join(FIGURES_OUTPUT, f'figure3_row1.{ext}')
        fig.savefig(out, dpi=600, bbox_inches='tight')
        print(f'Saved {out}')

    plt.close(fig)
    return all_results, analyzer


if __name__ == '__main__':
    main()
