#!/usr/bin/env python3
"""
Figure 3 Row 2: Panel c
  Left: 3 bar charts (VISp, CA1, RSPd) showing clustering significance
        across sES frequencies (8, 28, 140 Hz) with frequency-specific colours.
  Right: Heatmap summarising clustering significance for all recorded areas.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config.paths import FIGURES_OUTPUT
from config.plotting import (mm_to_inch, apply_nature_style,
                              remove_top_right_spines, FREQUENCY_COLORS)
from fig3_waveform import Figure3ExtendedAnalysis


def plot_freq_bars(area_results, ax, show_ylabel=True):
    """Single bar-chart panel for one area across 3 frequencies."""
    freqs_order = ['sine_8Hz', 'sine_28Hz', 'sine_140Hz']
    freq_nums = [8, 28, 140]
    labels = ['8 Hz', '28 Hz', '140 Hz']

    log_p, n_clust, colors = [], [], []
    for fs, fn in zip(freqs_order, freq_nums):
        if fs in area_results:
            r = area_results[fs]
            lp = -np.log10(r['cluster_test_pval']) if r['cluster_test_pval'] > 0 else 30
            log_p.append(lp)
            n_clust.append(r['n_clusters'])
        else:
            log_p.append(0)
            n_clust.append(0)
        colors.append(FREQUENCY_COLORS[fn])

    bars = ax.bar(labels, log_p, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.4)

    # Significance threshold
    ax.axhline(-np.log10(0.05), color='red', ls='--', alpha=0.6, lw=0.75)

    # Cluster count annotations
    for bar, nc in zip(bars, n_clust):
        if nc > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(nc), ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_ylim(0, max(log_p) * 1.25 if max(log_p) > 0 else 5)
    if show_ylabel:
        ax.set_ylabel('cluster significance\n' + r'$-\log_{10}$(p-value)', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xlabel('sES frequency', fontsize=7)
    remove_top_right_spines(ax)


def plot_heatmap(multi_freq_results, ax):
    """Summary heatmap: areas × frequencies."""
    areas = sorted(multi_freq_results.keys())
    frequencies = [8, 28, 140]

    mat = np.zeros((len(areas), len(frequencies)))
    n_clust_mat = np.zeros_like(mat)

    for i, area in enumerate(areas):
        for j, freq in enumerate(frequencies):
            fs = f'sine_{freq}Hz'
            if fs in multi_freq_results[area]:
                r = multi_freq_results[area][fs]
                mat[i, j] = min(-np.log10(r['cluster_test_pval']), 10)
                n_clust_mat[i, j] = r['n_clusters']

    im = ax.imshow(mat, cmap='Reds', aspect='auto', vmin=0, vmax=10)

    ax.set_xticks(range(len(frequencies)))
    ax.set_xticklabels([f'{f} Hz' for f in frequencies], fontsize=6)
    ax.set_yticks(range(len(areas)))
    ax.set_yticklabels(areas, fontsize=5)
    ax.set_xlabel('sES frequency', fontsize=7)

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='8%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(r'$-\log_{10}$(p-value)', rotation=270, labelpad=10, fontsize=7)
    cbar.set_ticks([0, 2, 4, 6, 8, 10])
    cbar.set_ticklabels(['0', '2', '4', '6', '8', '≥10'])
    cbar.ax.tick_params(labelsize=5)

    # Significance threshold line on colorbar
    sig_pos = -np.log10(0.05)
    cbar.ax.axhline(sig_pos, color='green', ls='--', lw=0.75)


def main():
    apply_nature_style()

    # Run analysis
    analyzer = Figure3ExtendedAnalysis()
    analyzer.load_and_filter_data()
    multi_freq_results = analyzer.analyze_multi_frequency_clustering()

    # --- Build Row 2 figure ---
    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(55)))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1.3], wspace=0.4,
                  left=0.08, right=0.95, top=0.88, bottom=0.22)

    target_areas = ['VISp', 'CA1', 'RSPd']
    for idx, area in enumerate(target_areas):
        ax = fig.add_subplot(gs[idx])
        if area in multi_freq_results:
            plot_freq_bars(multi_freq_results[area], ax,
                           show_ylabel=(idx == 0))
        ax.set_title(area, fontsize=8)

    # Heatmap
    ax_hm = fig.add_subplot(gs[3])
    plot_heatmap(multi_freq_results, ax_hm)

    fig.text(0.01, 0.94, 'c', fontsize=12, fontweight='bold')

    for ext in ['pdf', 'png']:
        out = os.path.join(FIGURES_OUTPUT, f'figure3_row2.{ext}')
        fig.savefig(out, dpi=600, bbox_inches='tight')
        print(f'Saved {out}')

    plt.close(fig)
    return multi_freq_results, analyzer


if __name__ == '__main__':
    main()
