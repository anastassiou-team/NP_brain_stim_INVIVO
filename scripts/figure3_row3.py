#!/usr/bin/env python3
"""
Figure 3 Row 3: Panel d
  Left: 3 bar charts (VISp, CA1, RSPd) showing effect sizes (η²) for
        associations between EAP waveform properties and cluster membership
        at 28 Hz.  Orange = FDR-corrected p < 0.05, gray = non-significant.
  Right: Heatmap of -log10(FDR-corrected p) across all areas and waveform
         properties.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config.paths import FIGURES_OUTPUT
from config.plotting import mm_to_inch, apply_nature_style, remove_top_right_spines
from fig3_waveform import Figure3ExtendedAnalysis


FEATURE_LABELS = ['duration', 'halfwidth', 'rep_slope', 'REP']


def plot_waveform_bars(results, ax, show_ylabel=True):
    """Bar chart of effect sizes for one area, coloured by significance."""
    features = results['waveform_features']
    effect_sizes = results['waveform_effect_sizes']
    p_corr = results['waveform_p_corrected']

    labels = [f.replace('waveform_', '') for f in features]

    colors = ['darkorange' if p < 0.05 else 'gray'
              for p in p_corr]

    bars = ax.bar(labels, effect_sizes, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=0.4)

    # Small-effect threshold
    ax.axhline(0.2, color='gray', ls=':', alpha=0.5, lw=0.75)

    # Asterisks for significance
    for bar, p in zip(bars, p_corr):
        if p < 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    '*', ha='center', va='bottom', fontsize=9,
                    fontweight='bold')

    if show_ylabel:
        ax.set_ylabel('effect size', fontsize=7)
    ax.set_xlabel('EAP waveform properties', fontsize=7)
    ax.set_ylim(0, 0.35)
    ax.tick_params(labelsize=6)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    remove_top_right_spines(ax)


def plot_waveform_heatmap(waveform_results, ax):
    """Heatmap of -log10(FDR p) for all areas × waveform features."""
    areas = sorted(waveform_results.keys())
    features = FEATURE_LABELS

    mat = np.zeros((len(areas), len(features)))

    for i, area in enumerate(areas):
        r = waveform_results[area]
        wf = r['waveform_features']
        pvals = r.get('waveform_p_corrected', r['waveform_p_values'])
        for j, feat in enumerate(features):
            full = f'waveform_{feat}'
            if full in wf:
                idx = wf.index(full)
                p = pvals[idx]
                mat[i, j] = min(-np.log10(p) if p > 0 else 5, 5)

    im = ax.imshow(mat, cmap='Reds', aspect='auto', vmin=0, vmax=5)

    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=6)
    ax.set_yticks(range(len(areas)))
    ax.set_yticklabels(areas, fontsize=5)
    ax.set_xlabel('EAP waveform properties', fontsize=7)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='8%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(r'$-\log_{10}$(p-value)', rotation=270, labelpad=10,
                   fontsize=7)
    cbar.set_ticks([0, 1, 2, 3, 4, 5])
    cbar.set_ticklabels(['0', '1', '2', '3', '4', '≥5'])
    cbar.ax.tick_params(labelsize=5)


def main():
    apply_nature_style()

    # Run analysis
    analyzer = Figure3ExtendedAnalysis()
    analyzer.load_and_filter_data()
    waveform_results = analyzer.analyze_waveform_cluster_relationship()

    # --- Build Row 3 figure ---
    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(55)))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1.3], wspace=0.4,
                  left=0.08, right=0.95, top=0.88, bottom=0.28)

    target_areas = ['VISp', 'CA1', 'RSPd']
    for idx, area in enumerate(target_areas):
        ax = fig.add_subplot(gs[idx])
        if area in waveform_results:
            plot_waveform_bars(waveform_results[area], ax,
                               show_ylabel=(idx == 0))
        ax.set_title(area, fontsize=8)

    # Heatmap
    ax_hm = fig.add_subplot(gs[3])
    plot_waveform_heatmap(waveform_results, ax_hm)

    fig.text(0.01, 0.94, 'd', fontsize=12, fontweight='bold')

    for ext in ['pdf', 'png']:
        out = os.path.join(FIGURES_OUTPUT, f'figure3_row3.{ext}')
        fig.savefig(out, dpi=600, bbox_inches='tight')
        print(f'Saved {out}')

    plt.close(fig)
    return waveform_results


if __name__ == '__main__':
    main()
