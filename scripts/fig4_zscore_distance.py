"""
Figure 4e – Absolute z-scored firing-rate change vs distance from sES
electrode, split by FS (blue) and RS (red).

One panel per frequency (8, 28, 140 Hz) at 5 µA, 100 ms window.
Statistics: Mann-Whitney FS vs RS at each distance bin with BH correction.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from config.paths import FIGURES_OUTPUT
from config.experiments import (
    FREQUENCIES, FIGURE4_DISTANCE_BINS, CELL_TYPE_COLORS,
)
from config.plotting import FREQUENCY_COLORS, apply_nature_style, remove_top_right_spines
from src.data_loading_transient import load_transient_data
from src.analysis_transient import (
    compute_zscore_by_distance, compare_fs_rs_at_bins,
)
from src.statistics import pval_to_stars


def main():
    apply_nature_style()
    df = load_transient_data()

    amplitude = 5
    eval_window = '100ms'
    freqs = FREQUENCIES  # 8, 28, 140
    bins = FIGURE4_DISTANCE_BINS
    ylim_plot = 10

    fig, axes = plt.subplots(1, len(freqs), figsize=(15, 3))

    for i, freq in enumerate(freqs):
        ax = axes[i]

        # Compute binned z-scores for FS and RS
        fs = compute_zscore_by_distance(
            df, freq, amplitude, 'FS', bins, eval_window)
        rs = compute_zscore_by_distance(
            df, freq, amplitude, 'RS', bins, eval_window)

        # Plot RS
        ax.plot(rs['distances'], rs['means'], color=CELL_TYPE_COLORS['RS'],
                label='RS')
        ax.errorbar(rs['distances'], rs['means'], yerr=rs['se'],
                    fmt='o', color=CELL_TYPE_COLORS['RS'],
                    ecolor=CELL_TYPE_COLORS['RS'],
                    elinewidth=1, capsize=5, capthick=1)

        # Plot FS
        ax.plot(fs['distances'], fs['means'], color=CELL_TYPE_COLORS['FS'],
                label='FS')
        ax.errorbar(fs['distances'], fs['means'], yerr=fs['se'],
                    fmt='o', color=CELL_TYPE_COLORS['FS'],
                    ecolor=CELL_TYPE_COLORS['FS'],
                    elinewidth=1, capsize=5, capthick=1)

        # Statistics: FS vs RS at each bin
        stats = compare_fs_rs_at_bins(
            df, freq, amplitude, bins, eval_window)
        for bi, pval_corr in enumerate(stats['bin_pvals_corrected']):
            stars = pval_to_stars(pval_corr)
            if stars is not None:
                y_annot = max(fs['means'][bi], rs['means'][bi]) + \
                          max(fs['se'][bi], rs['se'][bi]) + 0.3
                ax.text(fs['distances'][bi] - 0.2, y_annot, stars,
                        color='k', fontsize=20)

        ax.set_xlim(0, len(bins) - 0.5)
        ax.set_ylim(0, ylim_plot)
        ax.set_xticks(bins[1:])
        ax.set_xlabel('distance from sES / mm', size=16,
                       fontweight='normal', fontname='Arial')
        ax.set_ylabel('spike rate (|z-score|)', size=16,
                       fontweight='normal', fontname='Arial')
        ax.tick_params(axis='both', labelsize=16)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Arial')
            label.set_fontweight('normal')
        remove_top_right_spines(ax)

        # Print slope interaction p-value
        print(f'{freq} Hz – OLS slope interaction p = '
              f'{stats["ols_interaction_pval"]:.4g}')

    plt.tight_layout()
    os.makedirs(os.path.join(FIGURES_OUTPUT, 'fig4'), exist_ok=True)
    out = os.path.join(FIGURES_OUTPUT, 'fig4', 'figure4e_zscore_distance.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
