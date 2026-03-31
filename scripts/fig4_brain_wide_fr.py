"""
Figure 4b – Average firing-rate change in a 100 ms window after sES onset
compared to baseline, across brain areas and protocols (5 µA).

Three rows (one per frequency: 8, 28, 140 Hz), each showing paired
pre → stim lines for every unit, with median trend lines and
FDR-corrected significance dots.

Generates two versions:
  - figure4b_brain_wide_fr_ttest.png  (one-sample t-test on paired diffs)
  - figure4b_brain_wide_fr_mwu.png    (Mann-Whitney U, unpaired)

Both use the same 10s pre-stim baseline, BH-FDR correction, and
significance thresholds.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from config.paths import FIGURES_OUTPUT
from config.experiments import FREQUENCIES, FIGURE4_AREAS
from config.plotting import FREQUENCY_COLORS, apply_nature_style, remove_top_right_spines
from src.data_loading_transient import load_transient_data, select_units
from src.analysis_transient import analyze_all_areas_transient
from src.statistics import get_significance_level


def plot_figure4b(df, test, out_path):
    """Generate one version of figure 4b with the specified statistical test."""
    apply_nature_style()

    amplitude = 5
    # 10s pre-stim baseline with 100ms stim-on window
    pre_col = 'meanFR_pre_10s'
    stim_col = 'meanFR_stimOn_01s'
    freqs = FREQUENCIES  # 8, 28, 140
    areas = FIGURE4_AREAS
    ymax = 230

    fig, axes = plt.subplots(len(freqs), 1, figsize=(16, 8))

    for f, freq in enumerate(freqs):
        ax = axes[f]
        color = FREQUENCY_COLORS[freq]

        # Statistical test + FDR correction across all areas
        results = analyze_all_areas_transient(
            df, freq, amplitude, areas, '100ms',
            pre_col=pre_col, stim_col=stim_col, test=test)

        clus_ = 0
        for a, area in enumerate(areas):
            mask = select_units(df, freq, amplitude, area=area)
            sub = df.loc[mask]

            fr_pre = sub[pre_col].values
            fr_stim = sub[stim_col].values

            # Background bar
            ax.bar(clus_ + 1, ymax, width=2.6, color='lightgrey', alpha=0.1)

            # Area labels on top row only
            if f == 0:
                ax.text(clus_ + 1, ymax + 3, area,
                        rotation=45, color='k', size=16,
                        fontweight='normal', fontname='Arial')

            # Paired lines (pre → stim for each unit)
            x_pre = np.random.uniform(-0.2, 0.2, len(fr_pre)) + clus_
            x_stim = np.random.uniform(-0.2, 0.2, len(fr_stim)) + 2 + clus_
            ax.plot([x_pre, x_stim], [fr_pre, fr_stim],
                    color='k', alpha=0.3, linewidth=0.5)

            # Median trend line
            ax.plot([clus_, clus_ + 2],
                    [np.median(fr_pre), np.median(fr_stim)],
                    color=color, linewidth=2)

            # Significance dots (FDR-corrected)
            if area in results:
                pval = results[area]['pval_corrected']
                n_dots = get_significance_level(pval)
                if n_dots > 0:
                    offsets = {1: 0.75, 2: 0.5, 3: 0.1, 4: -0.3}
                    x_off = offsets.get(n_dots, 0)
                    ax.text(clus_ + x_off, ymax - 10,
                            '●' * n_dots, color=color)

            ax.set_ylabel('spike rate / Hz', multialignment='center',
                          size=16, fontweight='normal', fontname='Arial')
            ax.set_xticks([])
            ax.tick_params(axis='y', labelsize=16)
            for label in ax.get_yticklabels():
                label.set_fontname('Arial')
                label.set_fontweight('normal')
            ax.set_xlim([-2, len(areas) * 3 + 2])
            ax.set_ylim([0, ymax])
            remove_top_right_spines(ax)
            clus_ += 3

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'Saved {out_path}')
    plt.close(fig)


def main():
    df = load_transient_data()
    fig_dir = os.path.join(FIGURES_OUTPUT, 'fig4')

    # Fix random seed so all figures have identical jittered lines
    for test, suffix in [('ttest', '_ttest'), ('mwu', '_mwu'), ('wilcoxon', '_wilcoxon')]:
        np.random.seed(42)
        out = os.path.join(fig_dir, f'figure4b_brain_wide_fr{suffix}.png')
        plot_figure4b(df, test, out)


if __name__ == '__main__':
    main()
