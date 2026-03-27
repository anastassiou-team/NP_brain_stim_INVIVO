"""
Figure 4b – Average firing-rate change in a 100 ms window after sES onset
compared to baseline, across brain areas and protocols (5 µA).

Three rows (one per frequency: 8, 28, 140 Hz), each showing paired
pre → stim lines for every unit, with median trend lines and
FDR-corrected significance dots.

NOTE: The significance dots use Mann-Whitney U tests (matching the
original analysis code).  The figure legend in the manuscript refers to
"paired t-test on within-unit firing rate differences" — verify and
reconcile before submission.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from config.paths import FIGURES_OUTPUT
from config.experiments import FREQUENCIES, FIGURE4_AREAS
from config.plotting import FREQUENCY_COLORS, apply_nature_style, remove_top_right_spines
from src.data_loading_transient import load_transient_data, select_units, get_fr_columns
from src.statistics import fdr_correct, get_significance_level


def main():
    apply_nature_style()
    df = load_transient_data()

    amplitude = 5
    eval_window = '100ms'
    pre_col, stim_col, _ = get_fr_columns(eval_window)
    freqs = FREQUENCIES  # 8, 28, 140
    areas = FIGURE4_AREAS
    ymax = 230

    fig, axes = plt.subplots(len(freqs), 1, figsize=(16, 8))

    for f, freq in enumerate(freqs):
        ax = axes[f]
        color = FREQUENCY_COLORS[freq]

        clus_ = 0
        all_pvalues = []

        for a, area in enumerate(areas):
            mask = select_units(df, freq, amplitude, area=area)
            sub = df.loc[mask]

            fr_pre = sub[pre_col].values
            fr_stim = sub[stim_col].values

            # Mann-Whitney U test
            if len(fr_pre) > 0 and len(fr_stim) > 0:
                _, pval = mannwhitneyu(fr_pre, fr_stim)
            else:
                pval = np.nan
            all_pvalues.append(pval)

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

        # FDR-corrected significance dots
        corrected = fdr_correct(all_pvalues)
        clus_ = 0
        dot = '●'
        for a, area in enumerate(areas):
            n_dots = get_significance_level(corrected[a])
            if n_dots > 0:
                # Position adjustment matches original code
                offsets = {1: 0.75, 2: 0.5, 3: 0.1, 4: -0.3}
                x_off = offsets.get(n_dots, 0)
                ax.text(clus_ + x_off, ymax - 10,
                        dot * n_dots, color=color)
            clus_ += 3

    plt.tight_layout()
    out = os.path.join(FIGURES_OUTPUT, 'figure4b_brain_wide_fr.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
