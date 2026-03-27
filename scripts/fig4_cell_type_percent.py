"""
Figure 4d – Percent of RS and FS units showing a transient decrease (D)
or increase (I) of firing rate, across stimulation protocols at 5 µA.

Bar plot with mean ± SE across animals.  Pairwise Mann-Whitney U tests
across frequencies for each cell type, FDR-corrected.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config.paths import FIGURES_OUTPUT
from config.experiments import FREQUENCIES, CELL_TYPE_COLORS
from config.plotting import FREQUENCY_COLORS, apply_nature_style, remove_top_right_spines
from src.data_loading_transient import load_transient_data
from src.analysis_transient import (
    compute_transient_percentages, pairwise_mannwhitney_across_freqs,
)
from src.statistics import pval_to_stars


def main():
    apply_nature_style()
    df = load_transient_data()

    amplitude = 5
    freqs = FREQUENCIES  # 8, 28, 140
    subjects = df['mouse'].unique()
    freq_pairs = [(freqs[0], freqs[1]),
                  (freqs[0], freqs[2]),
                  (freqs[1], freqs[2])]

    # ── Compute per-subject percentages ───────────────────────────────────
    perc_by_freq = {}
    all_means = pd.DataFrame()
    all_se = pd.DataFrame()

    for freq in freqs:
        perc = compute_transient_percentages(df, freq, amplitude, subjects)
        perc_by_freq[freq] = perc

        mean_perc = perc.astype(float).mean()
        se_perc = perc.astype(float).std() / np.sqrt(len(subjects))

        # Reshape into MultiIndex columns (freq, cell_type) indexed by D/I
        mean_df = mean_perc.unstack()
        se_df = se_perc.unstack()
        mean_df.columns = pd.MultiIndex.from_product([[freq], mean_df.columns])
        se_df.columns = pd.MultiIndex.from_product([[freq], se_df.columns])

        all_means = pd.concat([all_means, mean_df], axis=1)
        all_se = pd.concat([all_se, se_df], axis=1)

    # ── Statistics ────────────────────────────────────────────────────────
    pvals = pairwise_mannwhitney_across_freqs(perc_by_freq, freq_pairs)

    # ── Bar positions ─────────────────────────────────────────────────────
    transient_types = ['D', 'I']
    bar_width = 0.1
    condition_gap = 0.05
    block_gap = 0.3

    positions = {t: [] for t in transient_types}
    current_pos = 0
    for t in transient_types:
        for _ in freqs:
            fs_pos = current_pos
            rs_pos = current_pos + bar_width
            positions[t].append((fs_pos, rs_pos))
            current_pos += 2 * bar_width + condition_gap
        current_pos += block_gap

    flat_positions = []
    for t in transient_types:
        for pair in positions[t]:
            flat_positions.extend(pair)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_idx = 0
    for t in transient_types:
        for ci, freq in enumerate(freqs):
            for j, cell in enumerate(['FS', 'RS']):
                value = all_means.loc[t, (freq, cell)]
                err = all_se.loc[t, (freq, cell)]
                pos = flat_positions[bar_idx]
                ax.bar(pos, value, width=bar_width,
                       color=CELL_TYPE_COLORS[cell],
                       edgecolor=None, linewidth=0,
                       label=cell if bar_idx == 0 else '')
                ax.errorbar(pos, value, yerr=err,
                            fmt='none', ecolor='black',
                            elinewidth=1.2, capsize=3)
                bar_idx += 1

    # ── Significance annotations ──────────────────────────────────────────
    line_offset = 5
    line_height_step = 2.5
    annotation_offsets = {
        t: {(cell, i): 0 for cell in ['FS', 'RS'] for i in range(len(freqs))}
        for t in transient_types
    }

    for t in transient_types:
        for cell in ['RS', 'FS']:
            color_stat = CELL_TYPE_COLORS[cell]
            for f1, f2 in freq_pairs:
                idx1 = list(freqs).index(f1)
                idx2 = list(freqs).index(f2)
                col_idx = 0 if cell == 'FS' else 1
                x1 = positions[t][idx1][col_idx]
                x2 = positions[t][idx2][col_idx]

                try:
                    y1 = all_means.loc[t, (f1, cell)]
                    y2 = all_means.loc[t, (f2, cell)]
                except KeyError:
                    continue
                topy = max(y1, y2)

                # Match original stacking for increase block
                if t == 'I' and (f1, f2) != (freqs[0], freqs[1]):
                    top_all = all_means.loc[t].max()
                    topy = top_all - 3 if cell == 'FS' else top_all + 4

                key1 = (cell, idx1)
                key2 = (cell, idx2)
                used = max(annotation_offsets[t].get(key1, 0),
                           annotation_offsets[t].get(key2, 0))
                y = topy + line_offset + used * line_height_step

                stars = pval_to_stars(pvals.get((t, cell, f1, f2), np.nan))
                if stars is not None:
                    ax.plot([x1, x1, x2, x2],
                            [y, y + 0.2, y + 0.2, y],
                            lw=1.0, color=color_stat)
                    ax.text((x1 + x2) / 2, y + 0.25, stars,
                            ha='center', va='bottom', fontsize=12,
                            color=color_stat)
                    annotation_offsets[t][key1] = used + 1
                    annotation_offsets[t][key2] = used + 1

    # ── X-axis labels ─────────────────────────────────────────────────────
    xtick_positions = []
    xtick_labels = []
    for t in transient_types:
        for ci, freq in enumerate(freqs):
            center = np.mean(positions[t][ci])
            xtick_positions.append(center)
            xtick_labels.append(f'{freq} Hz')

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=16, fontname='Arial')
    ax.set_ylabel('Percent of Units (%)', size=16,
                   fontweight='normal', fontname='Arial')
    ax.set_ylim(0, max(30, ax.get_ylim()[1]))
    ax.tick_params(axis='y', labelsize=16)
    for label in ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontweight('normal')
    remove_top_right_spines(ax)

    plt.tight_layout()
    os.makedirs(os.path.join(FIGURES_OUTPUT, 'fig4'), exist_ok=True)
    out = os.path.join(FIGURES_OUTPUT, 'fig4', 'figure4d_cell_type_percent.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
