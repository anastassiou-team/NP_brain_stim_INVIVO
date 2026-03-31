"""
Figure 4a – Distribution of transient effect durations across protocols.

One histogram per frequency (8, 28, 140 Hz) at 5 µA.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from config.paths import FIGURES_OUTPUT
from config.experiments import FREQUENCIES
from config.plotting import FREQUENCY_COLORS, apply_nature_style, remove_top_right_spines
from src.data_loading_transient import load_transient_data, select_units


def main():
    apply_nature_style()
    df = load_transient_data()

    amplitude = 5
    freqs = FREQUENCIES  # 8, 28, 140 Hz – left to right

    fig, axes = plt.subplots(1, len(freqs), figsize=(10, 5))

    for c, freq in enumerate(freqs):
        ax = axes[c]
        mask = select_units(df, freq, amplitude)
        sub = df.loc[mask]

        onsets = sub['transient_onset'].values
        offsets = sub['transient_offset'].values
        valid = ~np.isnan(onsets) & ~np.isnan(offsets)
        duration = offsets[valid] - onsets[valid]

        color = FREQUENCY_COLORS[freq]
        bins = np.linspace(0, 5, 25)
        ax.hist(duration, bins=bins, color=color, edgecolor='k', alpha=0.7)

        median_dur = np.median(duration)
        ax.axvline(median_dur, color='k', linestyle='--', linewidth=2,
                   label=f'median = {median_dur:.2f}s')
        ax.text(median_dur + 0.1, 300,
                f'median = {median_dur:.2f}s',
                size=24, fontweight='normal', fontname='Arial')

        ax.set_xlim([0, 2.5])
        ax.set_ylim([0, 350])
        ax.set_xlabel('duration / s', size=24, fontweight='normal', fontname='Arial')
        ax.set_ylabel('')
        ax.tick_params(axis='both', labelsize=24)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Arial')
            label.set_fontweight('normal')
        remove_top_right_spines(ax)

    plt.tight_layout()
    os.makedirs(os.path.join(FIGURES_OUTPUT, 'fig4'), exist_ok=True)
    out = os.path.join(FIGURES_OUTPUT, 'fig4', 'figure4a_transient_duration.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
