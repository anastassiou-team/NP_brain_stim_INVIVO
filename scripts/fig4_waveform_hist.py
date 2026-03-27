"""
Figure 4c (partial) – Spike-waveform duration histogram for FS / RS
classification.

Shows the bimodal distribution with the 0.4 ms threshold.
Only VIS cortex units at 140 Hz, 5 µA.

NOTE: The full panel 4c also includes single-trial rasters, PSTHs, and
example spike waveforms that require trial-level data not in the
transient CSV.  Those sub-panels are not generated here.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt

from config.paths import FIGURES_OUTPUT
from config.experiments import FS_THRESHOLD
from config.plotting import apply_nature_style, remove_top_right_spines
from src.data_loading_transient import load_transient_data, select_units


def main():
    apply_nature_style()
    df = load_transient_data()

    freq = 140
    amplitude = 5
    area = 'VIS'

    mask_fs = select_units(df, freq, amplitude, cell_type='FS',
                           area=area, apply_min_fr=False)
    mask_rs = select_units(df, freq, amplitude, cell_type='RS',
                           area=area, apply_min_fr=False)

    dur_fs = df.loc[mask_fs, 'waveform_duration']
    dur_rs = df.loc[mask_rs, 'waveform_duration']

    fig = plt.figure(figsize=(3, 5))
    plt.hist(dur_rs, color='red', bins=35)
    plt.hist(dur_fs, color='blue', bins=25)
    plt.xlim([0, 1.1])
    plt.xticks([0, FS_THRESHOLD, 0.8], fontsize=20, fontname='Arial')
    plt.yticks(fontsize=20, fontname='Arial')
    plt.vlines(x=FS_THRESHOLD, ymin=0, ymax=40,
               linestyle='--', linewidth=2, color='k')
    plt.text(0.20, 35, 'FS', color='b', fontsize=20)
    plt.text(0.9, 35, 'RS', color='r', fontsize=20)
    remove_top_right_spines(plt.gca())
    plt.tight_layout()

    out = os.path.join(FIGURES_OUTPUT, 'figure4c_waveform_histogram.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
