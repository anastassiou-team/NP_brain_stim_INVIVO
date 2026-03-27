# Sinusoidal electrical stimulation effects across the entire brain

Code and data to reproduce the figures in:

Irene Rembado, Soo Yeun Lee, Lydia C. Marks, Leslie D. Claar, Areg Peltekian,
Christof Koch and Costas A. Anastassiou (2026) "Selective gating of neural modulation through frequency- and behavior-dependent modes
during cortical electrical stimulation" (under review)

## Figures

Pre-generated composite figures are included in `figures/`.
To regenerate any panel, run the corresponding script from the repo root.

| Figure | Script | Output |
|--------|--------|--------|
| Fig. 2 | `python scripts/fig2_entrainment.py` | `figures/figure2_complete.png` |
| Fig. 3, row 1 | `python scripts/figure3_row1.py` | `figures/figure3_row1.png` |
| Fig. 3, row 2 | `python scripts/figure3_row2.py` | `figures/figure3_row2.png` |
| Fig. 3, row 3 | `python scripts/figure3_row3.py` | `figures/figure3_row3.png` |
| Fig. 4a | `python scripts/fig4_duration.py` | `figures/figure4a_transient_duration.png` |
| Fig. 4b | `python scripts/fig4_brain_wide_fr.py` | `figures/figure4b_brain_wide_fr.png` |
| Fig. 4c | `python scripts/fig4_waveform_hist.py` | `figures/figure4c_waveform_histogram.png` |
| Fig. 4d | `python scripts/fig4_cell_type_percent.py` | `figures/figure4d_cell_type_percent.png` |
| Fig. 4e | `python scripts/fig4_zscore_distance.py` | `figures/figure4e_zscore_distance.png` |
| Fig. S4 | `python scripts/figS4_mmr_entrainment.py` | `figures/fig2_mmr/figure2_mmr_complete.png` |
| Fig. S (VL-MMR) | `python scripts/figS_vl_mmr_corr.py` | `figures/fig2_vl_vs_mmr/` |

To regenerate all figures at once:

```bash
python run_all.py              # all figures
python run_all.py fig2 fig3    # only matching scripts
```

## Installation

```bash
conda env create -f environment.yml
conda activate neuropixels
```

## Usage

Run any figure script from the repository root:

```bash
python scripts/fig2_entrainment.py
```

Output PNGs are written to `figures/`.

## Repository structure

```
├── README.md
├── LICENSE                  MIT
├── environment.yml          Conda environment specification
├── run_all.py               Regenerate every figure (with optional filter)
├── structures.json          Allen Brain Atlas structure definitions
├── manifest.json            Allen Brain Atlas data manifest
├── config/                  Configuration (paths, brain areas, plotting, experiments)
├── src/                     Core analysis library (data loading, statistics, fitting, plotting)
├── scripts/                 One script per figure (see table above)
├── data/
│   ├── units/               Per-area unit CSVs  (freq × amplitude × area)
│   ├── waveform_features/   Waveform feature CSVs
│   ├── unified_data_VL_angle_MMR.csv   All units merged (VL, angle, MMR)
│   ├── unified_data_transient.csv      Transient firing-rate data (Fig. 4)
│   └── connectivity/        Allen Mouse Brain connectivity matrices
└── figures/                 Generated figures
```

## Data

Unit-level CSVs: `units_{frequency}Hz_{amplitude}_{brain_area}.csv`
Waveform features: `{brain_area}_wf_features_only.csv`
Unified datasets: `unified_data_VL_angle_MMR.csv`, `unified_data_transient.csv`

## License

MIT — see [LICENSE](LICENSE).
