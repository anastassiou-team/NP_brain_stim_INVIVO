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
| Fig. 3, row 1 | `python scripts/fig3_clustering.py` | `figures/figure3_clustering_NEW_LAYOUT.png` |
| Fig. 3, rows 2-3 | `python scripts/fig3_waveform.py` | `figures/figure3_complete_analysis.png` |
| Fig. S4 | `python scripts/figS4_mmr_entrainment.py` | `figures/fig2_mmr/figure2_mmr_complete.png` |
| Fig. S (VL-MMR) | `python scripts/figS_vl_mmr_corr.py` | `figures/fig2_vl_vs_mmr/` |

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
├── structures.json          Allen Brain Atlas structure definitions
├── manifest.json            Allen Brain Atlas data manifest
├── config/                  Configuration (paths, brain areas, plotting, experiments)
├── src/                     Core analysis library (data loading, statistics, fitting, plotting)
├── scripts/                 One script per figure (see table above)
├── data/
│   ├── units/               Per-area unit CSVs  (freq × amplitude × area)
│   ├── waveform_features/   Waveform feature CSVs
│   └── connectivity/        Allen Mouse Brain connectivity matrices
└── figures/                 Generated figures (git-ignored)
```

## Data

Unit-level CSVs: `units_{frequency}Hz_{amplitude}_{brain_area}.csv`
Waveform features: `{brain_area}_wf_features_only.csv`

## License

MIT — see [LICENSE](LICENSE).
