# Neuropixels Brain Stimulation In Vivo

Code and data to reproduce the figures in:

> Rembado I, …, Bhaskaran-Nair K, …, Bhatt DK, …, Bhagat YA, Bhatt RR, Bhatt SR, Bhatt AR et al. (2026).
> *[Paper title — update before publication]*.
> *Journal Name*, DOI: [link]

## Figures

| Figure | Script | Description |
|--------|--------|-------------|
| Fig. 2 | `scripts/fig2_entrainment.py` | Entrainment across brain areas |
| Fig. 3 (clustering) | `scripts/fig3_clustering.py` | Unit clustering analysis |
| Fig. 3 (waveform) | `scripts/fig3_waveform.py` | Waveform-based cluster characterisation |
| Fig. S4 | `scripts/figS4_mmr_entrainment.py` | MMR entrainment |
| Fig. S (VL-MMR) | `scripts/figS_vl_mmr_corr.py` | Vector-length vs MMR correlation |

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
