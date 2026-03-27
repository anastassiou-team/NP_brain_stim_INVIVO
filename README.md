# Sinusoidal electrical stimulation effects across the entire brain

<p align="center">
  <img src="figures/banner.png" width="100%" alt="Brain-wide effects of sinusoidal electrical stimulation">
</p>

## Overview

Electrical stimulation is widely used to modulate neuronal activity, yet its effects on neuronal circuits in vivo remain poorly understood. This, in turn, has hindered the principled design of stimulation protocols and raised questions about reproducibility that constrain the field's translational impact. Here we combine cortical sinusoidal electrical stimulation (sES) with Neuropixels recordings to characterize stimulation-driven responses in more than 2,700 well-isolated neurons across 53 brain areas in 14 behaving, head-fixed mice. We uncover two distinct, concurrent modes of neural modulation. First is a sustained, brain-wide spike-phase entrainment effect that depends on stimulation frequency: entrainment to slow stimulation is supported by non-synaptic electric field propagation while anatomical connectivity dominates entrainment to higher stimulation frequencies. Second, we find a transient, spatially localized spike-rate modulation mainly mediated through anatomical connectivity that only emerges at high stimulation frequencies by selectively recruiting inhibitory neurons. We show that the two distinct modes are differentially shaped by behavior. By identifying how stimulation frequency governs the mechanism of neural engagement and how behavioral state selectively gates brain-wide entrainment but not local inhibitory recruitment, our results provide a mechanistic foundation for designing targeted, reproducible neuromodulation strategies.

This repository contains the data and analysis code to reproduce all main and supplementary figures in the study below. 

**Citation:**

> Rembado, I., Lee, S.Y., Marks, L.C., Claar, L.D., Peltekian, A.,
> Koch, C. & Anastassiou, C.A. (2026). Selective gating of neural modulation
> through frequency- and behavior-dependent modes during cortical electrical
> stimulation. *Under review.*

## Figures

Pre-generated figures are included in `figures/`.
To regenerate any panel, run the corresponding script from the repo root.

| Figure | Description | Script | Output |
|--------|-------------|--------|--------|
| Fig. 1f | Analytical model of bipolar ES field: Ve contour map with recording probes at multiple angles (point-source approximation in resistive medium) | `python scripts/fig1f_analytical_model.py` | `figures/figure1f_bipolar_model.png` |
| Fig. 2 | Brain-wide phase entrainment: paired pre vs stim vector length (VL) across 30+ areas at each frequency/amplitude, with spatial decay fits and regression analysis | `python scripts/fig2_entrainment.py` | `figures/figure2_complete.png` |
| Fig. 3, row 1 | K-means clustering of VL responses (pre vs stim) in select areas, plus bubble chart of clustering significance vs distance | `python scripts/figure3_row1.py` | `figures/figure3_row1.png` |
| Fig. 3, row 2 | Frequency dependence of clustering: bar charts and heatmap of clustering significance across areas and frequencies | `python scripts/figure3_row2.py` | `figures/figure3_row2.png` |
| Fig. 3, row 3 | Waveform–cluster associations: effect sizes for EAP waveform features vs cluster identity across areas | `python scripts/figure3_row3.py` | `figures/figure3_row3.png` |
| Fig. 4a | Distribution of transient firing-rate effect durations at each frequency | `python scripts/fig4_duration.py` | `figures/figure4a_transient_duration.png` |
| Fig. 4b | Brain-wide firing-rate changes during the first 100 ms of stimulation, paired pre vs stim across areas and frequencies | `python scripts/fig4_brain_wide_fr.py` | `figures/figure4b_brain_wide_fr.png` |
| Fig. 4c | Bimodal waveform-duration histogram used to classify units as fast-spiking (FS) or regular-spiking (RS) | `python scripts/fig4_waveform_hist.py` | `figures/figure4c_waveform_histogram.png` |
| Fig. 4d | Percentage of FS and RS units showing transient firing-rate decreases or increases, compared across frequencies | `python scripts/fig4_cell_type_percent.py` | `figures/figure4d_cell_type_percent.png` |
| Fig. 4e | Distance-dependent z-scored firing-rate changes for FS vs RS units at each frequency | `python scripts/fig4_zscore_distance.py` | `figures/figure4e_zscore_distance.png` |
| Fig. S4 | Replication of Fig. 2 using mean modulation ratio (MMR) instead of VL as the entrainment metric | `python scripts/figS4_mmr_entrainment.py` | `figures/fig2_mmr/figure2_mmr_complete.png` |
| Fig. S (VL-MMR) | Analysis of VL–MMR discrepancies: when and why MMR exceeds VL, indicating bimodal phase-locking | `python scripts/figS_vl_mmr_corr.py` | `figures/fig2_vl_vs_mmr/` |

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

Alternatively, with pip:

```bash
pip install -r requirements.txt
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
├── requirements.txt         Pip requirements (alternative to conda)
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
│   ├── connectivity/        Allen Mouse Brain connectivity matrices
│   └── README.md            Full data dictionary
└── figures/                 Generated figures
```

## Statistical methods

Statistical tests are implemented in `src/statistics.py` and `src/analysis_transient.py`. The analyses use:

- **One-sample t-tests on paired differences** (`scipy.stats.ttest_1samp`) for within-unit pre vs stim comparisons (Figures 2, 4b).
- **Mann–Whitney U tests** (`scipy.stats.mannwhitneyu`) for unpaired comparisons between cell types and across frequency conditions (Figures 4d, 4e).
- **Benjamini–Hochberg FDR correction** (`statsmodels.stats.multitest.multipletests`, method `fdr_bh`) applied across brain areas or distance bins within each figure panel.
- **OLS interaction tests** (`statsmodels.formula.api.ols`, `y ~ x * group`) for comparing spatial decay slopes between FS and RS populations (Figure 4e).
- **K-means clustering with silhouette-score optimization** (`sklearn.cluster.KMeans`) for identifying response subgroups (Figure 3), implemented in `scripts/fig3_clustering.py`.

## Data

Detailed column-by-column data dictionaries are in [`data/README.md`](data/README.md).

- **Unit-level CSVs** (`data/units/`): one file per frequency × amplitude × brain area combination, covering 29 gray-matter regions that met minimum unit-count thresholds for the published analyses. Contains VL, MMR, spike counts, firing rates, waveform features, and distances for every recorded unit.
- **Transient data** (`data/unified_data_transient.csv`): firing-rate summary statistics in multiple time windows, transient onset/offset, and cell-type classification for Figure 4. Retains all recorded units across the full ~53-area recording coverage (45 parent gray-matter areas) before any unit-count filtering.
- **Connectivity matrices** (`data/connectivity/`): Allen Mouse Brain Atlas structural connectivity, used for regression analyses of entrainment strength vs anatomical connection weight.

## Contact

Costas A. Anastassiou — [Anastassiou Lab](https://anastassioulab.org), Cedars-Sinai Medical Center & California Institute of Technology

## License

MIT — see [LICENSE](LICENSE).
