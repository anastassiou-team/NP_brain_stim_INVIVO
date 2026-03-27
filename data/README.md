# Data dictionary

## units/

One CSV per combination of stimulation frequency, amplitude, and brain area.

**Naming:** `units_{frequency}Hz_{amplitude}_{brain_area}.csv`
- Frequencies: 8, 28, 140 Hz
- Amplitudes: 1, 5 µA
- Brain areas: ~30 regions (CA1, VISp, MOp, ACAd, SUB, etc.)

| Column | Units | Description |
|--------|-------|-------------|
| mouse | — | Mouse identifier |
| probe_position | — | `close` or `far` relative to stimulation electrode |
| unitID | — | Unit identifier within session |
| VL_pre | 0–1 | Vector length (phase-locking strength) during pre-stim baseline |
| VL_stimOn | 0–1 | Vector length during stimulation |
| VL_post | 0–1 | Vector length during post-stim recovery |
| angle_pre_ch15 | rad | Mean preferred phase angle, baseline |
| angle_stimOn_ch15 | rad | Mean preferred phase angle, stimulation |
| angle_post_ch15 | rad | Mean preferred phase angle, recovery |
| MMR_Pre | 0–1 | Modulation magnitude ratio, baseline |
| MMR_stimOn | 0–1 | Modulation magnitude ratio, stimulation |
| MMR_Post | 0–1 | Modulation magnitude ratio, recovery |
| Nspikes_pre | count | Spike count, baseline |
| Nspikes_StimOn | count | Spike count, stimulation |
| Nspikes_post | count | Spike count, recovery |
| Fspikes_pre | Hz | Firing rate, baseline |
| Fspikes_StimOn | Hz | Firing rate, stimulation |
| Fspikes_post | Hz | Firing rate, recovery |
| inst_amp_pre_peak_ch | µV | Peak extracellular action-potential amplitude, baseline |
| inst_amp_StimOn_peak_ch | µV | Peak EAP amplitude, stimulation |
| inst_amp_post_peak_ch | µV | Peak EAP amplitude, recovery |
| distance_peakch_stim_tip | mm | Distance from peak recording channel to stimulation electrode tip |
| error_waveform | 0–1 | Spike-sorting waveform error; units with error > 0.1 are excluded |
| universal_ID | — | Unique identifier: `{mouse}{probe_position}{unitID}` |
| waveform_halfwidth | ms | Trough-to-half-repolarisation duration |
| waveform_duration | ms | Total action-potential duration |
| waveform_REP | 0–1 | Repolarisation slope ratio |
| cluster | 0 or 1 | Waveform-based cell-type cluster assignment |

### Key metrics

- **Vector Length (VL):** magnitude of the mean unit vector in the complex plane, `|mean(exp(i * phase))|`. Ranges from 0 (no phase locking) to 1 (perfect phase locking).
- **Modulation Magnitude Ratio (MMR):** alternative entrainment measure that is more robust for multimodal phase distributions.
- **Three time windows:** every per-unit metric is computed separately for pre-stimulation baseline, stimulation-on, and post-stimulation recovery epochs.

---

## waveform_features/

One CSV per brain area, containing only waveform-derived features and cluster assignments.

**Naming:** `{brain_area}_wf_features_only.csv`

| Column | Units | Description |
|--------|-------|-------------|
| cluster | 0 or 1 | Cell-type cluster from k-means on waveform features |
| mouse | — | Mouse identifier |
| probe_position | — | `close` or `far` |
| unitID | — | Unit identifier |
| waveform_halfwidth | ms | Trough-to-half-repolarisation duration |
| waveform_duration | ms | Total action-potential duration |
| waveform_REP | 0–1 | Repolarisation slope ratio |
| distance_peakch_stim_tip | mm | Distance to stimulation electrode |
| universal_ID | — | Unique unit identifier |

---

## unified_data_VL_angle_MMR.csv

All units across all frequency/amplitude conditions merged into a single table (29 262 rows). Used by the clustering and VL-vs-MMR analyses.

Includes all columns from the per-area unit files plus:

| Column | Units | Description |
|--------|-------|-------------|
| stim_current | µA | Stimulation amplitude (1 or 5) |
| stim_freq | — | Stimulus label, e.g. `sine_8Hz` |
| pvalue_ch15_pre | — | Rayleigh-test p-value for phase concentration, baseline |
| pvalue_ch15_stimOn | — | Rayleigh-test p-value, stimulation |
| pvalue_ch15_post | — | Rayleigh-test p-value, recovery |
| peak_ch | — | Recording channel with maximum spike amplitude |
| area_peak_ch | — | Brain-area label at peak channel |
| depth_peak_ch | µm | Recording depth of peak channel |
| area_stim_tip | — | Brain area at stimulation electrode tip |

---

## unified_data_transient.csv

Transient firing-rate data for Figure 4 (29 262 rows). Each row represents one unit in one stimulation condition. Unlike the per-area unit CSVs, this file uses a long format with `stim_freq` and `stim_current` as columns, and brain-area names are layer-specific (e.g., `ACAd5`, `MOp2/3`, `VISp4`).

Units are classified as fast-spiking (FS, waveform duration ≤ 0.4 ms) or regular-spiking (RS, > 0.4 ms). Transient effects are detected as significant firing-rate deviations in the first 100 ms after stimulation onset.

| Column | Units | Description |
|--------|-------|-------------|
| mouse | — | Mouse identifier |
| probe_position | — | `close` or `far` relative to stimulation electrode |
| stim_current | µA | Stimulation amplitude (1 or 5) |
| stim_freq | — | Stimulus label: `sine_8Hz`, `sine_28Hz`, or `sine_140Hz` |
| unitID | — | Unit identifier within session |
| peak_ch | — | Recording channel with maximum spike amplitude |
| waveform_duration | ms | Total action-potential duration (FS ≤ 0.4 ms, RS > 0.4 ms) |
| waveform_REP | 0–1 | Repolarization slope ratio |
| transient_type | D / I / NaN | Transient effect classification: D = decrease, I = increase, NaN = no significant transient |
| transient_onset | s | Onset time of transient effect relative to stimulation start |
| transient_offset | s | Offset time of transient effect |
| transient_peak_value | Hz | Peak firing-rate deviation during the transient |
| meanFR_pre_10s | Hz | Mean firing rate during 10 s pre-stimulation baseline |
| stdFR_pre_10s | Hz | Std of firing rate during 10 s pre-stimulation baseline |
| meanFR_pre_1s | Hz | Mean firing rate during 1 s pre-stimulation baseline |
| stdFR_pre_1s | Hz | Std of firing rate during 1 s pre-stimulation baseline |
| meanFR_stimOn_10s | Hz | Mean firing rate during full 10 s stimulation epoch |
| meanFR_stimOn_1s | Hz | Mean firing rate during first 1 s of stimulation |
| meanFR_stimOn_01s | Hz | Mean firing rate during first 100 ms of stimulation |
| meanFR_stimOn_last_01s | Hz | Mean firing rate during last 100 ms of stimulation |
| meanFR_stimOn_last_1s | Hz | Mean firing rate during last 1 s of stimulation |
| area_peak_ch | — | Brain-area label at peak channel (layer-specific, e.g., `VISp4`, `CA1`) |
| depth_peak_ch | µm | Recording depth of peak channel |
| error_waveform | 0–1 | Spike-sorting waveform error; units with error ≥ 0.1 are excluded |
| area_stim_tip | — | Brain area at stimulation electrode tip |
| distance_peakch_stim_tip | mm | Distance from peak recording channel to stimulation electrode tip |

### Key derived quantities

- **Z-scored firing rate** (Figure 4e): `(meanFR_stimOn_01s − meanFR_pre_1s) / stdFR_pre_1s`, absolute value taken.
- **Transient duration** (Figure 4a): `transient_offset − transient_onset` for units with non-NaN transient classification.
- **Minimum baseline FR filter**: units with `meanFR_pre_10s ≤ 0.5 Hz` are excluded from all Figure 4 analyses.

---

## connectivity/

Allen Mouse Brain Atlas structural connectivity data.

| File | Description |
|------|-------------|
| connectivity.csv | Normalised inter- and intra-hemispheric connection strength and density between stimulation sites and recorded areas |
| connection_strength.csv | Full strength matrix (all areas × hemispheres) |
| connection_density.csv | Full density matrix |
| normalized_connection_strength.csv | Normalised strength matrix |
| normalized_connection_density.csv | Normalised density matrix |
| normalized_connection_density_ipsi_ctx.csv | Ipsilateral cortical density subset |
| manifest.json | Allen SDK data manifest |
