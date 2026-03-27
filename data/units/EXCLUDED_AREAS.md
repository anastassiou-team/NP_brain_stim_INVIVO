# Excluded area files

The following 5 area labels have unit CSV files in this directory but are
**excluded from all published figure analyses**. The files are retained for
completeness and provenance.

| File pattern | Label | Reason for exclusion |
|---|---|---|
| `units_*_ccb.csv` | ccb (corpus callosum body) | White matter fiber tract, not a gray-matter brain area |
| `units_*_cing.csv` | cing (cingulum bundle) | White matter fiber tract, not a gray-matter brain area |
| `units_*_DG.csv` | DG (dentate gyrus aggregate) | Redundant aggregate of DG-mo, DG-po, and DG-sg, which are already present as separate files |
| `units_*_anterior_cingulate.csv` | anterior_cingulate | Legacy label that maps to SSp-tr — duplicate of `units_*_SSp_tr.csv` |
| `units_*_ss_ctx.csv` | ss_ctx | Legacy label that also maps to SSp-tr — duplicate of `units_*_SSp_tr.csv` |
