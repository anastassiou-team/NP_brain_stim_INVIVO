"""
Configuration for MMR (Mean Modulation Ratio) analysis

MMR is a phase coupling metric that handles bimodal and multimodal distributions
better than vector length. It measures the non-uniformity of phase distributions.

MMR = 1 - mean(normalized_phase_distribution)
Range: [0, 1] where 0 = uniform, 1 = maximally concentrated
"""

# MMR column names in the CSV files
MMR_COLUMNS = {
    'pre': 'MMR_Pre',
    'stim': 'MMR_StimOn'
}

# MMR-specific plotting parameters
MMR_PLOT_PARAMS = {
    'ylabel_pre': 'MMR',
    'ylabel_stim': 'MMR', 
    'ylabel_delta': 'ΔMMR',
    'ylabel_delta_absolute': '|ΔMMR|',
    'metric_name': 'Mean Modulation Ratio',
    'metric_abbrev': 'MMR'
}

# MMR ranges for plotting (adjust based on actual data distribution)
MMR_PLOT_RANGES = {
    'mmr_min': 0.0,
    'mmr_max': 1.0,
    'delta_mmr_min': -0.5,
    'delta_mmr_max': 0.5,
    'abs_delta_mmr_min': 0.0,
    'abs_delta_mmr_max': 0.9
}

# Output folders specific to MMR analysis
MMR_OUTPUT = {
    'figure_folder': 'fig2_mmr',
    'complete_figure': 'figure2_mmr_complete.png',
    'rows5and6_figure': 'figure2_mmr_rows5and6.png',
    'brief_file': 'project_brief_fig2_mmr.md',
    'legend_file': 'figure_legend_fig2_mmr.md'
}
