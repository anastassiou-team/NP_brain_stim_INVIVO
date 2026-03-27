"""
Generate Complete Figure 2: Phase entrainment across brain areas

Complete 6-row layout:
- Rows 1-4: Paired pre vs stim comparisons (VL across areas)
- Row 5: Spatial decay analysis (3 frequencies, 2 panels each)
- Row 6: Regression analysis (3 frequencies, 2 panels each)

NOTE: Ensure that in src/plotting_utils.py, the plot_paired_comparison_row function
uses "vector length" (lowercase) instead of "Vector Length" for y-axis labels.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from config.paths import FIGURES_OUTPUT, BASE_DIR
from config.brain_areas import get_display_name, ALL_BRAIN_AREAS
from config.experiments import FREQUENCIES, DEFAULT_ANIMAL_LIST, FIGURE2_PARAMS, FIGURE2_SPATIAL_PARAMS
from config.plotting import (
    mm_to_inch, DOUBLE_COLUMN_WIDTH,
    FREQUENCY_COLORS, STATS_PARAMS, SPATIAL_DECAY_OPTIONS,
    SPATIAL_DECAY_PARAMS
)
from src.plotting_utils import (
    setup_nature_figure, remove_top_right_spines,
    plot_paired_comparison_row, format_indicator_axis
)
from src.data_loading import load_paired_condition_data
from src.analysis_core import (
    apply_fdr_correction , analyze_all_areas, identify_entrained_areas,
    prepare_spatial_decay_data, get_entrained_area_indices,
    sort_areas_by_distance
)
from src.fitting import fit_spatial_decay


# ============================================================================
# ROW 5: SPATIAL DECAY FUNCTIONS
# ============================================================================

def plot_spatial_decay_twopanel(
    spatial_data,
    fit_results,
    entrained_indices,
    entrained_distances,
    frequency,
    axes,
    show_ylabel=True
):
    """Create 2-panel spatial decay plot with significance indicators"""
    color = FREQUENCY_COLORS[frequency]
    params = SPATIAL_DECAY_PARAMS
    
    # Unpack axes
    ax_high_left = axes[0]
    ax_low_left = axes[1]
    ax_scatter = axes[2]
    ax_high_right = axes[3]
    ax_low_right = axes[4]
    ax_summary = axes[5]
    
    marker_size = params['marker_size_top']**2
    
    # === Left panel significance indicators ===
    ax_high_left.scatter(
        spatial_data['area_distances'], np.ones(len(spatial_data['area_distances'])),
        color='k', marker='o', s=marker_size,
        alpha=params['marker_alpha_dim']
    )
    if len(entrained_distances['high']) > 0:
        ax_high_left.scatter(
            entrained_distances['high'], np.ones(len(entrained_distances['high'])),
            facecolor=color, edgecolors=color,
            marker='o', s=marker_size,
            linewidth=0.5, alpha=1.0
        )
    format_indicator_axis(ax_high_left, '0.001', params)
    ax_high_left.set_xlim(params['xrange'])
    
    ax_low_left.scatter(
        spatial_data['area_distances'], np.ones(len(spatial_data['area_distances'])),
        color='k', marker='o', s=marker_size,
        alpha=params['marker_alpha_dim']
    )
    if len(entrained_distances['low']) > 0:
        ax_low_left.scatter(
            entrained_distances['low'], np.ones(len(entrained_distances['low'])),
            color=color, marker='o', s=marker_size,
            alpha=1.0
        )
    format_indicator_axis(ax_low_left, '0.01', params)
    ax_low_left.set_xlim(params['xrange'])
    
    # === Right panel significance indicators ===
    ax_high_right.scatter(
        spatial_data['area_distances'], np.ones(len(spatial_data['area_distances'])),
        color='k', marker='o', s=marker_size,
        alpha=params['marker_alpha_dim']
    )
    if len(entrained_distances['high']) > 0:
        ax_high_right.scatter(
            entrained_distances['high'], np.ones(len(entrained_distances['high'])),
            facecolor=color, edgecolors=color,
            marker='o', s=marker_size,
            linewidth=0.5, alpha=1.0
        )
    format_indicator_axis(ax_high_right, '0.001', params)
    ax_high_right.set_xlim(params['xrange'])
    
    ax_low_right.scatter(
        spatial_data['area_distances'], np.ones(len(spatial_data['area_distances'])),
        color='k', marker='o', s=marker_size,
        alpha=params['marker_alpha_dim']
    )
    if len(entrained_distances['low']) > 0:
        ax_low_right.scatter(
            entrained_distances['low'], np.ones(len(entrained_distances['low'])),
            color=color, marker='o', s=marker_size,
            alpha=1.0
        )
    format_indicator_axis(ax_low_right, '0.01', params)
    ax_low_right.set_xlim(params['xrange'])
    
    # === Left panel: Unit-level scatter with fit ===
    ax_scatter.plot(
        spatial_data['distances'],
        spatial_data['delta_vl'],
        'o', color='k',
        markersize=params['scatter_size'],
        alpha=params['scatter_alpha'],
        zorder=1
    )
    
    ax_scatter.plot(
        fit_results['x_pred'],
        fit_results['y_pred'],
        params['fit_line_style'],
        color=color,
        linewidth=params['fit_line_width'],
        zorder=2
    )
    
    ax_scatter.set_xlabel('distance / mm', fontsize=8)
    if show_ylabel:
        ax_scatter.set_ylabel('|ΔVL|' if SPATIAL_DECAY_OPTIONS['use_absolute_difference'] else 'ΔVL', 
                              fontsize=8)
    ax_scatter.set_xlim(params['xrange'])
    ax_scatter.set_ylim([-0.05, 0.9])
    ax_scatter.set_xticks(params['xticks'])
    ax_scatter.set_yticks(params['yticks_delta'])
    ax_scatter.tick_params(labelsize=7)
    remove_top_right_spines(ax_scatter)
    
    # === Right panel: Area-level summary with labels ===
    ax_summary.scatter(
        spatial_data['area_distances'],
        spatial_data['area_delta_vl'],
        color='k',
        s=params['scatter_size']**2,
        alpha=params['scatter_alpha']
    )
    
    if len(entrained_indices['high']) > 0:
        ax_summary.scatter(
            spatial_data['area_distances'][entrained_indices['high']],
            spatial_data['area_delta_vl'][entrained_indices['high']],
            s=params['scatter_size']**2,
            facecolor=color,
            edgecolors=color,
            linewidth=0.5
        )
        
        for idx in entrained_indices['high']:
            display_name = get_display_name(spatial_data['areas'][idx])
            ax_summary.annotate(
                display_name,
                (spatial_data['area_distances'][idx], 
                 spatial_data['area_delta_vl'][idx]),
                textcoords='offset points',
                xytext=params['label_offset'],
                fontsize=params['label_fontsize'],
                color=color,
                ha='left'
            )
    
    ax_summary.set_xlabel('distance / mm', fontsize=8)
    ax_summary.set_xlim(params['xrange'])
    ax_summary.set_ylim([-0.02, 0.4])
    ax_summary.set_xticks(params['xticks'])
    ax_summary.set_yticks(params['yticks_vl'])
    ax_summary.tick_params(labelsize=7)
    remove_top_right_spines(ax_summary)


# ============================================================================
# ROW 6: REGRESSION FUNCTIONS
# ============================================================================

def load_connectivity_data():
    """Load connectivity data from Allen Brain Atlas"""
    conn_path = os.path.join(BASE_DIR, 'data', 'connectivity', 'connectivity.csv')
    
    try:
        conn = pd.read_csv(conn_path)
        return conn
    except FileNotFoundError:
        print(f"  Warning: Connectivity file not found")
        return pd.DataFrame()


def prepare_regression_data_with_connectivity(amplitude, frequency, animal_list=None):
    """Prepare regression data including connectivity metrics"""
    if animal_list is None:
        animal_list = DEFAULT_ANIMAL_LIST
    
    area_results = analyze_all_areas(
        ALL_BRAIN_AREAS, amplitude, frequency, FREQUENCIES, animal_list,
        test_type=STATS_PARAMS['test_type']
    )
    
    conn = load_connectivity_data()
    
    areas, areas_renamed = [], []
    delta_vl_list, distances, lfp_amplitudes = [], [], []
    
    for area, data in area_results.items():
        areas.append(area)
        areas_renamed.append(get_display_name(area))
        delta_vl_list.append(data['stats']['effect_size'])
        distances.append(data['distance'])
        
        _, _, full_data = load_paired_condition_data(
            area, amplitude, frequency, FREQUENCIES, animal_list
        )
        if full_data is not None and 'inst_amp_StimOn_peak_ch' in full_data.columns:
            lfp_amplitudes.append(full_data['inst_amp_StimOn_peak_ch'].median())
        else:
            lfp_amplitudes.append(np.nan)
    
    conn_metrics = {
        'strength_inter': [], 'strength_intra': [],
        'density_inter': [], 'density_intra': []
    }
    
    for area_renamed in areas_renamed:
        area_conn = conn[conn['projection_area'] == area_renamed]
        if len(area_conn) > 0:
            conn_metrics['strength_inter'].append(area_conn['norm_strength inter'].values[0])
            conn_metrics['strength_intra'].append(area_conn['norm_strength intra'].values[0])
            conn_metrics['density_inter'].append(area_conn['norm_density inter'].values[0])
            conn_metrics['density_intra'].append(area_conn['norm_density intra'].values[0])
        else:
            for key in conn_metrics:
                conn_metrics[key].append(np.nan)
    
    return {
        'areas': areas,
        'delta_vl': np.array(delta_vl_list),
        'distances': np.array(distances),
        'lfp_amplitudes': np.array(lfp_amplitudes),
        'connectivity': conn_metrics
    }


def run_hierarchical_regression(X_full, y, variable_names):
    """Run 3 nested regression models"""
    results = {}
    
    X1 = X_full[['distance']]
    X1_const = sm.add_constant(X1)
    model1 = sm.OLS(y, X1_const).fit()
    results['model1'] = {
        'r_squared': model1.rsquared,
        'aic': model1.aic
    }
    
    X2 = X_full[['distance', 'lfp_amplitude']]
    X2_const = sm.add_constant(X2)
    model2 = sm.OLS(y, X2_const).fit()
    results['model2'] = {
        'r_squared': model2.rsquared,
        'delta_r_squared': model2.rsquared - model1.rsquared
    }
    
    X3_const = sm.add_constant(X_full)
    model3 = sm.OLS(y, X3_const).fit()
    
    coef_series = pd.Series(model3.params[1:].values, index=variable_names)
    pval_series = pd.Series(model3.pvalues[1:].values, index=variable_names)
    
    results['model3'] = {
        'r_squared': model3.rsquared,
        'aic': model3.aic,
        'delta_r_squared': model3.rsquared - model2.rsquared,
        'coefficients': coef_series,
        'pvalues': pval_series
    }
    
    return results


def plot_regression_twopanel(hierarchical_results, frequency, axes, show_ylabels=True):
    """Create 2-panel regression plot"""
    color = FREQUENCY_COLORS[frequency]
    model3 = hierarchical_results['model3']
    
    ax1 = axes[0]
    coefs = model3['coefficients']
    pvals = model3['pvalues']
    
    fixed_order = ['distance', 'LFP amplitude', 'inter strength', 
                   'intra strength', 'inter density', 'intra density']
    coefs_ordered = coefs.reindex(fixed_order)
    pvals_ordered = pvals.reindex(fixed_order)
    
    bar_colors = ['red' if p < 0.05 else 'lightgray' for p in pvals_ordered]
    y_pos = np.arange(len(coefs_ordered))
    ax1.barh(y_pos, coefs_ordered.values, color=bar_colors, alpha=0.8, 
             edgecolor='black', linewidth=0.5)
    ax1.axvline(0, color='k', linestyle='--', linewidth=0.5)
    
    ax1.set_yticks(y_pos)
    if show_ylabels:
        ax1.set_yticklabels(coefs_ordered.index, fontsize=7, rotation=45, ha='right')
    else:
        ax1.set_yticklabels([])
    
    ax1.set_xlabel('Standardized β', fontsize=8)
    ax1.tick_params(labelsize=7)
    remove_top_right_spines(ax1)
    
    ax2 = axes[1]
    
    model_names = ['Dist', 'Dist+LFP', 'Full']
    r_squared_vals = [
        hierarchical_results['model1']['r_squared'],
        hierarchical_results['model2']['r_squared'],
        hierarchical_results['model3']['r_squared']
    ]
    delta_r_squared = [
        hierarchical_results['model1']['r_squared'],
        hierarchical_results['model2']['delta_r_squared'],
        hierarchical_results['model3']['delta_r_squared']
    ]
    
    bottoms = [0, r_squared_vals[0], r_squared_vals[1]]
    colors_stack = ['#87CEEB', '#FFB347', '#9370DB']
    
    x_positions = np.arange(3)
    for i, (delta, bottom, col) in enumerate(zip(delta_r_squared, bottoms, colors_stack)):
        ax2.bar(x_positions[i], delta, bottom=bottom, color=col, 
               alpha=0.8, edgecolor='black', width=0.7, linewidth=0.5)
    
    for i, (delta, bottom) in enumerate(zip(delta_r_squared, bottoms)):
        if delta > 0.04:
            ax2.text(x_positions[i], bottom + delta/2, f"{delta:.2f}", 
                    ha='center', va='center', fontsize=7, weight='bold')
    
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(model_names, fontsize=7, rotation=45, ha='right')
    ax2.set_ylabel('R²', fontsize=8)
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(labelsize=7)
    remove_top_right_spines(ax2)


# ============================================================================
# MAIN FUNCTION: GENERATE COMPLETE FIGURE 2
# ============================================================================

def generate_complete_figure2(save_figures=True):
    """
    Generate complete Figure 2 with all 6 rows
    
    Layout:
    - Rows 1-4: Paired comparison plots (VL across areas)
    - Row 5: Spatial decay analysis
    - Row 6: Regression analysis
    """
    setup_nature_figure()
    
    print("\n" + "="*80)
    print("GENERATING COMPLETE FIGURE 2")
    print("="*80)
    
    # Create main figure
    fig_height = mm_to_inch(250)  # Tall enough for all 6 rows
    fig = plt.figure(figsize=(mm_to_inch(DOUBLE_COLUMN_WIDTH), fig_height))
    
    # Main grid: 6 rows
    # Rows 1-4 are comparison plots (equal height)
    # Row 5 is spatial decay (includes significance indicators)
    # Row 6 is regression
    gs_main = GridSpec(6, 1, 
                       height_ratios=[0.8, 0.8, 0.8, 0.8, 1.2, 0.9],
                       hspace=0.35,
                       top=0.97, bottom=0.03,
                       left=0.08, right=0.98)
    
    # ========================================================================
    # ROWS 1-4: PAIRED COMPARISONS
    # ========================================================================
    
    print("\n" + "="*80)
    print("ROWS 1-4: Paired Comparisons")
    print("="*80)
    
    for row_idx, (amplitude, frequency) in enumerate(FIGURE2_PARAMS):
        print(f"\nRow {row_idx + 1}: {amplitude}µA, {frequency}Hz")
        
        # Analyze areas
        results = analyze_all_areas(
            ALL_BRAIN_AREAS, amplitude, frequency, FREQUENCIES,
            DEFAULT_ANIMAL_LIST, test_type=STATS_PARAMS['test_type']
        )
        # Deploy BH FDR-correction for multiple comparisons...
        results = apply_fdr_correction(results, alpha=0.05)

        sorted_areas, sorted_distances = sort_areas_by_distance(results)
        
        # === Verify which p-values are used ===
        test_area = list(results.keys())[0]
        print(f"\n  DEBUG - First area: {test_area}")
        print(f"    Raw p-value:    {results[test_area]['stats']['pval']:.6f}")
        print(f"    FDR p-value:    {results[test_area]['stats']['pval_corrected']:.6f}")
        print(f"    Value passed to plotting: {results[test_area]['stats'].get('pval_corrected', results[test_area]['stats']['pval']):.6f}")
        
        # Check what identify_entrained_areas uses
        entrained = identify_entrained_areas(results, thresh_low=0.01, thresh_high=0.001, use_corrected=True)
        print(f"    Is entrained (FDR p<0.01)? {test_area in entrained['low']}")
        
        # Create axis for this row
        ax = fig.add_subplot(gs_main[row_idx])
        
        # Prepare comparison data
        comparison_data = {}
        for area in sorted_areas:
            comparison_data[area] = {
                'pre': results[area]['pre'],
                'stim': results[area]['stim'],
                'stats': results[area]['stats']
            }
        
        # Plot (only show labels on first row)
        show_labels = (row_idx == 0)
        plot_paired_comparison_row(
            sorted_areas,
            comparison_data,
            amplitude,
            frequency,
            show_area_labels=show_labels,
            ax=ax
        )

    # ========================================================================
    # ROW 5: SPATIAL DECAY
    # ========================================================================
    
    print("\n" + "="*80)
    print("ROW 5: Spatial Decay Analysis")
    print("="*80)
    
    # Create nested GridSpec for Row 5
    gs_row5 = GridSpecFromSubplotSpec(3, 6, subplot_spec=gs_main[4],
                                       height_ratios=[0.12, 0.12, 1.0],
                                       hspace=0.02, wspace=0.35)
    
    display_names = ['distance', 'LFP amplitude', 'inter strength', 
                    'intra strength', 'inter density', 'intra density']
    
    for idx, (amplitude, frequency) in enumerate(FIGURE2_SPATIAL_PARAMS):
        print(f"\n{frequency} Hz:")
        color = FREQUENCY_COLORS[frequency]
        
        # Get area results
        area_results = analyze_all_areas(
            ALL_BRAIN_AREAS, amplitude, frequency, FREQUENCIES,
            DEFAULT_ANIMAL_LIST, test_type=STATS_PARAMS['test_type']
        )

        area_results = apply_fdr_correction(area_results, alpha=0.05)

        # Report FDR correction effect
        n_raw_low = sum(1 for area in area_results if area_results[area]['stats']['pval_raw'] <= 0.01)
        n_raw_high = sum(1 for area in area_results if area_results[area]['stats']['pval_raw'] <= 0.001)
        n_fdr_low = sum(1 for area in area_results if area_results[area]['stats']['pval_corrected'] <= 0.01)
        n_fdr_high = sum(1 for area in area_results if area_results[area]['stats']['pval_corrected'] <= 0.001)

        print(f"  Before FDR: p<0.01: {n_raw_low}, p<0.001: {n_raw_high}")
        print(f"  After FDR:  p<0.01: {n_fdr_low}, p<0.001: {n_fdr_high}")

        entrained = identify_entrained_areas(
            area_results,
            thresh_low=STATS_PARAMS['entrainment_thresh_low'],
            thresh_high=STATS_PARAMS['entrainment_thresh_high'],
            use_corrected=True
        )
        
        # Prepare spatial decay data
        spatial_data = prepare_spatial_decay_data(
            ALL_BRAIN_AREAS, amplitude, frequency, FREQUENCIES,
            DEFAULT_ANIMAL_LIST,
            use_absolute=SPATIAL_DECAY_OPTIONS['use_absolute_difference']
        )
        
        # Fit spatial decay
        fit_results = fit_spatial_decay(
            spatial_data['distances'],
            spatial_data['delta_vl'],
            model='inverse_power_law_twooffsets'
        )
        
        # Get entrained indices and distances
        entrained_indices = get_entrained_area_indices(
            spatial_data['areas'], entrained
        )
        
        entrained_distances = {
            'low': spatial_data['area_distances'][entrained_indices['low']] if len(entrained_indices['low']) > 0 else np.array([]),
            'high': spatial_data['area_distances'][entrained_indices['high']] if len(entrained_indices['high']) > 0 else np.array([])
        }
        
        print(f"  Units: {len(spatial_data['distances'])}, Entrained areas: {len(entrained['high'])}")
        
        # Create axes
        ax_high_left = fig.add_subplot(gs_row5[0, idx*2])
        ax_low_left = fig.add_subplot(gs_row5[1, idx*2])
        ax_scatter = fig.add_subplot(gs_row5[2, idx*2])
        
        ax_high_right = fig.add_subplot(gs_row5[0, idx*2 + 1])
        ax_low_right = fig.add_subplot(gs_row5[1, idx*2 + 1])
        ax_summary = fig.add_subplot(gs_row5[2, idx*2 + 1])
        
        show_ylabel = (idx == 0)
        plot_spatial_decay_twopanel(
            spatial_data, fit_results, entrained_indices, entrained_distances,
            frequency, 
            [ax_high_left, ax_low_left, ax_scatter, 
             ax_high_right, ax_low_right, ax_summary],
            show_ylabel=show_ylabel
        )
        
        # Add title centered between panels
        pos_left = ax_scatter.get_position()
        pos_right = ax_summary.get_position()
        center_x = (pos_left.x0 + pos_right.x1) / 2
        
        # Position relative to row 5 top
        fig.text(
            center_x,
            gs_main[4].get_position(fig).y1 + 0.01,
            f"{amplitude} µA, {frequency} Hz",
            ha='center',
            fontsize=9,
            weight='bold',
            color=color,
            transform=fig.transFigure
        )
    
    # ========================================================================
    # ROW 6: REGRESSION
    # ========================================================================
    
    print("\n" + "="*80)
    print("ROW 6: Regression Analysis")
    print("="*80)
    
    gs_row6 = GridSpecFromSubplotSpec(1, 6, subplot_spec=gs_main[5],
                                       wspace=0.35)
    
    for idx, (amplitude, frequency) in enumerate(FIGURE2_SPATIAL_PARAMS):
        print(f"\n{frequency} Hz:")
        
        # Prepare regression data
        reg_data = prepare_regression_data_with_connectivity(amplitude, frequency)
        
        # Remove NaN
        valid_mask = ~(
            np.isnan(reg_data['delta_vl']) |
            np.isnan(reg_data['distances']) |
            np.isnan(reg_data['lfp_amplitudes']) |
            np.any([np.isnan(reg_data['connectivity'][k]) 
                   for k in reg_data['connectivity'].keys()], axis=0)
        )
        
        # Build predictor matrix
        X_dict = {
            'distance': reg_data['distances'][valid_mask],
            'lfp_amplitude': reg_data['lfp_amplitudes'][valid_mask],
            'strength_inter': np.array(reg_data['connectivity']['strength_inter'])[valid_mask],
            'strength_intra': np.array(reg_data['connectivity']['strength_intra'])[valid_mask],
            'density_inter': np.array(reg_data['connectivity']['density_inter'])[valid_mask],
            'density_intra': np.array(reg_data['connectivity']['density_intra'])[valid_mask]
        }
        
        X = pd.DataFrame(X_dict)
        y = reg_data['delta_vl'][valid_mask]
        
        # Standardize and regress
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        hierarchical_results = run_hierarchical_regression(X_scaled, y, display_names)
        
        print(f"  R²={hierarchical_results['model3']['r_squared']:.3f}, " +
              f"ΔR²(conn)={hierarchical_results['model3']['delta_r_squared']:.3f}")
        
        # Create panels
        ax_coef = fig.add_subplot(gs_row6[0, idx*2])
        ax_model = fig.add_subplot(gs_row6[0, idx*2 + 1])
        
        show_ylabels = (idx == 0)
        plot_regression_twopanel(
            hierarchical_results, frequency, [ax_coef, ax_model],
            show_ylabels=show_ylabels
        )
    
    # Save
    if save_figures:
        os.makedirs(os.path.join(FIGURES_OUTPUT, 'fig2'), exist_ok=True)
        filename = os.path.join(FIGURES_OUTPUT, 'fig2', 'figure2_complete.png')
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        print(f"\nSaved: {filename}")
    
    plt.close('all')

    print("\n" + "="*80)
    print("COMPLETE FIGURE 2 GENERATION FINISHED!")
    print("="*80)


if __name__ == '__main__':
    generate_complete_figure2(save_figures=True)
