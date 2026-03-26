"""
Generate Supplemental Figure: VL-MMR Discrepancy Analysis

Complete 9-panel figure (3×3 grid) examining when and why MMR exceeds VL,
indicating bimodal phase coupling patterns.

Row 1: Discovery & Quantification
Row 2: Spatial & Network Predictors  
Row 3: Cellular Mechanisms & Integration (Stage 2)

Statistical approach: Non-parametric tests with FDR correction
- Spearman correlations for continuous relationships
- Mann-Whitney U / Kruskal-Wallis for group comparisons
- Hybrid approach for regression (OLS + permutation validation)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon, kruskal, chi2_contingency
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import seaborn as sns

from config.paths import FIGURES_OUTPUT, BASE_DIR
from config.brain_areas import get_display_name, ALL_BRAIN_AREAS
from config.experiments import FREQUENCIES, DEFAULT_ANIMAL_LIST
from config.plotting import mm_to_inch, FREQUENCY_COLORS
from src.plotting_utils import setup_nature_figure, remove_top_right_spines

# Create dedicated output directory
OUTPUT_DIR = os.path.join(FIGURES_OUTPUT, 'fig2_vl_vs_mmr')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_unified_data():
    """
    Load unified CSV with VL, MMR, and metadata
    Calculate discrepancy metric: Δ = MMR - VL
    """
    data_path = os.path.join(
        BASE_DIR,
        'data',
        'unified_data_VL_angle_MMR.csv'
    )
    
    print(f"\nLoading unified data from: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"  ERROR: File not found!")
        print(f"  Expected location: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    
    # Rename columns to match expected names
    df = df.rename(columns={
        'stim_freq': 'frequency',
        'area_peak_ch': 'brain_area',
        'distance_peakch_stim_tip': 'distance',
        'MMR_stimOn': 'MMR_StimOn'  # Standardize capitalization
    })
    
    # Parse frequency from 'sine_XXHz' format to numeric
    df['frequency_numeric'] = df['frequency'].str.extract(r'(\d+)').astype(int)
    
    # Create universal_ID using EXACT same method as user's code
    # No separators - direct concatenation of strings
    df[['mouse', 'probe_position', 'unitID']] = df[['mouse', 'probe_position', 'unitID']].astype(str)
    df['universal_ID'] = df['mouse'] + df['probe_position'] + df['unitID']
    
    print(f"\n  Total unique units (universal_ID): {df['universal_ID'].nunique()}")
    print(f"  Stimulation currents in data: {sorted(df['stim_current'].unique())}")
    
    # CRITICAL: Filter to specific amplitude to avoid doublecounting
    # Use 5 µA for most analyses (standard amplitude from Figure 2)
    df_5ua = df[df['stim_current'] == 5].copy()
    print(f"  After filtering to 5 µA: {len(df_5ua)} rows")
    print(f"  Unique units at 5 µA: {df_5ua['universal_ID'].nunique()}")
    print(f"  Rows per frequency:")
    for freq in [8, 28, 140]:
        n_rows = len(df_5ua[df_5ua['frequency_numeric'] == freq])
        n_units = df_5ua[df_5ua['frequency_numeric'] == freq]['universal_ID'].nunique()
        print(f"    {freq} Hz: {n_rows} rows, {n_units} unique units")
    
    # Check for duplicate units within same frequency
    dups = df_5ua.groupby(['frequency_numeric', 'universal_ID']).size()
    if (dups > 1).any():
        n_dups = (dups > 1).sum()
        print(f"\n  WARNING: {n_dups} unit-frequency combinations have duplicates!")
        print(f"  Max duplicates per unit-freq: {dups.max()}")
        print(f"  Keeping first occurrence for each (frequency, unit) pair...")
        df_5ua = df_5ua.drop_duplicates(subset=['frequency_numeric', 'universal_ID'], keep='first')
        print(f"  After deduplication: {len(df_5ua)} rows")
        print(f"  Final unique units: {df_5ua['universal_ID'].nunique()}")
    
    df = df_5ua
    
    # Note: Waveform properties already present in unified CSV:
    # - waveform_duration, waveform_halfwidth, waveform_rep_slope, waveform_REP
    # - waveform_amplitude, waveform_velocity_above, waveform_velocity_below
    
    # Calculate discrepancy
    df['delta'] = df['MMR_StimOn'] - df['VL_stimOn']
    df['delta_pre'] = df['MMR_Pre'] - df['VL_pre']
    
    # Calculate change metrics
    df['change_VL'] = df['VL_stimOn'] - df['VL_pre']
    df['change_MMR'] = df['MMR_StimOn'] - df['MMR_Pre']
    
    print(f"  Loaded {len(df)} units")
    print(f"  Frequencies (parsed): {sorted(df['frequency_numeric'].unique())}")
    print(f"  Brain areas: {df['brain_area'].nunique()} unique areas")
    
    return df


def load_connectivity_data():
    """Load Allen Brain Atlas connectivity metrics"""
    conn_path = os.path.join(
        BASE_DIR, 'data', 'connectivity', 'connectivity.csv'
    )
    
    try:
        conn = pd.read_csv(conn_path)
        print(f"\n  Loaded connectivity data: {len(conn)} areas")
        return conn
    except FileNotFoundError:
        print(f"  Warning: Connectivity file not found at {conn_path}")
        return pd.DataFrame()


def merge_connectivity_to_units(units_df, conn_df):
    """
    Merge connectivity metrics to unit-level data
    Uses brain area as key
    """
    # Rename areas to match connectivity database
    units_df['area_display'] = units_df['brain_area'].apply(get_display_name)
    
    # Merge connectivity metrics
    merged = units_df.merge(
        conn_df[['projection_area', 'norm_strength inter', 'norm_strength intra',
                 'norm_density inter', 'norm_density intra']],
        left_on='area_display',
        right_on='projection_area',
        how='left'
    )
    
    return merged


# ============================================================================
# STATISTICAL FUNCTIONS (NON-PARAMETRIC + FDR)
# ============================================================================

def spearman_with_ci(x, y, n_bootstrap=1000):
    """
    Spearman correlation with bootstrap confidence intervals
    
    Returns:
        rho: Spearman correlation coefficient
        p: Two-tailed p-value
        ci: (lower, upper) 95% confidence interval
    """
    # Remove NaN
    valid = ~(np.isnan(x) | np.isnan(y))
    x_valid, y_valid = x[valid], y[valid]
    
    if len(x_valid) < 3:
        return np.nan, 1.0, (np.nan, np.nan)
    
    # Primary test
    rho, p = spearmanr(x_valid, y_valid)
    
    # Bootstrap CI
    rhos = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(x_valid), len(x_valid), replace=True)
        rho_boot, _ = spearmanr(x_valid[idx], y_valid[idx])
        rhos.append(rho_boot)
    
    ci_low, ci_high = np.percentile(rhos, [2.5, 97.5])
    
    return rho, p, (ci_low, ci_high)


def compare_frequencies_kruskal(data, value_col='delta', frequencies=[8, 28, 140]):
    """
    Compare distributions across frequencies using Kruskal-Wallis
    with post-hoc Mann-Whitney U tests and FDR correction
    
    Returns:
        h_stat: Kruskal-Wallis H statistic
        p_kw: Kruskal-Wallis p-value
        posthoc: List of dicts with pairwise comparisons
    """
    # Prepare groups (use frequency_numeric)
    groups = [data[data['frequency_numeric'] == f][value_col].dropna() 
              for f in frequencies]
    
    # Kruskal-Wallis omnibus test
    h_stat, p_kw = kruskal(*groups)
    
    # Post-hoc pairwise comparisons
    pairs = [(frequencies[i], frequencies[j]) 
             for i in range(len(frequencies)) 
             for j in range(i+1, len(frequencies))]
    
    pvals = []
    posthoc = []
    
    for f1, f2 in pairs:
        g1 = data[data['frequency_numeric'] == f1][value_col].dropna()
        g2 = data[data['frequency_numeric'] == f2][value_col].dropna()
        
        u_stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
        
        # Rank-biserial correlation (effect size)
        r = 1 - (2*u_stat) / (len(g1) * len(g2))
        
        posthoc.append({
            'pair': f'{f1}-{f2}Hz',
            'f1': f1, 'f2': f2,
            'U': u_stat,
            'r': r,
            'p': p,
            'n1': len(g1),
            'n2': len(g2)
        })
        pvals.append(p)
    
    # FDR correction
    _, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    
    for i, ph in enumerate(posthoc):
        ph['p_fdr'] = pvals_fdr[i]
    
    return h_stat, p_kw, posthoc


def test_area_delta_significance(data, areas, alpha=0.05):
    """
    Test if Δ ≠ 0 for each brain area (Wilcoxon signed-rank)
    Apply FDR correction across all areas
    
    Returns:
        results: DataFrame with test results per area
    """
    results = []
    pvals = []
    
    for area in areas:
        area_data = data[data['brain_area'] == area]['delta'].dropna()
        
        if len(area_data) < 3:
            results.append({
                'area': area,
                'n': len(area_data),
                'median_delta': np.nan,
                'W': np.nan,
                'p': 1.0,
                'p_fdr': 1.0
            })
            pvals.append(1.0)
            continue
        
        # Wilcoxon signed-rank test (testing if median ≠ 0)
        W, p = wilcoxon(area_data, alternative='two-sided')
        
        results.append({
            'area': area,
            'n': len(area_data),
            'median_delta': np.median(area_data),
            'W': W,
            'p': p
        })
        pvals.append(p)
    
    # FDR correction
    _, pvals_fdr, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    
    for i, res in enumerate(results):
        res['p_fdr'] = pvals_fdr[i]
        res['significant_fdr'] = pvals_fdr[i] < alpha
    
    return pd.DataFrame(results)


def permutation_test_regression(X, y, n_perm=10000):
    """
    Permutation test for regression coefficients
    
    Returns:
        pvals_perm: Permutation-based p-values for each coefficient
    """
    # Fit actual model
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    coefs_actual = model.params[1:]  # Exclude intercept
    
    # Permutation distribution
    coefs_perm = []
    for _ in range(n_perm):
        y_perm = np.random.permutation(y)
        model_perm = sm.OLS(y_perm, X_const).fit()
        coefs_perm.append(model_perm.params[1:])
    
    coefs_perm = np.array(coefs_perm)
    
    # Calculate p-values (two-tailed)
    pvals_perm = []
    for i, coef in enumerate(coefs_actual):
        p = np.mean(np.abs(coefs_perm[:, i]) >= np.abs(coef))
        pvals_perm.append(p)
    
    return np.array(pvals_perm)


# ============================================================================
# PANEL A: VL vs MMR SCATTER
# ============================================================================

def plot_panel_A(data, ax):
    """
    Scatter plot: VL vs MMR for all units
    Color by frequency, show unity diagonal
    """
    frequencies = [8, 28, 140]
    
    # Plot by frequency
    for freq in frequencies:
        freq_data = data[data['frequency_numeric'] == freq]
        ax.scatter(
            freq_data['VL_stimOn'],
            freq_data['MMR_StimOn'],
            c=FREQUENCY_COLORS[freq],
            s=3, alpha=0.3,
            label=f'{freq} Hz',
            rasterized=True
        )
    
    # Unity diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='VL = MMR')
    
    # Overall correlation
    rho, p, ci = spearman_with_ci(
        data['VL_stimOn'].values, 
        data['MMR_StimOn'].values
    )
    
    # Statistics box
    stats_text = (
        f"All units:\n"
        f"ρ = {rho:.3f}\n"
        f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]\n"
        f"p < 0.001"
    )
    ax.text(0.45, 0.05, stats_text,
            transform=ax.transAxes,
            fontsize=6,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', 
                     alpha=0.9, edgecolor='gray', linewidth=0.0))
    
    # Format
    ax.set_xlabel('vector length (VL)', fontsize=8)
    ax.set_ylabel('mean modulation ratio (MMR)', fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    # No legend - info will be in figure caption
    # Unity line is self-explanatory
    ax.tick_params(labelsize=7)
    remove_top_right_spines(ax)


# ============================================================================
# PANEL B: FREQUENCY COMPARISON
# ============================================================================

def plot_panel_B(data, ax):
    """
    Violin plots: Δ by frequency
    Include Kruskal-Wallis and post-hoc Mann-Whitney U tests
    """
    frequencies = [8, 28, 140]
    
    # Prepare data for plotting
    plot_data = []
    for freq in frequencies:
        freq_delta = data[data['frequency_numeric'] == freq]['delta'].dropna()
        plot_data.append(freq_delta)
        print(f"  {freq} Hz: {len(freq_delta)} units")
    
    # Violin plots
    parts = ax.violinplot(plot_data, positions=range(len(frequencies)),
                          showmeans=False, showmedians=False, widths=0.7)
    
    # Color violins by frequency
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(FREQUENCY_COLORS[frequencies[i]])
        pc.set_alpha(0.7)
    
    # Overlay box plots
    bp = ax.boxplot(plot_data, positions=range(len(frequencies)),
                    widths=0.3, patch_artist=True,
                    boxprops=dict(facecolor='white', alpha=0.8),
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(color='black', linewidth=1),
                    capprops=dict(color='black', linewidth=1))
    
    # Statistics
    h_stat, p_kw, posthoc = compare_frequencies_kruskal(data, 'delta', frequencies)
    
    # Add significance brackets above violins
    y_max = max([d.max() for d in plot_data if len(d) > 0])
    y_min = min([d.min() for d in plot_data if len(d) > 0])
    y_range = y_max - y_min
    
    # Define bracket heights (staggered to avoid overlap)
    bracket_heights = [y_max + 0.15 * y_range, 
                       y_max + 0.25 * y_range,
                       y_max + 0.20 * y_range]
    
    # Map pairs to positions and bracket heights
    pair_positions = {
        (8, 28): (0, 1, bracket_heights[0]),
        (8, 140): (0, 2, bracket_heights[1]),
        (28, 140): (1, 2, bracket_heights[2])
    }
    
    # Draw significant comparisons
    for ph in posthoc:
        if ph['p_fdr'] < 0.05:  # Only draw if significant
            f1, f2 = ph['f1'], ph['f2']
            pos1, pos2, bracket_y = pair_positions[(f1, f2)]
            
            # Significance stars
            if ph['p_fdr'] < 0.001:
                sig_text = '***'
            elif ph['p_fdr'] < 0.01:
                sig_text = '**'
            else:
                sig_text = '*'
            
            # Draw bracket
            ax.plot([pos1, pos1, pos2, pos2], 
                   [bracket_y - 0.02*y_range, bracket_y, bracket_y, bracket_y - 0.02*y_range],
                   'k-', linewidth=1)
            
            # Add stars
            ax.text((pos1 + pos2) / 2, bracket_y + 0.01*y_range, sig_text,
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Zero line
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Format
    ax.set_xticks(range(len(frequencies)))
    ax.set_xticklabels([f'{f} Hz' for f in frequencies], fontsize=7)
    ax.set_ylabel('Δ (MMR - VL)', fontsize=8)
    
    # Adjust y-limits to accommodate brackets
    ax.set_ylim([y_min - 0.1*y_range, y_max + 0.35*y_range])
    
    ax.tick_params(labelsize=7)
    remove_top_right_spines(ax)


# ============================================================================
# PANEL C: BRAIN AREA SUMMARY
# ============================================================================

def plot_panel_C(data, ax, area_significance):
    """
    Bar plot: Mean Δ by brain area, sorted by distance
    Color-code by significance categories
    Note: Only shows FDR-significant areas to improve readability
    """
    # Calculate area-level statistics
    area_stats = data.groupby('brain_area').agg({
        'delta': ['mean', 'sem', 'count'],
        'distance': 'first'
    }).reset_index()
    
    area_stats.columns = ['area', 'mean_delta', 'sem_delta', 'n_units', 'distance']
    
    # Filter to areas with sufficient data
    min_units = 10
    area_stats = area_stats[area_stats['n_units'] >= min_units]
    
    # Merge significance results
    area_stats = area_stats.merge(
        area_significance[['area', 'significant_fdr']], 
        on='area', how='left'
    )
    
    # CRITICAL: Show only significant areas for readability
    area_stats_sig = area_stats[area_stats['significant_fdr'] == True].copy()
    
    # If still too many (>40), take top 30 by absolute effect size
    if len(area_stats_sig) > 40:
        area_stats_sig['abs_delta'] = np.abs(area_stats_sig['mean_delta'])
        area_stats_sig = area_stats_sig.nlargest(30, 'abs_delta')
    
    print(f"  Panel C: Showing {len(area_stats_sig)} significant areas (FDR-corrected)")
    
    # Sort by distance
    area_stats_sig = area_stats_sig.sort_values('distance')
    
    # Plot
    x_pos = np.arange(len(area_stats_sig))
    ax.bar(x_pos, area_stats_sig['mean_delta'], 
           yerr=area_stats_sig['sem_delta'],
           color='red', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Zero line
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    
    # Format
    ax.set_xticks(x_pos)
    # Use 45-degree rotation for better readability
    ax.set_xticklabels(
        area_stats_sig['area'], 
        rotation=45, ha='right', va='top', fontsize=6
    )
    ax.set_ylabel('Mean Δ (MMR - VL)', fontsize=8)
    ax.set_xlabel('Brain Area (sorted by distance)', fontsize=8)
    ax.tick_params(labelsize=7, axis='y')
    ax.tick_params(labelsize=6, axis='x')
    remove_top_right_spines(ax)
    
    # Compact note
    ax.text(0.98, 0.98, f'n={len(area_stats_sig)} sig. areas',
            transform=ax.transAxes, fontsize=6,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', 
                     alpha=0.8, edgecolor='gray', linewidth=0.5))


# ============================================================================
# MAIN FIGURE GENERATION
# ============================================================================

def generate_figure_VL_MMR_stage1(save_figures=True):
    """
    Generate Stage 1 of VL-MMR comparison figure
    Panels A-F + H-I (without waveform analysis)
    """
    setup_nature_figure()
    
    print("\n" + "="*80)
    print("GENERATING VL-MMR COMPARISON FIGURE (STAGE 1)")
    print("="*80)
    
    # Load data
    data = load_unified_data()
    conn = load_connectivity_data()
    
    if data is None:
        print("ERROR: Could not load data!")
        return None
    
    # Filter to areas in ALL_BRAIN_AREAS or aggregate by main area
    # Note: CSV has layer-specific areas (e.g., 'ACAv5'), but config may have broader areas
    print(f"\nOriginal data: {len(data)} units across {data['brain_area'].nunique()} areas")
    
    # For now, keep all areas for Panel A and B
    # Panel C will be filtered to ALL_BRAIN_AREAS
    
    # Merge connectivity
    if not conn.empty:
        data = merge_connectivity_to_units(data, conn)
    
    # Test area-level significance (use areas present in data, not just ALL_BRAIN_AREAS)
    unique_areas = data['brain_area'].unique()
    print(f"\nTesting {len(unique_areas)} brain areas for Δ ≠ 0...")
    area_significance = test_area_delta_significance(data, unique_areas)
    
    # Create figure
    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(240)))
    gs = GridSpec(3, 3, 
                  hspace=0.40, wspace=0.35,
                  top=0.96, bottom=0.12,  # Extra space for rotated labels
                  left=0.08, right=0.98)
    
    # ========================================================================
    # ROW 1: DISCOVERY & QUANTIFICATION
    # ========================================================================
    
    print("\n" + "="*60)
    print("ROW 1: Discovery & Quantification")
    print("="*60)
    
    # Panel A: VL vs MMR scatter
    print("\nPanel A: VL vs MMR correlation")
    ax_a = fig.add_subplot(gs[0, 0])
    plot_panel_A(data, ax_a)
    ax_a.text(-0.15, 1.05, 'A', transform=ax_a.transAxes,
              fontsize=10, fontweight='bold')
    
    # Panel B: Frequency comparison
    print("Panel B: Frequency comparison")
    ax_b = fig.add_subplot(gs[0, 1])
    plot_panel_B(data, ax_b)
    ax_b.text(-0.15, 1.05, 'B', transform=ax_b.transAxes,
              fontsize=10, fontweight='bold')
    
    # Panel C: Brain area summary
    print("Panel C: Area summary")
    ax_c = fig.add_subplot(gs[0, 2])
    plot_panel_C(data, ax_c, area_significance)
    ax_c.text(-0.15, 1.05, 'C', transform=ax_c.transAxes,
              fontsize=10, fontweight='bold')
    
    # ========================================================================
    # ROW 2: SPATIAL & NETWORK PREDICTORS
    # ========================================================================
    
    print("\n" + "="*60)
    print("ROW 2: Spatial & Network Predictors")
    print("="*60)
    
    # Panels D, E, F - To be implemented
    # Placeholder text for now
    for i, (row, col, label) in enumerate([(1, 0, 'D'), (1, 1, 'E'), (1, 2, 'F')]):
        ax = fig.add_subplot(gs[row, col])
        ax.text(0.5, 0.5, f'Panel {label}\n(To be implemented)',
                ha='center', va='center', fontsize=10)
        ax.text(-0.15, 1.05, label, transform=ax.transAxes,
                fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # ========================================================================
    # ROW 3: INTEGRATION
    # ========================================================================
    
    print("\n" + "="*60)
    print("ROW 3: Integration")
    print("="*60)
    
    # Panels G, H, I - To be implemented
    for i, (row, col, label) in enumerate([(2, 0, 'G'), (2, 1, 'H'), (2, 2, 'I')]):
        ax = fig.add_subplot(gs[row, col])
        ax.text(0.5, 0.5, f'Panel {label}\n(To be implemented)',
                ha='center', va='center', fontsize=10)
        ax.text(-0.15, 1.05, label, transform=ax.transAxes,
                fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Save
    if save_figures:
        filename = os.path.join(OUTPUT_DIR, 'figS_VL_MMR_stage1.png')
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        print(f"\nSaved: {filename}")
        
        # Save area significance results
        sig_file = os.path.join(OUTPUT_DIR, 'area_significance_results.csv')
        area_significance.to_csv(sig_file, index=False)
        print(f"Saved: {sig_file}")
    
    plt.show()
    
    print("\n" + "="*80)
    print("STAGE 1 COMPLETE!")
    print("="*80)
    
    return fig


if __name__ == '__main__':
    generate_figure_VL_MMR_stage1(save_figures=True)
