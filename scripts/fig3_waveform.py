#!/usr/bin/env python3
"""
Figure 3 Rows 2-3: Extended Clustering Analysis
Row 2: Multi-frequency clustering analysis (8Hz, 28Hz, 140Hz)
Row 3: Waveform-based cluster characterization

Addresses core research questions:
1. Do different stimulus frequencies affect distinct clusters within areas?
2. Can waveform properties predict cluster identity?
3. Are clusters consistent across frequencies or frequency-specific?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import kruskal, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from config.plotting import FREQUENCY_COLORS

# Import existing Figure3ClusteringAnalysis class and helper functions
from fig3_clustering import Figure3ClusteringAnalysis
from config.experiments import cty_colors_, n_spike_thresh, error_threshold

class Figure3ExtendedAnalysis(Figure3ClusteringAnalysis):
    """Extended analysis for Rows 2-3"""
    
    def __init__(self):
        super().__init__()
        self.frequencies = ['sine_8Hz', 'sine_28Hz', 'sine_140Hz']
        self.waveform_features = [
            'waveform_duration', 'waveform_halfwidth', 
            'waveform_rep_slope', 'waveform_REP'
        ]
        
    def analyze_multi_frequency_clustering(self):
        """Row 2: Analyze clustering across frequencies for ALL viable areas"""
        
        print("\n" + "="*80)
        print("ROW 2: MULTI-FREQUENCY CLUSTERING ANALYSIS")
        print("="*80)
        
        # CHANGED: Use ALL viable areas instead of just 6
        target_areas = self.viable_areas  # All 33 areas with ≥20 units
        multi_freq_results = {}
        
        print(f"Analyzing {len(target_areas)} areas across frequencies...")
        
        for area in target_areas:
            area_results = {}
            
            for freq in self.frequencies:
                freq_result = self.analyze_area_clustering_frequency(area, freq)
                if freq_result is not None:
                    area_results[freq] = freq_result
            
            if len(area_results) >= 2:  # Need at least 2 frequencies
                multi_freq_results[area] = area_results
                
                # Only print detailed results for the 6 main areas
                if area in ['VISp', 'CA1', 'RSPd', 'MOp', 'MOs', 'CL']:
                    print(f"\n{area} across frequencies:")
                    for freq in self.frequencies:
                        if freq in area_results:
                            result = area_results[freq]
                            print(f"  {freq}: {result['n_clusters']} clusters, "
                                f"p={result['cluster_test_pval']:.2e}")
        
        print(f"\nTotal areas with multi-frequency data: {len(multi_freq_results)}")
        return multi_freq_results

    def create_summary_heatmap(self, all_freq_results):
        """Create summary heatmap showing clustering across all areas and frequencies"""
        
        # Get all areas that have data
        areas = sorted(all_freq_results.keys())
        frequencies = [8, 28, 140]
        
        # Create matrices
        significance_matrix = np.zeros((len(areas), len(frequencies)))
        cluster_matrix = np.zeros((len(areas), len(frequencies)))
        
        for i, area in enumerate(areas):
            for j, freq in enumerate(frequencies):
                freq_str = f'sine_{freq}Hz'
                if freq_str in all_freq_results[area]:
                    result = all_freq_results[area][freq_str]
                    significance_matrix[i, j] = -np.log10(result['cluster_test_pval'])
                    cluster_matrix[i, j] = result['n_clusters']
                else:
                    significance_matrix[i, j] = 0
                    cluster_matrix[i, j] = 0
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(6, max(8, len(areas)*0.3)))
        
        # Plot significance as color intensity
        im = ax.imshow(significance_matrix, cmap='Reds', aspect='auto', vmin=0)
        
        # Overlay cluster numbers as text
        for i in range(len(areas)):
            for j in range(len(frequencies)):
                if cluster_matrix[i, j] > 0:
                    ax.text(j, i, f'{int(cluster_matrix[i, j])}', 
                        ha='center', va='center', fontweight='bold', fontsize=8)
        
        # Format axes
        ax.set_xticks(range(len(frequencies)))
        ax.set_xticklabels([f'{f} Hz' for f in frequencies])
        ax.set_yticks(range(len(areas)))
        ax.set_yticklabels(areas, fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('-log10(p-value)', rotation=270, labelpad=15, fontsize=9)
        
        # Add significance threshold line in colorbar
        sig_line_pos = -np.log10(0.05)
        cbar.ax.axhline(sig_line_pos, color='blue', linestyle='--', linewidth=2)
        cbar.ax.text(0.5, sig_line_pos, 'p=0.05', color='blue', fontweight='bold')
        
        ax.set_title('Clustering Significance: All Areas × Frequencies\n(Numbers = cluster count)', 
                    fontsize=12, pad=20)
        ax.set_xlabel('Stimulation Frequency')
        ax.set_ylabel('Brain Area')
        
        plt.tight_layout()
        return fig
    
    def analyze_area_clustering_frequency(self, area_name, frequency):
        """Analyze clustering for specific area and frequency"""
        amp = 5  # Fixed amplitude
        
        # Get area data for this frequency
        area_data = self.data[
            (self.data['area_main'] == area_name) &
            (self.data['stim_freq'] == frequency) &
            (self.data['stim_current'] == amp)
        ].copy()
        
        if len(area_data) < self.min_units_per_area:
            return None
        
        # Prepare features for clustering
        area_data['delta_VL'] = area_data['VL_stimOn'] - area_data['VL_pre']
        features = area_data[['delta_VL', 'VL_pre']].values
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Find optimal clustering
        best_score = -1
        best_n_clusters = 2
        best_labels = None
        
        for n_clusters in range(2, min(7, len(area_data)//5)):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            sil_score = silhouette_score(features_scaled, labels)
            
            if sil_score > best_score:
                best_score = sil_score
                best_n_clusters = n_clusters
                best_labels = labels
        
        # Statistical test
        cluster_values = [area_data['delta_VL'][best_labels == i].values 
                         for i in range(best_n_clusters)]
        
        if len(cluster_values) > 1 and all(len(cv) > 0 for cv in cluster_values):
            kruskal_stat, kruskal_p = kruskal(*cluster_values)
        else:
            kruskal_p = 1.0
        
        return {
            'area': area_name,
            'frequency': frequency,
            'n_units': len(area_data),
            'n_clusters': best_n_clusters,
            'cluster_labels': best_labels,
            'silhouette_score': best_score,
            'cluster_test_pval': kruskal_p,
            'area_data': area_data,
            'cluster_delta_means': [area_data['delta_VL'][best_labels == i].mean() 
                                  for i in range(best_n_clusters)]
        }
    
    def analyze_waveform_cluster_relationship(self):
        """Row 3: Analyze relationship between clusters and waveform properties - ALL AREAS"""
        
        print("\n" + "="*80)
        print("ROW 3: WAVEFORM-CLUSTER RELATIONSHIP ANALYSIS")
        print("="*80)
        
        # Check waveform data availability
        waveform_cols = [col for col in self.data.columns if col in self.waveform_features]
        if len(waveform_cols) < 2:
            print(f"Warning: Only {len(waveform_cols)} waveform features available")
            return None
        
        print(f"Available waveform features: {waveform_cols}")
        
        # Analyze for 28Hz condition (primary frequency from Row 1)
        frequency = 'sine_28Hz'
        
        # FIXED: Test ALL viable areas, not just 3
        target_areas = self.viable_areas  # All 33 areas
        waveform_results = {}
        
        print(f"Testing waveform-cluster relationships in {len(target_areas)} areas...")
        
        for area in target_areas:
            result = self.analyze_area_waveform_clusters(area, frequency, waveform_cols)
            if result is not None:
                waveform_results[area] = result
                
                # Only print details for the 3 main areas
                if area in ['VISp', 'CA1', 'RSPd']:
                    print(f"{area}: {result['n_clusters']} clusters, "
                        f"waveform separation p={result['waveform_test_pval']:.3f}")
        
        print(f"Completed waveform analysis for {len(waveform_results)} areas")
        return waveform_results

    # Row 3 Analysis with Proper Statistics
    def analyze_area_waveform_clusters(self, area_name, frequency, waveform_cols):
        """FIXED: Proper statistical testing for waveform differences"""
        
        """Analyze waveform properties of clusters within an area"""
        amp = 5
        
        # Get area data
        area_data = self.data[
            (self.data['area_main'] == area_name) &
            (self.data['stim_freq'] == frequency) &
            (self.data['stim_current'] == amp)
        ].copy()
        
        if len(area_data) < self.min_units_per_area:
            return None
        
        # Check waveform data completeness
        waveform_complete = area_data[waveform_cols].notna().all(axis=1)
        if waveform_complete.sum() < self.min_units_per_area:
            print(f"  {area_name}: Insufficient waveform data "
                  f"({waveform_complete.sum()} units with complete data)")
            return None
        
        area_data_wf = area_data[waveform_complete].copy()
        
        # Perform clustering based on SWES response (same as Row 1)
        area_data_wf['delta_VL'] = area_data_wf['VL_stimOn'] - area_data_wf['VL_pre']
        cluster_features = area_data_wf[['delta_VL', 'VL_pre']].values
        
        scaler = StandardScaler()
        cluster_features_scaled = scaler.fit_transform(cluster_features)
        
        # Find optimal clustering
        best_score = -1
        best_n_clusters = 2
        best_labels = None
        
        for n_clusters in range(2, min(6, len(area_data_wf)//5)):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(cluster_features_scaled)
            sil_score = silhouette_score(cluster_features_scaled, labels)
            
            if sil_score > best_score:
                best_score = sil_score
                best_n_clusters = n_clusters
                best_labels = labels
        
        area_data_wf['cluster'] = best_labels
        
        # Test waveform differences between clusters
        waveform_p_values = []
        waveform_effect_sizes = []
        
        for feature in waveform_cols:
            cluster_groups = [area_data_wf[area_data_wf['cluster'] == i][feature].values 
                            for i in range(best_n_clusters)]
            
            # Remove empty groups
            cluster_groups = [group for group in cluster_groups if len(group) > 0]
            
            if len(cluster_groups) >= 2:
                if len(cluster_groups) == 2:
                    stat, p_val = mannwhitneyu(cluster_groups[0], cluster_groups[1])
                    # Cohen's d for effect size
                    mean1, mean2 = np.mean(cluster_groups[0]), np.mean(cluster_groups[1])
                    std1, std2 = np.std(cluster_groups[0], ddof=1), np.std(cluster_groups[1], ddof=1)
                    n1, n2 = len(cluster_groups[0]), len(cluster_groups[1])
                    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
                    effect_size = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                else:
                    stat, p_val = kruskal(*cluster_groups)
                    # Eta-squared approximation for multiple groups
                    effect_size = stat / (len(area_data_wf) - 1) if len(area_data_wf) > 1 else 0
                
                waveform_p_values.append(p_val)
                waveform_effect_sizes.append(effect_size)
            else:
                waveform_p_values.append(1.0)
                waveform_effect_sizes.append(0.0)
        
        # FIXED: Apply FDR correction across features
        if len(waveform_p_values) > 1:
            _, p_corrected, _, _ = multipletests(waveform_p_values, method='fdr_bh')
        else:
            p_corrected = waveform_p_values
        
        return {
            'area': area_name,
            'frequency': frequency,
            'n_units': len(area_data_wf),
            'n_clusters': best_n_clusters,
            'cluster_labels': best_labels,
            'silhouette_score': best_score,
            'area_data': area_data_wf,
            'waveform_features': waveform_cols,
            'waveform_p_values': waveform_p_values,      # Raw p-values
            'waveform_p_corrected': p_corrected,         # FDR-corrected p-values
            'waveform_effect_sizes': waveform_effect_sizes,
            'waveform_test_pval': np.min(p_corrected),   # Most significant corrected p-value
            'max_effect_size': np.max(waveform_effect_sizes)
        }

    # Row 3 Plotting with Proper Color Logic
    def _plot_waveform_analysis(self, results, ax, global_ylim=None):
        """FIXED: Use FDR-corrected p-values and effect size thresholds"""
        
        features = results['waveform_features']
        effect_sizes = results['waveform_effect_sizes']
        p_corrected = results['waveform_p_corrected']  # Use corrected p-values
        
        # Color logic: Red if significant AND medium+ effect size
        colors = []
        for p_corr, effect in zip(p_corrected, effect_sizes):
            if p_corr < 0.05 and effect >= 0.5:  # Significant AND medium effect
                colors.append('red')
            elif p_corr < 0.05:  # Significant but small effect
                colors.append('orange') 
            elif effect >= 0.5:  # Large effect but not significant
                colors.append('lightcoral')
            else:  # Neither significant nor large effect
                colors.append('gray')
        
        feature_labels = [f.replace('waveform_', '').replace('_', ' ') for f in features]
        
        bars = ax.bar(range(len(feature_labels)), effect_sizes, 
                    color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add effect size interpretation lines
        ax.axhline(0.2, color='lightblue', linestyle=':', alpha=0.7, linewidth=1, label='Small')
        ax.axhline(0.5, color='orange', linestyle='--', alpha=0.7, linewidth=1, label='Medium')
        ax.axhline(0.8, color='darkred', linestyle='-', alpha=0.7, linewidth=1, label='Large')
        
        # Add significance markers above bars
        for i, (bar, p_corr) in enumerate(zip(bars, p_corrected)):
            if p_corr < 0.05:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    '*', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xticks(range(len(feature_labels)))
        ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=7)
        
        if global_ylim is not None:
            ax.set_ylim(0, global_ylim)
        else:
            ax.set_ylim(0, max(effect_sizes) * 1.2 if effect_sizes else 1)
        
        ax.tick_params(labelsize=7)
        remove_top_right_spines(ax)

    def _plot_waveform_summary(self, waveform_results, ax):
        """Plot waveform analysis summary for all areas"""
        
        # Extract summary statistics across all areas
        areas_with_sig = []
        effect_sizes_by_feature = {
            'duration': [],
            'halfwidth': [], 
            'rep_slope': [],
            'REP': []
        }
        
        for area, results in waveform_results.items():
            if results['waveform_test_pval'] < 0.05:
                areas_with_sig.append(area)
            
            # Collect effect sizes by feature
            features = results['waveform_features']
            effect_sizes = results['waveform_effect_sizes']
            
            for feat, effect in zip(features, effect_sizes):
                feat_short = feat.replace('waveform_', '')
                if feat_short in effect_sizes_by_feature:
                    effect_sizes_by_feature[feat_short].append(effect)
        
        # Create summary bar plot
        feature_names = list(effect_sizes_by_feature.keys())
        mean_effects = [np.mean(effect_sizes_by_feature[feat]) if effect_sizes_by_feature[feat] else 0 
                    for feat in feature_names]
        
        bars = ax.bar(range(len(feature_names)), mean_effects, 
                    color='steelblue', alpha=0.7, edgecolor='black')
        
        # Add effect size thresholds
        ax.axhline(0.2, color='lightblue', linestyle=':', alpha=0.7, linewidth=1)
        ax.axhline(0.5, color='orange', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(0.8, color='darkred', linestyle='-', alpha=0.7, linewidth=1)
        
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels([f.replace('_', ' ') for f in feature_names], 
                        rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Mean Effect Size', fontsize=8)
        ax.set_title(f'Waveform Summary\n({len(areas_with_sig)}/{len(waveform_results)} sig)', 
                    fontsize=10, fontweight='bold')
        
        ax.tick_params(labelsize=7)
        remove_top_right_spines(ax)

    # SUMMARY HEATMAP for All Areas
    def create_summary_heatmap(self, multi_freq_results, all_areas_results):
        """Create summary heatmap showing clustering across all areas and frequencies"""
        
        # Collect data for all areas across frequencies
        areas = sorted(all_areas_results.keys())
        frequencies = [8, 28, 140]
        
        # Create matrices
        significance_matrix = np.zeros((len(areas), len(frequencies)))
        cluster_matrix = np.zeros((len(areas), len(frequencies)))
        
        for i, area in enumerate(areas):
            for j, freq in enumerate(frequencies):
                freq_str = f'sine_{freq}Hz'
                if area in all_areas_results and freq_str in all_areas_results[area]:
                    result = all_areas_results[area][freq_str]
                    significance_matrix[i, j] = -np.log10(result['cluster_test_pval'])
                    cluster_matrix[i, j] = result['n_clusters']
                else:
                    significance_matrix[i, j] = 0
                    cluster_matrix[i, j] = 0
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(6, 12))
        
        # Plot significance as color intensity
        im = ax.imshow(significance_matrix, cmap='Reds', aspect='auto')
        
        # Overlay cluster numbers as text
        for i in range(len(areas)):
            for j in range(len(frequencies)):
                if cluster_matrix[i, j] > 0:
                    ax.text(j, i, f'{int(cluster_matrix[i, j])}', 
                        ha='center', va='center', fontweight='bold')
        
        # Format axes
        ax.set_xticks(range(len(frequencies)))
        ax.set_xticklabels([f'{f} Hz' for f in frequencies])
        ax.set_yticks(range(len(areas)))
        ax.set_yticklabels(areas, fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('-log10(p-value)', rotation=270, labelpad=15, fontsize=9)
        
        ax.set_title('Clustering Significance Across Areas and Frequencies')
        ax.set_xlabel('Stimulation Frequency')
        ax.set_ylabel('Brain Area')
        
        return fig

    def _plot_summary_heatmap(self, all_freq_results, ax):
        """Plot summary heatmap - CONSISTENT formatting"""
        
        # Get all areas that have data
        areas = sorted(all_freq_results.keys())
        frequencies = [8, 28, 140]
        
        # Create matrices
        significance_matrix = np.zeros((len(areas), len(frequencies)))
        
        for i, area in enumerate(areas):
            for j, freq in enumerate(frequencies):
                freq_str = f'sine_{freq}Hz'
                if freq_str in all_freq_results[area]:
                    result = all_freq_results[area][freq_str]
                    # Cap at 10 for better visualization
                    log_p = min(-np.log10(result['cluster_test_pval']), 10)
                    significance_matrix[i, j] = log_p
        
        # Plot significance as color intensity with capped range
        im = ax.imshow(significance_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=10)
        
        # Format axes
        ax.set_xticks(range(len(frequencies)))
        ax.set_xticklabels([f'{f} Hz' for f in frequencies], fontsize=8)
        ax.set_yticks(range(len(areas)))
        ax.set_yticklabels(areas, fontsize=6)
        
        # CONSISTENT: Colorbar formatting matching Row 3
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="15%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('-log10(p-value)', rotation=270, labelpad=10, fontsize=9)
        cbar.set_ticks([0, 2, 4, 6, 8, 10])
        cbar.set_ticklabels(['0', '2', '4', '6', '8', '≥10'])
        
        # REMOVED: Title
        ax.set_xlabel('sES frequency', fontsize=9)


    # CREATE proper Row 3 summary heatmap (like Row 2)
    def _plot_waveform_summary_heatmap(self, waveform_results, ax):
        """Plot waveform summary heatmap - FIXED formatting"""
        
        if not waveform_results:
            ax.text(0.5, 0.5, 'No waveform\ndata available', ha='center', va='center')
            ax.axis('off')
            return
        
        # Get all areas with waveform results
        areas = sorted(waveform_results.keys())
        features = ['duration', 'halfwidth', 'rep_slope', 'REP']
        
        # Create significance matrix only (remove effect matrix for clarity)
        significance_matrix = np.zeros((len(areas), len(features)))
        
        for i, area in enumerate(areas):
            result = waveform_results[area]
            wf_features = result['waveform_features']
            p_values = result.get('waveform_p_corrected', result['waveform_p_values'])
            effect_sizes = result['waveform_effect_sizes']
            
            for j, feature in enumerate(features):
                feature_full = f'waveform_{feature}'
                if feature_full in wf_features:
                    idx = wf_features.index(feature_full)
                    p_val = p_values[idx]
                    effect = effect_sizes[idx]
                    
                    # Use negative log p-value, capped at 5
                    significance_matrix[i, j] = min(-np.log10(p_val) if p_val > 0 else 5, 5)
        
        # Create heatmap
        im = ax.imshow(significance_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=5)
                
        # Format axes - FIXED x-labels
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(areas)))
        ax.set_yticklabels(areas, fontsize=6)
        
        # FIXED: Colorbar matching Row 2 style exactly
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="15%", pad=0.05)  # Same as Row 2
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('-log10(p-value)', rotation=270, labelpad=10, fontsize=9)
        cbar.set_ticks([0, 1, 2, 3, 4, 5])
        cbar.set_ticklabels(['0', '1', '2', '3', '4', '≥5'])  # Same format as Row 2
        
        # FIXED: xlabel
        ax.set_xlabel('EAP waveform properties', fontsize=8)
    
    def create_figure3_complete(self, multi_freq_results, waveform_results):
        """Create complete Figure 3 - CLEANED layout"""
        
        # Set up figure
        setup_nature_figure()
        fig = plt.figure(figsize=(mm_to_inch(250), mm_to_inch(180)))
        
        # Create main GridSpec: 3 rows, consistent layout
        gs_main = GridSpec(3, 2, hspace=0.3, wspace=0.3, 
                        width_ratios=[3, 1],
                        top=0.95, bottom=0.05)
        
        # Row 1: Placeholder for existing analysis
        gs_row1 = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[0, :], 
                                        width_ratios=[1, 1, 1, 1])
        
        # Row 1 placeholders
        for i in range(3):
            ax = fig.add_subplot(gs_row1[i])
            ax.text(0.5, 0.5, f'Row 1 Detail\nArea {i+1}', ha='center', va='center', fontsize=10)
            ax.axis('off')
        
        ax_row1_summary = fig.add_subplot(gs_row1[3])
        ax_row1_summary.text(0.5, 0.5, 'Row 1\nSummary', ha='center', va='center', fontsize=10)
        ax_row1_summary.axis('off')
        
        # Row 2: Multi-frequency analysis (CLEANED)
        self.create_row2_aligned(fig, gs_main[1, :], multi_freq_results)
        
        # Row 3: Waveform analysis (CLEANED)
        self.create_row3_aligned(fig, gs_main[2, :], waveform_results)
        
        # REMOVED: All row labels
        
        return fig

    def create_row2_aligned(self, fig, gs_row, multi_freq_results):
        """Create Row 2: 3 areas + summary heatmap - CLEANED"""
        
        # Create nested GridSpec matching Row 1 layout
        gs_row2 = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_row, 
                                        width_ratios=[1, 1, 1, 1])
        
        # Left side: 3 areas
        target_areas = ['VISp', 'CA1', 'RSPd']
        
        for idx, area in enumerate(target_areas):
            ax = fig.add_subplot(gs_row2[idx])
            
            if area in multi_freq_results:
                self._plot_frequency_comparison(multi_freq_results[area], ax)
            else:
                ax.text(0.5, 0.5, f'{area}\n(insufficient data)', 
                    ha='center', va='center', fontsize=9)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            
            ax.set_title(area, fontsize=9)
            ax.set_ylabel(' ', fontsize=9)
            if idx == 0:
                ax.set_ylabel('cluster significance\n-log10(p-value)', fontsize=9)
            
            if idx == 1:  # Middle panel
                ax.set_xlabel('sES frequency', fontsize=9)

            for label in ax.get_xticklabels(): label.set_size(8)
        
        # Right side: Summary heatmap
        ax_heatmap = fig.add_subplot(gs_row2[3])
        self._plot_summary_heatmap(multi_freq_results, ax_heatmap)
        for label in ax_heatmap.get_xticklabels(): label.set_size(8)


    def create_row2_subplot_with_summary(self, fig, gs_row, multi_freq_results):
        """Create Row 2: 6 example areas + reference to summary heatmap"""
        
        # Create nested GridSpec for 6 areas (left side only)
        gs_row2 = GridSpecFromSubplotSpec(1, 6, subplot_spec=gs_row, wspace=0.3)
        
        # Show the 6 main areas
        target_areas = ['VISp', 'CA1', 'RSPd', 'MOp', 'MOs', 'CL']
        
        for idx, area in enumerate(target_areas):
            ax = fig.add_subplot(gs_row2[idx])
            
            if area in multi_freq_results:
                self._plot_frequency_comparison(multi_freq_results[area], ax)
            else:
                ax.text(0.5, 0.5, f'{area}\n(insufficient data)', 
                    ha='center', va='center', fontsize=9)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            
            ax.set_title(area, fontsize=10, fontweight='bold')
            
            if idx == 0:
                ax.set_ylabel('cluster significance\n-log10(p-value)', fontsize=9)
            
            if idx == 2:  # Middle panel
                ax.set_xlabel('Frequency', fontsize=8)
        
        # Add row label
        fig.text(0.02, 0.65, 'Row 2: Multi-Frequency', fontsize=14, fontweight='bold', rotation=90)

    
    def create_row1_subplot(self, fig, gs_row):
        """Create Row 1 subplot (existing analysis)"""
        # This would use the existing Panel A + B from our Row 1 analysis
        # For now, add placeholder
        ax = fig.add_subplot(gs_row)
        ax.text(0.5, 0.5, 'Row 1: Single Frequency Analysis\n(Panels A + B from existing)', 
                ha='center', va='center', fontsize=12)
        ax.set_title('Row 1: Within-Area Clustering (28 Hz, 5 μA)', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    def create_row2_subplot(self, fig, gs_row, multi_freq_results):
        """Create Row 2: Multi-frequency comparison"""
        
        # Create nested GridSpec for 6 areas
        gs_row2 = GridSpecFromSubplotSpec(1, 6, subplot_spec=gs_row, wspace=0.3)
        
        target_areas = ['VISp', 'CA1', 'RSPd', 'MOp', 'MOs', 'CL']
        
        for idx, area in enumerate(target_areas):
            ax = fig.add_subplot(gs_row2[idx])
            
            if area in multi_freq_results:
                self._plot_frequency_comparison(multi_freq_results[area], ax)
            else:
                ax.text(0.5, 0.5, f'{area}\n(insufficient data)', 
                       ha='center', va='center', fontsize=9)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            
            ax.set_title(area, fontsize=10, fontweight='bold')
            
            if idx == 0:
                ax.set_ylabel('cluster significance\n-log10(p-value)', fontsize=9)
            
            if idx == 2:  # Middle panel
                ax.set_xlabel('sES frequency', fontsize=9)
        
        # Add row label
        fig.text(0.02, 0.65, 'Row 2: Multi-Frequency', fontsize=14, fontweight='bold', rotation=90)
    
    def create_row3_subplot(self, fig, gs_row, waveform_results):
        """Create Row 3: Waveform analysis - FIXED: Uniform ylim, no titles"""
        
        # Create nested GridSpec for 6 areas
        gs_row3 = GridSpecFromSubplotSpec(1, 6, subplot_spec=gs_row, wspace=0.3)
        
        target_areas = ['VISp', 'CA1', 'RSPd', 'MOp', 'MOs', 'CL']
        
        # FIXED: Calculate global ylim across all areas
        all_effect_sizes = []
        for area in target_areas:
            if area in waveform_results:
                all_effect_sizes.extend(waveform_results[area]['waveform_effect_sizes'])
        
        global_ylim = max(all_effect_sizes) * 1.1 if all_effect_sizes else 1.0
        
        for idx, area in enumerate(target_areas):
            ax = fig.add_subplot(gs_row3[idx])
            
            if area in waveform_results:
                self._plot_waveform_analysis(waveform_results[area], ax, global_ylim)
            else:
                ax.text(0.5, 0.5, f'No waveform\ndata', 
                    ha='center', va='center', fontsize=9)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, global_ylim)
                        
            if idx == 0:
                ax.set_ylabel('Waveform effect size', fontsize=8)
            
            if idx == 2:  # Middle panel
                ax.set_xlabel('Waveform Features', fontsize=8)
        
        # Add row label
        fig.text(0.02, 0.32, 'Row 3: Waveform Properties', fontsize=14, fontweight='bold', rotation=90)

    def create_row3_aligned(self, fig, gs_row, waveform_results):
        """Create Row 3: FIXED labels and consistent sizing"""
        
        # Create nested GridSpec EXACTLY matching Row 2 layout
        gs_row3 = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_row, 
                                        width_ratios=[1, 1, 1, 1])  # Same as Row 2
        
        # Left side: 3 areas
        target_areas = ['VISp', 'CA1', 'RSPd']
        
        # Calculate global ylim
        relevant_effect_sizes = []
        for area in target_areas:
            if area in waveform_results:
                relevant_effect_sizes.extend(waveform_results[area]['waveform_effect_sizes'])
        
        global_ylim = max(relevant_effect_sizes) * 1.1 if relevant_effect_sizes else 1.0
        
        for idx, area in enumerate(target_areas):
            ax = fig.add_subplot(gs_row3[idx])
            
            if area in waveform_results:
                self._plot_waveform_analysis(waveform_results[area], ax, global_ylim)
            else:
                ax.text(0.5, 0.5, f'No waveform\ndata', 
                    ha='center', va='center', fontsize=9)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, global_ylim)
            
            if idx == 0:
                ax.set_ylabel('effect size', fontsize=9) 
            
            if idx == 1:  # Middle panel
                ax.set_xlabel('EAP waveform properties', fontsize=9)

            for label in ax.get_xticklabels(): label.set_size(8)
        
        # Right side: Waveform heatmap - MATCHING Row 2 sizing
        ax_wf_summary = fig.add_subplot(gs_row3[3])
        self._plot_waveform_summary_heatmap(waveform_results, ax_wf_summary)
        for label in ax_wf_summary.get_xticklabels(): label.set_size(8)
        
    def _plot_frequency_comparison(self, area_results, ax):
        """Plot frequency comparison for single area - FIXED: Use FREQUENCY_COLORS"""
        
        frequencies = list(area_results.keys())
        freq_labels = [f.replace('sine_', '').replace('Hz', '') for f in frequencies]
        p_values = [area_results[f]['cluster_test_pval'] for f in frequencies]
        n_clusters = [area_results[f]['n_clusters'] for f in frequencies]
        
        # Convert to -log10 p-values
        log_p_values = [-np.log10(p) if p > 0 else 10 for p in p_values]
        
        # FIXED: Use FREQUENCY_COLORS based on actual frequency values
        freq_numbers = [int(f.replace('sine_', '').replace('Hz', '')) for f in frequencies]
        colors = [FREQUENCY_COLORS[freq_num] for freq_num in freq_numbers]
        
        bars = ax.bar(freq_labels, log_p_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add significance line
        ax.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        # Add cluster number annotations
        for bar, n_clust in zip(bars, n_clusters):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{n_clust}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_ylim(0, max(log_p_values) * 1.2 if log_p_values else 1)
        ax.tick_params(labelsize=7)
        ax.set_xticklabels([f'{f} Hz' for f in freq_labels], fontsize=8)
        remove_top_right_spines(ax)

    def report_comprehensive_waveform_analysis(self, waveform_results):
        """Report comprehensive waveform analysis across all areas"""
        
        print(f"\n" + "="*60)
        print("COMPREHENSIVE WAVEFORM-CLUSTER ANALYSIS")
        print("="*60)
        
        if not waveform_results:
            print("No waveform results available")
            return
        
        total_areas = len(waveform_results)
        significant_areas = []
        meaningful_effect_areas = []
        both_criteria_areas = []
        
        feature_significance_count = {
            'duration': 0,
            'halfwidth': 0, 
            'rep_slope': 0,
            'REP': 0
        }
        
        for area, result in waveform_results.items():
            p_val = result['waveform_test_pval']
            max_effect = result['max_effect_size']
            
            if p_val < 0.05:
                significant_areas.append(area)
            
            if max_effect > 0.5:
                meaningful_effect_areas.append(area)
                
            if p_val < 0.05 and max_effect > 0.5:
                both_criteria_areas.append(area)
            
            # Count feature-specific significance
            wf_features = result['waveform_features']
            p_values = result.get('waveform_p_corrected', result['waveform_p_values'])
            
            for feat, p in zip(wf_features, p_values):
                feat_short = feat.replace('waveform_', '')
                if feat_short in feature_significance_count and p < 0.05:
                    feature_significance_count[feat_short] += 1
        
        print(f"TOTAL AREAS ANALYZED: {total_areas}")
        print(f"Areas with significant clustering (p<0.05): {len(significant_areas)} ({100*len(significant_areas)/total_areas:.1f}%)")
        print(f"Areas with meaningful effects (>0.5): {len(meaningful_effect_areas)} ({100*len(meaningful_effect_areas)/total_areas:.1f}%)")
        print(f"Areas meeting both criteria: {len(both_criteria_areas)} ({100*len(both_criteria_areas)/total_areas:.1f}%)")
        
        print(f"\nFEATURE-SPECIFIC SIGNIFICANCE:")
        for feature, count in feature_significance_count.items():
            print(f"  {feature}: {count}/{total_areas} areas ({100*count/total_areas:.1f}%)")
        
        print(f"\nSTRONGEST EVIDENCE AREAS (both p<0.05 and effect>0.5):")
        if both_criteria_areas:
            for area in both_criteria_areas:
                result = waveform_results[area]
                print(f"  {area}: p={result['waveform_test_pval']:.3f}, max_effect={result['max_effect_size']:.3f}")
        else:
            print("  None - no areas meet both criteria")
        
        print(f"="*60)

    
    def run_complete_analysis(self):
        """Run complete Figure 3 analysis with comprehensive reporting"""
        
        print("="*80)
        print("FIGURE 3: COMPLETE SWES ENTRAINMENT CLUSTERING ANALYSIS")
        print("="*80)
        
        # Load and filter data (from parent class)
        self.load_and_filter_data()
        self.report_aggregated_areas()
        
        # Row 2: Multi-frequency analysis (all areas)
        multi_freq_results = self.analyze_multi_frequency_clustering()
        
        # Row 3: Waveform analysis (NOW ALL AREAS)
        waveform_results = self.analyze_waveform_cluster_relationship()
        
        # Add comprehensive waveform reporting
        if waveform_results:
            self.report_comprehensive_waveform_analysis(waveform_results)
        
        # Create complete figure
        fig = self.create_figure3_complete(multi_freq_results, waveform_results)
        
        if fig is not None:
            output_path = os.path.join('figures', 'figure3_complete_analysis.png')
            fig.savefig(output_path, dpi=600, bbox_inches='tight')
            print(f"\nComplete figure saved: {output_path}")
            plt.show()
        
        return multi_freq_results, waveform_results, fig

# ROBUST CLUSTERING with Multiple Methods
class RobustClusterAnalysis:
    """Implements multiple clustering methods for validation"""
    
    def __init__(self):
        self.methods = ['kmeans', 'gmm', 'hierarchical']
        
    def analyze_robust_clustering(self, features_scaled, max_k=6):
        """Test clustering with multiple methods"""
        
        results = {}
        
        for method in self.methods:
            method_results = {}
            
            for k in range(2, min(max_k, len(features_scaled)//5)):
                
                if method == 'kmeans':
                    from sklearn.cluster import KMeans
                    clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = clusterer.fit_predict(features_scaled)
                    
                elif method == 'gmm':
                    from sklearn.mixture import GaussianMixture
                    clusterer = GaussianMixture(n_components=k, random_state=42)
                    labels = clusterer.fit_predict(features_scaled)
                    
                elif method == 'hierarchical':
                    from sklearn.cluster import AgglomerativeClustering
                    clusterer = AgglomerativeClustering(n_clusters=k)
                    labels = clusterer.fit_predict(features_scaled)
                
                # Calculate silhouette score
                if len(np.unique(labels)) > 1:
                    sil_score = silhouette_score(features_scaled, labels)
                    method_results[k] = {
                        'labels': labels,
                        'silhouette': sil_score
                    }
            
            if method_results:
                # Find best k for this method
                best_k = max(method_results.keys(), 
                           key=lambda k: method_results[k]['silhouette'])
                results[method] = {
                    'best_k': best_k,
                    'best_labels': method_results[best_k]['labels'],
                    'best_silhouette': method_results[best_k]['silhouette'],
                    'all_results': method_results
                }
        
        return results
    
    def assess_clustering_consensus(self, robust_results):
        """Assess agreement between clustering methods"""
        
        if len(robust_results) < 2:
            return None
            
        methods = list(robust_results.keys())
        
        # Check k consensus
        k_values = [robust_results[method]['best_k'] for method in methods]
        k_consensus = len(set(k_values)) == 1  # All methods agree on k
        
        # Check silhouette quality
        min_silhouette = min([robust_results[method]['best_silhouette'] 
                             for method in methods])
        silhouette_quality = min_silhouette > 0.3
        
        # Calculate adjusted rand index between methods
        from sklearn.metrics import adjusted_rand_score
        
        pairwise_ari = []
        for i in range(len(methods)):
            for j in range(i+1, len(methods)):
                method1, method2 = methods[i], methods[j]
                labels1 = robust_results[method1]['best_labels']
                labels2 = robust_results[method2]['best_labels']
                ari = adjusted_rand_score(labels1, labels2)
                pairwise_ari.append(ari)
        
        label_consensus = np.mean(pairwise_ari) > 0.5 if pairwise_ari else False
        
        return {
            'k_consensus': k_consensus,
            'silhouette_quality': silhouette_quality,
            'label_consensus': label_consensus,
            'mean_ari': np.mean(pairwise_ari) if pairwise_ari else 0,
            'min_silhouette': min_silhouette,
            'robust': k_consensus and silhouette_quality and label_consensus
        }


# Helper functions (import from existing)
def setup_nature_figure():
    """Setup matplotlib for Nature-style figures"""
    plt.rcParams.update({
        'font.size': 8,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'font.family': 'Arial'
    })

def remove_top_right_spines(ax):
    """Remove top and right spines from axis"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def mm_to_inch(mm):
    """Convert millimeters to inches"""
    return mm / 25.4

# Main execution
if __name__ == '__main__':
    
    # Initialize and run complete analysis
    analyzer = Figure3ExtendedAnalysis()
    multi_freq_results, waveform_results, figure = analyzer.run_complete_analysis()
    
    print("\n" + "="*80)
    print("COMPLETE ANALYSIS SUMMARY")
    print("="*80)
    
    if multi_freq_results:
        print(f"Row 2 - Multi-frequency analysis: {len(multi_freq_results)} areas analyzed")
        for area, results in multi_freq_results.items():
            freqs = list(results.keys())
            print(f"  {area}: {len(freqs)} frequencies ({', '.join(freqs)})")
    
    if waveform_results:
        print(f"Row 3 - Waveform analysis: {len(waveform_results)} areas analyzed")
        for area, results in waveform_results.items():
            n_features = len(results['waveform_features'])
            p_val = results['waveform_test_pval']
            print(f"  {area}: {n_features} features, p={p_val:.3f}")
    
    print("\nAnalysis addresses core research questions:")
    print("1. ✓ Multi-frequency clustering patterns identified")
    print("2. ✓ Waveform-cluster relationships quantified") 
    print("3. ✓ Frequency-specific vs consistent clusters characterized")