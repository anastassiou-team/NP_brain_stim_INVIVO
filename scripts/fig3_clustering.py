"""
Figure 3: SWES Entrainment Clustering Analysis
Analyzes whether units within brain areas show distinct clustering patterns
in their response to SWES.

Key Research Question:
Do units within a brain area respond uniformly to SWES, or do distinct clusters
emerge with different entrainment properties?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from statsmodels.stats.multitest import multipletests
import re

from config.paths import FIGURES_OUTPUT, BASE_DIR
from config.experiments import DEFAULT_ANIMAL_LIST
from config.plotting import mm_to_inch
from config.experiments import cty_colors_
from src.plotting_utils import setup_nature_figure, remove_top_right_spines

from config.plotting import FREQUENCY_COLORS

class Figure3ClusteringAnalysis:
    """Enhanced clustering analysis with proper area aggregation"""
    
    def __init__(self):
        freq_focus = 28
        self.target_condition = ('sine_'+str(freq_focus)+'Hz', 5)  # Frequency, amplitude (μA)
        self.min_units_per_area = 20
        self.data = None
        self.viable_areas = None
        
        # Quality filtering parameters (from mouse_spec_analysis_parameters)
        self.n_spike_thresh = 51
        self.error_threshold = 0.1
    
    def create_comprehensive_area_mapping(self):
        """
        Create comprehensive mapping from layer-specific areas to parent areas
        Based on enhanced_vl_mmr_analysis.py approach
        """
        mapping = {
            # ===== VISUAL AREAS =====
            # Primary visual area (VISp)
            'VISp1': 'VISp',
            'VISp2/3': 'VISp', 
            'VISp4': 'VISp',
            'VISp5': 'VISp',
            'VISp6a': 'VISp',
            'VISp6b': 'VISp',
            
            # Lateral visual area (VISl)
            'VISl1': 'VISl',
            'VISl2/3': 'VISl',
            'VISl4': 'VISl', 
            'VISl5': 'VISl',
            'VISl6a': 'VISl',
            'VISl6b': 'VISl',
            
            # Anterolateral visual area (VISal)
            'VISal1': 'VISal',
            'VISal2/3': 'VISal',
            'VISal4': 'VISal',
            'VISal5': 'VISal',
            'VISal6a': 'VISal',
            'VISal6b': 'VISal',
            
            # Anteromedial visual area (VISam)
            'VISam1': 'VISam',
            'VISam2/3': 'VISam',
            'VISam4': 'VISam',
            'VISam5': 'VISam',
            'VISam6a': 'VISam',
            'VISam6b': 'VISam',
            
            # Posteromedial visual area (VISpm)
            'VISpm1': 'VISpm',
            'VISpm2/3': 'VISpm',
            'VISpm4': 'VISpm',
            'VISpm5': 'VISpm',
            'VISpm6a': 'VISpm',
            'VISpm6b': 'VISpm',
            
            # ===== MOTOR AREAS =====
            # Primary motor area (MOp)
            'MOp1': 'MOp',
            'MOp2/3': 'MOp',
            'MOp5': 'MOp',
            'MOp6a': 'MOp',
            'MOp6b': 'MOp',
            
            # Secondary motor area (MOs)
            'MOs1': 'MOs',
            'MOs2/3': 'MOs',
            'MOs5': 'MOs',
            'MOs6a': 'MOs',
            'MOs6b': 'MOs',
            
            # ===== SOMATOSENSORY AREAS =====
            # Primary somatosensory area, barrel field (SSp-bfd)
            'SSp-bfd1': 'SSp-bfd',
            'SSp-bfd2/3': 'SSp-bfd',
            'SSp-bfd4': 'SSp-bfd',
            'SSp-bfd5': 'SSp-bfd',
            'SSp-bfd6a': 'SSp-bfd',
            'SSp-bfd6b': 'SSp-bfd',
            
            # Primary somatosensory area, trunk (SSp-tr)
            'SSp-tr1': 'SSp-tr',
            'SSp-tr2/3': 'SSp-tr',
            'SSp-tr4': 'SSp-tr',
            'SSp-tr5': 'SSp-tr',
            'SSp-tr6a': 'SSp-tr',
            'SSp-tr6b': 'SSp-tr',
            
            # Primary somatosensory area, lower limb (SSp-ll)
            'SSp-ll1': 'SSp-ll',
            'SSp-ll2/3': 'SSp-ll',
            'SSp-ll4': 'SSp-ll',
            'SSp-ll5': 'SSp-ll',
            'SSp-ll6a': 'SSp-ll',
            'SSp-ll6b': 'SSp-ll',
            
            # ===== ANTERIOR CINGULATE AREAS =====
            # Anterior cingulate area, dorsal part (ACAd)
            'ACAd1': 'ACAd',
            'ACAd2/3': 'ACAd',
            'ACAd5': 'ACAd',
            'ACAd6a': 'ACAd',
            'ACAd6b': 'ACAd',
            
            # Anterior cingulate area, ventral part (ACAv)
            'ACAv1': 'ACAv',
            'ACAv2/3': 'ACAv',
            'ACAv5': 'ACAv',
            'ACAv6a': 'ACAv',
            'ACAv6b': 'ACAv',
            
            # ===== RETROSPLENIAL AREAS =====
            # Retrosplenial area, dorsal part, layer 5 (RSPd)
            'RSPd1': 'RSPd',
            'RSPd2/3': 'RSPd',
            'RSPd4': 'RSPd',
            'RSPd5': 'RSPd',
            'RSPd6a': 'RSPd',
            'RSPd6b': 'RSPd',
            
            # Retrosplenial area, lateral agranular part (RSPagl)
            'RSPagl1': 'RSPagl',
            'RSPagl2/3': 'RSPagl',
            'RSPagl5': 'RSPagl',
            'RSPagl6a': 'RSPagl',
            'RSPagl6b': 'RSPagl',
            
            # ===== AUDITORY AREAS =====
            # Primary auditory area (AUDp)
            'AUDp1': 'AUDp',
            'AUDp2/3': 'AUDp',
            'AUDp4': 'AUDp',
            'AUDp5': 'AUDp',
            'AUDp6a': 'AUDp',
            'AUDp6b': 'AUDp',
            
            # Dorsal auditory area (AUDd)
            'AUDd1': 'AUDd',
            'AUDd2/3': 'AUDd',
            'AUDd4': 'AUDd',
            'AUDd5': 'AUDd',
            'AUDd6a': 'AUDd',
            'AUDd6b': 'AUDd',
            
            # ===== ORBITAL AREAS =====
            # Orbital area, lateral part (ORBl)
            'ORBl1': 'ORBl',
            'ORBl2/3': 'ORBl',
            'ORBl5': 'ORBl',
            'ORBl6a': 'ORBl',
            'ORBl6b': 'ORBl',
            
            # Orbital area, medial part (ORBm)
            'ORBm1': 'ORBm',
            'ORBm2/3': 'ORBm',
            'ORBm5': 'ORBm',
            'ORBm6a': 'ORBm',
            'ORBm6b': 'ORBm',
            
            # ===== PRELIMBIC AND INFRALIMBIC =====
            # Prelimbic area (PL)
            'PL1': 'PL',
            'PL2/3': 'PL',
            'PL5': 'PL',
            'PL6a': 'PL',
            'PL6b': 'PL',
            
            # Infralimbic area (ILA)
            'ILA1': 'ILA',
            'ILA2/3': 'ILA',
            'ILA5': 'ILA',
            'ILA6a': 'ILA',
            'ILA6b': 'ILA',
            
            # ===== TEMPORAL ASSOCIATION AREAS =====
            # Temporal association areas (TEa)
            'TEa1': 'TEa',
            'TEa2/3': 'TEa',
            'TEa4': 'TEa',
            'TEa5': 'TEa',
            'TEa6a': 'TEa',
            'TEa6b': 'TEa',
            
            # ===== ENTORHINAL AREAS =====
            # Entorhinal area, lateral part (ENTl)
            'ENTl1': 'ENTl',
            'ENTl2/3': 'ENTl',
            'ENTl4': 'ENTl',
            'ENTl5': 'ENTl',
            'ENTl6a': 'ENTl',
            'ENTl6b': 'ENTl',
            
            # Entorhinal area, medial part (ENTm)
            'ENTm1': 'ENTm',
            'ENTm2/3': 'ENTm',
            'ENTm4': 'ENTm',
            'ENTm5': 'ENTm',
            'ENTm6a': 'ENTm',
            'ENTm6b': 'ENTm',
            
            # ===== ADDITIONAL PATTERN-BASED MAPPINGS =====
            # Note: Areas not explicitly mapped will be handled by the pattern matching below
        }
        
        return mapping
    
    def _map_to_main_area(self, area_name):
        """
        Map layer-specific area names to main area names
        FIXES: VISa aggregation and DG aggregation
        """
        if pd.isna(area_name):
            return area_name
            
        area_str = str(area_name)
        
        # First, handle explicit mappings for common patterns
        explicit_mappings = {
            # Visual areas - all layers → main area
            'VISp1': 'VISp', 'VISp2/3': 'VISp', 'VISp4': 'VISp', 'VISp5': 'VISp', 'VISp6a': 'VISp', 'VISp6b': 'VISp',
            'VISpm1': 'VISpm', 'VISpm2/3': 'VISpm', 'VISpm4': 'VISpm', 'VISpm5': 'VISpm', 'VISpm6a': 'VISpm', 'VISpm6b': 'VISpm',
            'VISam1': 'VISam', 'VISam2/3': 'VISam', 'VISam4': 'VISam', 'VISam5': 'VISam', 'VISam6a': 'VISam', 'VISam6b': 'VISam',
            'VISal1': 'VISal', 'VISal2/3': 'VISal', 'VISal4': 'VISal', 'VISal5': 'VISal', 'VISal6a': 'VISal', 'VISal6b': 'VISal',
            'VISl1': 'VISl', 'VISl2/3': 'VISl', 'VISl4': 'VISl', 'VISl5': 'VISl', 'VISl6a': 'VISl', 'VISl6b': 'VISl',
            
            # FIXED: VISa aggregation - include VISa2/3
            'VISa2/3': 'VISa', 'VISa4': 'VISa', 'VISa5': 'VISa', 'VISa6a': 'VISa', 'VISa6b': 'VISa', 'VISa1': 'VISa',
            
            # Motor areas - all layers → main area  
            'MOp1': 'MOp', 'MOp2/3': 'MOp', 'MOp5': 'MOp', 'MOp6a': 'MOp', 'MOp6b': 'MOp',
            'MOs1': 'MOs', 'MOs2/3': 'MOs', 'MOs5': 'MOs', 'MOs6a': 'MOs', 'MOs6b': 'MOs',
            
            # Somatosensory areas
            'SSp-bfd1': 'SSp-bfd', 'SSp-bfd2/3': 'SSp-bfd', 'SSp-bfd4': 'SSp-bfd', 'SSp-bfd5': 'SSp-bfd', 'SSp-bfd6a': 'SSp-bfd', 'SSp-bfd6b': 'SSp-bfd',
            'SSp-tr1': 'SSp-tr', 'SSp-tr2/3': 'SSp-tr', 'SSp-tr4': 'SSp-tr', 'SSp-tr5': 'SSp-tr', 'SSp-tr6a': 'SSp-tr', 'SSp-tr6b': 'SSp-tr',
            'SSp-ll1': 'SSp-ll', 'SSp-ll2/3': 'SSp-ll', 'SSp-ll4': 'SSp-ll', 'SSp-ll5': 'SSp-ll', 'SSp-ll6a': 'SSp-ll', 'SSp-ll6b': 'SSp-ll',
            
            # Anterior cingulate areas
            'ACAd1': 'ACAd', 'ACAd2/3': 'ACAd', 'ACAd5': 'ACAd', 'ACAd6a': 'ACAd', 'ACAd6b': 'ACAd',
            'ACAv1': 'ACAv', 'ACAv2/3': 'ACAv', 'ACAv5': 'ACAv', 'ACAv6a': 'ACAv', 'ACAv6b': 'ACAv',
            
            # Retrosplenial areas
            'RSPd1': 'RSPd', 'RSPd2/3': 'RSPd', 'RSPd4': 'RSPd', 'RSPd5': 'RSPd', 'RSPd6a': 'RSPd', 'RSPd6b': 'RSPd',
            'RSPagl1': 'RSPagl', 'RSPagl2/3': 'RSPagl', 'RSPagl5': 'RSPagl', 'RSPagl6a': 'RSPagl', 'RSPagl6b': 'RSPagl',
            
            # Auditory areas
            'AUDp1': 'AUDp', 'AUDp2/3': 'AUDp', 'AUDp4': 'AUDp', 'AUDp5': 'AUDp', 'AUDp6a': 'AUDp', 'AUDp6b': 'AUDp',
            'AUDd1': 'AUDd', 'AUDd2/3': 'AUDd', 'AUDd4': 'AUDd', 'AUDd5': 'AUDd', 'AUDd6a': 'AUDd', 'AUDd6b': 'AUDd',
            
            # Orbital areas
            'ORBl1': 'ORBl', 'ORBl2/3': 'ORBl', 'ORBl5': 'ORBl', 'ORBl6a': 'ORBl', 'ORBl6b': 'ORBl',
            'ORBm1': 'ORBm', 'ORBm2/3': 'ORBm', 'ORBm5': 'ORBm', 'ORBm6a': 'ORBm', 'ORBm6b': 'ORBm',
            
            # Prelimbic and infralimbic
            'PL1': 'PL', 'PL2/3': 'PL', 'PL5': 'PL', 'PL6a': 'PL', 'PL6b': 'PL',
            'ILA1': 'ILA', 'ILA2/3': 'ILA', 'ILA5': 'ILA', 'ILA6a': 'ILA', 'ILA6b': 'ILA',
            
            # Temporal association areas
            'TEa1': 'TEa', 'TEa2/3': 'TEa', 'TEa4': 'TEa', 'TEa5': 'TEa', 'TEa6a': 'TEa', 'TEa6b': 'TEa',
            
            # Entorhinal areas
            'ENTl1': 'ENTl', 'ENTl2/3': 'ENTl', 'ENTl4': 'ENTl', 'ENTl5': 'ENTl', 'ENTl6a': 'ENTl', 'ENTl6b': 'ENTl',
            'ENTm1': 'ENTm', 'ENTm2/3': 'ENTm', 'ENTm4': 'ENTm', 'ENTm5': 'ENTm', 'ENTm6a': 'ENTm', 'ENTm6b': 'ENTm',
            
            # FIXED: DG aggregation - aggregate all DG subfields
            'DG-mo': 'DG', 'DG-po': 'DG', 'DG-sg': 'DG',
        }
        
        # Check explicit mappings first
        if area_str in explicit_mappings:
            return explicit_mappings[area_str]
        
        # SPECIAL CASE: Keep hippocampal CA subfields distinct (CA1, CA2, CA3) but aggregate DG
        if area_str.startswith(('CA1', 'CA2', 'CA3')):
            return area_str  # Keep as-is, don't aggregate
        elif area_str.startswith('DG-'):
            return 'DG'  # Aggregate all DG subfields
        
        # Pattern-based mapping for any remaining areas
        # Remove layer suffixes: numbers, 2/3, 6a, 6b, etc.
        import re
        
        # Remove trailing layer designations
        area_main = re.sub(r'[1-6][a-z]?$', '', area_str)  # Remove 1, 2, 3, 4, 5, 6, 6a, 6b
        area_main = re.sub(r'2/3$', '', area_main)         # Remove 2/3
        
        # If we successfully removed a layer suffix, return the base area
        if area_main != area_str and area_main:
            return area_main
        
        # Keep original for areas already at the main level (like AV, CL, SUB, etc.)
        return area_str
    
    def load_and_filter_data(self):
        """Load data with quality filtering and proper area aggregation"""
        
        print("Loading and filtering data...")
        
        # Find data file
        possible_paths = [
            os.path.join(BASE_DIR, "data", "unified_data_VL_angle_MMR.csv"),
        ]
        
        data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            print(f"  ERROR: Could not find unified data file!")
            print(f"  Tried paths:")
            for path in possible_paths:
                print(f"    {path}")
            return None
        
        print(f"  Found data file at: {data_path}")
        df = pd.read_csv(data_path)
        
        # Apply quality filters (from mouse_spec_analysis_parameters)
        quality_mask = (
            (df['Nspikes_pre'] >= self.n_spike_thresh) & 
            (df['Nspikes_StimOn'] >= self.n_spike_thresh) &
            (df['error_waveform'] <= self.error_threshold)
        )
        
        df_filtered = df[quality_mask].copy()
        
        print(f"Data loaded: {len(df)} total rows")
        print(f"After quality filtering: {len(df_filtered)} rows")
        
        # Create universal_ID to identify unique physical units
        df_filtered['universal_ID'] = (
            df_filtered['mouse'].astype(str) + 
            df_filtered['probe_position'].astype(str) + 
            df_filtered['unitID'].astype(str)
        )
        
        unique_units = df_filtered['universal_ID'].nunique()
        print(f"Unique physical units: {unique_units}")
        
        # Map to aggregated areas - FIXED VERSION
        df_filtered['area_main'] = df_filtered['area_peak_ch'].apply(self._map_to_main_area)
        
        # DEBUG: Test the mapping function directly
        print(f"\nDEBUG: Direct function test:")
        test_areas = ['MOs5', 'MOp6a', 'VISp4', 'VISpm5', 'ACAd2/3']
        for test_area in test_areas:
            result = self._map_to_main_area(test_area)
            print(f"  Direct test: {test_area} → {result}")
        
        # Check if mapping is working - focus on cortical areas
        print(f"\nDEBUG: Area mapping verification (focusing on cortical areas):")
        mapping_check = df_filtered[['area_peak_ch', 'area_main']].drop_duplicates()
        
        # Show motor areas specifically
        motor_areas = mapping_check[mapping_check['area_peak_ch'].str.contains('MOp|MOs', na=False)]
        print("Motor areas from dataframe:")
        for _, row in motor_areas.head(10).iterrows():
            print(f"  {row['area_peak_ch']} → {row['area_main']}")
        
        # Show visual areas specifically
        visual_areas = mapping_check[mapping_check['area_peak_ch'].str.contains('VIS', na=False)]
        print("Visual areas from dataframe:")
        for _, row in visual_areas.head(10).iterrows():
            print(f"  {row['area_peak_ch']} → {row['area_main']}")
        
        # Get target condition data
        target_condition_data = df_filtered[
            (df_filtered['stim_freq'] == self.target_condition[0]) &
            (df_filtered['stim_current'] == self.target_condition[1])
        ]
        
        print(f"Units in target condition ({self.target_condition[0]}, {self.target_condition[1]} μA): {len(target_condition_data)}")
        
        # Find areas with sufficient units for clustering analysis (using aggregated areas)
        area_counts = target_condition_data['area_main'].value_counts()
        viable_areas = area_counts[area_counts >= self.min_units_per_area].index.tolist()
        
        print(f"Aggregated areas with ≥{self.min_units_per_area} units: {len(viable_areas)}")
        print("Top aggregated areas by unit count:")
        for area in area_counts.head(15).index:
            count = area_counts[area]
            viable = "✓" if count >= self.min_units_per_area else "✗"
            print(f"  {area}: {count} units {viable}")
        
        self.data = df_filtered
        self.viable_areas = viable_areas
        return df_filtered
    
    def analyze_area_clustering(self, area_name):
        """Analyze clustering for a single aggregated area"""
        freq, amp = self.target_condition
        
        # Get all units for this aggregated area in target condition
        area_data = self.data[
            (self.data['area_main'] == area_name) &
            (self.data['stim_freq'] == freq) &
            (self.data['stim_current'] == amp)
        ].copy()
        
        if len(area_data) < self.min_units_per_area:
            return None
        
        # Prepare features for clustering: ΔVL (change from pre to stim)
        area_data['delta_VL'] = area_data['VL_stimOn'] - area_data['VL_pre']
        
        # Create feature matrix
        features = area_data[['delta_VL', 'VL_pre']].values
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Test different numbers of clusters (2-6)
        best_score = -1
        best_n_clusters = 2
        best_labels = None
        
        for n_clusters in range(2, min(7, len(area_data)//5)):  # At least 5 units per cluster
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            
            # Calculate silhouette score
            sil_score = silhouette_score(features_scaled, labels)
            
            if sil_score > best_score:
                best_score = sil_score
                best_n_clusters = n_clusters
                best_labels = labels
        
        # Test for significant differences between clusters using Kruskal-Wallis
        cluster_values = [area_data['delta_VL'][best_labels == i].values 
                         for i in range(best_n_clusters)]
        
        if len(cluster_values) > 1 and all(len(cv) > 0 for cv in cluster_values):
            kruskal_stat, kruskal_p = kruskal(*cluster_values)
        else:
            kruskal_p = 1.0
        
        return {
            'area': area_name,
            'n_units': len(area_data),
            'n_clusters': best_n_clusters,
            'cluster_labels': best_labels,
            'silhouette_score': best_score,
            'cluster_test_pval': kruskal_p,
            'area_data': area_data,
            'cluster_delta_means': [area_data['delta_VL'][best_labels == i].mean() 
                                  for i in range(best_n_clusters)],
            'cluster_delta_stds': [area_data['delta_VL'][best_labels == i].std() 
                                 for i in range(best_n_clusters)]
        }
    
    def analyze_all_areas_clustering(self):
        """Run clustering analysis on ALL viable AGGREGATED areas"""
        freq, amp = self.target_condition
        
        print(f"\nRunning comprehensive clustering analysis on aggregated areas...")
        print(f"Analyzing {len(self.viable_areas)} aggregated areas with ≥{self.min_units_per_area} units")
        
        all_results = {}
        
        for aggregated_area in self.viable_areas:
            result = self.analyze_area_clustering(aggregated_area)
            if result is not None:
                all_results[aggregated_area] = result
        
        print(f"\nCompleted analysis on {len(all_results)} aggregated areas")
        
        # DEBUG: Show distances for verification
        print(f"\nDEBUG: Aggregated areas and distances:")
        target_data = self.data[
            (self.data['stim_freq'] == freq) &
            (self.data['stim_current'] == amp)
        ]
        
        for area in list(all_results.keys())[:10]:  # Show first 10
            area_data = target_data[target_data['area_main'] == area]
            if len(area_data) > 0:
                mean_dist = area_data['distance_peakch_stim_tip'].mean()
                print(f"  {area}: {mean_dist:.1f} mm, {len(area_data)} units")
        
        # Show significant clustering results
        significant_results = {area: res for area, res in all_results.items() 
                             if res['cluster_test_pval'] < 0.05}
        
        print(f"\nAreas with significant clustering (p < 0.05): {len(significant_results)}/{len(all_results)}")
        
        return all_results
    
    def select_top_areas(self, all_results, n_areas=3):
        """Select top areas based on clustering quality and significance"""
        
        # Filter to significant areas only
        significant_areas = {area: res for area, res in all_results.items() 
                           if res['cluster_test_pval'] < 0.05}
        
        print(f"\nAreas with significant clustering (p < 0.05): {len(significant_areas)}/{len(all_results)}")
        
        if len(significant_areas) < n_areas:
            print(f"Warning: Only {len(significant_areas)} significant areas found, showing all")
            significant_areas = all_results
        
        # Sort by: 1) Number of clusters (descending), 2) p-value (ascending)
        sorted_areas = sorted(significant_areas.items(), 
                            key=lambda x: (-x[1]['n_clusters'], x[1]['cluster_test_pval']))
        
        # Select top n areas
        selected_areas = dict(sorted_areas[:n_areas])
        
        print(f"\nSelected top {len(selected_areas)} areas:")
        for area, res in selected_areas.items():
            print(f"  {area}: {res['n_clusters']} clusters, p = {res['cluster_test_pval']:.2e}, "
                  f"sil = {res['silhouette_score']:.3f}, n = {res['n_units']}")
        
        return selected_areas
    
    def generate_summary_stats(self, selected_results, total_areas):
        """Generate summary statistics for the selected areas"""
        
        print("\n" + "="*60)
        print("FIGURE 3 ROW 1: CLUSTERING SUMMARY STATISTICS")
        print("="*60)
        
        print(f"\nAREAS ANALYZED: {len(selected_results)}/{total_areas}")
        print(f"CONDITION: {self.target_condition[0]}, {self.target_condition[1]} μA")
        
        for area_name, results in selected_results.items():
            print(f"\n{area_name}:")
            print(f"  Units: {results['n_units']}")
            print(f"  Clusters: {results['n_clusters']}")
            print(f"  Silhouette score: {results['silhouette_score']:.3f}")
            print(f"  Cluster significance: p = {results['cluster_test_pval']:.2e} (Kruskal-Wallis)")
            print(f"  Cluster ΔVL means:")
            
            for cluster_id in range(results['n_clusters']):
                cluster_mean = results['cluster_delta_means'][cluster_id]
                cluster_std = results['cluster_delta_stds'][cluster_id]
                print(f"    Cluster {cluster_id+1}: {cluster_mean:.3f} ± {cluster_std:.3f}")
        
        # Overall summary
        silhouette_scores = [res['silhouette_score'] for res in selected_results.values()]
        significant_areas = [area for area, res in selected_results.items() 
                           if res['cluster_test_pval'] < 0.05]
        
        print(f"\nOVERALL SUMMARY:")
        print(f"  Mean silhouette score: {np.mean(silhouette_scores):.3f}")
        print(f"  Areas with significant clustering: {len(significant_areas)}/{len(selected_results)}")
        print(f"  Significant areas: {significant_areas}")
    
    def report_aggregated_areas(self):
        """Report the complete list of areas after aggregation for verification"""

        print("\n" + "="*60)
        print("COMPLETE AREA AGGREGATION REPORT")
        print("="*60)

        # Get mapping from original to aggregated
        mapping_data = self.data[['area_peak_ch', 'area_main']].drop_duplicates()
        mapping_data = mapping_data.sort_values('area_main')

        # Group by aggregated area
        aggregated_areas = {}
        for _, row in mapping_data.iterrows():
            original = row['area_peak_ch']
            aggregated = row['area_main']

            if aggregated not in aggregated_areas:
                aggregated_areas[aggregated] = []
            aggregated_areas[aggregated].append(original)

        # Report aggregation results
        print(f"Total original areas: {len(mapping_data)}")
        print(f"Total aggregated areas: {len(aggregated_areas)}")
        print(f"\nAggregation details:")

        for aggregated_area in sorted(aggregated_areas.keys()):
            original_areas = sorted(aggregated_areas[aggregated_area])

            if len(original_areas) == 1:
                print(f"  {aggregated_area}: {original_areas[0]} (no aggregation)")
            else:
                print(f"  {aggregated_area}: {', '.join(original_areas)} → {aggregated_area}")

