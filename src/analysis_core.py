"""
Core analysis functions that orchestrate data loading and statistics
"""
import numpy as np
from src.data_loading import load_paired_condition_data
from src.statistics import compute_area_statistics
from config.brain_areas import get_display_name
from config.plotting import STATS_PARAMS

from statsmodels.stats.multitest import multipletests

def analyze_all_areas(
    brain_areas,
    amplitude,
    frequency,
    frequencies_all,
    animal_list,
    test_type='trend_swarm'
):
    """
    Analyze all brain areas for a given stim condition
    
    Parameters
    ----------
    brain_areas : list
        List of brain area identifiers
    amplitude : int
        Stimulation amplitude (µA)
    frequency : int
        Stimulation frequency (Hz)
    frequencies_all : list
        All frequencies (needed for shared unit filtering)
    animal_list : list
        Animal IDs to include
    test_type : str
        Statistical test type
    
    Returns
    -------
    dict
        Results for each area: {area: {'pre': ..., 'stim': ..., 'stats': ...}}
    """
    results = {}
    
    for area in brain_areas:
        pre_data, stim_data, full_data = load_paired_condition_data(
            area, amplitude, frequency, frequencies_all, animal_list
        )
        
        if pre_data is None or len(pre_data) < STATS_PARAMS['min_units_per_area']:
            continue
        
        # Compute statistics
        stats = compute_area_statistics(
            pre_data['vl'].values,
            stim_data['vl'].values,
            test_type=test_type
        )
        
        results[area] = {
            'pre': pre_data['vl'].values,
            'stim': stim_data['vl'].values,
            'stats': stats,
            'distance': full_data['distance_peakch_stim_tip'].median(),
            'n_units': len(pre_data)
        }
    
    return results


def sort_areas_by_distance(results):
    """
    Sort brain areas by median distance from electrode
    
    Parameters
    ----------
    results : dict
        Analysis results from analyze_all_areas
    
    Returns
    -------
    sorted_areas : list
        Brain areas sorted by distance
    sorted_distances : list
        Corresponding distances
    """
    # Extract distances
    area_distance_pairs = [
        (area, data['distance']) for area, data in results.items()
    ]
    
    # Sort by distance
    area_distance_pairs.sort(key=lambda x: x[1])
    
    sorted_areas = [pair[0] for pair in area_distance_pairs]
    sorted_distances = [pair[1] for pair in area_distance_pairs]
    
    return sorted_areas, sorted_distances

def apply_fdr_correction(results, alpha=0.05):
    """
    Apply Benjamini-Hochberg FDR correction to p-values across brain areas
    
    Parameters
    ----------
    results : dict
        Analysis results from analyze_all_areas()
        Each entry contains: results[area]['stats']['pval']
    alpha : float
        Family-wise error rate for FDR control (default 0.05)
    
    Returns
    -------
    results : dict
        Modified in-place with added fields:
        - results[area]['stats']['pval_corrected']: FDR-adjusted p-value
        - results[area]['stats']['pval_raw']: Original uncorrected p-value (preserved)
        - results[area]['stats']['reject_fdr']: Boolean, significant after FDR
    
    Notes
    -----
    Uses Benjamini-Hochberg procedure (statsmodels.stats.multitest.multipletests)
    This controls FDR (expected proportion of false discoveries among rejections)
    rather than FWER (probability of any false positives).
    """
    # Extract areas and p-values
    areas = list(results.keys())
    p_values = [results[area]['stats']['pval'] for area in areas]
    
    # Apply Benjamini-Hochberg FDR correction
    reject, p_corrected, alpha_sidak, alpha_bonf = multipletests(
        p_values, 
        alpha=alpha, 
        method='fdr_bh'
    )
    
    # Store corrected p-values back in results
    for i, area in enumerate(areas):
        # Preserve original p-value
        results[area]['stats']['pval_raw'] = results[area]['stats']['pval']
        # Store corrected p-value
        results[area]['stats']['pval_corrected'] = p_corrected[i]
        # Store rejection decision
        results[area]['stats']['reject_fdr'] = reject[i]
    
    return results


def identify_entrained_areas(results, thresh_low=0.01, thresh_high=0.001, use_corrected=True):
    """
    Identify significantly entrained brain areas
    
    Parameters
    ----------
    results : dict
        Analysis results from analyze_all_areas()
        Should contain FDR-corrected p-values if use_corrected=True
    thresh_low : float
        Low significance threshold (default p<0.01)
    thresh_high : float
        High significance threshold (default p<0.001)
    use_corrected : bool
        If True, use FDR-corrected p-values (recommended for publication)
        If False, use raw uncorrected p-values
    
    Returns
    -------
    dict
        Contains two lists:
        - 'low': areas with p <= thresh_low
        - 'high': areas with p <= thresh_high
    """
    entrained_low = []
    entrained_high = []
    
    for area, data in results.items():
        # Choose which p-value to use
        if use_corrected and 'pval_corrected' in data['stats']:
            pval = data['stats']['pval_corrected']
        else:
            pval = data['stats']['pval']
        
        if pval <= thresh_low:
            entrained_low.append(area)
        if pval <= thresh_high:
            entrained_high.append(area)
    
    return {
        'low': entrained_low,
        'high': entrained_high
    }

def prepare_spatial_decay_data(
    brain_areas,
    amplitude,
    frequency,
    frequencies_all,
    animal_list,
    use_absolute=True
):
    """
    Aggregate unit-level data across all areas for spatial decay analysis
    
    Parameters
    ----------
    brain_areas : list
        Brain areas to include
    amplitude : int
        Stimulation amplitude (µA)
    frequency : int
        Stimulation frequency (Hz)
    frequencies_all : list
        All frequencies for shared unit filtering
    animal_list : list
        Animal IDs to include
    use_absolute : bool
        If True, use absolute difference |VL_stim - VL_pre|
        If False, use signed difference (VL_stim - VL_pre)
    
    Returns
    -------
    dict
        Contains: distances, delta_vl, area_distances, area_delta_vl
    """
    # Unit-level data (for scatter plot)
    all_distances = []
    all_delta_vl = []
    
    # Area-level data (for summary)
    area_distances = []
    area_delta_vl = []
    areas_with_data = []
    
    for area in brain_areas:
        pre_data, stim_data, full_data = load_paired_condition_data(
            area, amplitude, frequency, frequencies_all, animal_list
        )
        
        if pre_data is None or len(pre_data) < STATS_PARAMS['min_units_per_area']:
            continue
        
        # Unit-level
        distances = full_data['distance_peakch_stim_tip'].values
        delta_vl = stim_data['vl'].values - pre_data['vl'].values

        # Apply absolute value if requested
        if use_absolute:
            delta_vl = np.abs(delta_vl)
        
        all_distances.extend(distances)
        all_delta_vl.extend(delta_vl)
        
        # Area-level summary
        area_distances.append(np.median(distances))
        area_delta_vl.append(np.median(delta_vl))
        areas_with_data.append(area)
    
    return {
        'distances': np.array(all_distances),
        'delta_vl': np.array(all_delta_vl),
        'area_distances': np.array(area_distances),
        'area_delta_vl': np.array(area_delta_vl),
        'areas': areas_with_data
    }


def get_entrained_area_indices(areas_with_data, entrained_areas):
    """
    Get indices of entrained areas in the areas_with_data list
    
    Parameters
    ----------
    areas_with_data : list
        All areas with data
    entrained_areas : dict
        Contains 'low' and 'high' lists
    
    Returns
    -------
    dict
        Indices for 'low' and 'high' thresholds
    """
    ind_low = [i for i, area in enumerate(areas_with_data) 
               if area in entrained_areas['low']]
    ind_high = [i for i, area in enumerate(areas_with_data) 
                if area in entrained_areas['high']]
    
    return {'low': ind_low, 'high': ind_high}
