"""
Data loading functions for Neuropixels unit data
"""
import os
import pandas as pd
import numpy as np
from config.paths import DATA_PATH


def load_unit_data(brain_area, amplitude, frequency, animal_list=None):
    """
    Load unit data for a specific brain area, amplitude, and frequency
    
    Parameters
    ----------
    brain_area : str
        Brain area identifier
    amplitude : int
        Stimulation amplitude (µA)
    frequency : int
        Stimulation frequency (Hz)
    animal_list : list, optional
        List of animal IDs to include
    
    Returns
    -------
    pd.DataFrame
        Unit data
    """
    filename = f'units_{frequency}Hz_{amplitude}_{brain_area}.csv'
    filepath = os.path.join(DATA_PATH, filename)    
    try:
        data = pd.read_csv(filepath)
        
        # Filter by animal list if provided
        if animal_list is not None:
            data = data[data['mouse'].isin(animal_list)]
        
        return data
    except FileNotFoundError:
        return pd.DataFrame()


def get_shared_unit_ids(brain_area, amplitude, frequencies, animal_list=None):
    """
    Find units that exist across all specified frequencies
    
    Parameters
    ----------
    brain_area : str
        Brain area identifier
    amplitude : int
        Stimulation amplitude (µA)
    frequencies : list
        List of frequencies to check
    animal_list : list, optional
        Animal IDs to include
    
    Returns
    -------
    list
        Universal IDs of units present in all conditions
    """
    # Load data for all frequencies
    dfs = []
    for freq in frequencies:
        df = load_unit_data(brain_area, amplitude, freq, animal_list)
        if df.empty:
            return []
        dfs.append(df)
    
    # Find shared IDs
    shared_ids = set(dfs[0]['universal_ID'])
    for df in dfs[1:]:
        shared_ids &= set(df['universal_ID'])
    
    return list(shared_ids)


def load_shared_units_across_frequencies_mmr(
    brain_area, 
    amplitude, 
    frequencies, 
    animal_list=None
):
    """
    Load data for units present across all frequencies
    
    Parameters
    ----------
    brain_area : str
        Brain area identifier
    amplitude : int
        Stimulation amplitude (µA)
    frequencies : list
        List of 3 frequencies [freq1, freq2, freq3]
    animal_list : list, optional
        Animal IDs to include
    
    Returns
    -------
    dict
        Dictionary mapping frequency to DataFrame of shared units
    """
    shared_ids = get_shared_unit_ids(brain_area, amplitude, frequencies, animal_list)
    
    if not shared_ids:
        return {freq: pd.DataFrame() for freq in frequencies}
    
    # Load data for each frequency, filtered to shared IDs
    result = {}
    for freq in frequencies:
        df = load_unit_data(brain_area, amplitude, freq, animal_list)
        result[freq] = df[df['universal_ID'].isin(shared_ids)]
    
    return result


def extract_condition_data(df, frequency, condition='stim'):
    """
    Extract vector length and angle data for a specific condition
    
    Parameters
    ----------
    df : pd.DataFrame
        Unit data
    frequency : int
        Stimulation frequency (Hz)
    condition : str
        'stim' or 'pre'
    
    Returns
    -------
    pd.DataFrame
        Extracted data with columns: distance, cluster, universal_ID, vl, angle
    """
    if condition == 'stim':
        vl_key = 'MMR_stimOn'
        angle_key = f'angle_stimOn_ch15'
    elif condition == 'pre':
        vl_key = 'MMR_Pre'
        angle_key = f'angle_pre_ch15'
    else:
        raise ValueError(f"Unknown condition: {condition}")
    
    return pd.DataFrame({
        'distance': df['distance_peakch_stim_tip'],
        'cluster': df['cluster'],
        'universal_ID': df['universal_ID'],
        'vl': df[vl_key],
        'angle': df[angle_key]
    })


def load_paired_condition_data_mmr(
    brain_area,
    amplitude,
    frequency,
    frequencies_all,
    animal_list=None
):
    """
    Load paired pre and stim data for units present across all frequencies
    
    Parameters
    ----------
    brain_area : str
        Brain area identifier
    amplitude : int
        Stimulation amplitude (µA)
    frequency : int
        Target frequency for comparison
    frequencies_all : list
        All frequencies (needed to find shared units)
    animal_list : list, optional
        Animal IDs to include
    
    Returns
    -------
    pre_data : pd.DataFrame or None
        Pre-stim data
    stim_data : pd.DataFrame or None
        Stim-on data
    shared_units : pd.DataFrame or None
        Full data for shared units at target frequency
    """
    # Get shared units across all frequencies
    shared_data = load_shared_units_across_frequencies_mmr(
        brain_area, amplitude, frequencies_all, animal_list
    )
    
    df = shared_data.get(frequency)
    
    if df is None or df.empty:
        return None, None, None
    
    # Extract pre and stim conditions
    pre_data = extract_condition_data(df, frequency, 'pre')
    stim_data = extract_condition_data(df, frequency, 'stim')
    
    return pre_data, stim_data, df
