"""
Data loading functions for transient firing-rate analysis (Figure 4).

Loads the unified_data_transient.csv and provides filtered views
by frequency, amplitude, cell type, and brain area.
"""
import numpy as np
import pandas as pd

from config.paths import TRANSIENT_DATA_PATH
from config.experiments import (
    FS_THRESHOLD, MIN_FR_THRESHOLD, error_threshold,
    FREQ_STR_TO_INT,
)

_CACHED_DF = None


def load_transient_data(force_reload=False):
    """
    Load and lightly preprocess the transient CSV (cached after first call).

    Preprocessing
    -------------
    * Adds numeric ``freq`` column derived from ``stim_freq``.
    * Adds boolean ``filter_quality`` (error_waveform < threshold).
    * Adds boolean ``filter_min_fr`` (baseline FR > threshold).
    * Adds ``cell_type`` column ('FS' or 'RS').
    """
    global _CACHED_DF
    if _CACHED_DF is not None and not force_reload:
        return _CACHED_DF.copy()

    df = pd.read_csv(TRANSIENT_DATA_PATH)

    # Numeric frequency
    df['freq'] = df['stim_freq'].map(FREQ_STR_TO_INT)

    # Quality filters
    df['filter_quality'] = df['error_waveform'] <= error_threshold
    df['filter_min_fr'] = df['meanFR_pre_10s'] > MIN_FR_THRESHOLD

    # Cell type
    df['cell_type'] = np.where(
        df['waveform_duration'] <= FS_THRESHOLD, 'FS', 'RS'
    )

    _CACHED_DF = df
    return df.copy()


def select_units(df, freq, amplitude, cell_type='all', area=None,
                 apply_quality=True, apply_min_fr=True):
    """
    Return a boolean mask selecting units that match the given criteria.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_transient_data`.
    freq : int
        Numeric frequency (8, 28, or 140).
    amplitude : int
        Stimulation current (µA).
    cell_type : str
        'FS', 'RS', or 'all'.
    area : str or None
        If given, ``df['area_peak_ch'].str.contains(area)`` is applied
        (matches layer-specific sub-area names to parent areas).
    apply_quality : bool
        Require error_waveform < threshold.
    apply_min_fr : bool
        Require baseline FR > threshold.

    Returns
    -------
    pd.Series[bool]
    """
    mask = (df['freq'] == freq) & (df['stim_current'] == amplitude)

    if apply_quality:
        mask &= df['filter_quality']
    if apply_min_fr:
        mask &= df['filter_min_fr']
    if cell_type == 'FS':
        mask &= df['cell_type'] == 'FS'
    elif cell_type == 'RS':
        mask &= df['cell_type'] == 'RS'
    if area is not None:
        mask &= df['area_peak_ch'].str.contains(area, na=False)

    return mask


def get_fr_columns(eval_window='100ms'):
    """
    Return (pre_col, stim_col, std_col) column names for a given evaluation window.

    Supported windows: '100ms', '1s', '10s'.
    """
    if eval_window == '100ms':
        return 'meanFR_pre_1s', 'meanFR_stimOn_01s', 'stdFR_pre_1s'
    elif eval_window == '1s':
        return 'meanFR_pre_1s', 'meanFR_stimOn_1s', 'stdFR_pre_1s'
    elif eval_window == '10s':
        return 'meanFR_pre_10s', 'meanFR_stimOn_10s', 'stdFR_pre_10s'
    else:
        raise ValueError(f"Unknown eval_window: {eval_window}")
