"""
Analysis functions for transient firing-rate effects (Figure 4).

These operate on the DataFrame returned by
:func:`src.data_loading_transient.load_transient_data`.
"""
import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu
from src.statistics import fdr_correct, ols_interaction_test
from src.data_loading_transient import select_units, get_fr_columns


# ── Panel 4b helpers ──────────────────────────────────────────────────────

def compute_area_fr_stats(df, freq, amplitude, area, eval_window='100ms'):
    """
    Paired t-test on within-unit FR difference (stim − pre) for one area.

    Uses one-sample t-test on paired differences (same as Figures 2/3).

    Returns
    -------
    dict or None
        Keys: fr_pre, fr_stim, pval, n_units, median_pre, median_stim.
        None if no units pass filters.
    """
    from scipy.stats import ttest_1samp

    mask = select_units(df, freq, amplitude, cell_type='all', area=area)
    sub = df.loc[mask]
    if len(sub) == 0:
        return None

    pre_col, stim_col, _ = get_fr_columns(eval_window)
    fr_pre = sub[pre_col].values
    fr_stim = sub[stim_col].values

    diffs = fr_stim - fr_pre
    if len(diffs) < 2:
        return None

    t_stat, pval = ttest_1samp(diffs, 0, alternative='two-sided')

    return {
        'fr_pre': fr_pre,
        'fr_stim': fr_stim,
        'pval': pval,
        'n_units': len(diffs),
        'median_pre': np.median(fr_pre),
        'median_stim': np.median(fr_stim),
    }


def analyze_all_areas_transient(df, freq, amplitude, areas,
                                eval_window='100ms'):
    """
    Run :func:`compute_area_fr_stats` for every area, then FDR-correct.

    Returns
    -------
    dict
        Keyed by area.  Each value has the fields from
        ``compute_area_fr_stats`` plus ``pval_corrected``.
    """
    results = {}
    for area in areas:
        r = compute_area_fr_stats(df, freq, amplitude, area, eval_window)
        if r is not None:
            results[area] = r

    # FDR correction across areas
    area_keys = list(results.keys())
    raw_p = np.array([results[a]['pval'] for a in area_keys])
    corrected = fdr_correct(raw_p)
    for i, area in enumerate(area_keys):
        results[area]['pval_corrected'] = corrected[i]

    return results


# ── Panel 4d helpers ──────────────────────────────────────────────────────

def compute_transient_percentages(df, freq, amplitude, subjects):
    """
    Per-subject percentage of FS/RS units showing decrease (D) or increase (I).

    Parameters
    ----------
    df : pd.DataFrame
        Full transient DataFrame.
    freq : int
        Numeric frequency.
    amplitude : int
        Stimulation current (µA).
    subjects : array-like
        List of mouse IDs.

    Returns
    -------
    pd.DataFrame
        Index = subjects, columns = MultiIndex ('D'/'I', 'FS'/'RS').
    """
    cols = pd.MultiIndex.from_product([['D', 'I'], ['FS', 'RS']])
    result = pd.DataFrame(index=subjects, columns=cols, dtype=float)

    for subj in subjects:
        mask = select_units(df, freq, amplitude) & (df['mouse'] == subj)
        sub = df.loc[mask]
        if len(sub) == 0:
            continue

        for trans in ['D', 'I']:
            for ct in ['FS', 'RS']:
                n_total = (sub['cell_type'] == ct).sum()
                if n_total == 0:
                    continue
                n_type = ((sub['transient_type'] == trans) &
                          (sub['cell_type'] == ct)).sum()
                result.at[subj, (trans, ct)] = n_type / n_total * 100

    return result


def pairwise_mannwhitney_across_freqs(perc_by_freq, freq_pairs):
    """
    Mann-Whitney U tests comparing per-subject percentages across frequencies.

    Parameters
    ----------
    perc_by_freq : dict
        ``{freq: DataFrame}`` from :func:`compute_transient_percentages`.
    freq_pairs : list of tuples
        Frequency pairs to compare, e.g. ``[(8, 28), (8, 140), (28, 140)]``.

    Returns
    -------
    dict
        ``{(trans_type, cell_type, f1, f2): corrected_pval}``.
    """
    raw_pvals = {}
    keys = []

    for trans in ['D', 'I']:
        for ct in ['FS', 'RS']:
            pvals_this = []
            keys_this = []
            for f1, f2 in freq_pairs:
                arr1 = perc_by_freq[f1][(trans, ct)].astype(float).dropna().values
                arr2 = perc_by_freq[f2][(trans, ct)].astype(float).dropna().values
                if len(arr1) > 0 and len(arr2) > 0:
                    _, p = mannwhitneyu(arr1, arr2)
                else:
                    p = np.nan
                raw_pvals[(trans, ct, f1, f2)] = p
                pvals_this.append(p)
                keys_this.append((trans, ct, f1, f2))
            keys.extend(keys_this)

    # FDR-correct all comparisons together
    all_p = np.array([raw_pvals[k] for k in keys])
    corrected = fdr_correct(all_p)
    return {k: corrected[i] for i, k in enumerate(keys)}


# ── Panel 4e helpers ──────────────────────────────────────────────────────

def compute_zscore_by_distance(df, freq, amplitude, cell_type,
                               distance_bins, eval_window='100ms'):
    """
    Compute |z-scored FR| binned by distance from stimulation electrode.

    Z-score = (FR_stim − FR_pre) / std_pre  (absolute value taken).

    Parameters
    ----------
    df : pd.DataFrame
    freq : int
    amplitude : int
    cell_type : str
        'FS' or 'RS'.
    distance_bins : list
        Bin edges in mm, e.g. [0, 1, 2, 3, 4].
    eval_window : str

    Returns
    -------
    dict
        Keys: distances (bin means), means, se, n_per_bin,
        raw_distances, raw_zscores, groups (pd.Categorical).
    """
    mask = select_units(df, freq, amplitude, cell_type=cell_type)
    sub = df.loc[mask].copy()

    pre_col, stim_col, std_col = get_fr_columns(eval_window)
    z = (sub[stim_col] - sub[pre_col]) / sub[std_col]

    # Remove infinities
    valid = np.isfinite(z)
    z = z[valid]
    distances = sub.loc[valid, 'distance_peakch_stim_tip']

    abs_z = np.abs(z)
    groups = pd.cut(distances, distance_bins, right=True)

    d_mean = distances.groupby(groups).mean().values
    z_mean = abs_z.groupby(groups).apply(np.nanmean).values
    z_std = abs_z.groupby(groups).apply(np.nanstd).values
    n = abs_z.groupby(groups).count().values
    z_se = z_std / np.sqrt(n)

    return {
        'distances': d_mean,
        'means': z_mean,
        'se': z_se,
        'n_per_bin': n,
        'raw_distances': distances.values,
        'raw_zscores': abs_z.values,
        'groups': groups,
    }


def compare_fs_rs_at_bins(df, freq, amplitude, distance_bins,
                          eval_window='100ms'):
    """
    Mann-Whitney FS vs RS |z-score| at each distance bin, FDR-corrected.

    Also runs OLS interaction test on the full data.

    Returns
    -------
    dict
        Keys: bin_pvals_corrected (array), ols_interaction_pval (float).
    """
    fs = compute_zscore_by_distance(
        df, freq, amplitude, 'FS', distance_bins, eval_window)
    rs = compute_zscore_by_distance(
        df, freq, amplitude, 'RS', distance_bins, eval_window)

    # Per-bin Mann-Whitney
    fs_groups = pd.Series(np.abs(
        (df.loc[select_units(df, freq, amplitude, 'FS')].pipe(
            lambda d: (d[get_fr_columns(eval_window)[1]] -
                       d[get_fr_columns(eval_window)[0]]) /
                      d[get_fr_columns(eval_window)[2]])
        ))).dropna()

    # Simpler approach: recompute grouped values
    mask_fs = select_units(df, freq, amplitude, 'FS')
    mask_rs = select_units(df, freq, amplitude, 'RS')
    pre_col, stim_col, std_col = get_fr_columns(eval_window)

    def _zscore_series(mask):
        sub = df.loc[mask].copy()
        z = (sub[stim_col] - sub[pre_col]) / sub[std_col]
        valid = np.isfinite(z)
        return np.abs(z[valid]), sub.loc[valid, 'distance_peakch_stim_tip']

    z_fs, d_fs = _zscore_series(mask_fs)
    z_rs, d_rs = _zscore_series(mask_rs)

    g_fs = pd.cut(d_fs, distance_bins, right=True)
    g_rs = pd.cut(d_rs, distance_bins, right=True)

    bin_pvals = []
    for b in g_fs.cat.categories:
        vals_fs = z_fs[g_fs == b].dropna().values
        vals_rs = z_rs[g_rs == b].dropna().values
        if len(vals_fs) > 0 and len(vals_rs) > 0:
            _, p = mannwhitneyu(vals_fs, vals_rs)
        else:
            p = np.nan
        bin_pvals.append(p)

    corrected_bins = fdr_correct(bin_pvals)

    # OLS interaction test for slope difference
    x_all = np.concatenate([
        d_fs.groupby(g_fs).transform('mean').values,
        d_rs.groupby(g_rs).transform('mean').values,
    ])
    # Use raw per-unit distances and z-scores instead
    x_all = np.concatenate([d_fs.values, d_rs.values])
    y_all = np.concatenate([z_fs.values, z_rs.values])
    g_all = np.concatenate([
        np.full(len(z_fs), 'FS'),
        np.full(len(z_rs), 'RS'),
    ])
    ols_result = ols_interaction_test(x_all, y_all, g_all)

    return {
        'bin_pvals_corrected': corrected_bins,
        'ols_interaction_pval': ols_result['interaction_pval'],
    }
