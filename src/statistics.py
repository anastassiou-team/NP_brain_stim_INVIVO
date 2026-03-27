"""
Statistical analysis functions
"""
import numpy as np
from scipy.stats import mannwhitneyu, linregress, ttest_1samp, iqr, skew, skewtest
from scipy.optimize import curve_fit


def paired_ttest_with_effect(pre_data, stim_data):
    """
    Perform paired t-test on pre vs stim data (pairwise swarm attack)
    
    Parameters
    ----------
    pre_data : array-like
        Pre-stimulation values
    stim_data : array-like
        Stim-on values (must match pre_data length)
    
    Returns
    -------
    dict
        Contains: mean_diff, variance, t_stat, pval
    """
    pre_data = np.array(pre_data)
    stim_data = np.array(stim_data)
    
    # Calculate pairwise differences
    diffs = stim_data - pre_data
    
    # Statistics on differences
    mean_diff = np.mean(diffs)
    variance = np.var(diffs)
    
    # One-sample t-test against zero
    t_stat, pval = ttest_1samp(diffs, 0, alternative='two-sided')
    
    return {
        'mean_diff': mean_diff,
        'variance': variance,
        't_stat': t_stat,
        'pval': pval,
        'median_pre': np.median(pre_data),
        'median_stim': np.median(stim_data)
    }


def linear_regression_test(pre_data, stim_data):
    """
    Test for trend using linear regression
    
    Parameters
    ----------
    pre_data : array-like
        Pre-stimulation values
    stim_data : array-like
        Stim-on values
    
    Returns
    -------
    dict
        Contains: slope, intercept, r_value, pval, std_err
    """
    n = len(pre_data)
    X = np.concatenate([np.zeros(n), np.ones(n)])
    Y = np.concatenate([pre_data, stim_data])
    
    slope, intercept, r_value, pval, std_err = linregress(X, Y)
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'pval': pval,
        'std_err': std_err
    }


def mannwhitneyu_test(pre_data, stim_data):
    """
    Perform Mann-Whitney U test
    
    Parameters
    ----------
    pre_data : array-like
        Pre-stimulation values
    stim_data : array-like
        Stim-on values
    
    Returns
    -------
    dict
        Contains: statistic, pval, median_pre, median_stim
    """
    statistic, pval = mannwhitneyu(pre_data, stim_data)
    
    return {
        'statistic': statistic,
        'pval': pval,
        'median_pre': np.median(pre_data),
        'median_stim': np.median(stim_data)
    }


def compute_area_statistics(pre_data, stim_data, test_type='trend_swarm'):
    """
    Compute comprehensive statistics for a brain area
    
    Parameters
    ----------
    pre_data : array-like
        Pre-stimulation VL values
    stim_data : array-like
        Stim-on VL values
    test_type : str
        'trend_swarm', 'trend', or 'mannwhitneyu'
    
    Returns
    -------
    dict
        Comprehensive statistics including effect size, variance, p-value, etc.
    """
    pre_data = np.array(pre_data)
    stim_data = np.array(stim_data)
    diffs = stim_data - pre_data
    
    # Perform primary statistical test
    if test_type == 'trend_swarm':
        stats = paired_ttest_with_effect(pre_data, stim_data)
        effect_size = stats['mean_diff']
        intercept = np.mean(pre_data)
    elif test_type == 'trend':
        stats = linear_regression_test(pre_data, stim_data)
        effect_size = stats['slope']
        intercept = stats['intercept']
    else:  # mannwhitneyu
        stats = mannwhitneyu_test(pre_data, stim_data)
        effect_size = np.median(stim_data)
        intercept = np.median(pre_data)
    
    # Additional statistics
    skew_test = skewtest(diffs)
    
    return {
        'effect_size': effect_size,
        'variance': stats.get('variance', np.var(diffs)),
        'pval': stats['pval'],
        'intercept': intercept,
        'median_pre': np.median(pre_data),
        'median_stim': np.median(stim_data),
        'iqr': iqr(diffs),
        'skewness': skew(diffs),
        'skew_z': skew_test[0],
        'skew_p': skew_test[1],
        'n_units': len(pre_data)
    }


def cohens_d(data1, data2):
    """
    Calculate Cohen's d effect size
    
    Parameters
    ----------
    data1 : array-like
        First group
    data2 : array-like
        Second group
    
    Returns
    -------
    float
        Cohen's d effect size
    """
    mean_diff = np.mean(data1) - np.mean(data2)
    pooled_std = np.sqrt(
        (np.std(data1, ddof=1)**2 + np.std(data2, ddof=1)**2) / 2
    )
    return mean_diff / pooled_std


def get_significance_level(pval):
    """
    Convert p-value to significance level (number of markers)
    
    Parameters
    ----------
    pval : float
        P-value
    
    Returns
    -------
    int
        Number of significance markers (0-4)
    """
    if pval > 0.05:
        return 0
    elif pval <= 0.05 and pval > 0.01:
        return 1
    elif pval <= 0.01 and pval > 0.001:
        return 2
    elif pval <= 0.001 and pval > 0.0001:
        return 3
    else:  # pval <= 0.0001
        return 4


def fdr_correct(pvalues, alpha=0.05, method='fdr_bh'):
    """
    Apply Benjamini-Hochberg FDR correction to an array of p-values.

    Parameters
    ----------
    pvalues : array-like
        Raw p-values (NaN entries are passed through unchanged).
    alpha : float
        Family-wise error rate.
    method : str
        Correction method (default 'fdr_bh').

    Returns
    -------
    np.ndarray
        Corrected p-values (same length as input; NaN where input was NaN).
    """
    from statsmodels.stats.multitest import multipletests

    pvalues = np.asarray(pvalues, dtype=float)
    corrected = np.full_like(pvalues, np.nan)
    valid = ~np.isnan(pvalues)
    if valid.sum() == 0:
        return corrected
    _, corrected[valid], _, _ = multipletests(pvalues[valid], alpha=alpha, method=method)
    return corrected


def pval_to_stars(pval):
    """
    Convert a p-value to an asterisk string for annotation.

    Returns None if not significant (p >= 0.05).
    """
    if np.isnan(pval) or pval >= 0.05:
        return None
    if pval < 0.0001:
        return '****'
    if pval < 0.001:
        return '***'
    if pval < 0.01:
        return '**'
    return '*'


def ols_interaction_test(x, y, group_labels):
    """
    Test whether two groups have different slopes via OLS interaction term.

    Fits  y ~ x * group  and returns the p-value on the interaction term.

    Parameters
    ----------
    x : array-like
        Continuous predictor (e.g. distance).
    y : array-like
        Response variable.
    group_labels : array-like
        Group labels (e.g. 'FS' / 'RS').

    Returns
    -------
    dict
        Contains: interaction_pval, model_summary (str), coefficients (dict).
    """
    import statsmodels.formula.api as smf
    import pandas as pd

    df = pd.DataFrame({'x': x, 'y': y, 'group': group_labels})
    model = smf.ols('y ~ x * group', data=df).fit()

    # Find the interaction term (statsmodels names it e.g. 'x:group[T.RS]')
    interaction_keys = [k for k in model.pvalues.index if ':' in k]
    interaction_pval = model.pvalues[interaction_keys[0]] if interaction_keys else np.nan

    return {
        'interaction_pval': interaction_pval,
        'coefficients': dict(model.params),
        'model_summary': str(model.summary()),
    }
