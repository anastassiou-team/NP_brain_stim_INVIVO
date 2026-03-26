"""
Curve fitting functions for spatial decay analysis
"""
import numpy as np
from scipy.optimize import curve_fit


def inverse_power_law_offset(x, x0, n):
    """
    Inverse power law with single offset: f(x) = 1 / (x - x0)^n
    
    Parameters
    ----------
    x : array-like
        Independent variable (distance)
    x0 : float
        Horizontal offset
    n : float
        Power
    
    Returns
    -------
    array-like
        Function values
    """
    return 1 / np.power(x - x0, n)


def inverse_power_law_twooffsets(x, x0, x1, n):
    """
    Inverse power law with two offsets: f(x) = x1 + 1 / (x - x0)^n
    
    Parameters
    ----------
    x : array-like
        Independent variable (distance)
    x0 : float
        Horizontal offset
    x1 : float
        Vertical offset (asymptote)
    n : float
        Power
    
    Returns
    -------
    array-like
        Function values
    """
    return x1 + (1 / np.power(x - x0, n))


def linear_func(x, a, b):
    """
    Linear function: f(x) = a*x + b
    
    Parameters
    ----------
    x : array-like
        Independent variable
    a : float
        Slope
    b : float
        Intercept
    
    Returns
    -------
    array-like
        Function values
    """
    return (a * x) + b


def calculate_aic(y_data, y_model, num_params):
    """
    Calculate Akaike Information Criterion (AIC)
    
    Parameters
    ----------
    y_data : array-like
        Observed data
    y_model : array-like
        Model predictions
    num_params : int
        Number of parameters in the model
    
    Returns
    -------
    float
        AIC value
    """
    residuals = y_data - y_model
    sse = np.sum(residuals**2)
    n = len(y_data)
    aic = n * np.log(sse / n) + 2 * num_params
    return aic


def fit_spatial_decay(distances, delta_vl, model='inverse_power_law_twooffsets'):
    """
    Fit spatial decay model to ΔVL vs distance data
    
    Parameters
    ----------
    distances : array-like
        Distance from stimulation electrode (mm)
    delta_vl : array-like
        Change in vector length (stim - pre)
    model : str
        'linear', 'inverse_power_law_offset', or 'inverse_power_law_twooffsets'
    
    Returns
    -------
    dict
        Contains: params, covariance, x_pred, y_pred, aic
    """
    x_data = np.array(distances)
    y_data = np.array(delta_vl)
    
    # Generate prediction range
    x_pred = np.linspace(min(x_data), max(x_data), 100)
    
    if model == 'linear':
        initial_params = [-0.1, 0.01]
        params, covariance = curve_fit(linear_func, x_data, y_data, p0=initial_params)
        y_pred = linear_func(x_pred, *params)
        num_params = 2
        
    elif model == 'inverse_power_law_offset':
        initial_params = [-1.0, 2.0]
        params, covariance = curve_fit(
            inverse_power_law_offset, x_data, y_data, p0=initial_params
        )
        y_pred = inverse_power_law_offset(x_pred, *params)
        num_params = 2
        
    elif model == 'inverse_power_law_twooffsets':
        initial_params = [-1.0, -0.1, 2.0]
        params, covariance = curve_fit(
            inverse_power_law_twooffsets, x_data, y_data, p0=initial_params
        )
        y_pred = inverse_power_law_twooffsets(x_pred, *params)
        num_params = 3
        
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # Calculate AIC
    y_model = eval(f"{model}(x_data, *params)")
    aic = calculate_aic(y_data, y_model, num_params)
    
    return {
        'params': params,
        'covariance': covariance,
        'x_pred': x_pred,
        'y_pred': y_pred,
        'aic': aic,
        'model': model
    }


def compare_models(distances, delta_vl, models=None):
    """
    Compare multiple models using AIC
    
    Parameters
    ----------
    distances : array-like
        Distance data
    delta_vl : array-like
        ΔVL data
    models : list, optional
        List of model names to compare
    
    Returns
    -------
    dict
        Fit results for each model, sorted by AIC
    """
    if models is None:
        models = ['linear', 'inverse_power_law_offset', 'inverse_power_law_twooffsets']
    
    results = {}
    for model in models:
        try:
            results[model] = fit_spatial_decay(distances, delta_vl, model)
        except:
            print(f"Failed to fit {model}")
    
    # Sort by AIC
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['aic']))
    
    return sorted_results
