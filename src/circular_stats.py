"""
Circular statistics and vector operations for phase analysis
"""
import math
import numpy as np


def angular_to_cartesian(length, angle):
    """
    Convert angular (polar) coordinates to cartesian
    
    Parameters
    ----------
    length : float
        Vector length (radius)
    angle : float
        Angle in radians
    
    Returns
    -------
    x : float
        X coordinate
    y : float
        Y coordinate
    """
    x = length * math.cos(angle)
    y = length * math.sin(angle)
    return x, y


def cartesian_to_angular(x, y):
    """
    Convert cartesian coordinates to angular (polar)
    
    Parameters
    ----------
    x : float
        X coordinate
    y : float
        Y coordinate
    
    Returns
    -------
    length : float
        Vector length (radius)
    angle : float
        Angle in radians
    """
    length = math.sqrt(x**2 + y**2)
    angle = math.atan2(y, x)
    return length, angle


def subtract_vectors(length1, angle1, length2, angle2):
    """
    Subtract two vectors in angular coordinates
    
    Vector subtraction: result = vector1 - vector2
    
    Parameters
    ----------
    length1 : float
        Length of first vector
    angle1 : float
        Angle of first vector (radians)
    length2 : float
        Length of second vector
    angle2 : float
        Angle of second vector (radians)
    
    Returns
    -------
    result_length : float
        Length of result vector
    result_angle : float
        Angle of result vector (radians)
    """
    # Convert to cartesian
    x1, y1 = angular_to_cartesian(length1, angle1)
    x2, y2 = angular_to_cartesian(length2, angle2)
    
    # Subtract
    x_result = x1 - x2
    y_result = y1 - y2
    
    # Convert back to angular
    result_length, result_angle = cartesian_to_angular(x_result, y_result)
    
    return result_length, result_angle


def compute_vector_differences(pre_vl, pre_angle, stim_vl, stim_angle):
    """
    Compute vector differences for arrays of pre and stim data
    
    Parameters
    ----------
    pre_vl : array-like
        Pre-stim vector lengths
    pre_angle : array-like
        Pre-stim angles (radians)
    stim_vl : array-like
        Stim-on vector lengths
    stim_angle : array-like
        Stim-on angles (radians)
    
    Returns
    -------
    diff_vl : np.ndarray
        Difference vector lengths
    diff_angle : np.ndarray
        Difference angles (radians)
    """
    n = len(pre_vl)
    diff_vl = np.zeros(n)
    diff_angle = np.zeros(n)
    
    for i in range(n):
        diff_vl[i], diff_angle[i] = subtract_vectors(
            pre_vl[i], pre_angle[i],
            stim_vl[i], stim_angle[i]
        )
    
    return diff_vl, diff_angle
