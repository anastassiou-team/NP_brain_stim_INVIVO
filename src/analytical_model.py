"""
Analytical model of extracellular potential (Ve) from point-source
electrical stimulation in a homogeneous resistive medium.

Supports monopolar and bipolar electrode configurations.
Used to generate Figure 1e–f.
"""
import numpy as np
from scipy.signal import hilbert


# ── point-source potential ──────────────────────────────────────────

def ve_monopolar(xx, zz, stim, I0, rho, phase=np.pi / 2):
    """Ve from a single point source at a given stimulus phase."""
    r = np.sqrt((xx - stim[0]) ** 2 + (zz - stim[2]) ** 2)
    return (I0 * rho / (4 * np.pi)) * np.sin(phase) / r


def ve_bipolar(xx, zz, plus_stim, minus_stim, I0, rho, phase=np.pi / 2):
    """Ve from two opposed point sources (bipolar) at a given phase."""
    r_plus = np.sqrt((xx - plus_stim[0]) ** 2 + (zz - plus_stim[2]) ** 2)
    r_minus = np.sqrt((xx - minus_stim[0]) ** 2 + (zz - minus_stim[2]) ** 2)
    return (I0 * rho / (4 * np.pi)) * (
        np.sin(phase) / r_plus + np.sin(phase + np.pi) / r_minus
    )


# ── probe geometry ──────────────────────────────────────────────────

def probe_coordinates(probe_length, n_elec, offset, angle):
    """
    Return (n_elec, 3) array of electrode positions for a linear probe.

    Parameters
    ----------
    probe_length : float   Total probe length (m).
    n_elec : int           Number of electrodes.
    offset : array-like    (x, y, z) offset of the probe tip.
    angle : float          Tilt angle from the z-axis (radians).
    """
    coords = np.zeros((n_elec, 3))
    coords[:, 0] = np.linspace(0, probe_length * np.sin(angle), n_elec) + offset[0]
    coords[:, 1] = offset[1]
    coords[:, 2] = np.linspace(0, probe_length * np.cos(angle), n_elec) + offset[2]
    return coords


# ── Ve along electrodes ────────────────────────────────────────────

def _distances(coords, point):
    """Squared Euclidean distance from each electrode to *point*."""
    return np.sum((coords - np.array(point)) ** 2, axis=1)


def ve_along_probe(coords, plus_stim, minus_stim, I0, rho, phase=np.pi / 2):
    """
    Return (V_monopolar, V_bipolar) along the probe electrodes.

    Both are 1-D arrays of length n_elec.
    """
    d_plus = _distances(coords, plus_stim)
    d_minus = _distances(coords, minus_stim)
    coeff = I0 * rho / (4 * np.pi)
    V_mono = coeff * np.sin(phase) / np.sqrt(d_plus)
    V_bi = coeff * (np.sin(phase) / np.sqrt(d_plus)
                    + np.sin(phase + np.pi) / np.sqrt(d_minus))
    return V_mono, V_bi


# ── time-domain simulation ─────────────────────────────────────────

def simulate_timeseries(V_static, freq, duration=2.0, points_per_cycle=50):
    """
    Modulate a static Ve snapshot by sin(2πft) over *duration* seconds.

    Parameters
    ----------
    V_static : ndarray   Shape (n_elec,) or (n_elec, n_angles).
    freq : float         Stimulation frequency (Hz).
    duration : float     Total simulation time (s).
    points_per_cycle : int

    Returns
    -------
    V_t : ndarray   Shape (*V_static.shape, n_time).
    time : ndarray  Shape (n_time,).
    """
    dt = 1 / (freq * points_per_cycle)
    time = np.arange(0, duration, dt)
    phase = 2 * np.pi * freq * time
    V_t = V_static[..., np.newaxis] * np.sin(phase)
    return V_t, time


# ── instantaneous amplitude / phase via Hilbert transform ──────────

def instantaneous_amplitude(V_t):
    """
    Compute envelope (instantaneous amplitude) along the last axis.

    Parameters
    ----------
    V_t : ndarray  Trailing axis is time.

    Returns
    -------
    amplitude : ndarray  Same shape as V_t.
    """
    analytic = hilbert(V_t - np.median(V_t, axis=-1, keepdims=True), axis=-1)
    return np.abs(analytic)
