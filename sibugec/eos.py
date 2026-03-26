"""
sibugec.eos
===========
Equation-of-state (EoS) models for the two-phase fluid.

Two analytic parametrisations are provided:

* ``pplus``  – high-temperature (HT) branch.
* ``pminus`` – low-temperature (LT) branch.

A helper ``speed_of_sound_squared`` wraps both into a uniform callable
and ``load_custom_eos`` allows replacing the analytic models with
tabulated data read from a file.
"""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d


# ---------------------------------------------------------------------------
# Analytic EoS parametrisations
# ---------------------------------------------------------------------------

def pplus(e, eHT, p1, delta=10.0):
    """
    High-temperature pressure branch p+(e).

    Constructed so that:
      - p'(eHT) = 0  (pressure minimum at eHT)
      - p'(e)  → 1/3 as e → ∞  (conformal limit)
      - p(eHT) = p1

    Parameters
    ----------
    e     : array_like  Energy density.
    eHT   : float       Energy at the HT turning point.
    p1    : float       Pressure value at eHT.
    delta : float       Smoothing scale (default 10).

    Returns
    -------
    ndarray or float
        Pressure p+(e).  Values for e < eHT − 0.1 are set to NaN.
    """
    e = np.asarray(e)
    p_raw = (1.0 / 3.0) * delta * np.log(np.cosh((e - eHT) / delta))
    C = p1 - (1.0 / 3.0) * delta * np.log(np.cosh(0.0))
    p = p_raw + C
    p = np.where(e < eHT*0.9, np.nan, p)
    return p if p.shape != () else p.item()


def pminus(e, eTL, n=1):
    """
    Low-temperature pressure branch p−(e).

    Constructed so that:
      - p(0) = 0
      - p'(0) = 1/3
      - p'(eTL) = 0
      - p'(e) strictly decreasing

    Parameters
    ----------
    e   : array_like  Energy density.
    eTL : float       Energy at the LT turning point.
    n   : float       Shape exponent (default 1).

    Returns
    -------
    ndarray or float
        Pressure p−(e).  Values for e > eTL + 0.1 are set to NaN.
    """
    e = np.asarray(e)
    p = np.full_like(e, np.nan, dtype=float)
    mask = e <= eTL*1.1
    p[mask] = (1.0 / 3.0) * (
        e[mask] - e[mask] ** (n + 1) / ((n + 1) * eTL ** n)
    )
    return p if p.shape != () else p.item()


# ---------------------------------------------------------------------------
# Speed of sound
# ---------------------------------------------------------------------------

def speed_of_sound_squared(e_array, p_array):
    """
    Compute c_s²(e) = dp/de from tabulated (e, p) data.

    Parameters
    ----------
    e_array : array_like  Energy density grid (must be strictly increasing).
    p_array : array_like  Pressure values on the same grid.

    Returns
    -------
    CubicSpline
        Callable cs²(e).
    """
    e_array = np.asarray(e_array)
    p_array = np.asarray(p_array)
    cs2_vals = np.gradient(p_array, e_array)
    return CubicSpline(e_array, cs2_vals)


# ---------------------------------------------------------------------------
# Custom EoS loader
# ---------------------------------------------------------------------------

def load_custom_eos(filepath, delimiter=None, usecols=None):
    """
    Load a custom equation of state from a data file and return
    callable pressure and speed-of-sound functions for *both* branches.

    The file must contain at least two columns:
      column 0 – energy density  e
      column 1 – pressure        p(e)

    If a third column is present it is interpreted as c_s²(e) directly;
    otherwise c_s² is derived numerically from p(e).

    Parameters
    ----------
    filepath  : str   Path to the data file (whitespace or CSV).
    delimiter : str   Column delimiter (default: whitespace).
    usecols   : sequence of int, optional
                Columns to read (default: all columns).

    Returns
    -------
    p_func   : callable(e)  Pressure interpolant.
    cs2_func : callable(e)  Speed-of-sound-squared interpolant.
    e_min    : float        Minimum energy in the table.
    e_max    : float        Maximum energy in the table.

    Notes
    -----
    The returned callables are ``scipy.interpolate.CubicSpline`` objects
    evaluated with ``extrapolate=False``; querying outside [e_min, e_max]
    returns NaN.

    Examples
    --------
    >>> p_func, cs2_func, e_lo, e_hi = load_custom_eos("my_eos.dat")
    >>> # Use as the plus branch
    >>> from sibugec.eos import pplus
    >>> # Replace pplus with p_func in interactive_bubble_plot(...)
    """
    data = np.loadtxt(filepath, delimiter=delimiter, usecols=usecols)

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            "Custom EoS file must have at least two columns: energy and pressure."
        )

    e_raw = data[:, 0]
    p_raw = data[:, 1]

    # Sort by energy (ascending) and remove duplicate e values
    sort_idx = np.argsort(e_raw)
    e_raw = e_raw[sort_idx]
    p_raw = p_raw[sort_idx]
    e_unique, unique_idx = np.unique(e_raw, return_index=True)
    p_unique = p_raw[unique_idx]

    if len(e_unique) < 4:
        raise ValueError(
            "Custom EoS file must contain at least 4 distinct energy values."
        )

    p_func = CubicSpline(e_unique, p_unique, extrapolate=False)

    if data.shape[1] >= 3:
        cs2_raw = data[sort_idx, 2][unique_idx]
        cs2_func = CubicSpline(e_unique, cs2_raw, extrapolate=False)
    else:
        cs2_func = speed_of_sound_squared(e_unique, p_unique)

    e_min = float(e_unique[0])
    e_max = float(e_unique[-1])

    print(
        f"[SiBuGEC] Custom EoS loaded from '{filepath}': "
        f"{len(e_unique)} points, e ∈ [{e_min:.4g}, {e_max:.4g}]"
    )
    return p_func, cs2_func, e_min, e_max
