"""
sibugec.thermodynamics
======================
Thermodynamic quantities derived from the equation of state:

* Entropy density S(E) via ODE integration along each branch.
* Temperature T = dE/dS.
* Free energy F = −p.
* Critical temperature T_c where both free-energy branches cross.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, PchipInterpolator

from .eos import pplus, pminus


# ---------------------------------------------------------------------------
# Entropy ODEs  (ds/dE formulation)
# ---------------------------------------------------------------------------

def entropy_ODE_plus(s, e, eTH, p1, delta):
    """
    ODE for the HT entropy branch: dE/dS = (p+(E) + E) / S.

    Parameters
    ----------
    s     : float  Independent variable (entropy).
    e     : float   [E] current energy density.
    eTH   : float  HT turning-point energy.
    p1    : float  Pressure at eHT.
    delta : float  EoS smoothing scale.

    Returns
    -------
    list  [dE/dS]
    """
    return [(pplus(e, eTH, p1, delta) + e) / s]


def entropy_ODE_minus(s, e, eTL, n):
    """
    ODE for the LT entropy branch: dE/dS = (p−(E) + E) / S.

    Parameters
    ----------
    s   : float  Independent variable (entropy).
    e   : float   [E] current energy density.
    eTL : float  LT turning-point energy.
    n   : float  EoS shape exponent.

    Returns
    -------
    list  [dE/dS]
    """
    return [(pminus(e, eTL, n) + e) / s]


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------

def calculate_temperature(e_vals, s_vals):
    """
    Compute temperature as T = dE/dS from discrete (E, S) data.

    Parameters
    ----------
    e_vals : array_like  Energy density values.
    s_vals : array_like  Entropy density values (same length).

    Returns
    -------
    e_valid : ndarray  Energy values where T is physical.
    T_valid : ndarray  Corresponding temperature values.
    """
    e_vals = np.asarray(e_vals)
    s_vals = np.asarray(s_vals)

    if len(e_vals) < 2 or len(s_vals) < 2:
        return np.array([]), np.array([])

    temperature = np.gradient(e_vals, s_vals)
    valid = (temperature > 0.1) & (temperature < 100) & np.isfinite(temperature)
    return e_vals[valid], temperature[valid]


# ---------------------------------------------------------------------------
# Critical temperature
# ---------------------------------------------------------------------------

def find_critical_temperature(t_minus, f_minus, t_plus, f_plus):
    """
    Find the critical temperature T_c where F−(T) = F+(T).

    Uses linear interpolation of the free-energy difference onto a common
    temperature grid and locates the first sign change.

    Parameters
    ----------
    t_minus : ndarray  Temperatures for the LT branch.
    f_minus : ndarray  Free energies for the LT branch.
    t_plus  : ndarray  Temperatures for the HT branch.
    f_plus  : ndarray  Free energies for the HT branch.

    Returns
    -------
    float or None
        Critical temperature, or None if no crossing is found.
    """
    if any(len(a) == 0 for a in [t_minus, f_minus, t_plus, f_plus]):
        return None

    t_lo = max(t_minus.min(), t_plus.min())
    t_hi = min(t_minus.max(), t_plus.max())
    if t_lo >= t_hi:
        return None

    try:
        mask_m = (t_minus >= t_lo) & (t_minus <= t_hi)
        mask_p = (t_plus  >= t_lo) & (t_plus  <= t_hi)
        tm = t_minus[mask_m];  fm = f_minus[mask_m]
        tp = t_plus[mask_p];   fp = f_plus[mask_p]

        if len(tm) < 2 or len(tp) < 2:
            return None

        f_int_m = interp1d(tm[np.argsort(tm)], fm[np.argsort(tm)],
                           kind='linear', bounds_error=False, fill_value='extrapolate')
        f_int_p = interp1d(tp[np.argsort(tp)], fp[np.argsort(tp)],
                           kind='linear', bounds_error=False, fill_value='extrapolate')

        t_common = np.linspace(t_lo, t_hi, 1000)
        diff = f_int_m(t_common) - f_int_p(t_common)
        sign_changes = np.where(np.diff(np.sign(diff)))[0]

        if len(sign_changes) == 0:
            return None

        idx = sign_changes[0]
        t1, t2 = t_common[idx], t_common[idx + 1]
        d1, d2 = diff[idx], diff[idx + 1]
        return float(t1 - d1 * (t2 - t1) / (d2 - d1))

    except Exception as exc:
        print(f"[SiBuGEC] Warning in find_critical_temperature: {exc}")
        return None


# ---------------------------------------------------------------------------
# Integrated entropy branches (helper used by the EoS explorer GUI)
# ---------------------------------------------------------------------------

def integrate_entropy_branches(eTH, eTL, p1, sTL, sTH, delta, n, n_pts=1000):
    """
    Integrate the entropy ODE for both branches and return (E, S) arrays
    together with PchipInterpolator callables S(E).

    Parameters
    ----------
    eTH, eTL    : float  HT/LT turning-point energies.
    p1          : float  HT branch pressure at eHT.
    sTL, sTH    : float  Entropy anchors for each branch.
    delta       : float  HT EoS smoothing scale.
    n           : float  LT EoS shape exponent.
    n_pts       : int    Number of ODE evaluation points (default 1000).

    Returns
    -------
    e_minus, s_minus : ndarray  LT branch (E, S).
    e_plus,  s_plus  : ndarray  HT branch (E, S).
    s_minus_interp   : PchipInterpolator  S−(E).
    s_plus_interp    : PchipInterpolator  S+(E).
    """
    # LT branch — integrate backward then forward from the anchor
    y0 = [eTL]
    t_bw = np.flip(np.logspace(np.log10(1e-7), np.log10(sTL), n_pts))
    sol_bw = solve_ivp(entropy_ODE_minus, (t_bw[0], t_bw[-1]), y0,
                       t_eval=t_bw, args=(eTL, n),
                       method='RK45', rtol=1e-6, atol=1e-9)
    t_fw = np.flip(np.logspace(np.log10(10), np.log10(sTL), n_pts))
    sol_fw = solve_ivp(entropy_ODE_minus, (t_fw[0], t_fw[-1]), y0,
                       t_eval=t_fw, args=(eTL, n),
                       method='RK45', rtol=1e-6, atol=1e-9)
    e_minus = np.concatenate((np.flip(sol_fw.y[0][:-2]), sol_bw.y[0]))
    s_minus = np.concatenate((np.flip(sol_fw.t[:-2]),    sol_bw.t))

    # HT branch — backward then forward
    y0 = [eTH]
    t_bw = np.flip(np.logspace(np.log10(1e-7), np.log10(sTH), n_pts))
    sol_bw = solve_ivp(entropy_ODE_plus, (t_bw[0], t_bw[-1]), y0,
                       t_eval=t_bw, args=(eTH, p1, delta),
                       method='RK45', rtol=1e-6, atol=1e-9)
    t_fw = np.linspace(sTH, max(20.0, sTH + 15.0), n_pts)
    sol_fw = solve_ivp(entropy_ODE_plus, (t_fw[0], t_fw[-1]), y0,
                       t_eval=t_fw, args=(eTH, p1, delta),
                       method='RK45', rtol=1e-6, atol=1e-9)

    if sol_bw.success and sol_fw.success:
        e_plus = np.concatenate((sol_fw.y[0][1:], sol_bw.y[0]))
        s_plus = np.concatenate((sol_fw.t[1:],    sol_bw.t))
    elif sol_bw.success:
        e_plus, s_plus = sol_bw.y[0], sol_bw.t
    elif sol_fw.success:
        e_plus, s_plus = sol_fw.y[0], sol_fw.t
    else:
        e_plus = s_plus = np.array([])

    # Build monotone interpolants S(E)
    def _make_interp(e_arr, s_arr):
        e_u, idx = np.unique(e_arr, return_index=True)
        return PchipInterpolator(e_u, s_arr[idx])

    s_minus_interp = _make_interp(e_minus, s_minus)
    if len(e_plus) > 0:
        s_plus_interp = _make_interp(e_plus, s_plus)
    else:
        s_plus_interp = None

    return e_minus, s_minus, e_plus, s_plus, s_minus_interp, s_plus_interp
