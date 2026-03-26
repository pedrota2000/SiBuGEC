"""
sibugec.hydrodynamics
=====================
Self-similar hydrodynamic flow solvers for cosmological bubble walls.

Three solution types are supported:

* **Detonation** – supersonic wall, rarefaction behind the wall.
* **Deflagration** – subsonic wall, shock in front of the wall.
* **Hybrid** – detonation wall with a preceding rarefaction region.

The module also exports the relativistic fluid ODE ``system_minus``
and all junction-condition residuals so that callers can construct
custom solvers.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve, least_squares


# ---------------------------------------------------------------------------
# Relativistic fluid ODE (self-similar frame)
# ---------------------------------------------------------------------------

def system_minus(t, y, p_func, cs2_func, eT, p0):
    """
    Self-similar relativistic fluid equations in the variable t = 2/(xi·v·(xi−v)).

    State vector y = [v, xi, e].

    Parameters
    ----------
    t        : float     Integration variable.
    y        : array     [v, xi, e].
    p_func   : callable  Pressure function p(e).
    cs2_func : callable  Speed-of-sound squared c_s²(e).
    eT       : float     Unused turning-point argument (kept for API consistency).
    p0       : float     Unused pressure offset argument (kept for API consistency).

    Returns
    -------
    list  [dv/dt, dxi/dt, de/dt]
    """
    v, xi, e = y
    cs2 = cs2_func(e)
    dv_dt  = 2.0 * v * cs2 * (1.0 - v**2) * (1.0 - xi * v)
    dxi_dt = xi * ((xi - v)**2 - cs2 * (1.0 - xi * v)**2)
    de_dt  = 2.0 * (e + p_func(e, eT, p0)) * v * (xi - v)
    return [dv_dt, dxi_dt, de_dt]


# ---------------------------------------------------------------------------
# Junction-condition residuals
# ---------------------------------------------------------------------------

def _jc_residual(vp, vm, ep, em, pp_val, pm_val):
    """Core 2×1 junction-condition vector (energy–momentum conservation)."""
    JC1 = vp * vm * (ep - em) - (pp_val - pm_val)
    JC2 = vp * (ep + pm_val) - (em + pp_val) * vm
    if np.any(np.isnan([JC1, JC2])):
        return np.array([1e10, 1e10])
    return np.array([JC1, JC2])


def j_system_minus(minus, vp, ep, pplus, pminus, eHT, eLT, p0, p1):
    """Solve for (vm, em) given vp and ep."""
    vm, em = minus
    if np.ndim(ep) > 0:
        ep = ep.item()
    return _jc_residual(vp, vm, ep, em, pplus(ep, eHT, p1), pminus(em, eLT, p0))


def j_system_plus(plus, vm, em, pplus, pminus, eHT, eLT, p0, p1):
    """Solve for (vp, ep) given vm and em."""
    vp, ep = plus
    return _jc_residual(vp, vm, ep, em, pplus(ep, eHT, p1), pminus(em, eLT, p0))


def j_system_det_contour(unk, ep, em, pplus, pminus, eHT, eLT, p0, p1):
    """Solve for (vm, vp) given (ep, em) — used for detonation separator."""
    vm, vp = unk
    return _jc_residual(vp, vm, ep, em, pplus(ep, eHT, p1), pminus(em, eLT, p0))


def j_system_shock(xien, v_inter, e_inter, pplus, eHT, p1):
    """Junction condition for a same-phase shock in the HT fluid."""
    xi, ep = xien
    em = e_inter(xi)
    vm = (v_inter(xi) - xi) / (1.0 - v_inter(xi) * xi)
    vp = -xi
    val = _jc_residual(vp, vm, ep, em, pplus(ep, eHT, p1), pplus(em, eHT, p1))
    return np.where(np.isnan(val), 1e2, val)


def j_system_shock_dets(xien, v_inter, e_inter, pplus, pminus, eHT, eLT, p0, p1):
    """Cross-phase shock junction condition (used for limiting detonations)."""
    xi, ep = xien
    em = e_inter(xi)
    vm = (v_inter(xi) - xi) / (1.0 - v_inter(xi) * xi)
    vp = -xi
    val = _jc_residual(vp, vm, ep, em, pplus(ep, eHT, p1), pminus(em, eLT, p0))
    return np.where(np.isnan(val), 1e2, val)


def j_system_jouguet(vpem, ep, cs2minus, pplus, pminus, eHT, eLT, p0, p1):
    """Jouguet condition: vm = −c_s−(em)."""
    vp, em = vpem
    vm = -np.sqrt(np.maximum(0, cs2minus(em)))
    return _jc_residual(vp, vm, ep, em, pplus(ep, eHT, p1), pminus(em, eLT, p0))


def entropy_JC_det(vmemep, vp, pplus, pminus, eHT, eLT, p0, p1, splus, sminus):
    """Junction conditions + entropy conservation for detonation entropy separator."""
    vm, em, ep = vmemep
    gp = 1.0 / np.sqrt(1.0 - vp**2)
    gm = 1.0 / np.sqrt(1.0 - vm**2)
    if np.ndim(ep) > 0:
        ep = ep.item()
    res = _jc_residual(vp, vm, ep, em, pplus(ep, eHT, p1), pminus(em, eLT, p0))
    JC3 = gp * vp * splus(ep) - gm * vm * sminus(em)
    return [res[0], res[1], JC3] if not np.isnan(JC3) else [1e10, 1e10, 1e10]


def entropy_JC_def_wall(vpemep, vm, pplus, pminus, eHT, eLT, p0, p1, splus, sminus):
    """Junction conditions + entropy conservation for deflagration entropy separator."""
    vp, em, ep = vpemep
    gp = 1.0 / np.sqrt(1.0 - vp**2)
    gm = 1.0 / np.sqrt(1.0 - vm**2)
    if np.ndim(ep) > 0:
        ep = ep.item()
    res = _jc_residual(vp, vm, ep, em, pplus(ep, eHT, p1), pminus(em, eLT, p0))
    JC3 = gp * vp * splus(ep) - gm * vm * sminus(em)
    return [res[0], res[1], JC3] if not np.isnan(JC3) else [1e10, 1e10, 1e10]


# ---------------------------------------------------------------------------
# Monotonic-segment helper
# ---------------------------------------------------------------------------

def _find_monotonic_segment(arr, start_idx):
    """Return (backward_start, forward_end) indices of the longest monotonic
    segment through ``arr[start_idx]``."""
    n = len(arr)
    forward_end = start_idx
    if start_idx < n - 1:
        increasing = arr[start_idx + 1] > arr[start_idx]
        for i in range(start_idx + 1, n):
            if (increasing and arr[i] > arr[i - 1]) or \
               (not increasing and arr[i] < arr[i - 1]):
                forward_end = i
            else:
                break

    backward_start = start_idx
    if start_idx > 0:
        inc_back = arr[start_idx] > arr[start_idx - 1]
        for i in range(start_idx - 1, -1, -1):
            if (inc_back and arr[i] < arr[i + 1]) or \
               (not inc_back and arr[i] > arr[i + 1]):
                backward_start = i
            else:
                break

    return backward_start, forward_end


# ---------------------------------------------------------------------------
# Detonation solver
# ---------------------------------------------------------------------------

def find_deto(e_out, xiw, pplus, pminus, eTH, eTL, p0, p1,
              cs2_plus, cs2_minus, em=None, vm=None):
    """
    Compute a detonation solution for a given nucleation energy ``e_out``
    and wall velocity ``xiw``.

    Parameters
    ----------
    e_out    : float  Nucleation (outside) energy.
    xiw      : float  Wall velocity.
    pplus    : callable  HT pressure p+(e, eHT, p1).
    pminus   : callable  LT pressure p−(e, eTL, p0).
    eTH, eTL : float  HT/LT turning-point energies.
    p0, p1   : float  EoS pressure offsets.
    cs2_plus : callable  c_s²+(e).
    cs2_minus: callable  c_s²−(e).
    em, vm   : float, optional  Pre-computed junction values (skip JC solve).

    Returns
    -------
    eC : float or NaN
        Energy inside the bubble (centre energy).
    flow : list [v(xi), xi, e(xi)] or [NaN, NaN, NaN]
        Hydrodynamic profile arrays.
    """
    if np.ndim(e_out) > 0:
        e_out = e_out.item()

    if xiw < np.sqrt(np.maximum(0, cs2_plus(e_out))):
        return np.nan, np.nan

    if em is None and vm is None:
        vp = -xiw
        guesses = [(-0.9, eTL * 0.2), (-0.5 * xiw, eTL * 0.5)]
        sol_m = None
        res = None
        for guess in guesses:
            try:
                sol = least_squares(
                    j_system_minus, guess,
                    args=(vp, e_out, pplus, pminus, eTH, eTL, p0, p1),
                    bounds=([-1, 0.0], [0, eTL]),
                    ftol=1e-12, xtol=1e-12,
                )
            except Exception:
                continue
            res_tmp = j_system_minus(sol.x, vp, e_out, pplus, pminus, eTH, eTL, p0, p1)
            if (not np.any(np.isnan(res_tmp))) and np.mean(np.array(res_tmp)**2) < 1e-8:
                sol_m = sol.x
                res = res_tmp
                break
            sol_m = sol.x
            res = res_tmp
        no_JC = False
    else:
        sol_m = [vm, em]
        res = j_system_minus(sol_m, -xiw, e_out, pplus, pminus, eTH, eTL, p0, p1)
        no_JC = True

    if sol_m is None or np.any(np.isnan(sol_m)) or np.mean(np.array(res)**2) >= 1e-10:
        return np.nan, [np.nan, np.nan, np.nan]

    vm, em = sol_m
    init_point = [(vm + xiw) / (1.0 + xiw * vm), xiw, em]

    # Backward integration
    t_bw = np.linspace(0, -50, 5000)
    flow_bw = solve_ivp(system_minus, (t_bw[0], t_bw[-1]), init_point,
                        t_eval=t_bw, args=(pminus, cs2_minus, eTL, p0))
    # Forward integration
    t_fw = np.linspace(0, 2, 5000)
    flow_fw = solve_ivp(system_minus, (t_fw[0], t_fw[-1]), init_point,
                        t_eval=t_fw, args=(pminus, cs2_minus, eTL, p0))

    v_flow  = np.concatenate((np.flip(flow_bw.y[0]), flow_fw.y[0]))
    xi_flow = np.concatenate((np.flip(flow_bw.y[1]), flow_fw.y[1]))
    e_flow  = np.concatenate((np.flip(flow_bw.y[2]), flow_fw.y[2]))

    mask_e = e_flow <= init_point[2]
    v_flow  = np.flip(v_flow[mask_e])
    xi_flow = np.flip(xi_flow[mask_e])
    e_flow  = np.flip(e_flow[mask_e])

    if np.any(xi_flow > init_point[1]):
        return np.nan, np.nan

    diff_xi = np.diff(xi_flow)
    breaks = np.where(diff_xi > 0)[0]
    if len(breaks) > 0 and not no_JC:
        last_break = breaks[0]
        if np.abs(v_flow[last_break]) > 1e-6:
            return np.nan, np.nan
        xi_flow = xi_flow[:last_break + 1]
        v_flow  = v_flow[:last_break + 1]
        e_flow  = e_flow[:last_break + 1]

    if len(xi_flow) <= 1:
        return np.nan, np.nan

    xi_flow = np.concatenate((np.linspace(0, min(xi_flow), 1000),
                               np.flip(xi_flow),
                               np.linspace(max(xi_flow), 1, 1000)))
    v_flow  = np.concatenate((np.zeros(1000), np.flip(v_flow), np.zeros(1000)))
    e_flow  = np.concatenate((min(e_flow) * np.ones(1000),
                               np.flip(e_flow),
                               e_out * np.ones(1000)))
    eC = float(min(e_flow))
    return eC, [v_flow, xi_flow, e_flow]


# ---------------------------------------------------------------------------
# Deflagration solver
# ---------------------------------------------------------------------------

def find_def(eC, xiw, pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus):
    """
    Compute a deflagration solution for a given bubble energy ``eC``
    and wall velocity ``xiw``.

    Parameters
    ----------
    eC       : float  Energy inside the bubble (centre energy).
    xiw      : float  Wall velocity.
    pplus    : callable  HT pressure p+(e, eHT, p1).
    pminus   : callable  LT pressure p−(e, eTL, p0).
    eTH, eTL : float  HT/LT turning-point energies.
    p0, p1   : float  EoS pressure offsets.
    cs2_plus : callable  c_s²+(e).
    cs2_minus : callable  c_s²−(e).

    Returns
    -------
    eN : float or NaN
        Nucleation energy in front of the shock.
    flow : list [v(xi), xi, e(xi)] or None
        Hydrodynamic profile arrays.
    """
    em = eC
    vm = -xiw

    if np.ndim(eC) > 0:
        em = eC.item()

    from .hydrodynamics import _e_max_from_pressure  # lazy local helper
    e_max = _e_max_from_pressure(eTH, eTL, p0, p1, pminus, pplus)

    if np.sqrt(cs2_minus(em)) < xiw or cs2_minus(em) < 0:
        return np.nan, np.nan

    sol = least_squares(j_system_plus,
                        [-xiw * 0.1, eTH + (e_max - eTH) / 2.0],
                        args=(vm, em, pplus, pminus, eTH, eTL, p0, p1),
                        bounds=([-xiw, eTH], [0, np.inf]))
    vp, ep = sol.x
    res = j_system_plus([vp, ep], vm, em, pplus, pminus, eTH, eTL, p0, p1)
    if np.any(np.isnan(res)) or np.mean(np.array(res)**2) > 1e-10 or \
            ep < eTH or abs(vp) < 1e-15:
        sol = least_squares(j_system_plus,
                            [-xiw * 0.9, eTH * 1.05],
                            args=(vm, em, pplus, pminus, eTH, eTL, p0, p1),
                            bounds=([-xiw, eTH], [0, np.inf]))
        vp, ep = sol.x
        res = j_system_plus([vp, ep], vm, em, pplus, pminus, eTH, eTL, p0, p1)
        if np.any(np.isnan(res)) or np.mean(np.array(res)**2) > 1e-10 or \
                ep < eTH or abs(vp) < 1e-15:
            return np.nan, np.nan

    init_point = np.array([(vp + xiw) / (1.0 + xiw * vp), xiw, ep])
    dxi_dt = xiw * ((xiw - init_point[0])**2
                    - cs2_plus(init_point[2]) * (1.0 - xiw * init_point[0])**2)
    t_eval = np.linspace(0, 100, 1000) if dxi_dt > 0 else np.linspace(0, -100, 1000)

    flow = solve_ivp(system_minus, (t_eval[0], t_eval[-1]), init_point,
                     t_eval=t_eval, args=(pplus, cs2_plus, eTH, p1))
    v_f, xi_f, e_f = flow.y

    if np.any(np.diff(xi_f) < 0):
        cut = np.where(np.diff(xi_f) < 0)[0][0]
    else:
        cut = len(xi_f)
    v_f = v_f[:cut];  xi_f = xi_f[:cut];  e_f = e_f[:cut]

    mask = e_f <= init_point[2]
    v_f = v_f[mask][~np.isnan(v_f[mask])]
    xi_f = xi_f[mask][~np.isnan(xi_f[mask])]
    e_f  = e_f[mask][~np.isnan(e_f[mask])]

    if len(xi_f) <= 1:
        return np.nan, np.nan

    v_spl = CubicSpline(xi_f, v_f)
    e_spl = CubicSpline(xi_f, e_f)
    xi_dense = np.linspace(xiw, max(xi_f), 50)

    if v_spl(xi_dense[-1]) <= 1e-20:
        xi_sh = xi_dense[-1]
        eN = e_spl(xi_dense[-1])
        xi_out = np.concatenate((np.linspace(0, xiw, 1000),
                                  np.linspace(xiw, xi_sh, 1000),
                                  np.linspace(xi_sh, 1, 1000)))
        v_out  = np.concatenate((np.zeros(1000),
                                  v_spl(np.linspace(xiw, xi_sh, 1000)),
                                  np.zeros(1000)))
        e_out_ = np.concatenate((eC * np.ones(1000),
                                  e_spl(np.linspace(xiw, xi_sh, 1000)),
                                  eN * np.ones(1000)))
        return eN, [v_out, xi_out, e_out_]

    for xi_try in xi_dense:
        if xi_try < xiw or e_spl(xi_try) < eTH:
            continue
        sol_sh = least_squares(j_system_shock,
                               [xi_try, e_spl(xi_try)],
                               args=(v_spl, e_spl, pplus, eTH, p1),
                               bounds=([xiw, eTH], [max(xi_f), np.inf]))
        xi_sh, eN = sol_sh.x
        res = j_system_shock([xi_sh, eN], v_spl, e_spl, pplus, eTH, p1)
        if np.mean(np.array(res)**2) < 1e-10 and xi_sh <= max(xi_f):
            xi_out = np.concatenate((np.linspace(0, xiw, 1000),
                                      np.linspace(xiw, xi_sh, 1000),
                                      np.linspace(xi_sh, 1, 1000)))
            v_out  = np.concatenate((np.zeros(1000),
                                      v_spl(np.linspace(xiw, xi_sh, 1000)),
                                      np.zeros(1000)))
            e_out_ = np.concatenate((eC * np.ones(1000),
                                      e_spl(np.linspace(xiw, xi_sh, 1000)),
                                      eN * np.ones(1000)))
            return eN, [v_out, xi_out, e_out_]

    return np.nan, np.nan


# ---------------------------------------------------------------------------
# Hybrid solver
# ---------------------------------------------------------------------------

def find_hyb(emwall, xiw, pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus):
    """
    Compute a hybrid solution: detonation at the wall preceded by a
    rarefaction in the LT phase.

    Parameters
    ----------
    emwall   : float  Energy at the wall on the LT side.
    xiw      : float  Wall velocity.
    (others) : same as find_deto / find_def.

    Returns
    -------
    eN     : float or NaN  Nucleation energy.
    flow   : list or NaN   [v(xi), xi, e(xi)].
    emwall : float or NaN  Wall energy (returned for bookkeeping).
    """
    if isinstance(emwall, np.ndarray):
        emwall = emwall.item()
    if emwall > eTL:
        return np.nan, np.nan, np.nan

    vm = -np.sqrt(np.maximum(0, cs2_minus(emwall)))
    if (vm + xiw) / (1.0 + xiw * vm) < 0:
        return np.nan, np.nan, np.nan

    init_point = [(vm + xiw) / (1.0 + xiw * vm), xiw, emwall]
    if np.any(np.isnan(init_point)):
        return np.nan, np.nan, np.nan

    # Build rarefaction behind the wall
    t_fw = np.linspace(0,  2, 5000)
    t_bw = np.linspace(0, -50, 5000)
    flow_fw = solve_ivp(system_minus, (t_fw[0], t_fw[-1]), init_point,
                        t_eval=t_fw, args=(pminus, cs2_minus, eTL, p0))
    flow_bw = solve_ivp(system_minus, (t_bw[0], t_bw[-1]), init_point,
                        t_eval=t_bw, args=(pminus, cs2_minus, eTL, p0))

    v_r  = np.concatenate((np.flip(flow_bw.y[0]), flow_fw.y[0]))
    xi_r = np.concatenate((np.flip(flow_bw.y[1]), flow_fw.y[1]))
    e_r  = np.concatenate((np.flip(flow_bw.y[2]), flow_fw.y[2]))

    # Select the rarefaction portion
    cut = v_r <= init_point[0]
    xi_rare = xi_r[cut];  v_rare = v_r[cut];  e_rare = e_r[cut]

    if np.any(np.diff(xi_rare) < -1e-2) or np.any(xi_rare - init_point[1] > 0):
        return np.nan, np.nan, np.nan

    init_idx = len(flow_bw.y[0]) - 1
    mono_s, mono_e = _find_monotonic_segment(xi_rare, init_idx)
    xi_rare = xi_rare[mono_s:mono_e + 1]
    v_rare  = v_rare[mono_s:mono_e + 1]
    e_rare  = e_rare[mono_s:mono_e + 1]

    if abs(max(xi_rare) - init_point[1]) > 1e-2:
        return np.nan, np.nan, np.nan

    diff_xi = np.diff(xi_rare)
    if np.all(diff_xi > 0):
        v_rare_spl = CubicSpline(xi_rare, v_rare)
        e_rare_spl = CubicSpline(xi_rare, e_rare)
    elif np.all(diff_xi < 0):
        v_rare_spl = CubicSpline(np.flip(xi_rare), np.flip(v_rare))
        e_rare_spl = CubicSpline(np.flip(xi_rare), np.flip(e_rare))
    else:
        return np.nan, np.nan, np.nan

    if v_rare_spl(xiw) < 1e-3:
        return np.nan, np.nan, np.nan

    # Junction at the wall (HT side)
    from .hydrodynamics import _e_max_from_pressure  # avoid circular import
    e_max = _e_max_from_pressure(eTH, eTL, p0, p1, pminus, pplus)
    vm_wall = -np.sqrt(np.maximum(0, cs2_minus(emwall)))
    em = emwall
    sol = least_squares(j_system_plus,
                        [vm_wall / 2.0, (e_max + eTH) / 2.0],
                        args=(vm_wall, em, pplus, pminus, eTH, eTL, p0, p1),
                        bounds=([vm_wall, eTH], [0, np.inf]))
    vp, ep = sol.x
    res = j_system_plus([vp, ep], vm_wall, em, pplus, pminus, eTH, eTL, p0, p1)
    if np.any(np.isnan(res)) or np.mean(np.array(res)**2) > 1e-12 or \
            ep < eTH or abs(vp) < 1e-15:
        sol = least_squares(j_system_plus,
                            [vm_wall * 0.1, (e_max + eTH) / 2.0],
                            args=(vm_wall, em, pplus, pminus, eTH, eTL, p0, p1),
                            bounds=([vm_wall, eTH], [0, np.inf]))
        vp, ep = sol.x
        res = j_system_plus([vp, ep], vm_wall, em, pplus, pminus, eTH, eTL, p0, p1)
        if np.any(np.isnan(res)) or np.mean(np.array(res)**2) > 1e-10 or \
                ep < eTH or abs(vp) < 1e-15:
            return np.nan, np.nan, np.nan

    if ep < eTH:
        return np.nan, np.nan, np.nan

    # Integration in the HT region after the wall
    init_p = np.array([(vp + xiw) / (1.0 + xiw * vp), xiw, ep])
    dxi_dt = xiw * ((xiw - init_p[0])**2
                    - cs2_plus(init_p[2]) * (1.0 - xiw * init_p[0])**2)
    t_eval = np.linspace(0, 20, 100) if dxi_dt > 0 else np.linspace(0, -20, 100)

    flow_p = solve_ivp(system_minus, (t_eval[0], t_eval[-1]), init_p,
                       t_eval=t_eval, args=(pplus, cs2_plus, eTH, p1))
    v_f, xi_f, e_f = flow_p.y

    if np.any(np.diff(xi_f) < 0):
        cut = np.where(np.diff(xi_f) < 0)[0][0]
    else:
        cut = len(xi_f)
    v_f = v_f[:cut];  xi_f = xi_f[:cut];  e_f = e_f[:cut]

    mask = e_f <= init_p[2]
    valid = ~np.isnan(v_f[mask]) & ~np.isnan(xi_f[mask]) & ~np.isnan(e_f[mask])
    v_f  = v_f[mask][valid]
    xi_f = xi_f[mask][valid]
    e_f  = e_f[mask][valid]

    if len(xi_f) <= 1:
        return np.nan, np.nan, np.nan

    v_spl = CubicSpline(xi_f, v_f)
    e_spl = CubicSpline(xi_f, e_f)
    xi_dense = np.linspace(xiw, max(xi_f), 3)

    for xi_try in xi_dense:
        sol_sh = fsolve(j_system_shock,
                        [xi_try, e_spl(xi_try)],
                        args=(v_spl, e_spl, pplus, eTH, p1))
        xi_sh, eN = sol_sh
        res = j_system_shock([xi_sh, eN], v_spl, e_spl, pplus, eTH, p1)
        if np.mean(np.array(res)**2) < 1e-10 and \
                xi_sh < max(xi_f) and xi_sh > xiw and eN > eTH:
            xi_out = np.concatenate((
                np.linspace(0, min(xi_rare), 1000),
                np.linspace(min(xi_rare), xiw, 1000),
                np.linspace(xiw, xi_sh, 1000),
                np.linspace(xi_sh, 1, 1000),
            ))
            v_out = np.concatenate((
                np.zeros(1000),
                v_rare_spl(np.linspace(min(xi_rare), xiw, 1000)),
                v_spl(np.linspace(xiw, xi_sh, 1000)),
                np.zeros(1000),
            ))
            e_rare_vals = e_rare_spl(np.linspace(min(xi_rare), xiw, 1000))
            e_out_ = np.concatenate((
                min(e_rare_vals) * np.ones(1000),
                e_rare_vals,
                e_spl(np.linspace(xiw, xi_sh, 1000)),
                eN * np.ones(1000),
            ))
            if eN < eTH or eN <= min(e_out_) or min(e_out_) < 0:
                return np.nan, np.nan, np.nan
            return eN, [v_out, xi_out, e_out_], emwall

    return np.nan, np.nan, np.nan


# ---------------------------------------------------------------------------
# Private utility — e_max from pressure balance
# ---------------------------------------------------------------------------

def _e_max_from_pressure(eTH, eTL, p0, p1, pminus, pplus):
    """Return the energy e_max where p−(eTL) = p+(e_max)."""
    from scipy.optimize import fsolve as _fsolve
    target = pminus(eTL, eTL, p0) if p0 is not None else pminus(eTL, eTL)
    guess = eTL * 2 if eTL > eTH else eTH * 5
    result = _fsolve(lambda e: target - pplus(e, eTH, p1), guess)[0]
    return max(result, eTH + 0.2)
