"""
sibugec.separators
==================
Phase-space separator and contour finders.

These routines compute the boundary curves that delimit the regions of
(ξ_w, E_N) and (ξ_w, E_C) space occupied by each solution type.

Public functions
----------------
compute_def_separator            Deflagration Chapman–Jouguet boundary.
compute_det_separator            Detonation Chapman–Jouguet boundary.
compute_hyb_separator            Hybrid onset boundary.
limiting_detonation_contour_finder
jouguet_detonation_contour_finder
compute_entropy_separator_det    Entropy-production boundary for detonations.
compute_entropy_separator_def    Entropy-production boundary for deflagrations.
compute_alphan                   Convert nucleation energy to α_N.
eN_contour_finder                Contours of constant E_C in (ξ_w, E_N) space.
eC_contour_finder                Contours of constant E_N in (ξ_w, E_C) space.
entropy_checker                  Verify the entropy condition at each discontinuity.
"""

import numpy as np
from scipy.optimize import fsolve, least_squares, brentq
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from .hydrodynamics import (
    system_minus,
    j_system_minus,
    j_system_plus,
    j_system_det_contour,
    j_system_shock,
    j_system_shock_dets,
    j_system_jouguet,
    entropy_JC_det,
    entropy_JC_def_wall,
    find_deto,
    find_def,
    find_hyb,
    _e_max_from_pressure,
    _find_monotonic_segment,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _inf_point_equation(e, cs2_minus, eTL, p0, pminus):
    """Inflection-point condition for the LT branch rarefaction."""
    first  = 2.0 * (cs2_minus(e) - 1.0) * cs2_minus(e)
    second = -(e + pminus(e, eTL, p0)) * cs2_minus(e, nu=1)
    return first + second


def _emwall_cs_finder(e, cs, cs2_func):
    return cs2_func(e) - cs**2


# ---------------------------------------------------------------------------
# Separator computations
# ---------------------------------------------------------------------------

def compute_def_separator(xiw, pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus):
    """
    Locate the deflagration Chapman–Jouguet separator at wall speed ``xiw``.

    Returns
    -------
    (eC, eN) : (float, float) or (NaN, NaN)
    """
    sol = least_squares(
        lambda e: np.sqrt(cs2_minus(e)) - xiw,
        eTL / 2.0,
        bounds=(1e-6, eTL),
    )
    if abs(np.sqrt(cs2_minus(sol.x)) - xiw) < 1e-6:
        eN_sol = find_def(sol.x.item() - 1e-3, xiw,
                          pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus)[0]
        eC_val = float(np.array(sol.x).item())
        eN_val = float(np.array(eN_sol).item()) if np.ndim(eN_sol) > 0 else float(eN_sol)
        return eC_val, eN_val
    return np.nan, np.nan


def compute_det_separator(eN, pplus, pminus, eTH, eTL, p0, p1,
                          cs2_plus, cs2_minus):
    """
    Locate the detonation Chapman–Jouguet separator at nucleation energy ``eN``.

    Returns
    -------
    (xiw, eC) : (float, float) or (NaN, NaN)
    """
    sol = least_squares(
        j_system_det_contour,
        (-0.9, -0.9),
        args=(eN, eTL, pplus, pminus, eTH, eTL, p0, p1),
        bounds=([-1, -1], [0, 0]),
    )
    vm, vp = sol.x
    res = j_system_det_contour(sol.x, eN, eTL, pplus, pminus, eTH, eTL, p0, p1)
    if np.mean(np.array(res)**2) < 1e-10:
        eC_sol = find_deto(eN, -vp.item(), pplus, pminus, eTH, eTL, p0, p1,
                           cs2_plus, cs2_minus, em=eTL, vm=vm)[0]
        return float(-vp.item()), float(np.array(eC_sol).item() if np.ndim(eC_sol) > 0 else eC_sol)
    return np.nan, np.nan


def compute_hyb_separator(xiw, pplus, pminus, eTH, eTL, p0, p1,
                           cs2_plus, cs2_minus, inf_point_eq):
    """
    Locate the hybrid onset separator at wall speed ``xiw``.

    Parameters
    ----------
    inf_point_eq : callable(e)  Inflection-point residual in the LT branch.

    Returns
    -------
    (eN, eC) : (float, float) or (NaN, NaN)
    """
    sol = least_squares(inf_point_eq, eTL / 2.0, bounds=(1e-6, eTL))
    if abs(inf_point_eq(sol.x)) > 1e-6:
        return np.nan, np.nan
    if abs(np.sqrt(cs2_minus(sol.x)) - xiw) < 1e-6:
        return np.nan, np.nan
    eN_sol, flow, __ = find_hyb(sol.x.item() - 1e-3, xiw,
                                 pplus, pminus, eTH, eTL, p0, p1,
                                 cs2_plus, cs2_minus)
    if np.any(np.isnan(flow)):
        return np.nan, np.nan
    __, __, e_flow = flow
    return float(eN_sol), float(min(e_flow))


def limiting_detonation_contour_finder(xiw, pplus, pminus, eTH, eTL, p0, p1,
                                        cs2_plus, cs2_minus, inf_point_eq):
    """
    Compute the limiting detonation trajectory for wall speed ``xiw``.

    Returns
    -------
    (xi_sh, eN, eC) : each float or NaN
    """
    if np.ndim(xiw) > 0:
        xiw = xiw.item()

    sol_infl = least_squares(inf_point_eq, eTL / 2.0, bounds=(1e-6, eTL))
    emwall = sol_infl.x[0]
    cs2_wall = cs2_minus(emwall)
    if cs2_wall < 0:
        return np.nan, np.nan, np.nan

    v_init = (xiw - np.sqrt(cs2_wall)) / (1.0 + xiw * (-np.sqrt(cs2_wall)))
    if v_init < 0:
        emwall = least_squares(
            _emwall_cs_finder, eTL / 2.0,
            bounds=(0, eTL), args=(xiw, cs2_minus),
            ftol=1e-8, xtol=1e-8,
        ).x[0] + 1e-5
        cs2_wall = cs2_minus(emwall)
        v_init = (xiw - np.sqrt(cs2_wall)) / (1.0 + xiw * (-np.sqrt(cs2_wall)))
        if cs2_minus(sol_infl.x[0]) < xiw**2:
            return np.nan, np.nan, np.nan

    if abs(v_init) < 1e-10:
        return np.nan, np.nan, np.nan

    init_point = [v_init, xiw, emwall]
    flow_bw = solve_ivp(system_minus, (0, -100), init_point,
                        t_eval=np.linspace(0, -100, 1000),
                        args=(pminus, cs2_minus, eTL, p0),
                        method='RK45', rtol=1e-8, atol=1e-10)
    flow_fw = solve_ivp(system_minus, (0, 100), init_point,
                        t_eval=np.linspace(0, 100, 1000),
                        args=(pminus, cs2_minus, eTL, p0),
                        method='RK45', rtol=1e-8, atol=1e-10)

    v_flow  = np.concatenate((np.flip(flow_bw.y[0]), flow_fw.y[0]))
    xi_flow = np.concatenate((np.flip(flow_bw.y[1]), flow_fw.y[1]))
    e_flow  = np.concatenate((np.flip(flow_bw.y[2]), flow_fw.y[2]))

    breaks = np.where(np.diff(xi_flow) < 0)[0]
    if len(breaks) > 0:
        last = breaks[0]
        if abs(v_flow[last]) > 1e-6:
            return np.nan, np.nan, np.nan

    xi_u, idx_u = np.unique(xi_flow, return_index=True)
    v_u = v_flow[idx_u];  e_u = e_flow[idx_u]
    sel = xi_u <= 1.0
    v_int = CubicSpline(xi_u[sel], v_u[sel])
    e_int = CubicSpline(xi_u[sel], e_u[sel])

    for xi_cur in np.flip(np.linspace(xiw, 1.0, 50)):
        if xi_cur <= xiw or xi_cur > max(xi_u) or e_int(xi_cur) < eTH:
            continue
        sol = least_squares(
            j_system_shock_dets, (xi_cur, e_int(xi_cur)),
            args=(v_int, e_int, pplus, pminus, eTH, eTL, p0, p1),
            bounds=([xiw, eTH], [1, np.inf]),
            ftol=1e-10, xtol=1e-10,
        )
        xi_sh, eN = sol.x
        res = j_system_shock_dets(sol.x, v_int, e_int, pplus, pminus, eTH, eTL, p0, p1)
        if (np.mean(np.array(res)**2) < 1e-12 and xi_sh > xiw
                and xi_sh <= max(xi_u) and eN >= eTH and xi_sh <= 1 - 2e-5):
            return float(xi_sh), float(eN), float(min(e_u))

    return np.nan, np.nan, np.nan


def jouguet_detonation_contour_finder(eN, pplus, pminus, eTH, eTL, p0, p1,
                                       cs2_plus, cs2_minus):
    """
    Compute the Jouguet detonation point for nucleation energy ``eN``.

    Returns
    -------
    (xiw, eC) : (float, float) or (NaN, NaN)
    """
    ep = eN.item() if np.ndim(eN) > 0 else eN
    sol = least_squares(
        j_system_jouguet, (-0.9, eTL * 0.5),
        args=(ep, cs2_minus, pplus, pminus, eTH, eTL, p0, p1),
        bounds=([-1, 0.0], [0, eTL]),
    )
    vp, em = sol.x
    res = j_system_jouguet(sol.x, ep, cs2_minus, pplus, pminus, eTH, eTL, p0, p1)
    if np.mean(np.array(res)**2) < 1e-12:
        xiw = -vp
        eC = find_deto(eN, xiw, pplus, pminus, eTH, eTL, p0, p1,
                       cs2_plus, cs2_minus,
                       em=em, vm=-np.sqrt(cs2_minus(em)))[0]
        return float(xiw), float(np.array(eC).item() if np.ndim(eC) > 0 else eC)
    return np.nan, np.nan


def compute_entropy_separator_det(xiw, pplus, pminus, eTH, eTL, p0, p1,
                                   cs2_plus, cs2_minus, splus, sminus, e_max):
    """
    Entropy-production separator for detonations at wall speed ``xiw``.

    Returns
    -------
    (eN, eC) : (float, float) or (NaN, NaN)
    """
    vp = -xiw
    sol = least_squares(
        entropy_JC_det,
        (-0.975, eTL * 0.95, (e_max + eTH) / 2.0),
        args=(vp, pplus, pminus, eTH, eTL, p0, p1, splus, sminus),
        bounds=([-1, 0, eTH], [0, eTL, e_max]),
    )
    vm, em, ep = sol.x
    res = entropy_JC_det(sol.x, vp, pplus, pminus, eTH, eTL, p0, p1, splus, sminus)
    if np.mean(np.array(res)**2) < 1e-10:
        eC = find_deto(ep, -vp.item(), pplus, pminus, eTH, eTL, p0, p1,
                       cs2_plus, cs2_minus, em=em, vm=vm)[0]
        return float(ep.item()), float(np.array(eC).item() if np.ndim(eC) > 0 else eC)
    return np.nan, np.nan


def compute_entropy_separator_def(xiw, pplus, pminus, eTH, eTL, p0, p1,
                                   cs2_plus, cs2_minus, splus, sminus, e_max):
    """
    Entropy-production separator for deflagrations at wall speed ``xiw``.

    Returns
    -------
    (eC, eN) : (float, float) or (NaN, NaN)
    """
    vm = -xiw
    if eTH < eTL:
        x0 = (-0.1 * xiw, eTL * 0.5, (e_max + eTH) * 0.5)
    else:
        x0 = (-0.1 * xiw, eTL * 0.1, 1.1 * eTH)

    sol = least_squares(
        entropy_JC_def_wall, x0,
        args=(vm, pplus, pminus, eTH, eTL, p0, p1, splus, sminus),
        bounds=([-1, 0, eTH], [0, eTL, e_max]),
    )
    vp, em, ep = sol.x
    res = entropy_JC_def_wall(sol.x, vm, pplus, pminus, eTH, eTL, p0, p1, splus, sminus)
    if np.mean(np.array(res)**2) < 1e-10:
        eN = find_def(em, -vm.item(), pplus, pminus, eTH, eTL, p0, p1,
                      cs2_plus, cs2_minus)[0]
        return float(em.item()), float(np.array(eN).item() if np.ndim(eN) > 0 else eN)
    return np.nan, np.nan


# ---------------------------------------------------------------------------
# alpha_N
# ---------------------------------------------------------------------------

def compute_alphan(eN, pplus, pminus, eTH, eTL, p0, p1, splus, sminus, n, delta):
    """
    Convert nucleation energy ``eN`` to the strength parameter α_N.

    Parameters
    ----------
    splus, sminus : PchipInterpolator  Entropy interpolants S+(E), S−(E).

    Returns
    -------
    float or NaN
    """
    eN_val = float(eN.item() if np.ndim(eN) > 0 else eN)
    if np.isnan(eN_val):
        return np.nan

    Tn = 1.0 / splus.derivative()(eN_val)
    eh = eN_val
    ph = pplus(eh, eTH, p1, delta=delta)
    theta_h = 0.25 * (eh - 3.0 * ph)

    sol = least_squares(
        lambda e: sminus.derivative()(e) - 1.0 / Tn,
        eTL / 2.0,
        bounds=(1e-6, eTL),
    )
    el = sol.x[0]
    if (sminus.derivative()(el) - 1.0 / Tn)**2 > 1e-10 or el > eTL:
        return np.nan

    pl = pminus(el, eTL, n=n)
    theta_l = 0.25 * (el - 3.0 * pl)
    wh = eh + ph
    return float(4.0 / 3.0 * (theta_h - theta_l) / wh)


# ---------------------------------------------------------------------------
# Contour finders
# ---------------------------------------------------------------------------

def eN_contour_finder(xiw_list, eC_list, detonation_data, deflagration_data,
                       hybrid_data, pplus, pminus, eTH, eTL, p0, p1,
                       cs2_plus, cs2_minus):
    """
    For each ξ_w in ``xiw_list`` find the E_N values corresponding to the
    E_C bounds ``eC_list = (eC_minus, eC_plus)``.

    Returns
    -------
    eN_plus_contour, eN_minus_contour : ndarray of shape (N, 2)
    """
    def _safe_match_det(eN, xiw, eC):
        eC_sol, __ = find_deto(eN, xiw, pplus, pminus, eTH, eTL, p0, p1,
                                cs2_plus, cs2_minus)
        return (eC_sol if eC_sol is not None else np.nan) - eC

    def _safe_match_def(eC_in, xiw, eN):
        eN_sol, __ = find_def(eC_in, xiw, pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus)
        return (eN_sol if eN_sol is not None else np.nan) - eN

    def _safe_match_hyb(eC_in, xiw, eN):
        eN_sol, flow, __ = find_hyb(eC_in, xiw, pplus, pminus, eTH, eTL, p0, p1,
                                     cs2_plus, cs2_minus)
        return (eN_sol if eN_sol is not None else np.nan) - eN

    def _safe_match_hyb_eC(em, xiw, eC_target):
        __, flow, __ = find_hyb(em, xiw, pplus, pminus, eTH, eTL, p0, p1,
                                 cs2_plus, cs2_minus)
        if np.any(np.isnan(flow)):
            return 1e10
        return float(min(flow[2])) - eC_target

    eC_minus, eC_plus = eC_list
    eN_plus_contour = []
    eN_minus_contour = []

    for xiw_val in xiw_list:
        best_p = best_m = {'dist': np.inf, 'sol': None, 'type': None}

        for row in detonation_data:
            xiw_r, eN_r, eC_r = row[0], row[1], row[2]
            if xiw_r != xiw_val:
                continue
            dp = abs(eC_r - eC_plus);  dm = abs(eC_r - eC_minus)
            if dp < best_p['dist']:
                best_p = {'dist': dp, 'sol': row, 'type': 'detonation',
                          'above': eC_r >= eC_plus}
            if dm < best_m['dist']:
                best_m = {'dist': dm, 'sol': row, 'type': 'detonation',
                          'above': eC_r >= eC_minus}

        for row in deflagration_data:
            xiw_r, eN_r, eC_r = row[0], row[1], row[2]
            if xiw_r != xiw_val:
                continue
            dp = abs(eC_r - eC_plus);  dm = abs(eC_r - eC_minus)
            if dp < best_p['dist']:
                best_p = {'dist': dp, 'sol': row, 'type': 'deflagration',
                          'above': eC_r >= eC_plus}
            if dm < best_m['dist']:
                best_m = {'dist': dm, 'sol': row, 'type': 'deflagration',
                          'above': eC_r >= eC_minus}

        for row in hybrid_data:
            xiw_r, eN_r, eC_r = row[0], row[1], row[2]
            if xiw_r != xiw_val:
                continue
            dp = abs(eC_r - eC_plus);  dm = abs(eC_r - eC_minus)
            if dp < best_p['dist']:
                best_p = {'dist': dp, 'sol': row, 'type': 'hybrid',
                          'above': eC_r >= eC_plus}
            if dm < best_m['dist']:
                best_m = {'dist': dm, 'sol': row, 'type': 'hybrid',
                          'above': eC_r >= eC_minus}

        # Plus contour
        if best_p['type'] is not None:
            if not best_p['above']:
                eN_plus_contour.append([xiw_val, best_p['sol'][1]])
            elif best_p['type'] == 'deflagration':
                eN_sol, __ = find_def(eC_plus, xiw_val, pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus)
                eN_plus_contour.append([xiw_val, float(np.array(eN_sol).item() if np.ndim(eN_sol) > 0 else eN_sol)])
            elif best_p['type'] == 'detonation':
                eN_sol = fsolve(_safe_match_det, best_p['sol'][1], args=(xiw_val, eC_plus))
                eN_plus_contour.append([xiw_val, float(np.array(eN_sol).item() if np.ndim(eN_sol) > 0 else eN_sol)])
            elif best_p['type'] == 'hybrid':
                em_sol = fsolve(_safe_match_hyb_eC, best_p['sol'][-1], args=(xiw_val, eC_plus))
                eN_sol, __, __ = find_hyb(em_sol, xiw_val, pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus)
                eN_plus_contour.append([xiw_val, float(np.array(eN_sol).item() if np.ndim(eN_sol) > 0 else eN_sol)])

        # Minus contour
        if best_m['type'] is not None:
            eN_minus_contour.append([xiw_val, best_m['sol'][1]])

    return np.array(eN_plus_contour), np.array(eN_minus_contour)


def eC_contour_finder(xiw_list, eN_list, detonation_data, deflagration_data,
                       hybrid_data, pplus, pminus, eTH, eTL, p0, p1,
                       cs2_plus, cs2_minus):
    """
    For each ξ_w in ``xiw_list`` find the E_C values corresponding to the
    E_N bounds ``eN_list = (eN_minus, eN_plus)``.

    Returns
    -------
    eC_plus_contour, eC_minus_contour : ndarray of shape (N, 2)
    """
    def _safe_match_def(eC_in, xiw, eN):
        eN_sol, __ = find_def(eC_in, xiw, pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus)
        return (eN_sol if eN_sol is not None else np.nan) - eN

    def _safe_match_hyb(em, xiw, eN):
        eN_sol, __, __ = find_hyb(em, xiw, pplus, pminus, eTH, eTL, p0, p1,
                                   cs2_plus, cs2_minus)
        return (eN_sol if eN_sol is not None else np.nan) - eN

    eN_minus, eN_plus = eN_list
    eC_plus_contour = []
    eC_minus_contour = []

    for xiw_val in xiw_list:
        best_p = best_m = {'dist': np.inf, 'sol': None, 'type': None,
                           'above': False, 'below': False}

        for row in detonation_data:
            xiw_r, eN_r, eC_r = row[0], row[1], row[2]
            if xiw_r != xiw_val:
                continue
            dp = abs(eN_r - eN_plus);  dm = abs(eN_r - eN_minus)
            if dp < best_p['dist']:
                best_p = {'dist': dp, 'sol': row, 'type': 'detonation',
                          'above': eN_r >= eN_plus, 'below': eN_r <= eN_minus}
            if dm < best_m['dist']:
                best_m = {'dist': dm, 'sol': row, 'type': 'detonation',
                          'above': eN_r >= eN_minus, 'below': eN_r <= eN_minus}

        for row in deflagration_data:
            xiw_r, eN_r, eC_r = row[0], row[1], row[2]
            if xiw_r != xiw_val:
                continue
            dp = abs(eN_r - eN_plus);  dm = abs(eN_r - eN_minus)
            if dp < best_p['dist']:
                best_p = {'dist': dp, 'sol': row, 'type': 'deflagration',
                          'above': eN_r >= eN_plus, 'below': eN_r <= eN_minus}
            if dm < best_m['dist']:
                best_m = {'dist': dm, 'sol': row, 'type': 'deflagration',
                          'above': eN_r >= eN_minus, 'below': eN_r <= eN_minus}

        for row in hybrid_data:
            xiw_r, eN_r, eC_r = row[0], row[1], row[2]
            if xiw_r != xiw_val:
                continue
            dp = abs(eN_r - eN_plus);  dm = abs(eN_r - eN_minus)
            if dp < best_p['dist']:
                best_p = {'dist': dp, 'sol': row, 'type': 'hybrid',
                          'above': eN_r >= eN_plus, 'below': eN_r <= eN_minus}
            if dm < best_m['dist']:
                best_m = {'dist': dm, 'sol': row, 'type': 'hybrid',
                          'above': eN_r >= eN_minus, 'below': eN_r <= eN_minus}

        # Plus contour
        if best_p['type'] is not None:
            if not best_p['above']:
                eC_plus_contour.append([xiw_val, best_p['sol'][2]])
            elif best_p['type'] == 'deflagration':
                eC_sol = fsolve(_safe_match_def, best_p['sol'][2], args=(xiw_val, eN_plus))
                eC_plus_contour.append([xiw_val, float(np.array(eC_sol).item() if np.ndim(eC_sol) > 0 else eC_sol)])
            elif best_p['type'] == 'detonation':
                eC_sol, __ = find_deto(eN_plus, xiw_val, pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus)
                eC_plus_contour.append([xiw_val, float(np.array(eC_sol).item() if np.ndim(eC_sol) > 0 else eC_sol)])
            elif best_p['type'] == 'hybrid':
                em_sol = fsolve(_safe_match_hyb, best_p['sol'][-1], args=(xiw_val, eN_plus))
                __, flow, __ = find_hyb(em_sol, xiw_val, pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus)
                eC_plus_contour.append([xiw_val, float(min(flow[2]))])

        # Minus contour
        if best_m['type'] is not None:
            if best_m['below']:
                eC_minus_contour.append([xiw_val, best_m['sol'][2]])
            elif best_m['type'] == 'deflagration':
                eC_sol = fsolve(_safe_match_def, best_m['sol'][2], args=(xiw_val, eN_minus))
                eC_minus_contour.append([xiw_val, float(np.array(eC_sol).item() if np.ndim(eC_sol) > 0 else eC_sol)])
            elif best_m['type'] == 'detonation':
                eC_sol, __ = find_deto(eN_minus, xiw_val, pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus)
                eC_minus_contour.append([xiw_val, float(np.array(eC_sol).item() if np.ndim(eC_sol) > 0 else eC_sol)])
            elif best_m['type'] == 'hybrid':
                em_sol = fsolve(_safe_match_hyb, best_m['sol'][-1], args=(xiw_val, eN_minus))
                __, flow, __ = find_hyb(em_sol, xiw_val, pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus)
                eC_minus_contour.append([xiw_val, float(min(flow[2]))])

    return np.array(eC_plus_contour), np.array(eC_minus_contour)


# ---------------------------------------------------------------------------
# Entropy checker
# ---------------------------------------------------------------------------

def entropy_checker(detonation_data, deflagration_data, hybrid_data,
                    splus, sminus):
    """
    Verify the entropy condition ΔJ_s ≥ 0 at each discontinuity.

    Returns arrays of entropy jumps ΔJ_s = J_s+ − J_s− at:
    - detonation walls
    - deflagration walls
    - deflagration shocks
    - hybrid walls
    - hybrid shocks

    Negative values indicate entropy violation.
    """
    def _currents(v, xi, e, s_func):
        v_rest = (v - xi) / (1.0 - v * xi)
        g = 1.0 / np.sqrt(1.0 - v_rest**2)
        return v_rest * g * s_func(e)

    entropy_det     = []
    entropy_def     = []
    entropy_def_sh  = []
    entropy_hyb     = []
    entropy_hyb_sh  = []

    for xiw, eN, eC, v_flow, xi_flow, e_flow in detonation_data:
        pos = np.where(xi_flow == xiw)[0]
        if len(pos) == 0:
            entropy_det.append(np.nan);  continue
        i = pos[-1]
        vm = (v_flow[i-1] - xi_flow[i-1]) / (1.0 - v_flow[i-1]*xi_flow[i-1])
        gm = 1.0 / np.sqrt(1.0 - vm**2)
        Jm = vm * gm * sminus(e_flow[i-1])
        vp = -xi_flow[i-1]
        gp = 1.0 / np.sqrt(1.0 - vp**2)
        Jp = vp * gp * splus(e_flow[i])
        entropy_det.append(Jp - Jm)

    for xiw, eN, eC, v_flow, xi_flow, e_flow in deflagration_data:
        pos = np.where(xi_flow == xiw)[0]
        if len(pos) == 0:
            entropy_def.append(np.nan);  continue
        i = pos[0]
        vm = -xi_flow[i]
        gm = 1.0 / np.sqrt(1.0 - vm**2)
        Jm = vm * gm * sminus(e_flow[i])
        vp = (v_flow[i+1] - xi_flow[i+1]) / (1.0 - v_flow[i+1]*xi_flow[i+1])
        gp = 1.0 / np.sqrt(1.0 - vp**2)
        Jp = vp * gp * splus(e_flow[i+1])
        entropy_def.append(Jp - Jm)

    for xiw, eN, eC, v_flow, xi_flow, e_flow in deflagration_data:
        pos = np.where(e_flow == eN)[0]
        if len(pos) == 0:
            entropy_def_sh.append(np.nan);  continue
        i = pos[0]
        vm = (v_flow[i-1] - xi_flow[i-1]) / (1.0 - v_flow[i-1]*xi_flow[i-1])
        gm = 1.0 / np.sqrt(1.0 - vm**2)
        Jm = vm * gm * splus(e_flow[i-1])
        vp = -xi_flow[i]
        gp = 1.0 / np.sqrt(1.0 - vp**2)
        Jp = vp * gp * splus(e_flow[i])
        entropy_def_sh.append(Jp - Jm)

    for xiw, eN, eC, v_flow, xi_flow, e_flow, emwall in hybrid_data:
        pos = np.where(xi_flow == xiw)[0]
        if len(pos) == 0:
            entropy_hyb.append(np.nan);  continue
        i = pos[0]
        vm = (v_flow[i] - xi_flow[i]) / (1.0 - v_flow[i]*xi_flow[i])
        gm = 1.0 / np.sqrt(1.0 - vm**2)
        Jm = vm * gm * sminus(e_flow[i])
        vp = (v_flow[i+1] - xi_flow[i+1]) / (1.0 - v_flow[i+1]*xi_flow[i+1])
        gp = 1.0 / np.sqrt(1.0 - vp**2)
        Jp = vp * gp * splus(e_flow[i+1])
        entropy_hyb.append(Jp - Jm)

    for xiw, eN, eC, v_flow, xi_flow, e_flow, emwall in hybrid_data:
        pos = np.where(e_flow == eN)[0]
        if len(pos) == 0:
            entropy_hyb_sh.append(np.nan);  continue
        i = pos[0]
        vm = (v_flow[i-1] - xi_flow[i-1]) / (1.0 - v_flow[i-1]*xi_flow[i-1])
        gm = 1.0 / np.sqrt(1.0 - vm**2)
        Jm = vm * gm * splus(e_flow[i-1])
        vp = -xi_flow[i]
        gp = 1.0 / np.sqrt(1.0 - vp**2)
        Jp = vp * gp * splus(e_flow[i])
        entropy_hyb_sh.append(Jp - Jm)

    return (np.array(entropy_det),
            np.array(entropy_def),
            np.array(entropy_def_sh),
            np.array(entropy_hyb),
            np.array(entropy_hyb_sh))
