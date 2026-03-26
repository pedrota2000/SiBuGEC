"""
main_eos_explorer.py
====================
SiBuGEC — EoS Explorer (inverse-problem GUI)

Interactive GUI for exploring the equation of state and thermodynamic
quantities of the two-phase model.  Sliders control the EoS parameters
in real time; a button triggers the full bubble-phase-space computation.

Usage
-----
    python main_eos_explorer.py

Optional: set ``CUSTOM_EOS_FILE`` below to load tabulated EoS data from
a plain text file (two columns: energy density and pressure).

EoS parameters
--------------
eTH   : HT turning-point energy
eTL   : LT turning-point energy
p1    : Pressure at eHT  (p+(eHT) = p1)
sTL   : Entropy anchor for the LT branch
sTH   : Entropy anchor for the HT branch
delta : HT branch smoothing scale
n     : LT branch shape exponent
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.interpolate import interp1d, PchipInterpolator

from sibugec.eos import pplus, pminus, load_custom_eos
from sibugec.thermodynamics import (
    entropy_ODE_plus,
    entropy_ODE_minus,
    calculate_temperature,
    find_critical_temperature,
)
from sibugec.plotting import interactive_bubble_plot

# ---------------------------------------------------------------------------
# Optional: path to a custom EoS file.
# Set to None to use the built-in analytic parametrisation.
# The file must have at least two columns: energy density and pressure.
# An optional third column may contain c_s²(e) directly.
# ---------------------------------------------------------------------------
CUSTOM_EOS_FILE = None  # e.g. "my_eos.dat"

# ---------------------------------------------------------------------------
# Default initial parameter values
# ---------------------------------------------------------------------------
INITIAL_ETH   = 0.5
INITIAL_ETL   = 2.0
INITIAL_P1    = 0.0
INITIAL_STL   = 8.45
INITIAL_STH   = 2.5
INITIAL_DELTA = 1.0
INITIAL_N     = 4.0

# ---------------------------------------------------------------------------
# Custom EoS loading (optional)
# ---------------------------------------------------------------------------
custom_pplus_func  = None
custom_pminus_func = None
if CUSTOM_EOS_FILE is not None:
    try:
        _p_func, _cs2_func, _e_lo, _e_hi = load_custom_eos(CUSTOM_EOS_FILE)
        # Split at the midpoint of the energy range as a simple heuristic.
        # Users may adjust this logic for their specific data layout.
        _e_mid = (_e_lo + _e_hi) / 2.0
        print(
            f"[EoS Explorer] Custom EoS loaded. "
            f"Using e ≤ {_e_mid:.4g} as LT branch and e > {_e_mid:.4g} as HT branch."
        )
        # Wrap the single tabulated EoS into the expected two-argument signature
        custom_pplus_func  = lambda e, eHT, p1_arg, **kw: _p_func(np.asarray(e))
        custom_pminus_func = lambda e, eTL_arg, p0_arg, **kw: _p_func(np.asarray(e))
    except Exception as exc:
        print(f"[EoS Explorer] Warning: could not load custom EoS — {exc}")
        CUSTOM_EOS_FILE = None


# ---------------------------------------------------------------------------
# Resolution (increase for publication-quality results)
# ---------------------------------------------------------------------------
XIW_RESOLUTION     = 25
EN_RESOLUTION      = 25
CONTOUR_RESOLUTION = 250

# ---------------------------------------------------------------------------
# Global entropy-interpolant holders (updated by update_plots)
# ---------------------------------------------------------------------------
s_minus_interp = None
s_plus_interp  = None


# ---------------------------------------------------------------------------
# Helper: find critical temperature from current slider values
# ---------------------------------------------------------------------------
def _get_critical_temperature(eTH, eTL, p1, sTL, sTH, delta, n):
    """Integrate entropy ODEs and return the critical temperature."""
    try:
        # LT branch
        y0 = [eTL]
        t_bw = np.flip(np.logspace(np.log10(1e-7), np.log10(sTL), 500))
        sol_bw = solve_ivp(entropy_ODE_minus, (t_bw[0], t_bw[-1]), y0,
                           t_eval=t_bw, args=(eTL, n),
                           method='RK45', rtol=1e-6, atol=1e-9)
        t_fw = np.flip(np.logspace(np.log10(10), np.log10(sTL), 500))
        sol_fw = solve_ivp(entropy_ODE_minus, (t_fw[0], t_fw[-1]), y0,
                           t_eval=t_fw, args=(eTL, n),
                           method='RK45', rtol=1e-6, atol=1e-9)
        e_minus = np.concatenate((np.flip(sol_fw.y[0][:-2]), sol_bw.y[0]))
        s_minus = np.concatenate((np.flip(sol_fw.t[:-2]),    sol_bw.t))

        # HT branch
        y0 = [eTH]
        t_bw = np.flip(np.logspace(np.log10(1e-7), np.log10(sTH), 500))
        sol_bw = solve_ivp(entropy_ODE_plus, (t_bw[0], t_bw[-1]), y0,
                           t_eval=t_bw, args=(eTH, p1, delta),
                           method='RK45', rtol=1e-6, atol=1e-9)
        t_fw = np.linspace(sTH, max(20.0, sTH + 15.0), 500)
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
            return None

        e_m, T_m = calculate_temperature(e_minus, s_minus)
        e_p, T_p = calculate_temperature(e_plus,  s_plus)

        if len(e_m) == 0 or len(e_p) == 0:
            return None

        f_minus = -np.array([pminus(e, eTL, n) for e in e_m])
        f_plus  = -np.array([pplus(e, eTH, p1, delta) for e in e_p])
        return find_critical_temperature(T_m, f_minus, T_p, f_plus)

    except Exception as exc:
        print(f"[EoS Explorer] Error computing T_c: {exc}")
        return None


def _find_energy_at_tc(eTH, eTL, p1, sTL, sTH, t_c, delta, n, branch):
    """Find the energy on ``branch`` ('upper'|'lower') at temperature t_c."""
    try:
        if branch == 'upper':
            y0 = [eTH]
            t_bw = np.flip(np.logspace(np.log10(1e-7), np.log10(sTH), 500))
            sol_bw = solve_ivp(entropy_ODE_plus, (t_bw[0], t_bw[-1]), y0,
                               t_eval=t_bw, args=(eTH, p1, delta),
                               method='RK45', rtol=1e-6, atol=1e-9)
            t_fw = np.linspace(sTH, max(20.0, sTH + 15.0), 500)
            sol_fw = solve_ivp(entropy_ODE_plus, (t_fw[0], t_fw[-1]), y0,
                               t_eval=t_fw, args=(eTH, p1, delta),
                               method='RK45', rtol=1e-6, atol=1e-9)
            if sol_bw.success and sol_fw.success:
                e_arr = np.concatenate((sol_fw.y[0][1:], sol_bw.y[0]))
                s_arr = np.concatenate((sol_fw.t[1:],    sol_bw.t))
            elif sol_bw.success:
                e_arr, s_arr = sol_bw.y[0], sol_bw.t
            else:
                return None
        else:
            y0 = [eTL]
            t_bw = np.flip(np.logspace(np.log10(1e-7), np.log10(sTL), 500))
            sol_bw = solve_ivp(entropy_ODE_minus, (t_bw[0], t_bw[-1]), y0,
                               t_eval=t_bw, args=(eTL, n),
                               method='RK45', rtol=1e-6, atol=1e-9)
            t_fw = np.flip(np.logspace(np.log10(10), np.log10(sTL), 500))
            sol_fw = solve_ivp(entropy_ODE_minus, (t_fw[0], t_fw[-1]), y0,
                               t_eval=t_fw, args=(eTL, n),
                               method='RK45', rtol=1e-6, atol=1e-9)
            e_arr = np.concatenate((np.flip(sol_fw.y[0][:-2]), sol_bw.y[0]))
            s_arr = np.concatenate((np.flip(sol_fw.t[:-2]),    sol_bw.t))

        e_vals, T_vals = calculate_temperature(e_arr, s_arr)
        if len(e_vals) < 2:
            return None

        e_of_T = interp1d(T_vals, e_vals, kind='linear',
                          bounds_error=False, fill_value='extrapolate')
        energy = float(e_of_T(t_c))
        if not np.isfinite(energy):
            return None
        if branch == 'upper' and energy > eTH:
            return energy
        if branch == 'lower' and energy < eTL:
            return energy
        return None

    except Exception as exc:
        print(f"[EoS Explorer] Error in _find_energy_at_tc ({branch}): {exc}")
        return None


# ---------------------------------------------------------------------------
# Button callback: launch full bubble computation
# ---------------------------------------------------------------------------
def _compute_all_bubbles(event):
    eTH   = slider_eTH.val
    eTL   = slider_eTL.val
    p1    = slider_p1.val
    sTL   = slider_sTL.val
    sTH   = slider_sTH.val
    delta = slider_delta.val
    n     = slider_n.val

    print("\n[EoS Explorer] Computing bubble phase space…")
    print(f"  eTH={eTH:.3f}  eTL={eTL:.3f}  p1={p1:.3f}  delta={delta:.3f}  n={n:.3f}")

    t_c = _get_critical_temperature(eTH, eTL, p1, sTL, sTH, delta, n)

    if t_c is not None:
        e_at_tc_up = _find_energy_at_tc(eTH, eTL, p1, sTL, sTH, t_c, delta, n, 'upper')
        e_at_tc_lo = _find_energy_at_tc(eTH, eTL, p1, sTL, sTH, t_c, delta, n, 'lower')
        nuc_energies = (eTH, e_at_tc_up) if e_at_tc_up is not None and e_at_tc_up > eTH \
                       else (eTH, eTH + 2.0)
        bub_energies = (1e-6, e_at_tc_lo) if e_at_tc_lo is not None and e_at_tc_lo < eTL \
                       else (1e-6, eTL)
        print(f"  T_c = {t_c:.3f}")
    else:
        nuc_energies = (eTH, 3.0)
        bub_energies = (1e-6, eTL)
        print("  No T_c found — using default energy ranges")

    interactive_bubble_plot(
        eTH=eTH, eTL=eTL,
        p1=p1, p0=0.0,
        n=n, delta=delta,
        nucleation_energies_allowed=nuc_energies,
        bubble_energies_allowed=bub_energies,
        xiw_resolution=XIW_RESOLUTION,
        en_resolution=EN_RESOLUTION,
        contour_resolution=CONTOUR_RESOLUTION,
        show_eos=False,
        sminus=s_minus_interp,
        splus=s_plus_interp,
        sTL=sTL, sTH=sTH,
        saving=False,
        custom_pplus=custom_pplus_func,
        custom_pminus=custom_pminus_func,
    )


# ---------------------------------------------------------------------------
# Slider callback: update thermodynamic plots
# ---------------------------------------------------------------------------
def _update_plots(val):
    global s_minus_interp, s_plus_interp

    eTH   = slider_eTH.val
    eTL   = slider_eTL.val
    p1    = slider_p1.val
    sTL   = slider_sTL.val
    sTH   = slider_sTH.val
    delta = slider_delta.val
    n     = slider_n.val

    try:
        e_max = fsolve(lambda e: pminus(eTL, eTL, n) - pplus(e, eTH, p1, delta), eTH * 10)[0]
        e_max = max(e_max, eTH + 0.2)

        # ---- EoS plot ----
        e_vals   = np.linspace(0, e_max, 1000)
        p_p = pplus(e_vals, eTH, p1, delta)
        p_m = pminus(e_vals, eTL, n)
        line_plus_eos.set_data(e_vals, p_p)
        line_minus_eos.set_data(e_vals, p_m)
        ax1.set_xlim(0, e_max * 1.1)
        y_lo = np.nanmin([np.nanmin(p_p), np.nanmin(p_m)])
        y_hi = np.nanmax([np.nanmax(p_p), np.nanmax(p_m)])
        if np.isfinite(y_lo) and np.isfinite(y_hi):
            dy = y_hi - y_lo or 1.0
            ax1.set_ylim(y_lo - 0.1 * dy, y_hi + 0.1 * dy)

        # ---- Entropy ODE integration ----
        y0 = [eTL]
        t_bw_m = np.flip(np.logspace(np.log10(1e-7), np.log10(sTL), 1000))
        sol_bw_m = solve_ivp(entropy_ODE_minus, (t_bw_m[0], t_bw_m[-1]), y0,
                             t_eval=t_bw_m, args=(eTL, n),
                             method='RK45', rtol=1e-6, atol=1e-9, dense_output=True)
        t_fw_m = np.flip(np.logspace(np.log10(10), np.log10(sTL), 1000))
        sol_fw_m = solve_ivp(entropy_ODE_minus, (t_fw_m[0], t_fw_m[-1]), y0,
                             t_eval=t_fw_m, args=(eTL, n),
                             method='RK45', rtol=1e-6, atol=1e-9, dense_output=True)
        e_m = np.concatenate((np.flip(sol_fw_m.y[0][:-2]), sol_bw_m.y[0]))
        s_m = np.concatenate((np.flip(sol_fw_m.t[:-2]),    sol_bw_m.t))

        y0 = [eTH]
        t_bw_p = np.flip(np.logspace(np.log10(1e-7), np.log10(sTH), 1000))
        sol_bw_p = solve_ivp(entropy_ODE_plus, (t_bw_p[0], t_bw_p[-1]), y0,
                             t_eval=t_bw_p, args=(eTH, p1, delta),
                             method='RK45', rtol=1e-6, atol=1e-9, dense_output=True)
        t_fw_p = np.linspace(sTH, max(20.0, sTH + 15.0), 1000)
        sol_fw_p = solve_ivp(entropy_ODE_plus, (t_fw_p[0], t_fw_p[-1]), y0,
                             t_eval=t_fw_p, args=(eTH, p1, delta),
                             method='RK45', rtol=1e-6, atol=1e-9, dense_output=True)

        if sol_bw_p.success and sol_fw_p.success:
            e_p = np.concatenate((sol_fw_p.y[0][1:], sol_bw_p.y[0]))
            s_p = np.concatenate((sol_fw_p.t[1:],    sol_bw_p.t))
        elif sol_bw_p.success:
            e_p, s_p = sol_bw_p.y[0], sol_bw_p.t
        elif sol_fw_p.success:
            e_p, s_p = sol_fw_p.y[0], sol_fw_p.t
        else:
            e_p, s_p = np.array([]), np.array([])

        # Update entropy plot
        line_minus_entropy.set_data(e_m, s_m)
        line_plus_entropy.set_data(e_p, s_p)

        # Build interpolants
        e_u, idx = np.unique(e_m, return_index=True)
        s_minus_interp = PchipInterpolator(e_u, s_m[idx])
        if len(e_p) > 0:
            e_u2, idx2 = np.unique(e_p, return_index=True)
            s_plus_interp = PchipInterpolator(e_u2, s_p[idx2])
        else:
            s_plus_interp = None

        # ---- Temperature, free energy, S/T³, S vs T ----
        t_c = None
        for (e_arr, s_arr, p_func, e_T, p_par,
             line_eT, line_fT, line_st3, line_sT) in [
            (e_m, s_m, lambda e: pminus(e, eTL, n), eTL, None,
             line_minus_e_vs_t, line_minus_f_vs_t, line_minus_s_t3, line_minus_s_vs_t),
            (e_p, s_p, lambda e: pplus(e, eTH, p1, delta), eTH, p1,
             line_plus_e_vs_t, line_plus_f_vs_t, line_plus_s_t3, line_plus_s_vs_t),
        ]:
            if len(e_arr) < 3:
                for ln in (line_eT, line_fT, line_st3, line_sT):
                    ln.set_data([], [])
                continue

            e_v, T_v = calculate_temperature(e_arr, s_arr)
            if len(e_v) == 0:
                for ln in (line_eT, line_fT, line_st3, line_sT):
                    ln.set_data([], [])
                continue

            line_eT.set_data(T_v, e_v)
            f_v = -np.array([p_func(e) for e in e_v])
            line_fT.set_data(T_v, f_v)

            try:
                sort_idx = np.argsort(s_arr)
                s_sorted = s_arr[sort_idx]
                e_sorted = e_arr[sort_idx]
                if s_sorted[-1] > s_sorted[0]:
                    e_of_s = interp1d(s_sorted, e_sorted, kind='linear',
                                      bounds_error=False, fill_value='extrapolate')
                    s_dense = np.linspace(s_sorted[0], s_sorted[-1], 2000)
                    e_dense = e_of_s(s_dense)
                    T_dense = np.gradient(e_dense, s_dense)
                    ok = (T_dense > 1e-3) & np.isfinite(T_dense)
                    if np.any(ok):
                        line_st3.set_data(T_dense[ok], s_dense[ok] / T_dense[ok]**3)
                        line_sT.set_data(T_dense[ok], s_dense[ok])
                    else:
                        line_st3.set_data([], [])
                        line_sT.set_data([], [])
                else:
                    line_st3.set_data([], [])
                    line_sT.set_data([], [])
            except Exception as exc:
                print(f"[EoS Explorer] Warning S/T³: {exc}")
                line_st3.set_data([], []);  line_sT.set_data([], [])

        # ---- Critical temperature ----
        T_m_vals = line_minus_e_vs_t.get_xdata()
        f_m_vals = line_minus_f_vs_t.get_ydata()
        T_p_vals = line_plus_e_vs_t.get_xdata()
        f_p_vals = line_plus_f_vs_t.get_ydata()
        if all(len(a) > 0 for a in [T_m_vals, f_m_vals, T_p_vals, f_p_vals]):
            t_c = find_critical_temperature(
                np.array(T_m_vals), np.array(f_m_vals),
                np.array(T_p_vals), np.array(f_p_vals))

        # Remove old T_c lines
        for ax in [ax3, ax4, ax5, ax6]:
            for ln in [l for l in ax.lines if getattr(l, '_is_tc_line', False)]:
                ln.remove()

        if t_c is not None and np.isfinite(t_c):
            for ax in [ax3, ax4, ax5, ax6]:
                ln = ax.axvline(t_c, color='red', linestyle='--', alpha=0.7,
                                label=f'T_c = {t_c:.3f}')
                ln._is_tc_line = True
            ax3.legend()

        # ---- Auto-scale axes ----
        _autoscale(ax2, [line_minus_entropy, line_plus_entropy], xy='both')
        _autoscale(ax3, [line_minus_e_vs_t, line_plus_e_vs_t], xy='both')
        _autoscale(ax4, [line_minus_f_vs_t, line_plus_f_vs_t], xy='both')
        _autoscale(ax5, [line_minus_s_t3,   line_plus_s_t3],   xy='both')
        _autoscale(ax6, [line_minus_s_vs_t,  line_plus_s_vs_t], xy='both')

    except Exception as exc:
        print(f"[EoS Explorer] Error in update: {exc}")
        for ln in [line_plus_eos, line_minus_eos,
                   line_minus_entropy, line_plus_entropy,
                   line_minus_e_vs_t, line_plus_e_vs_t,
                   line_minus_f_vs_t, line_plus_f_vs_t,
                   line_minus_s_t3,  line_plus_s_t3,
                   line_minus_s_vs_t, line_plus_s_vs_t]:
            ln.set_data([], [])

    fig.canvas.draw_idle()


def _autoscale(ax, lines, xy='both'):
    all_x, all_y = [], []
    for ln in lines:
        xd, yd = ln.get_data()
        if len(xd) > 0:
            all_x.extend(np.asarray(xd)[np.isfinite(xd)])
            all_y.extend(np.asarray(yd)[np.isfinite(yd)])
    if all_x and (xy in ('both', 'x')):
        lo, hi = min(all_x), max(all_x)
        r = hi - lo or 1.0
        ax.set_xlim(lo - 0.1 * r, hi + 0.1 * r)
    if all_y and (xy in ('both', 'y')):
        lo, hi = min(all_y), max(all_y)
        r = hi - lo or 1.0
        ax.set_ylim(lo - 0.1 * r, hi + 0.1 * r)


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 21))
gs_main = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
ax1 = fig.add_subplot(gs_main[0, 0])  # EoS
ax2 = fig.add_subplot(gs_main[0, 1])  # S vs E
ax3 = fig.add_subplot(gs_main[1, 0])  # E vs T
ax4 = fig.add_subplot(gs_main[1, 1])  # F vs T
ax5 = fig.add_subplot(gs_main[2, 0])  # S/T³ vs T
ax6 = fig.add_subplot(gs_main[2, 1])  # S vs T

plt.subplots_adjust(bottom=0.18)

# Sliders
_sh, _ss, _bm = 0.02, 0.02, 0.015
ax_eTH   = plt.axes([0.10, _bm + 6 * _ss, 0.35, _sh])
ax_eTL   = plt.axes([0.10, _bm + 5 * _ss, 0.35, _sh])
ax_p1    = plt.axes([0.55, _bm + 6 * _ss, 0.35, _sh])
ax_sTL   = plt.axes([0.55, _bm + 5 * _ss, 0.35, _sh])
ax_sTH   = plt.axes([0.10, _bm + 4 * _ss, 0.35, _sh])
ax_delta = plt.axes([0.55, _bm + 4 * _ss, 0.35, _sh])
ax_n     = plt.axes([0.10, _bm + 3 * _ss, 0.35, _sh])
ax_btn   = plt.axes([0.55, _bm + 3 * _ss, 0.35, _sh])

slider_eTH   = Slider(ax_eTH,   'eTH',   0.01, 2.0,  valinit=INITIAL_ETH,   valfmt='%.2f')
slider_eTL   = Slider(ax_eTL,   'eTL',   0.1,  20.0, valinit=INITIAL_ETL,   valfmt='%.2f')
slider_p1    = Slider(ax_p1,    'p1',   -5.0,  0.5,  valinit=INITIAL_P1,    valfmt='%.2f')
slider_sTL   = Slider(ax_sTL,   'sTL',   0.1,  20.0, valinit=INITIAL_STL,   valfmt='%.2f')
slider_sTH   = Slider(ax_sTH,   'sTH',   0.01, 20.0, valinit=INITIAL_STH,   valfmt='%.2f')
slider_delta = Slider(ax_delta, 'delta', 0.1,  50.0, valinit=INITIAL_DELTA, valfmt='%.2f')
slider_n     = Slider(ax_n,     'n',     0.1,  10.0, valinit=INITIAL_N,     valfmt='%.2f')

button_compute = Button(ax_btn, 'Compute all bubbles')
button_compute.on_clicked(_compute_all_bubbles)

# Plot lines
line_plus_eos,   = ax1.plot([], [], label='p+', color='tab:blue',   lw=2)
line_minus_eos,  = ax1.plot([], [], label='p−', color='tab:orange', lw=2)
line_minus_entropy, = ax2.plot([], [], label='LT branch', color='tab:orange', lw=2)
line_plus_entropy,  = ax2.plot([], [], label='HT branch', color='tab:blue',   lw=2)
line_minus_e_vs_t,  = ax3.plot([], [], label='LT branch', color='tab:orange', lw=2)
line_plus_e_vs_t,   = ax3.plot([], [], label='HT branch', color='tab:blue',   lw=2)
line_minus_f_vs_t,  = ax4.plot([], [], label='LT branch', color='tab:orange', lw=2)
line_plus_f_vs_t,   = ax4.plot([], [], label='HT branch', color='tab:blue',   lw=2)
line_minus_s_t3,    = ax5.plot([], [], label='LT branch', color='tab:orange', lw=2)
line_plus_s_t3,     = ax5.plot([], [], label='HT branch', color='tab:blue',   lw=2)
line_minus_s_vs_t,  = ax6.plot([], [], label='LT branch', color='tab:orange', lw=2)
line_plus_s_vs_t,   = ax6.plot([], [], label='HT branch', color='tab:blue',   lw=2)

# Axis labels
for ax, xl, yl in [
    (ax1, r'$\mathcal{E}$',   r'$\mathcal{P}(\mathcal{E})$'),
    (ax2, r'$\mathcal{E}$',   r'$S$'),
    (ax3, r'$T$',             r'$\mathcal{E}(T)$'),
    (ax4, r'$T$',             r'$\mathcal{F}(T)$'),
    (ax5, r'$T$',             r'$S/T^3$'),
    (ax6, r'$T$',             r'$S$'),
]:
    ax.set_xlabel(xl);  ax.set_ylabel(yl)
    ax.legend();        ax.grid(True)

# Connect sliders
for sl in (slider_eTH, slider_eTL, slider_p1, slider_sTL,
           slider_sTH, slider_delta, slider_n):
    sl.on_changed(_update_plots)

# Initial render
_update_plots(None)

plt.show()
