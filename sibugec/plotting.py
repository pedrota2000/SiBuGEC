"""
sibugec.plotting
================
Interactive phase-space bubble plot for SiBuGEC.

The main entry point is ``interactive_bubble_plot``.  It computes all
separator curves and hydrodynamic flows, then displays a live
matplotlib figure where clicking on any point shows the corresponding
velocity and energy profiles.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import concurrent.futures
import pickle
import os

from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline

from .eos import pplus as _pplus_analytic, pminus as _pminus_analytic
from .hydrodynamics import (
    system_minus, find_deto, find_def, find_hyb, _e_max_from_pressure
)
from .separators import (
    compute_def_separator,
    compute_det_separator,
    compute_hyb_separator,
    limiting_detonation_contour_finder,
    jouguet_detonation_contour_finder,
    compute_entropy_separator_det,
    compute_entropy_separator_def,
    compute_alphan,
    eN_contour_finder,
    eC_contour_finder,
    entropy_checker,
)


def interactive_bubble_plot(eTH, eTL, nucleation_energies_allowed, bubble_energies_allowed,
                          p0=0.0, p1=0.0, xiw_resolution=25, en_resolution=50,
                          contour_resolution=500, show_eos=True,
                          sminus=None, splus=None, n=4.0, delta=1.0,
                          sTL=None, sTH=None, saving=False, load_eos=None,
                          custom_pplus=None, custom_pminus=None,
                          custom_cs2_plus=None, custom_cs2_minus=None,
                          output_dir="."):
    """
    Compute and display the interactive phase-space bubble plot.

    Parameters
    ----------
    eTH, eTL : float
        HT/LT turning-point energies.
    nucleation_energies_allowed : tuple (eN_lo, eN_hi)
        Allowed nucleation energy range.
    bubble_energies_allowed : tuple (eC_lo, eC_hi)
        Allowed bubble energy range.
    p0, p1 : float, optional
        EoS pressure offsets (default 0.0).
    xiw_resolution : int
        Number of wall-velocity grid points (default 25).
    en_resolution : int
        Number of energy grid points per axis (default 50).
    show_eos : bool
        Show EoS plot before computing (default True).
    sminus, splus : callable, optional
        Entropy interpolants S−(E), S+(E) for entropy-condition checks.
    n : float
        LT branch shape exponent.
    delta : float
        HT branch smoothing scale.
    sTL, sTH : float, optional
        Entropy anchor values (used only when ``saving=True``).
    saving : bool
        Pickle the results to ``output_dir`` (default False).
    load_eos : str, optional
        Path to a previously saved ``.pkl`` file; overrides all EoS parameters.
    custom_pplus : callable(e, eHT, p1), optional
        Replacement for the built-in HT pressure function.
    custom_pminus : callable(e, eTL, p0), optional
        Replacement for the built-in LT pressure function.
    custom_cs2_plus : callable(e), optional
        Replacement for the built-in HT speed-of-sound-squared.
    custom_cs2_minus : callable(e), optional
        Replacement for the built-in LT speed-of-sound-squared.
    output_dir : str
        Directory for output files (default current directory).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes  (the (ξ_w, E_N) axis)
    """

    # ------------------------------------------------------------------
    # Optionally load EoS parameters from a saved pickle
    # ------------------------------------------------------------------
    results_dict = {
        'EoS_parameters': {
            'eTL': eTL, 'eTH': eTH, 'p0': p0, 'p1': p1,
            'n': n, 'delta': delta, 'sTL': sTL, 'sTH': sTH,
        },
        'Global_contours': {},
        'Local_contours': {},
        'Hydro_flows': {},
    }

    if load_eos is not None:
        print(f"[SiBuGEC] Loading EoS from '{load_eos}'")
        with open(load_eos, "rb") as f:
            results_dict = pickle.load(f)
        ep = results_dict['EoS_parameters']
        eTL = ep['eTL'];  eTH   = ep['eTH']
        p0  = ep['p0'];   p1    = ep['p1']
        n   = ep['n'];    delta = ep['delta']
        sTL = ep['sTL'];  sTH   = ep['sTH']

    # ------------------------------------------------------------------
    # Resolve EoS callables (analytic or custom)
    # ------------------------------------------------------------------
    if custom_pplus is not None:
        pplus  = custom_pplus
    else:
        def pplus(e, eHT, p1_arg=p1, delta=delta):
            return _pplus_analytic(e, eHT, p1_arg, delta)

    if custom_pminus is not None:
        pminus = custom_pminus
    else:
        def pminus(e, eTL_arg, p0_arg=p0, n=n):
            return _pminus_analytic(e, eTL_arg, n)

    # ------------------------------------------------------------------
    # Energy scale
    # ------------------------------------------------------------------
    if eTL > eTH:
        e_max = fsolve(lambda e: pminus(eTL, eTL, p0) - pplus(e, eTH, p1), eTL * 2)[0]
    else:
        e_max = fsolve(lambda e: pminus(eTL, eTL, p0) - pplus(e, eTH, p1), eTH * 5)[0]
    print(f"[SiBuGEC] e_max = {e_max:.4f}")

    # ------------------------------------------------------------------
    # Speed of sound
    # ------------------------------------------------------------------
    if custom_cs2_plus is not None:
        cs2_plus = custom_cs2_plus
    else:
        e_plus  = np.linspace(eTH * 0.9, e_max * 5, 5000)
        p_plus  = pplus(e_plus, eTH, p1)
        cs2_plus = CubicSpline(e_plus, np.gradient(p_plus, e_plus))

    if custom_cs2_minus is not None:
        cs2_minus = custom_cs2_minus
    else:
        e_minus  = np.linspace(0.0, eTL * 1.1, 5000)
        p_minus  = pminus(e_minus, eTL, p0)
        cs2_minus = CubicSpline(e_minus, np.gradient(p_minus, e_minus))

    # ------------------------------------------------------------------
    # Inflection-point equation
    # ------------------------------------------------------------------
    def inf_point_eq(e):
        first  = 2.0 * (cs2_minus(e) - 1.0) * cs2_minus(e)
        second = -(e + pminus(e, eTL, p0)) * cs2_minus(e, nu=1)
        return first + second

    from scipy.optimize import least_squares as _ls
    e_CSP = _ls(inf_point_eq, eTL / 2.0, bounds=(1e-6, eTL)).x[0]
    print(f"[SiBuGEC] Energy at critical sound point (CSP): {e_CSP:.4f}")

    # Save EoS profiles
    _save_txt(output_dir, "pofe_minus.txt",
              np.stack((e_minus if custom_cs2_minus is None else np.linspace(0, eTL * 1.1, 100),
                        pminus(e_minus if custom_cs2_minus is None else np.linspace(0, eTL * 1.1, 100),
                               eTL, p0)), axis=1))
    _save_txt(output_dir, "pofe_plus.txt",
              np.stack((e_plus if custom_cs2_plus is None else np.linspace(eTH * 0.9, e_max * 5, 100),
                        pplus(e_plus if custom_cs2_plus is None else np.linspace(eTH * 0.9, e_max * 5, 100),
                              eTH, p1)), axis=1))

    # ------------------------------------------------------------------
    # Optionally show EoS
    # ------------------------------------------------------------------
    if show_eos:
        _plot_eos(pplus, pminus, eTH, eTL, p0, p1, e_max)

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------
    xiw_vals = np.linspace(0.01, 0.99, xiw_resolution)
    nucleation_energies = np.linspace(eTH + 1e-3, e_max, en_resolution)
    bubble_energies     = np.linspace(1e-6, eTL - 1e-3, en_resolution)

    # ------------------------------------------------------------------
    # Figure scaffold
    # ------------------------------------------------------------------
    fig_det = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.3], hspace=0.3, wspace=0.2)
    ax_det   = fig_det.add_subplot(gs[0, 0])
    ax_C     = fig_det.add_subplot(gs[0, 1])
    ax_alpha = fig_det.add_subplot(gs[1, :])

    ax_C.plot(np.sqrt(cs2_minus(e_CSP)), e_CSP, 'xr', markersize=10, label='CSP')
    for ax, xlabel, ylabel in [
        (ax_det,   r'$\xi_w$', r'$\mathcal{E}_N$'),
        (ax_C,     r'$\xi_w$', r'$\mathcal{E}_C$'),
        (ax_alpha, r'$\xi_w$', r'$\alpha_N$'),
    ]:
        ax.set_xlabel(xlabel);  ax.set_ylabel(ylabel);  ax.grid(True)
    ax_det.set_xlim(-0.01, 1.01)
    fig_det.suptitle("Computing separators…")
    plt.show(block=False)
    fig_det.canvas.draw(); fig_det.canvas.flush_events()

    # Helper: alpha_N with current EoS parameters
    def _alphan(eN):
        if sminus is None or splus is None:
            return np.nan
        return compute_alphan(eN, pplus, pminus, eTH, eTL, p0, p1,
                               splus, sminus, n, delta)

    # ------------------------------------------------------------------
    # Separator curves (parallel)
    # ------------------------------------------------------------------
    xi_temporal = np.linspace(1e-2, 0.99, contour_resolution * 10)
    eN_temporal = np.linspace(eTH, e_max, contour_resolution * 10)

    # Deflagration separator
    fig_det.suptitle("Computing deflagration separators…"); fig_det.canvas.draw(); fig_det.canvas.flush_events()
    with concurrent.futures.ThreadPoolExecutor() as ex:
        results = list(ex.map(
            lambda xiw: compute_def_separator(
                xiw, pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus),
            xi_temporal))
    def_eC = np.array([r[0] for r in results], dtype=float)
    def_eN = np.array([r[1] for r in results], dtype=float)
    mask = ~np.isnan(def_eC) & ~np.isnan(def_eN)
    def_eC, def_eN, xi_def = def_eC[mask], def_eN[mask], xi_temporal[mask]
    ax_C.plot(xi_def, def_eC, '--k')
    ax_det.plot(xi_def, def_eN, '--k')
    ax_alpha.plot(xi_def, np.array(list(map(_alphan, def_eN))), '--k')
    results_dict['Global_contours']['deflagration'] = {
        'xi': xi_def, 'eC': def_eC, 'eN': def_eN,
        'alpha': np.array(list(map(_alphan, def_eN)))}

    # Detonation separator
    fig_det.suptitle("Computing detonation separators…"); fig_det.canvas.draw(); fig_det.canvas.flush_events()
    with concurrent.futures.ThreadPoolExecutor() as ex:
        results = list(ex.map(
            lambda eN: compute_det_separator(
                eN, pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus),
            eN_temporal))
    det_xiw = np.array([r[0] for r in results], dtype=float)
    det_eC  = np.array([r[1] for r in results], dtype=float)
    mask = ~np.isnan(det_xiw) & ~np.isnan(det_eC)
    det_xiw, det_eC, det_eN = det_xiw[mask], det_eC[mask], np.array(eN_temporal)[mask]
    ax_det.plot(det_xiw, det_eN, '--k')
    ax_C.plot(det_xiw, det_eC, '--k')
    ax_alpha.plot(det_xiw, np.array(list(map(_alphan, det_eN))), '--k')
    results_dict['Global_contours']['detonation'] = {
        'xi': det_xiw, 'eC': det_eC, 'eN': det_eN,
        'alpha': np.array(list(map(_alphan, det_eN)))}

    # Hybrid separator
    fig_det.suptitle("Computing hybrid separators…"); fig_det.canvas.draw(); fig_det.canvas.flush_events()
    with concurrent.futures.ThreadPoolExecutor() as ex:
        results = list(ex.map(
            lambda xiw: compute_hyb_separator(
                xiw, pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus, inf_point_eq),
            xi_temporal))
    hyb_eN = np.array([r[0] for r in results], dtype=float)
    hyb_eC = np.array([r[1] for r in results], dtype=float)
    mask = ~np.isnan(hyb_eN) & ~np.isnan(hyb_eC)
    hyb_eN, hyb_eC, xi_hyb = hyb_eN[mask], hyb_eC[mask], xi_temporal[mask]
    ax_C.plot(xi_hyb, hyb_eC, '--g')
    ax_det.plot(xi_hyb, hyb_eN, '--g')
    ax_alpha.plot(xi_hyb, np.array(list(map(_alphan, hyb_eN))), '--g')
    results_dict['Local_contours']['hybrid'] = {
        'xi': xi_hyb, 'eC': hyb_eC, 'eN': hyb_eN,
        'alpha': np.array(list(map(_alphan, hyb_eN)))}

    # Limiting detonation contour
    fig_det.suptitle("Computing limiting detonations…"); fig_det.canvas.draw(); fig_det.canvas.flush_events()
    xiw_v2 = np.linspace(0.3, 1 - 1e-3, int(contour_resolution // 2.5))
    with concurrent.futures.ThreadPoolExecutor() as ex:
        results = list(ex.map(
            lambda xiw: limiting_detonation_contour_finder(
                xiw, pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus, inf_point_eq),
            xiw_v2))
    ld_xish = np.array([r[0] for r in results], dtype=float)
    ld_eN   = np.array([r[1] for r in results], dtype=float)
    ld_eC   = np.array([r[2] for r in results], dtype=float)
    m1 = ~(np.isnan(ld_eN) | np.isnan(ld_xish))
    m2 = ~(np.isnan(ld_eC) | np.isnan(ld_xish))
    ax_det.plot(ld_xish[m1], ld_eN[m1], '--r', lw=2)
    ax_C.plot(ld_xish[m2],   ld_eC[m2], '--r', lw=2)
    ax_alpha.plot(ld_xish[m1], np.array(list(map(_alphan, ld_eN[m1]))), '--r', lw=2)
    results_dict['Local_contours']['detonation'] = {
        'xi': ld_xish[m1], 'eC': ld_eC[m2], 'eN': ld_eN[m1],
        'alpha': np.array(list(map(_alphan, ld_eN[m1])))}

    # Jouguet detonations
    fig_det.suptitle("Computing Jouguet detonations…"); fig_det.canvas.draw(); fig_det.canvas.flush_events()
    eN_jouguet = np.linspace(eTH * 1.01, e_max, contour_resolution)
    with concurrent.futures.ThreadPoolExecutor() as ex:
        results = list(ex.map(
            lambda eN: jouguet_detonation_contour_finder(
                eN, pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus),
            eN_jouguet))
    j_xiw = np.array([r[0] for r in results], dtype=float)
    j_eC  = np.array([r[1] for r in results], dtype=float)
    mask = ~np.isnan(j_xiw) & ~np.isnan(j_eC)
    j_xiw, j_eC, j_eN = j_xiw[mask], j_eC[mask], eN_jouguet[mask]
    max_hyb_xiw = np.max(xi_hyb) if len(xi_hyb) > 0 else 0.0
    final_mask = j_xiw >= max_hyb_xiw
    j_xiw, j_eC, j_eN = j_xiw[final_mask], j_eC[final_mask], j_eN[final_mask]
    ax_det.plot(j_xiw, j_eN, '--b', lw=2)
    ax_C.plot(j_xiw, j_eC, '--b', lw=2)
    ax_alpha.plot(j_xiw, np.array(list(map(_alphan, j_eN))), '--b', lw=2)
    results_dict['Local_contours']['jouguet'] = {
        'xi': j_xiw, 'eC': j_eC, 'eN': j_eN,
        'alpha': np.array(list(map(_alphan, j_eN)))}

    # # Entropy separator — detonations
    # fig_det.suptitle("Computing entropy detonation contour…"); fig_det.canvas.draw(); fig_det.canvas.flush_events()
    # if sminus is not None and splus is not None:
    #     xiw_entr = np.linspace(0.3, 1 - 1e-3, contour_resolution)
    #     with concurrent.futures.ThreadPoolExecutor() as ex:
    #         results = list(ex.map(
    #             lambda xiw: compute_entropy_separator_det(
    #                 xiw, pplus, pminus, eTH, eTL, p0, p1,
    #                 cs2_plus, cs2_minus, splus, sminus, e_max),
    #             xiw_entr))
    #     entr_det_eN = np.array([r[0] for r in results], dtype=float)
    #     entr_det_eC = np.array([r[1] for r in results], dtype=float)
    #     mask = ~np.isnan(entr_det_eN) & ~np.isnan(entr_det_eC)
    #     ax_det.plot(xiw_entr[mask], entr_det_eN[mask], ':b', lw=2)
    #     ax_C.plot(xiw_entr[mask],   entr_det_eC[mask], ':b', lw=2)
    #     ax_alpha.plot(xiw_entr[mask], np.array(list(map(_alphan, entr_det_eN[mask]))), ':b', lw=2)
    #
    #     # Entropy separator — deflagrations
    #     fig_det.suptitle("Computing entropy deflagration contour…"); fig_det.canvas.draw(); fig_det.canvas.flush_events()
    #     xiw_entr = np.linspace(0.01, 1 - 1e-3, contour_resolution)
    #     with concurrent.futures.ThreadPoolExecutor() as ex:
    #         results = list(ex.map(
    #             lambda xiw: compute_entropy_separator_def(
    #                 xiw, pplus, pminus, eTH, eTL, p0, p1,
    #                 cs2_plus, cs2_minus, splus, sminus, e_max),
    #             xiw_entr))
    #     entr_def_eC = np.array([r[0] for r in results], dtype=float)
    #     entr_def_eN = np.array([r[1] for r in results], dtype=float)
    #     mask = ~np.isnan(entr_def_eN) & ~np.isnan(entr_def_eC)
    #     ax_C.plot(xiw_entr[mask],   entr_def_eC[mask], ':b', lw=2)
    #     ax_det.plot(xiw_entr[mask], entr_def_eN[mask], ':b', lw=2)
    #     ax_alpha.plot(xiw_entr[mask], np.array(list(map(_alphan, entr_def_eN[mask]))), ':b', lw=2)

    fig_det.suptitle(None); fig_det.canvas.draw()

    # ------------------------------------------------------------------
    # Progress bars
    # ------------------------------------------------------------------
    bh, bs, by = 0.015, 0.003, 0.94
    ax_bar_det = fig_det.add_axes([0.15, by, 0.7, bh])
    ax_bar_def = fig_det.add_axes([0.15, by - (bh + bs), 0.7, bh])
    ax_bar_hyb = fig_det.add_axes([0.15, by - 2 * (bh + bs), 0.7, bh])
    for axb in (ax_bar_det, ax_bar_def, ax_bar_hyb):
        axb.set_xticks([]);  axb.set_yticks([])
        axb.set_xlim(0, 1);  axb.set_ylim(0, 1)
    det_rect = ax_bar_det.barh([0.5], [0], height=1.0, color='red')[0]
    def_rect = ax_bar_def.barh([0.5], [0], height=1.0, color='blue')[0]
    hyb_rect = ax_bar_hyb.barh([0.5], [0], height=1.0, color='green')[0]
    for axb, rect, label in [
        (ax_bar_det, det_rect, "Detonations"),
        (ax_bar_def, def_rect, "Deflagrations"),
        (ax_bar_hyb, hyb_rect, "Hybrids"),
    ]:
        axb.text(-0.02, 0.5, label, va='center', ha='right', fontsize=8,
                 transform=axb.transAxes)
        axb.text(1.02, 0.5, "0%", va='center', ha='left', fontsize=8,
                 transform=axb.transAxes, gid=label + '_pct')

    def _pct_text(ax, label):
        for t in ax.texts:
            if t.get_gid() == label + '_pct':
                return t
        return None

    det_pct = _pct_text(ax_bar_det, "Detonations")
    def_pct = _pct_text(ax_bar_def, "Deflagrations")
    hyb_pct = _pct_text(ax_bar_hyb, "Hybrids")

    def _update_bar(rect, pct_text, frac, ax, total, done):
        rect.set_width(frac)
        if pct_text:
            pct_text.set_text(f"{frac*100:.1f}%")
        if done % max(1, total // 25) == 0:
            ax.figure.canvas.draw_idle()
            fig_det.canvas.flush_events()

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------
    def _det_worker(args):
        xiw, e = args
        __, flow = find_deto(e, xiw, pplus, pminus, eTH, eTL, p0, p1,
                              cs2_plus, cs2_minus)
        return xiw, e, flow

    def _def_worker(args):
        xiw, eC = args
        eN, flow = find_def(eC, xiw, pplus, pminus, eTH, eTL, p0, p1,
                             cs2_plus, cs2_minus)
        return xiw, eC, eN, flow

    def _hyb_worker(args):
        xiw, em = args
        eN, flow, emwall = find_hyb(em, xiw, pplus, pminus, eTH, eTL, p0, p1,
                                     cs2_plus, cs2_minus)
        return xiw, eN, flow, emwall

    # ------------------------------------------------------------------
    # Compute flows
    # ------------------------------------------------------------------
    detonation_flows   = []
    deflagration_flows = []
    hybrid_flows       = []

    # Detonations
    XIW, E = np.meshgrid(xiw_vals, nucleation_energies, indexing='ij')
    tasks = [(XIW[i, j], E[i, j]) for i in range(XIW.shape[0]) for j in range(XIW.shape[1])]
    total = len(tasks);  done = 0
    with concurrent.futures.ThreadPoolExecutor() as ex:
        for xiw, e, flow in ex.map(_det_worker, tasks):
            done += 1
            _update_bar(det_rect, det_pct, done / total, ax_bar_det, total, done)
            if flow is not None and not np.isnan(flow).any():
                v_f, xi_f, e_f = flow
                detonation_flows.append((xiw, e, float(min(e_f)), v_f, xi_f, e_f))
    _batch_plot(ax_det, ax_C, ax_alpha, detonation_flows, 'ro', 'Detonations',
                5, _alphan)
    results_dict['Hydro_flows']['detonation'] = _pack_flows(detonation_flows, _alphan)

    # Deflagrations
    XIW, E = np.meshgrid(xiw_vals, bubble_energies, indexing='ij')
    tasks = [(XIW[i, j], E[i, j]) for i in range(XIW.shape[0]) for j in range(XIW.shape[1])]
    total = len(tasks);  done = 0
    with concurrent.futures.ThreadPoolExecutor() as ex:
        for xiw, eC, eN, flow in ex.map(_def_worker, tasks):
            done += 1
            _update_bar(def_rect, def_pct, done / total, ax_bar_def, total, done)
            if flow is None or np.isnan(eN):
                continue
            v_f, xi_f, e_f = flow
            deflagration_flows.append((xiw, eN, eC, v_f, xi_f, e_f))
    _batch_plot(ax_det, ax_C, ax_alpha, deflagration_flows, 'bo', 'Deflagration',
                4, _alphan)
    results_dict['Hydro_flows']['deflagration'] = _pack_flows(deflagration_flows, _alphan)

    # Hybrids
    XIW, E = np.meshgrid(xiw_vals, bubble_energies, indexing='ij')
    tasks = [(XIW[i, j], E[i, j]) for i in range(XIW.shape[0]) for j in range(XIW.shape[1])]
    total = len(tasks);  done = 0
    with concurrent.futures.ThreadPoolExecutor() as ex:
        for xiw, eN, flow, emwall in ex.map(_hyb_worker, tasks):
            done += 1
            _update_bar(hyb_rect, hyb_pct, done / total, ax_bar_hyb, total, done)
            if flow is None or np.isnan(eN):
                continue
            v_f, xi_f, e_f = flow
            hybrid_flows.append((xiw, eN, float(min(e_f)), v_f, xi_f, e_f, emwall))
    _batch_plot(ax_det, ax_C, ax_alpha, hybrid_flows, 'go', 'Hybrids', 3, _alphan)
    results_dict['Hydro_flows']['hybrid'] = _pack_flows(hybrid_flows, _alphan)

    # Remove progress bars
    ax_bar_det.remove();  ax_bar_def.remove();  ax_bar_hyb.remove()

    # ------------------------------------------------------------------
    # Allowed-energy contours
    # ------------------------------------------------------------------
    eN_plus, eN_minus = eN_contour_finder(
        xiw_vals, bubble_energies_allowed,
        detonation_flows, deflagration_flows, hybrid_flows,
        pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus)
    eC_plus, eC_minus = eC_contour_finder(
        xiw_vals, nucleation_energies_allowed,
        detonation_flows, deflagration_flows, hybrid_flows,
        pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus)

    ax_det.axhspan(*nucleation_energies_allowed, color='orange', alpha=0.2,
                   label=r'Allowed $\mathcal{E}_N$')
    if len(eN_plus) > 0 and len(eN_minus) > 0:
        ax_det.fill_between(eN_plus[:, 0], eN_plus[:, 1], eN_minus[:, 1],
                            color='green', alpha=0.2, label=r'Allowed $\mathcal{E}_C$')
    ax_C.axhspan(*bubble_energies_allowed, color='green', alpha=0.2,
                 label=r'Allowed $\mathcal{E}_C$')
    if len(eC_plus) > 0 and len(eC_minus) > 0:
        ax_C.fill_between(eC_plus[:, 0], eC_plus[:, 1], eC_minus[:, 1],
                          color='orange', alpha=0.2, label=r'Allowed $\mathcal{E}_N$')

    # ------------------------------------------------------------------
    # Entropy filtering
    # ------------------------------------------------------------------
    if sminus is not None and splus is not None:
        (bad_det, bad_def, bad_def_sh,
         bad_hyb, bad_hyb_sh) = entropy_checker(
            detonation_flows, deflagration_flows, hybrid_flows, splus, sminus)

        # _entropy_overlay(ax_det, ax_C, ax_alpha, detonation_flows,
        #                  bad_det,    'k*', 2.5, _alphan)
        # _entropy_overlay(ax_det, ax_C, ax_alpha, deflagration_flows,
        #                  bad_def_sh, 'yd', 2.5, _alphan)
        # _entropy_overlay(ax_det, ax_C, ax_alpha, deflagration_flows,
        #                  bad_def,    'k*', 2.5, _alphan)
        # _entropy_overlay(ax_det, ax_C, ax_alpha, hybrid_flows,
        #                  bad_hyb_sh, 'yd', 2.5, _alphan)
        # _entropy_overlay(ax_det, ax_C, ax_alpha, hybrid_flows,
        #                  bad_hyb,    'k*', 2.5, _alphan)

    # ------------------------------------------------------------------
    # Final plot decoration
    # ------------------------------------------------------------------
    for ax in (ax_det, ax_C, ax_alpha):
        ax.legend()
    ax_det.set_title(
        rf"Interactive Solutions for EoS $e_{{TL}}-e_{{TH}} = {eTL - eTH:.2f}$"
        + "\n(Click on points to see flow details)"
    )
    ax_det.set_xlim(-0.01, 1.01)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if saving:
        save_path = os.path.join(
            output_dir,
            f"sibugec_results_{eTH:.3f}_{eTL:.3f}_{p0:.3f}.pkl",
        )
        with open(save_path, 'wb') as f:
            pickle.dump(results_dict, f)
        print(f"[SiBuGEC] Results saved to '{save_path}'")

    # ------------------------------------------------------------------
    # Click handler
    # ------------------------------------------------------------------
    fig_det.canvas.mpl_connect(
        'button_press_event',
        lambda ev: _on_click(ev, ax_det, ax_C, ax_alpha,
                             detonation_flows, deflagration_flows, hybrid_flows,
                             output_dir, _alphan),
    )

    plt.show()
    return fig_det, ax_det


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _save_txt(output_dir, filename, data):
    path = os.path.join(output_dir, filename)
    try:
        np.savetxt(path, data)
    except OSError:
        pass   # non-fatal if directory is read-only


def _plot_eos(pplus, pminus, eTH, eTL, p0, p1, e_max):
    e_vals = np.linspace(0, e_max, 500)
    fig_eos, ax_eos = plt.subplots(figsize=(8, 5))
    ax_eos.plot(e_vals, pplus(e_vals, eTH, p1), label='p+', color='tab:blue', lw=2)
    ax_eos.plot(e_vals, pminus(e_vals, eTL, p0), label='p−', color='tab:orange', lw=2)
    ax_eos.set_xlabel(r'$\mathcal{E}$')
    ax_eos.set_ylabel(r'$\mathcal{P}(\mathcal{E})$')
    ax_eos.set_title('Equation of State')
    ax_eos.legend();  ax_eos.grid(True)
    fig_eos.tight_layout();  fig_eos.show()


def _batch_plot(ax_det, ax_C, ax_alpha, flows, marker, label, ms, alphan_func):
    xiws = [r[0] for r in flows]
    es   = [r[1] for r in flows]
    eCs  = [r[2] for r in flows]
    ax_det.plot(xiws, es,   marker, label=label, ms=ms)
    ax_C.plot(xiws,   eCs,  marker, label=label, ms=ms)
    ax_alpha.plot(xiws, np.array(list(map(alphan_func, es))), marker, label=label, ms=ms)


def _pack_flows(flows, alphan_func):
    xiws = np.array([r[0] for r in flows])
    es   = np.array([r[1] for r in flows])
    eCs  = np.array([r[2] for r in flows])
    return {
        'xi': xiws, 'eC': eCs, 'eN': es,
        'alpha': np.array(list(map(alphan_func, es))),
        'flows': flows,
    }


def _entropy_overlay(ax_det, ax_C, ax_alpha, flows, bad_arr, marker, ms, alphan_func):
    if len(bad_arr) == 0:
        return
    xiws = np.array([r[0] for r in flows])
    es   = np.array([r[1] for r in flows])
    eCs  = np.array([r[2] for r in flows])
    sel  = bad_arr < 0
    if not np.any(sel):
        return
    ax_det.plot(xiws[sel], es[sel],   marker, ms=ms)
    ax_C.plot(xiws[sel],   eCs[sel],  marker, ms=ms)
    ax_alpha.plot(xiws[sel], np.array(list(map(alphan_func, es[sel]))), marker, ms=ms)


def _on_click(event, ax_det, ax_C, ax_alpha,
              detonation_data, deflagration_data, hybrid_data,
              output_dir, alphan_func):
    if event.inaxes not in (ax_det, ax_C, ax_alpha):
        return
    xiw_click = event.xdata
    ey_click  = event.ydata
    if xiw_click is None or ey_click is None:
        return

    is_eN_plot    = (event.inaxes == ax_det)
    is_eC_plot    = (event.inaxes == ax_C)
    is_alpha_plot = (event.inaxes == ax_alpha)
    y_name = 'E_N' if is_eN_plot else ('E_C' if is_eC_plot else 'Alpha')
    print(f"[SiBuGEC] Click at ξ_w={xiw_click:.3f}, {y_name}={ey_click:.3f}")

    best_dist = np.inf
    best_sol  = None
    best_type = None

    for row in detonation_data:
        xiw, eN, eC = row[0], row[1], row[2]
        if is_eN_plot:
            ey = eN
        elif is_eC_plot:
            ey = eC
        else:
            ey = alphan_func(eN)
        d = np.hypot(xiw - xiw_click, ey - ey_click)
        if d < best_dist:
            best_dist = d;  best_sol = row + (np.nan,);  best_type = 'Detonation'

    for row in deflagration_data:
        xiw, eN, eC = row[0], row[1], row[2]
        if is_eN_plot:
            ey = eN
        elif is_eC_plot:
            ey = eC
        else:
            ey = alphan_func(eN)
        d = np.hypot(xiw - xiw_click, ey - ey_click)
        if d < best_dist:
            best_dist = d;  best_sol = row + (np.nan,);  best_type = 'Deflagration'

    for row in hybrid_data:
        xiw, eN, eC = row[0], row[1], row[2]
        if is_eN_plot:
            ey = eN
        elif is_eC_plot:
            ey = eC
        else:
            ey = alphan_func(eN)
        d = np.hypot(xiw - xiw_click, ey - ey_click)
        if d < best_dist:
            best_dist = d;  best_sol = row;  best_type = 'Hybrid'

    if best_sol is None:
        return

    xiw_s, eN_s, eC_s, v_f, xi_f, e_f = best_sol[:6]
    print(f"[SiBuGEC] → {best_type}: ξ_w={xiw_s:.3f}, E_N={eN_s:.3f}, E_C={eC_s:.3f}")

    y_label = 'E_N' if is_eN_plot else ('E_C' if is_eC_plot else 'Alpha')
    y_val   = eN_s  if is_eN_plot else (eC_s if is_eC_plot else alphan_func(eN_s))

    fig_f, (ax_v, ax_e) = plt.subplots(1, 2, figsize=(12, 5))
    ax_v.plot(xi_f, v_f, 'b-', lw=2, label=best_type)
    ax_v.set_xlabel(r'$\xi$');  ax_v.set_ylabel(r'$v(\xi)$')
    ax_v.set_title(f"Velocity — {best_type}\nξ_w={xiw_s:.3f}, {y_label}={y_val:.3f}")
    ax_v.grid(True);  ax_v.legend()

    ax_e.plot(xi_f, e_f, 'r-', lw=2, label=best_type)
    ax_e.set_xlabel(r'$\xi$');  ax_e.set_ylabel(r'$e(\xi)$')
    ax_e.set_title(f"Energy — {best_type}\nξ_w={xiw_s:.3f}, {y_label}={y_val:.3f}")
    ax_e.grid(True);  ax_e.legend()

    plt.tight_layout();  plt.show()

    _save_txt(output_dir, "vofxi.txt", np.stack((xi_f, v_f), axis=1))
    _save_txt(output_dir, "eofxi.txt", np.stack((xi_f, e_f), axis=1))
