"""
main_bubble_solver.py
=====================
SiBuGEC — Standalone Bubble Phase-Space Solver

Directly launches the interactive bubble phase-space computation without
the EoS-explorer GUI.  Useful when EoS parameters are already known.

Usage
-----
    python main_bubble_solver.py

Configuration
-------------
Edit the parameter block below before running.  All parameters are
documented in ``sibugec.plotting.interactive_bubble_plot``.

Custom EoS
----------
Set ``CUSTOM_EOS_FILE`` to the path of a plain-text file with at least
two columns (energy density, pressure).  If set, the analytic p±
parametrisations are bypassed; the same tabulated curve is used for
both branches, split at ``CUSTOM_EOS_SPLIT_ENERGY``.  Adjust as needed.
"""

import numpy as np
from sibugec.plotting import interactive_bubble_plot
from sibugec.eos import load_custom_eos

# ---------------------------------------------------------------------------
# EoS parameters — analytic parametrisation
# ---------------------------------------------------------------------------
ETH   = 0.5    # HT turning-point energy
ETL   = 2.0    # LT turning-point energy
P1    = 0.0    # Pressure at eHT  (p+(eHT) = p1)
P0    = 0.0    # Pressure offset for LT branch (usually 0)
N     = 4.0    # LT branch shape exponent
DELTA = 1.0    # HT branch smoothing scale
STL   = 8.45   # LT entropy anchor
STH   = 2.5    # HT entropy anchor

# ---------------------------------------------------------------------------
# Energy ranges for phase-space sampling
# ---------------------------------------------------------------------------
NUCLEATION_ENERGY_RANGE = (ETH, ETH + 1.5)   # (E_N_min, E_N_max)
BUBBLE_ENERGY_RANGE     = (1e-6, ETL)         # (E_C_min, E_C_max)

# ---------------------------------------------------------------------------
# Resolution (increase for publication-quality results)
# ---------------------------------------------------------------------------
XIW_RESOLUTION     = 50   # Wall-velocity grid points
EN_RESOLUTION      = 50   # Energy grid points per axis
CONTOUR_RESOLUTION = 500  # Resolution for separator/contour lines

# ---------------------------------------------------------------------------
# Optional: path to a custom EoS file (set to None to use analytic model)
# ---------------------------------------------------------------------------
CUSTOM_EOS_FILE         = None   # e.g.  "my_eos.dat"
CUSTOM_EOS_SPLIT_ENERGY = None   # float; if None, use midpoint of energy range

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
SAVE_RESULTS = False   # Set True to pickle the results
OUTPUT_DIR   = "."     # Directory for output files

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    custom_pplus  = None
    custom_pminus = None
    custom_cs2_p  = None
    custom_cs2_m  = None

    if CUSTOM_EOS_FILE is not None:
        p_func, cs2_func, e_lo, e_hi = load_custom_eos(CUSTOM_EOS_FILE)

        split = CUSTOM_EOS_SPLIT_ENERGY if CUSTOM_EOS_SPLIT_ENERGY is not None \
                else (e_lo + e_hi) / 2.0

        print(f"[SiBuGEC] Custom EoS split at e = {split:.4g}")
        print(f"         e < {split:.4g} → LT branch,  e ≥ {split:.4g} → HT branch")

        # Wrap tabulated EoS to match the (e, eT, p_offset) call signature
        custom_pplus  = lambda e, eHT, p1_arg, **kw: p_func(np.asarray(e))
        custom_pminus = lambda e, eTL_arg, p0_arg, **kw: p_func(np.asarray(e))
        custom_cs2_p  = cs2_func
        custom_cs2_m  = cs2_func

    fig, ax = interactive_bubble_plot(
        eTH=ETH,
        eTL=ETL,
        p1=P1,
        p0=P0,
        n=N,
        delta=DELTA,
        sTL=STL,
        sTH=STH,
        nucleation_energies_allowed=NUCLEATION_ENERGY_RANGE,
        bubble_energies_allowed=BUBBLE_ENERGY_RANGE,
        xiw_resolution=XIW_RESOLUTION,
        en_resolution=EN_RESOLUTION,
        contour_resolution=CONTOUR_RESOLUTION,
        show_eos=True,
        saving=SAVE_RESULTS,
        output_dir=OUTPUT_DIR,
        custom_pplus=custom_pplus,
        custom_pminus=custom_pminus,
        custom_cs2_plus=custom_cs2_p,
        custom_cs2_minus=custom_cs2_m,
    )
