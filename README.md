# SiBuGEC
## Self-similar Bubble Expansion with General Equation of State Code

SiBuGEC computes self-similar hydrodynamic bubble solutions
(detonations, deflagrations, and hybrids) for cosmological first-order
phase transitions, for a general equation of state.

---

## Features

- Interactive EoS explorer with real-time thermodynamic plots (GUI).
- Full phase-space computation: detonations, deflagrations, and hybrids.
- Automatic separator curves: Chapman–Jouguet, Jouguet, hybrid onset,
  limiting detonation, and entropy-production boundaries.
- Click-to-inspect: click any point in the phase-space plot to display
  the corresponding velocity and energy profiles.
- Built-in α_N (phase-transition strength) computation.
- **Custom EoS support**: load tabulated (e, p) data from a plain-text
  file and replace the analytic parametrisations entirely. ** WORK IN PROGRESS **
- Parallel computation via `concurrent.futures`.
- Optional result pickling for post-processing.

---

## Installation

```bash
pip install numpy scipy matplotlib
```

No additional packages are required.  Clone or download the repository
and run scripts from its root directory.

---

## File structure

```
sibugec/
    __init__.py          Package entry point and public API
    eos.py               EoS models (pplus, pminus) and custom EoS loader
    thermodynamics.py    Entropy ODEs, temperature, free energy, T_c
    hydrodynamics.py     Junction conditions and flow solvers
    separators.py        Phase-space separator and contour finders
    plotting.py          Interactive phase-space bubble plot

main_eos_explorer.py     Interactive EoS explorer GUI (start here)
main_bubble_solver.py    Standalone bubble phase-space solver
```

---

## Quick start

### 1. EoS explorer (recommended starting point)

```bash
python main_eos_explorer.py
```

Use the sliders to tune the EoS parameters and watch the thermodynamic
quantities update in real time.  Click **"Compute all bubbles"** to
launch the full phase-space computation.

### 2. Standalone bubble solver

Edit the parameter block at the top of `main_bubble_solver.py`, then:

```bash
python main_bubble_solver.py
```

### 3. Scripting

```python
from sibugec.plotting import interactive_bubble_plot

fig, ax = interactive_bubble_plot(
    eTH=0.5, eTL=2.0,
    p1=0.0, p0=0.0, n=4.0, delta=1.0,
    nucleation_energies_allowed=(0.5, 2.0),
    bubble_energies_allowed=(1e-6, 2.0),
    xiw_resolution=50,
    en_resolution=50,
)
```

---

## Custom EoS (WIP)

To replace the analytic model with tabulated data, provide a plain-text
file with two (or three) columns:

```
# e   p(e)   [optional: cs2(e)]
0.0   0.000
0.1   0.033
0.5   0.155
...
```

Then set `CUSTOM_EOS_FILE` in either entry-point script:

```python
# main_bubble_solver.py
CUSTOM_EOS_FILE = "my_eos.dat"
```

The loader (`sibugec.eos.load_custom_eos`) fits a cubic spline to the
data and returns callable `p(e)` and `c_s²(e)` functions that are
passed directly to the solver.  If a third column is present it is used
as `c_s²(e)` directly; otherwise it is derived numerically from `p(e)`.

For advanced use (separate HT and LT tables), load the two files
separately and pass them via `custom_pplus`, `custom_pminus`,
`custom_cs2_plus`, and `custom_cs2_minus` arguments of
`interactive_bubble_plot`.

---

## EoS parametrisation

The built-in two-phase model uses:

**High-temperature (HT) branch:**
```
p+(e) = (δ/3) ln cosh((e − e_TH)/δ) + C
```
where C is fixed by p+(e_TH) = p₁.

**Low-temperature (LT) branch:**
```
p−(e) = (1/3) [ e − e^(n+1) / ((n+1) e_TL^n) ]
```

---

## Parameters

| Symbol  | Description                              | Default |
|---------|------------------------------------------|---------|
| `eTH`   | HT branch turning-point energy           | 0.5     |
| `eTL`   | LT branch turning-point energy           | 2.0     |
| `p1`    | HT pressure at `eTH`                     | 0.0     |
| `delta` | HT smoothing scale                       | 1.0     |
| `n`     | LT shape exponent                        | 4.0     |
| `sTH`   | HT entropy anchor (sets T scale)         | 2.5     |
| `sTL`   | LT entropy anchor (sets T scale)         | 8.45    |

---

## Output files

When `saving=True` (or clicking "Compute all bubbles" with the flag
enabled), results are saved as a pickle file:

```
sibugec_results_<eTH>_<eTL>_<p0>.pkl
```

Two profile files are written to `output_dir` after each click event:

- `vofxi.txt` — velocity profile v(ξ)
- `eofxi.txt` — energy profile e(ξ)

---

## License

Released under the MIT License.  See `LICENSE` for details.
