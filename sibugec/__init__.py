"""
SiBuGEC — Self-similar Bubble Expansion with General Equation of State Code

A Python package for computing hydrodynamic bubble solutions
(detonations, deflagrations, and hybrids) arising during cosmological
first-order phase transitions, for a general equation of state.

Modules
-------
eos           : Equation-of-state models and custom EoS loader
thermodynamics: Entropy ODEs, temperature, free energy, critical temperature
hydrodynamics : Junction conditions and flow solvers
separators    : Phase-space separator and contour finders
plotting      : Interactive phase-space bubble plot
"""

from .eos import pplus, pminus, speed_of_sound_squared, load_custom_eos
from .thermodynamics import (
    entropy_ODE_plus,
    entropy_ODE_minus,
    calculate_temperature,
    find_critical_temperature,
)
from .hydrodynamics import find_deto, find_def, find_hyb
from .plotting import interactive_bubble_plot

__all__ = [
    "pplus",
    "pminus",
    "speed_of_sound_squared",
    "load_custom_eos",
    "entropy_ODE_plus",
    "entropy_ODE_minus",
    "calculate_temperature",
    "find_critical_temperature",
    "find_deto",
    "find_def",
    "find_hyb",
    "interactive_bubble_plot",
]
