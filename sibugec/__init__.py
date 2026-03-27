def print_logo():
    """Print the SiBuGEC ASCII logo to the terminal."""
    print(r"""
   _____ _ _           _____ ______ _____
  / ____(_) |         / ____|  ____/ ____|
 | (___  _| |__  _   _| |  __| |__ | |
  \___ \| | '_ \| | | | | |_ |  __|| |
  ____) | | |_) | |_| | |__| | |___| |____
 |_____/|_|_.__/ \__,_|\_____|______\_____|
    """)


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
    "print_logo",
    "entropy_ODE_plus",
    "entropy_ODE_minus",
    "calculate_temperature",
    "find_critical_temperature",
    "find_deto",
    "find_def",
    "find_hyb",
    "interactive_bubble_plot",
]
