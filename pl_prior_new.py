import numpy as np
from enterprise.signals.parameter import function
import enterprise.constants as const

# Redefine the powerlaw prior to be compatible with the new gp_bases.

@function
def powerlaw(f):

    df = np.diff(f[::2])
    return (
        (10 ** log10_A) ** 2 / 12.0 / np.pi ** 2 * const.fyr ** (gamma - 3) * f[:-2] ** (-gamma) * np.repeat(df, 2)
    )
