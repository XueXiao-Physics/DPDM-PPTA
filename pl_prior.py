# gp_priors.py
"""Utilities module containing various useful
functions for use in other modules.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy.stats

from enterprise.signals import parameter
from enterprise.signals.parameter import function
import enterprise.constants as const


@function
def powerlaw(f, log10_A=-16, gamma=5, components=2):
    #df = np.diff(np.concatenate((np.array([0]), f[::components])))
    df = np.diff(f[::components])
    return (
        (10 ** log10_A) ** 2 / 12.0 / np.pi ** 2 * const.fyr ** (gamma - 3) * f[:-2] ** (-gamma) * np.repeat(df, components)
    )
