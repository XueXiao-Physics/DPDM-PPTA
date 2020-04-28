# gp_bases.py
"""Utilities module containing various useful
functions for use in other modules.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from enterprise.signals.parameter import function

######################################
# Fourier-basis signal functions #####
######################################

__all__ = [
    "createfourierdesignmatrix_red",
    "createfourierdesignmatrix_dm",
    "createfourierdesignmatrix_env",
    "createfourierdesignmatrix_ephem",
    "createfourierdesignmatrix_eph",
]


@function
def createfourierdesignmatrix_red(
    toas, nmodes=30, Tspan=None, logf=False, fmin=None, fmax=None, pshift=False, modes=None
):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013
    :param toas: vector of time series in seconds
    :param nmodes: number of fourier coefficients to use
    :param freq: option to output frequencies
    :param Tspan: option to some other Tspan
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency
    :param pshift: option to add random phase shift
    :param modes: option to provide explicit list or array of
                  sampling frequencies

    :return: F: fourier design matrix
    :return: f: Sampling frequencies
    """

    T = Tspan if Tspan is not None else toas.max() - toas.min()

    # define sampling frequencies
    if modes is not None:
        nmodes = len(modes)
        f = modes
    elif fmin is None and fmax is None and not logf:
        # make sure partially overlapping sets of modes
        # have identical frequencies
        #f = 1.0 * np.arange(1, nmodes + 1) / T
        f = 1.0 / T + np.arange(0, nmodes + 1) / T / 2.0
    else:
        # more general case

        if fmin is None:
            fmin = 1 / T

        if fmax is None:
            fmax = nmodes / T

        if logf:
            f = np.logspace(np.log10(fmin), np.log10(fmax), nmodes)
        else:
            f = np.linspace(fmin, fmax, nmodes)

    # add random phase shift to basis functions
    ranphase = np.random.uniform(0.0, 2 * np.pi, nmodes) if pshift else np.zeros(nmodes)

    Ffreqs = np.repeat(f, 2)

    f = f[:-1]

    N = len(toas)

    F = np.zeros((N, 2 * nmodes))

    # The sine/cosine modes
    F[:, ::2] = np.sin(2 * np.pi * toas[:, None] * f[None, :] + ranphase[None, :])
    F[:, 1::2] = np.cos(2 * np.pi * toas[:, None] * f[None, :] + ranphase[None, :])

    return F, Ffreqs


@function
def createfourierdesignmatrix_dm(
    toas, freqs, nmodes=30, Tspan=None, pshift=False, fref=1400, logf=False, fmin=None, fmax=None, modes=None
):
    """
    Construct DM-variation fourier design matrix. Current
    normalization expresses DM signal as a deviation [seconds]
    at fref [MHz]

    :param toas: vector of time series in seconds
    :param freqs: radio frequencies of observations [MHz]
    :param nmodes: number of fourier coefficients to use
    :param Tspan: option to some other Tspan
    :param pshift: option to add random phase shift
    :param fref: reference frequency [MHz]
    :param logf: use log frequency spacing
    :param fmin: lower sampling frequency
    :param fmax: upper sampling frequency
    :param modes: option to provide explicit list or array of
                  sampling frequencies

    :return: F: DM-variation fourier design matrix
    :return: f: Sampling frequencies
    """

    # get base fourier design matrix and frequencies
    F, Ffreqs = createfourierdesignmatrix_red(
        toas, nmodes=nmodes, Tspan=Tspan, logf=logf, fmin=fmin, fmax=fmax, pshift=pshift, modes=modes
    )

    # compute the DM-variation vectors
    Dm = (fref / freqs) ** 2

    return F * Dm[:, None], Ffreqs

