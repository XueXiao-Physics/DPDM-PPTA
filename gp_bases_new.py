import numpy as np
from enterprise.signals.parameter import function



# Redefine the fourier design matrix to be that of a 2T uniform.



@function
def createfourierdesignmatrix_red(toas, nmodes=30, Tspan=None):


    T = Tspan if Tspan is not None else toas.max() - toas.min()
    f = 1.0 / T + np.arange(0, nmodes + 1) / T / 2.0 # 2T uniform
    Ffreqs = np.repeat(f, 2)

    
    F = np.zeros((len(toas), 2 * nmodes))

    
    # The sine/cosine modes
    F[:, ::2] = np.sin(2 * np.pi * toas[:, None] * f[None, :-1])
    F[:, 1::2] = np.cos(2 * np.pi * toas[:, None] * f[None, :-1])
    
    # Ffreqs has (nmodes+1)*2 length.
    return F, Ffreqs


@function
def createfourierdesignmatrix_dm(toas, freqs, fref=1400., nmodes=30, Tspan=None):


    F, Ffreqs = createfourierdesignmatrix_red(toas, nmodes=nmodes, Tspan=Tspan)

    Dm = (fref / freqs) ** 2

    return F * Dm[:, None], Ffreqs

