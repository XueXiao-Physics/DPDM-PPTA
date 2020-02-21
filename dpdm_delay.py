from enterprise.signals import parameter
from enterprise.signals import deterministic_signals
import scipy.constants as sc
import scipy.stats as ss
import numpy as np







def GammaPrior(value, a, loc, scale):
    """Prior function for Uniform parameters."""

    return ss.gamma.pdf(value, a, loc, scale)



def GammaSampler(a, loc, scale, size=None):
    """Sampling function for Uniform parameters."""

    return ss.gamma.rvs(a, loc, scale, size=size)



def Gamma(a, loc, scale, size=None):

    class Gamma(parameter.Parameter):
        _size = size
        _prior = parameter.Function(GammaPrior, a=a, loc=loc, scale=scale)
        _sampler = staticmethod(GammaSampler)
        _typename = parameter._argrepr("Gamma", a=a, loc=loc, scale=scale)

    return Gamma


@parameter.function
def delay_U1B(toas, log10_ma, log10_eps, normA_e, normA_p, phase_e, phase_p):

    e   = np.sqrt( sc.alpha * 4 * np.pi )
    rvt2s  = sc.hbar / ( sc.electron_volt )
    bq2m = 2 * sc.electron_volt / ( ( sc.m_p + sc.m_n ) * sc.c ** 2) * 1e9

    ma = 10**log10_ma
    eps = 10**log10_eps
    A0 = 2.48e-12 * (1.0 / ma) * (1.0 / np.sqrt(3))
    twopif = 2 * np.pi * ma * sc.electron_volt / sc.h
    q2m_e = bq2m
    q2m_p = bq2m

    dx_e = - A0 * normA_e * eps * e * (1.0 / ma) * q2m_e * np.sin(twopif * toas + phase_e)
    dx_p = - A0 * normA_p * eps * e * (1.0 / ma) * q2m_p * np.sin(twopif * toas + phase_p)

    return (dx_p - dx_e) * rvt2s


@parameter.function
def delay_U1BL(toas, log10_ma, log10_eps, normA_e, normA_p, phase_e, phase_p):

    e   = np.sqrt( sc.alpha * 4 * np.pi )
    rvt2s  = sc.hbar / ( sc.electron_volt )
    bq2m = 2 * sc.electron_volt / ( ( sc.m_p + sc.m_n ) * sc.c ** 2) * 1e9

    ma = 10**log10_ma   
    eps = 10**log10_eps
    A0 = 2.48e-12 * (1.0 / ma) * (1.0 / np.sqrt(3))
    twopif = 2 * np.pi * ma * sc.electron_volt / sc.h
    q2m_e = bq2m / 2.
    q2m_p = bq2m

    dx_e = - A0 * normA_e * eps * e * (1.0 / ma) * q2m_e * np.sin(twopif * toas + phase_e)
    dx_p = - A0 * normA_p * eps * e * (1.0 / ma) * q2m_p * np.sin(twopif * toas + phase_p)


    return (dx_p - dx_e) * rvt2s



def dpdm_block(model=None,log10_ma=None,log10_eps=None,normA_e=None,
                normA_p=None,phase_e=None,phase_p=None):

    name='x_dp'

    if log10_ma == None:
        log10_ma  = parameter.Uniform(-23.0, -21.0)(name + '_log10_ma')

    log10_eps = parameter.Uniform(-28.0, -16.0)(name + '_log10_eps')
    normA_e = Gamma(3.156, 0, 0.281)(name + "_normA_e")
    normA_p = Gamma(3.156, 0, 0.281)
    phase_e = parameter.Uniform(0, 2 * np.pi)(name + '_phase_e')
    phase_p = parameter.Uniform(0, 2 * np.pi)

    if model == "U1B":
        delay = delay_U1B(log10_ma = log10_ma,
                        log10_eps = log10_eps,
                        normA_e = normA_e,
                        normA_p = normA_p,
                        phase_e = phase_e,
                        phase_p = phase_p)
    if model == "U1BL":
        delay = delay_U1BL(log10_ma = log10_ma,
                        log10_eps = log10_eps,
                        normA_e = normA_e,
                        normA_p = normA_p,
                        phase_e = phase_e,
                        phase_p = phase_p)

    dpdm = deterministic_signals.Deterministic(delay, name=name)
    return dpdm



