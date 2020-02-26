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
def delay_U1B(toas, pos, log10_ma, log10_eps, normA1_e, normA2_e, normA3_e, normA_p, phase1_e, phase2_e, phase3_e, phase_p):

    e   = np.sqrt( sc.alpha * 4 * np.pi )
    rvt2s  = sc.hbar / ( sc.electron_volt )
    bq2m = 2 * sc.electron_volt / ( ( sc.m_p + sc.m_n ) * sc.c ** 2) * 1e9

    ma = 10**log10_ma   
    eps = 10**log10_eps
    A0 = 2.48e-12 * (1.0 / ma) * (1.0 / np.sqrt(3.0))
    twopif = 2 * np.pi * ma * sc.electron_volt / sc.h

    q2m_e = bq2m
    q2m_p = bq2m

    dx_e1 = - A0 * normA1_e * eps * e * (1.0 / ma) * q2m_e * np.sin(twopif * toas + phase1_e)
    dx_e2 = - A0 * normA2_e * eps * e * (1.0 / ma) * q2m_e * np.sin(twopif * toas + phase2_e)
    dx_e3 = - A0 * normA3_e * eps * e * (1.0 / ma) * q2m_e * np.sin(twopif * toas + phase3_e)    
    dx_p = - A0 * normA_p * eps * e * (1.0 / ma) * q2m_p * np.sin(twopif * toas + phase_p)

    dx_e = pos[0]*dx_e1 + pos[1]*dx_e2 + pos[2]*dx_e3

    return (dx_p - dx_e) * rvt2s


@parameter.function
def delay_U1BL(toas, pos, log10_ma, log10_eps, normA1_e, normA2_e, normA3_e, normA_p, phase1_e, phase2_e, phase3_e, phase_p):

    e   = np.sqrt( sc.alpha * 4 * np.pi )
    rvt2s  = sc.hbar / ( sc.electron_volt )
    bq2m = 2 * sc.electron_volt / ( ( sc.m_p + sc.m_n ) * sc.c ** 2) * 1e9

    ma = 10**log10_ma   
    eps = 10**log10_eps
    A0 = 2.48e-12 * (1.0 / ma) * (1.0 / np.sqrt(3.0))
    twopif = 2 * np.pi * ma * sc.electron_volt / sc.h
    q2m_e = bq2m / 2.
    q2m_p = bq2m
        
    dx_e1 = - A0 * normA1_e * eps * e * (1.0 / ma) * q2m_e * np.sin(twopif * toas + phase1_e)
    dx_e2 = - A0 * normA2_e * eps * e * (1.0 / ma) * q2m_e * np.sin(twopif * toas + phase2_e)
    dx_e3 = - A0 * normA3_e * eps * e * (1.0 / ma) * q2m_e * np.sin(twopif * toas + phase3_e)    
    dx_p = - A0 * normA_p * eps * e * (1.0 / ma) * q2m_p * np.sin(twopif * toas + phase_p)

    dx_e = pos[0]*dx_e1 + pos[1]*dx_e2 + pos[2]*dx_e3

    return (dx_p - dx_e) * rvt2s



def dpdm_block(model=None,log10_ma=None,log10_eps=None,normA1_e=None,normA2_e=None,normA3_e=None,
                normA_p=None,phase1_e=None,phase2_e=None,phase3_e=None,phase_p=None):

    name='x_dp'

    if log10_ma == None:
        log10_ma  = parameter.Uniform(-23.0, -21.0)(name + '_log10_ma')

    log10_eps = parameter.Uniform(-28.0, -16.0)(name + '_log10_eps')

    normA1_e = Gamma(3.141, 0, 0.282)(name + "_normA1_e")
    normA2_e = Gamma(3.141, 0, 0.282)(name + "_normA2_e")
    normA3_e = Gamma(3.141, 0, 0.282)(name + "_normA3_e")

    normA_p = Gamma(3.141, 0, 0.282)

    phase1_e = parameter.Uniform(0, 2 * np.pi)(name + '_phase1_e')
    phase2_e = parameter.Uniform(0, 2 * np.pi)(name + '_phase2_e')
    phase3_e = parameter.Uniform(0, 2 * np.pi)(name + '_phase3_e')

    phase_p = parameter.Uniform(0, 2 * np.pi)

    if model == "U1B":
        delay = delay_U1B(log10_ma = log10_ma,
                        log10_eps = log10_eps,
                        normA1_e = normA1_e,
                        normA2_e = normA2_e,
                        normA3_e = normA3_e,
                        normA_p = normA_p,
                        phase1_e = phase1_e,
                        phase2_e = phase2_e,
                        phase3_e = phase3_e,
                        phase_p = phase_p)
        print("model:U1B")

    if model == "U1BL":
        delay = delay_U1BL(log10_ma = log10_ma,
                        log10_eps = log10_eps,
                        normA1_e = normA1_e,
                        normA2_e = normA2_e,
                        normA3_e = normA3_e,
                        normA_p = normA_p,
                        phase1_e = phase1_e,
                        phase2_e = phase2_e,
                        phase3_e = phase3_e,
                        phase_p = phase_p)
        print("model:U1B-L")

    dpdm = deterministic_signals.Deterministic(delay, name=name)
    return dpdm



