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


# U(1)_B delay
@parameter.function
def delay_U1B(toas, pos, log10_ma, log10_eps, normA2_1_e, normA2_2_e, normA2_3_e, normA2_p, phase1_e, phase2_e, phase3_e, phase_p):

    e   = np.sqrt( sc.alpha * 4 * np.pi )
    rvt2s  = sc.hbar / ( sc.electron_volt )
    bq2m = 2 * sc.electron_volt / ( ( sc.m_p + sc.m_n ) * sc.c ** 2) * 1e9

    ma = 10**log10_ma   
    eps = 10**log10_eps
    A0 = 2.48e-12 * (1.0 / ma) * (1.0 / np.sqrt(3.0))
    twopif = 2 * np.pi * ma * sc.electron_volt / sc.h

    q2m_e = bq2m
    q2m_p = bq2m

    dx_e1 = - A0 * np.sqrt(normA2_1_e) * eps * e * (1.0 / ma) * q2m_e * np.sin(twopif * toas + phase1_e)
    dx_e2 = - A0 * np.sqrt(normA2_2_e) * eps * e * (1.0 / ma) * q2m_e * np.sin(twopif * toas + phase2_e)
    dx_e3 = - A0 * np.sqrt(normA2_3_e) * eps * e * (1.0 / ma) * q2m_e * np.sin(twopif * toas + phase3_e)    
    dx_p = - A0 * np.sqrt(normA2_p) * eps * e * (1.0 / ma) * q2m_p * np.sin(twopif * toas + phase_p)

    dx_e = pos[0]*dx_e1 + pos[1]*dx_e2 + pos[2]*dx_e3

    return (dx_p - dx_e) * rvt2s


#U(1)_{B-L} delay
@parameter.function
def delay_U1BL(toas, pos, log10_ma, log10_eps, normA2_1_e, normA2_2_e, normA2_3_e, normA2_p, phase1_e, phase2_e, phase3_e, phase_p):

    e   = np.sqrt( sc.alpha * 4 * np.pi )
    rvt2s  = sc.hbar / ( sc.electron_volt )
    bq2m = 2 * sc.electron_volt / ( ( sc.m_p + sc.m_n ) * sc.c ** 2) * 1e9

    ma = 10**log10_ma   
    eps = 10**log10_eps
    A0 = 2.48e-12 * (1.0 / ma) * (1.0 / np.sqrt(3.0))
    twopif = 2 * np.pi * ma * sc.electron_volt / sc.h
    q2m_e = bq2m / 2.
    q2m_p = bq2m
        
    dx_e1 = - A0 * np.sqrt(normA2_1_e) * eps * e * (1.0 / ma) * q2m_e * np.sin(twopif * toas + phase1_e)
    dx_e2 = - A0 * np.sqrt(normA2_2_e) * eps * e * (1.0 / ma) * q2m_e * np.sin(twopif * toas + phase2_e)
    dx_e3 = - A0 * np.sqrt(normA2_3_e) * eps * e * (1.0 / ma) * q2m_e * np.sin(twopif * toas + phase3_e)    
    dx_p = - A0 * np.sqrt(normA2_p) * eps * e * (1.0 / ma) * q2m_p * np.sin(twopif * toas + phase_p)

    dx_e = pos[0]*dx_e1 + pos[1]*dx_e2 + pos[2]*dx_e3

    return (dx_p - dx_e) * rvt2s


# the correlated case of U(1)_B delay
@parameter.function
def delay_U1B_corr(toas, pos, log10_ma, log10_eps, normA2_1_e, normA2_2_e, normA2_3_e, phase1_e, phase2_e, phase3_e, phase_p):

    e   = np.sqrt( sc.alpha * 4 * np.pi )
    rvt2s  = sc.hbar / ( sc.electron_volt )
    bq2m = 2 * sc.electron_volt / ( ( sc.m_p + sc.m_n ) * sc.c ** 2) * 1e9

    ma = 10**log10_ma   
    eps = 10**log10_eps
    A0 = 2.48e-12 * (1.0 / ma) * (1.0 / np.sqrt(3.0))
    twopif = 2 * np.pi * ma * sc.electron_volt / sc.h

    q2m_e = bq2m
    q2m_p = bq2m

    dx_e1 = - A0 * np.sqrt(normA2_1_e) * eps * e * (1.0 / ma) * q2m_e * np.sin(twopif * toas + phase1_e)
    dx_e2 = - A0 * np.sqrt(normA2_2_e) * eps * e * (1.0 / ma) * q2m_e * np.sin(twopif * toas + phase2_e)
    dx_e3 = - A0 * np.sqrt(normA2_3_e) * eps * e * (1.0 / ma) * q2m_e * np.sin(twopif * toas + phase3_e)

    dx_p1 = - A0 * np.sqrt(normA2_1_e) * eps * e * (1.0 / ma) * q2m_p * np.sin(twopif * toas + phase1_e + phase_p)
    dx_p2 = - A0 * np.sqrt(normA2_2_e) * eps * e * (1.0 / ma) * q2m_p * np.sin(twopif * toas + phase2_e + phase_p)
    dx_p3 = - A0 * np.sqrt(normA2_3_e) * eps * e * (1.0 / ma) * q2m_p * np.sin(twopif * toas + phase3_e + phase_p)

    dx_e = pos[0]*dx_e1 + pos[1]*dx_e2 + pos[2]*dx_e3
    dx_p = pos[0]*dx_p1 + pos[1]*dx_p2 + pos[2]*dx_p3

    return (dx_p - dx_e) * rvt2s


# the correlated case of U(1)_{B-L} delay
@parameter.function
def delay_U1BL_corr(toas, pos, log10_ma, log10_eps, normA2_1_e, normA2_2_e, normA2_3_e, phase1_e, phase2_e, phase3_e, phase_p):

    e   = np.sqrt( sc.alpha * 4 * np.pi )
    rvt2s  = sc.hbar / ( sc.electron_volt )
    bq2m = 2 * sc.electron_volt / ( ( sc.m_p + sc.m_n ) * sc.c ** 2) * 1e9

    ma = 10**log10_ma   
    eps = 10**log10_eps
    A0 = 2.48e-12 * (1.0 / ma) * (1.0 / np.sqrt(3.0))
    twopif = 2 * np.pi * ma * sc.electron_volt / sc.h
    q2m_e = bq2m / 2.
    q2m_p = bq2m
        
    dx_e1 = - A0 * np.sqrt(normA2_1_e) * eps * e * (1.0 / ma) * q2m_e * np.sin(twopif * toas + phase1_e)
    dx_e2 = - A0 * np.sqrt(normA2_2_e) * eps * e * (1.0 / ma) * q2m_e * np.sin(twopif * toas + phase2_e)
    dx_e3 = - A0 * np.sqrt(normA2_3_e) * eps * e * (1.0 / ma) * q2m_e * np.sin(twopif * toas + phase3_e)

    dx_p1 = - A0 * np.sqrt(normA2_1_e) * eps * e * (1.0 / ma) * q2m_p * np.sin(twopif * toas + phase1_e + phase_p)
    dx_p2 = - A0 * np.sqrt(normA2_2_e) * eps * e * (1.0 / ma) * q2m_p * np.sin(twopif * toas + phase2_e + phase_p)
    dx_p3 = - A0 * np.sqrt(normA2_3_e) * eps * e * (1.0 / ma) * q2m_p * np.sin(twopif * toas + phase3_e + phase_p)

    dx_e = pos[0]*dx_e1 + pos[1]*dx_e2 + pos[2]*dx_e3
    dx_p = pos[0]*dx_p1 + pos[1]*dx_p2 + pos[2]*dx_p3

    return (dx_p - dx_e) * rvt2s




def dpdm_block(model=None,log10_ma=None,log10_eps=None,normA2_1_e=None,normA2_2_e=None,normA2_3_e=None,
                normA2_p=None,phase1_e=None,phase2_e=None,phase3_e=None,phase_p=None):

    name='x_dp'

    if log10_ma == None:
        log10_ma  = parameter.Uniform(-23.0, -21.0)(name + '_log10_ma')
    if log10_eps == None:
        log10_eps = parameter.Uniform(-28.0, -16.0)(name + '_log10_eps')

    normA2_1_e = Gamma(1, 0, 1)(name + "_normA2_1_e")
    normA2_2_e = Gamma(1, 0, 1)(name + "_normA2_2_e")
    normA2_3_e = Gamma(1, 0, 1)(name + "_normA2_3_e")

    normA2_p = Gamma(1, 0, 1)

    phase1_e = parameter.Uniform(0, 2 * np.pi)(name + '_phase1_e')
    phase2_e = parameter.Uniform(0, 2 * np.pi)(name + '_phase2_e')
    phase3_e = parameter.Uniform(0, 2 * np.pi)(name + '_phase3_e')

    phase_p = parameter.Uniform(0, 2 * np.pi)

    if model == "U1B":
        delay = delay_U1B(log10_ma = log10_ma,
                        log10_eps = log10_eps,
                        normA2_1_e = normA2_1_e,
                        normA2_2_e = normA2_2_e,
                        normA2_3_e = normA2_3_e,
                        normA2_p = normA2_p,
                        phase1_e = phase1_e,
                        phase2_e = phase2_e,
                        phase3_e = phase3_e,
                        phase_p = phase_p)
        print("model:U1B")

    if model == "U1BL":
        delay = delay_U1BL(log10_ma = log10_ma,
                        log10_eps = log10_eps,
                        normA2_1_e = normA2_1_e,
                        normA2_2_e = normA2_2_e,
                        normA2_3_e = normA2_3_e,
                        normA2_p = normA2_p,
                        phase1_e = phase1_e,
                        phase2_e = phase2_e,
                        phase3_e = phase3_e,
                        phase_p = phase_p)
        print("model:U1B-L")

    if model == "U1B_corr":
        delay = delay_U1B_corr(log10_ma = log10_ma,
                        log10_eps = log10_eps,
                        normA2_1_e = normA2_1_e,
                        normA2_2_e = normA2_2_e,
                        normA2_3_e = normA2_3_e,
                        phase1_e = phase1_e,
                        phase2_e = phase2_e,
                        phase3_e = phase3_e,
                        phase_p = phase_p)
        print("model:U1B, ***correlated**")

    if model == "U1BL_corr":
        delay = delay_U1BL_corr(log10_ma = log10_ma,
                        log10_eps = log10_eps,
                        normA2_1_e = normA2_1_e,
                        normA2_2_e = normA2_2_e,
                        normA2_3_e = normA2_3_e,
                        phase1_e = phase1_e,
                        phase2_e = phase2_e,
                        phase3_e = phase3_e,
                        phase_p = phase_p)
        print("model:U1B-L, ***correlated***")

    dpdm = deterministic_signals.Deterministic(delay, name=name)
    return dpdm



