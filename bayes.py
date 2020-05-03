import glob
import sys
import numpy as np

from enterprise.pulsar import Pulsar
from enterprise.signals import signal_base
from enterprise.signals import parameter
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals

import dpdm_delay

# import customized files to introduce 2T modes.
import gp_bases_new as gp_bases
import pl_prior as gp_priors

from enterprise.signals import selections

import multiprocess
from PTMCMCSampler.PTMCMCSampler import PTSampler

datadir = 'DR2'

parfiles = sorted(glob.glob(datadir + '/newpars/*.par'))
timfiles = sorted(glob.glob(datadir + '/toas/*.tim'))

psrs=[]
for ipsr in range(len(parfiles)):
    psr = Pulsar(parfiles[ipsr], timfiles[ipsr],ephem='DE436',clk="TT(BIPM2018)")
    psrs.append(psr)
    print(psr.name)

# red noise
nmodes = 30


log10_A = parameter.Constant()
gamma = parameter.Constant()
pl = gp_priors.powerlaw(log10_A=log10_A, gamma=gamma)
dm_basis = gp_bases.createfourierdesignmatrix_dm(nmodes=nmodes)
red_basis = gp_bases.createfourierdesignmatrix_red(nmodes=nmodes)
selection = selections.Selection(selections.by_band)

dmn = gp_signals.BasisGP(pl, dm_basis, name='dm', coefficients=False)
spn = gp_signals.BasisGP(pl, red_basis, name='sp',coefficients=False)
bdn = gp_signals.BasisGP(pl, red_basis, name='bd',coefficients=False,selection=selection)

# white noise
backend = selections.Selection(selections.by_backend)
efac = parameter.Constant()
equad = parameter.Constant()

ef = white_signals.MeasurementNoise(efac=efac,selection=backend)
eq = white_signals.EquadNoise(log10_equad=equad,selection=backend)
wnb = ef + eq

# timing model
tm = gp_signals.TimingModel()

# Bayes ephmeris
eph = deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)


# include best fit parameters
Dict = np.load('noise_params/noisepars_m'+str(nmodes)+'.npy',allow_pickle=True,encoding='bytes').item()

if len(Dict)!=638:
    raise



def single_task(log10_ma, name, ret, model, test=False):

        np.random.seed()

        log10_eps = parameter.Uniform(-28.0, -16.0)('x_dp_log10_eps')
        dp = dpdm_delay.dpdm_block(model=model,log10_ma=log10_ma,log10_eps=log10_eps)

        model0  = tm + wnb + dmn + spn + eph + dp
        model1  = tm + wnb + dmn + spn + eph + dp + bdn

        signal=[]
        for psr in psrs:
            if psr.name in ['J0437-4715','J1939+2134']:
                signal.append( model1( psr ) )
            else:
                signal.append( model0( psr ) )

        pta = signal_base.PTA(signal)
        pta.set_default_params(Dict)
        if test == True:
            return pta

        #outDir="bayes/"+model+"_m"+str(nmodes)+"/n"+str(i).zfill(3)+'/'
        outDir="/home/sdd/xuex/bayes/"+model+"_m"+str(nmodes)+'/'+name+'/'
#============================
        # to make the initial values
        xs = {par.name:par.sample() for par in pta.params}
        for parname in Dict.keys():
            if parname in xs.keys():
                xs[parname] = Dict[parname]

        x0 = np.hstack([xs[key] for key in sorted(xs.keys())])
        ndim = len(x0)
#============================

        x0[-19:-8] = 0.
        x0[-8] = -26.

        groups=[range(ndim),range(ndim-8,ndim)]

        print(pta.get_lnlikelihood(x0))


        covdiag = np.ones(ndim)*0.01

        sampler = PTSampler(ndim, pta.get_lnlikelihood, pta.get_lnprior,
                        cov=np.diag(covdiag),outDir=outDir,groups=groups,resume=False)

        sampler.sample(x0, 30000, isave=1000)

        #chain=np.loadtxt(outDir+'/chain_1.txt')

        #logep=chain[int(0.8*len(chain)):,-12]

        #ret.value = np.percentile(logep,95)

for names in ['A','B','C','D','E','F','G','H','I','J']:
    for it in [0,1,2,3]:
        a = 0 + 32*it
        b = min(32 + 32*it , 125)

        print(a,b,nmodes)

        jobs = []
        RETs={}

        for i in range(a,b): 

            RETs[i] = multiprocess.Manager().Value('i',0)

            logmamin = -23.5 + 2.5 * float(i) / float(125)
            logmamax = -23.5 + 2.5 * float(i+1) / float(125)
    
            log10_ma = parameter.Uniform(logmamin,logmamax)('x_dp_log10_ma')#n=0~99

            p = multiprocess.Process(target=single_task, args=(log10_ma, names+str(i).zfill(3) , RETs[i] , "U1B" ))

            jobs.append(p)

            p.start()

        for p in jobs:

            p.join()

