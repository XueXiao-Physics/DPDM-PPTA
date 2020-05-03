import glob
import sys
import numpy as np

from enterprise.pulsar import Pulsar
from enterprise.signals import signal_base
from enterprise.signals import parameter
from enterprise.signals import white_signals
from enterprise.signals import gp_signals

# import customized files to introduce 2T modes.
import gp_bases_new as gp_bases
import pl_prior as gp_priors

from enterprise.signals import selections

import multiprocess
from PTMCMCSampler.PTMCMCSampler import PTSampler

def get_pulsar_noise(pta , ret, ro):

        ndim = len(pta.params)


        groups0 = [[i,i+1] for i in range(0,ndim-1,2)]
        groups0.extend([range(ndim)])

        groups1 = [range(ndim)]
        outDir0='/home/sdd/xuex/noise/first_run_1/'+pta.pulsars[0]
        outDir1='/home/sdd/xuex/noise/second_run_1/'+pta.pulsars[0]

        if ro==False:
                x0 = np.zeros(ndim)
                x0=np.hstack([par.sample() for par in pta.params])
                cov0 = np.diag(np.ones(ndim)*0.01)

                sampler = PTSampler(ndim, pta.get_lnlikelihood, pta.get_lnprior,
                        cov0, groups=groups0 , outDir = outDir0, verbose=True)
                print(pta.pulsars[0]+'***starts')
                sampler.sample(x0, 50000,isave=1000)
                chain0 = np.loadtxt(outDir0+'/chain_1.txt')



                x1 = chain0[np.where(chain0==np.max(chain0[:,-3]))[0][0],:-4]
                cov1 = np.load(outDir0 + '/cov.npy')
                sampler = PTSampler(ndim, pta.get_lnlikelihood, pta.get_lnprior,
                        cov1 , groups=groups1 , outDir=outDir1, verbose=True)
                sampler.sample(x1, 100000, isave=1000)

        chain1 = np.loadtxt(outDir1+'/chain_1.txt')

        # End of the second run.


        # Return the ln-likelihood value of the best fit(maximal likelihood).

        MLHselect = chain1[np.where(chain1==np.max(chain1[:,-3]))[0][0],:]
        Dict = {pta.params[i].name:MLHselect[i] for i in range(ndim)}
        ret.value = (Dict,pta.get_lnlikelihood(Dict),pta.get_lnprior(Dict))
        print(pta.pulsars[0]+'***finished')


        # End of the function.

datadir = 'DR2'
parfiles = sorted(glob.glob(datadir + '/newpars/*.par'))
timfiles = sorted(glob.glob(datadir + '/toas/*.tim'))

psrs=[]
for ipsr in range(len(parfiles)):
    psr = Pulsar(parfiles[ipsr], timfiles[ipsr])
    psrs.append(psr)
    print(psr.name)

# red noise
nmodes = 30


log10_A = parameter.Uniform(-21,-9)
gamma = parameter.Uniform(0,7)
pl = gp_priors.powerlaw(log10_A=log10_A, gamma=gamma)
dm_basis = gp_bases.createfourierdesignmatrix_dm(nmodes=nmodes)
red_basis = gp_bases.createfourierdesignmatrix_red(nmodes=nmodes)
selection = selections.Selection(selections.by_band)

dmn = gp_signals.BasisGP(pl, dm_basis, name='dm', coefficients=False)
spn = gp_signals.BasisGP(pl, red_basis, name='sp',coefficients=False)
bdn = gp_signals.BasisGP(pl, red_basis, name='bd',coefficients=False,selection=selection)

# white noise
backend = selections.Selection(selections.by_backend)
efac = parameter.Uniform(0.01, 10.0)
equad = parameter.Uniform(-8.5, -5)

ef = white_signals.MeasurementNoise(efac=efac,selection=backend)
eq = white_signals.EquadNoise(log10_equad=equad,selection=backend)
wnb = ef + eq

# timing model
tm = gp_signals.TimingModel()

model0  = tm + wnb + dmn + spn
model1  = tm + wnb + dmn + spn + bdn
ptas=[]
for psr in psrs:
        if psr.name in ['J0437-4715','J1939+2134']:
                pta=signal_base.PTA( model1(psr) )
        else:
                pta=signal_base.PTA( model0(psr) )
        ptas.append(pta)
        print(psr.name)

jobs = []
RETs={}        
for i in range(len(psrs)):
        RETs[i] = multiprocess.Manager().Value('i',0)
        p = multiprocess.Process(target=get_pulsar_noise, args=(ptas[i],RETs[i],False))
        jobs.append(p)
        p.start()
for p in jobs:
        p.join()


# Return the sum of the Maximal Likelihood values.

MLHselect = [RET.value for RET in RETs.values()]


Dict = {}
for x in MLHselect:
        Dict.update(x[0])

np.save('noise_chain/noisepars_1_m'+str(nmodes),Dict)
