import glob
import subprocess
import multiprocessing
import os

parfiles = sorted(glob.glob('newpars/*.par'))
timfiles = sorted(glob.glob('toas/*.tim'))
name=[os.path.basename(p) for p in parfiles]

def run(p,t,n):
    subprocess.call('tempo2 -f ' + p + ' ' + t +' -nobs 100000 -newpar > out/' + n , shell=True)
    subprocess.call('mv new.par'+' newpars/' + n , shell=True)

jobs = []
for i in range(len(parfiles)):
    p = multiprocessing.Process(target=run, args=(parfiles[i],timfiles[i],name[i]))
    jobs.append(p)
    p.start()
for p in jobs:
    p.join()

