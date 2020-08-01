#!/usr/bin/env python
from pathlib import Path
import subprocess as sp

batch = '''#!/bin/bash
#SBATCH -p gpu
#SBATCH -o logs/{jobname}.log
#SBATCH -e logs/{jobname}.err
#SBATCH -J {jobname}
#SBATCH --gres gpu:1

'''


job = '''

python /home2/koji_ono/workspace/kiritani_ono_2018/scripts/run2.py with work_dir=/home2/koji_ono/workspace/kiritani_ono_2018/experiments/{jobname}-rtFalse dataset={ds} epochs=50 kernel_sizes_pool=[[1,1],[1,1],[2,12]]  rotate_traindata=False r_min={rmin} r_max={rmax} H={h} W={w} &

python /home2/koji_ono/workspace/kiritani_ono_2018/scripts/run2.py with work_dir=/home2/koji_ono/workspace/kiritani_ono_2018/experiments/{jobname}-rtTrue dataset={ds} epochs=50 kernel_sizes_pool=[[1,1],[1,1],[2,12]]  rotate_traindata=True r_min={rmin} r_max={rmax} H={h} W={w} &

wait

'''


datasets = ['mnist',
            'fmnist',
            'rotatedmnist',
            'mnistr',
            'mnistrts',
            'sim2mnist']

datasets = ['rotatedmnist',
            'mnistr',
            'mnistrts',
            'sim2mnist']

rmins = [0.01, 0.05, 0.1]
rmaxs = [0.6, 1.0]
Hs = [5, 7, 9, 12]
Ws = [12, 14, 16]

for h in Hs:
    _myjob = batch
    for ds in datasets:
        for rmin in rmins:
            for rmax in rmaxs:
                for w in Ws:
                    options = dict(
                        ds=ds, rmin=rmin, rmax=rmax,
                        h=h, w=w)
                    jobname = "exp4-lp-ds{ds}-rmin{rmin}-rmax{rmax}-h{h}-w{w}".format(**options)
                    _myjob += job.format(
                        jobname=jobname,
                        **options)
    exp_dir = Path('batch_scripts')
    exp_dir.mkdir(exist_ok=True)
    jobname_sh = (exp_dir / (jobname + ".sh"))
    with jobname_sh.open('w') as fp:
        fp.write(_myjob)
        if jobname_sh.exists():
            sp.run(['sbatch {}'.format(jobname_sh),], shell=True)

