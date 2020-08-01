#!/usr/bin/env python
from pathlib import Path
import subprocess as sp

batch = '''#!/bin/bash
#SBATCH -p gpu
#SBATCH -o logs/{jobname}.log
#SBATCH -e logs/{jobname}.err
#SBATCH -J {jobname}

python /home/koji.ono/mnt/kiritani_ono_2018/scripts/run2.py with work_dir=/home/koji.ono/mnt/kiritani_ono_2018/experiments/{jobname}_ng6 glimpse_layer={gl} patch_size={ps} glimpse_scale={gs} num_patches={np} glimpse_polar={gp} init_lr={lr} num_glimpses=6 num_workers=2 &
python /home/koji.ono/mnt/kiritani_ono_2018/scripts/run2.py with work_dir=/home/koji.ono/mnt/kiritani_ono_2018/experiments/{jobname}_ng8 glimpse_layer={gl} patch_size={ps} glimpse_scale={gs} num_patches={np} glimpse_polar={gp} init_lr={lr} num_glimpses=8 num_workers=2 &

wait
'''

glimpse_layers = ['cnn', 'fc']
patch_sizes = [8, 12, 16]
glimpse_scales = [2, 3, 4]
num_pathces = [1, 2]
glimpse_polar = [True, False]
init_lrs = [1e-3, 3e-4]

for gl in glimpse_layers:
    for ps in patch_sizes:
        for gs in glimpse_scales:
            for np in num_pathces:
                for gp in glimpse_polar:
                    for lr in init_lrs:
                        options = dict(
                            gl=gl, ps=ps, gs=gs,
                            np=np, gp=gp, lr=lr)
                        jobname = "rirvam-gl{gl}-ps{ps}-gs{gs}-np{np}-gp{gp}-lr{lr}".format(**options)
                        exp_dir = Path('batch_scripts')
                        exp_dir.mkdir(exist_ok=True)
                        jobname_sh = (exp_dir / (jobname + ".sh"))
                        with jobname_sh.open('w') as fp:
                            fp.write(batch.format(
                                jobname=jobname,
                                **options))
                        if jobname_sh.exists():
                            sp.run(['sbatch {}'.format(jobname_sh),], shell=True)
