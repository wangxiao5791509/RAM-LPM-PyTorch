#!/usr/bin/env python
from pathlib import Path
import subprocess as sp

batch = '''#!/bin/bash
#SBATCH -p gpu
#SBATCH -o logs/{jobname}.log
#SBATCH -e logs/{jobname}.err
#SBATCH -J {jobname}
#SBATCH --gres=gpu:1

python /home2/koji_ono/workspace/kiritani_ono_2018/scripts/run.py with polar_coordinate={pc} rot_angle={ra} arch=CNN2FC2 workdir=/home2/koji_ono/workspace/kiritani_ono_2018/experiments/{jobname}-archCNN2FC2 use_cuda=True num_workers=2 &
python /home2/koji_ono/workspace/kiritani_ono_2018/scripts/run.py with polar_coordinate={pc} rot_angle={ra} arch=CNN2FC2V2 workdir=/home2/koji_ono/workspace/kiritani_ono_2018/experiments/{jobname}-archCNN2FC2V2 use_cuda=True num_workers=2 &
python /home2/koji_ono/workspace/kiritani_ono_2018/scripts/run.py with polar_coordinate={pc} rot_angle={ra} arch=Net workdir=/home2/koji_ono/workspace/kiritani_ono_2018/experiments/{jobname}-archNet use_cuda=True num_workers=2 &
python /home2/koji_ono/workspace/kiritani_ono_2018/scripts/run.py with polar_coordinate={pc} rot_angle={ra} workdir=/home2/koji_ono/workspace/kiritani_ono_2018/experiments/{jobname}-archDefo use_cuda=True num_workers=2 &

wait
'''

polar_coordinates = [False, True]
rotate_angle = list(range(-70, 80, 10))

for pc in polar_coordinates:
    for ra in rotate_angle:
        jobname = f"pc{pc}-ra{ra}"
        exp_dir = Path('batch_scripts')
        exp_dir.mkdir(exist_ok=True)
        jobname_sh = (exp_dir / (jobname + ".sh"))
        with jobname_sh.open('w') as fp:
            fp.write(batch.format(
                jobname=jobname,
                pc=pc, ra=ra))
        if jobname_sh.exists():
            sp.run(['sbatch {}'.format(jobname_sh),], shell=True)
