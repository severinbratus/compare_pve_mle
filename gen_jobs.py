import sys
import os
import random


gpus_per_task_one = "#SBATCH --gpus-per-task=1"

template = """#!/bin/sh

#SBATCH --job-name=N{n:03}K{k:03}{postfix}
#SBATCH --partition={partition}
#SBATCH --account=innovation
#SBATCH --time=05:55:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1GB
{gpus_per_task_line}

module load 2023r1
module load python
module load py-numpy
module load py-scipy
module load py-matplotlib
module load py-pip
module load py-llvmlite
module load py-pandas

python -m pip install --user jax optax ray[tune] flax PyDTMC


cd compare_pve_mle
srun python exp2.py {n} {k} {n_seeds} > out_{mkey}.log
"""

n_seeds = 10

def gen_job(n, k):
    mkey = f"N{n:03}_K{k:03}"
    if not os.path.exists('jobs'):
        os.makedirs('jobs')
    if random.random() < .10:
        partition = 'gpu'
        postfix = 'gpu'
        gpus_per_task_line = gpus_per_task_one
    else:
        partition = 'compute'
        postfix = 'cpu'
        gpus_per_task_line = ''

    text = template.format(mkey=mkey, n=n, k=k, n_seeds=n_seeds, partition=partition, postfix=postfix, gpus_per_task_line=gpus_per_task_line)
    with open(f'jobs/job_{mkey}.sh', 'w') as f:
        f.write(text)

range_n = [8, 16, 32, 64, 128]
range_k = [10, 30, 50, 70, 90, 104]

args = sys.argv[1:]

if args:
    n, k = args
    n = int(n)
    k = int(k)
    gen_job(n, k)
else:
    for n in range_n:
        for k in range_k:
            gen_job(n, k)
            





