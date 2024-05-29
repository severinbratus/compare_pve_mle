import sys
import os
import random


gpus_per_task_one = "#SBATCH --gpus-per-task=1"

template = """#!/bin/sh

#SBATCH --job-name=N{n:03}K{k:03}S{s:03}{postfix}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --time=03:55:00
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
srun python exp2.py {n} {k} {s} > out_{jkey}.log
"""

acc_flag = int(os.getenv('ACC', '0'))
account = 'education-eemcs-courses-cse3000' if acc_flag else 'innovation'

n_seeds = 10

def gen_job(n, k, s):
    jkey = f"N{n:03}_K{k:03}_S{s:03}"
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

    text = template.format(jkey=jkey, n=n, k=k, s=s,
                           partition=partition, postfix=postfix, gpus_per_task_line=gpus_per_task_line,
                           account=account)
    with open(f'jobs/job_{jkey}.sh', 'w') as f:
        f.write(text)

range_n = [8, 16, 32, 64, 128]
range_k = [10, 30, 50, 70, 90, 104]
range_s = list(range(10))

args = sys.argv[1:]

if args:
    n, k, s = args
    n = int(n)
    k = int(k)
    s = int(s)
    gen_job(n, k, s)
else:
    for n in range_n:
        for k in range_k:
            for s in range_s:
                gen_job(n, k, s)
            
