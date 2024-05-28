import sys

template = """#!/bin/sh

#SBATCH --job-name=N{n:03}K{k:03}
#SBATCH --partition=gpu
#SBATCH --account=innovation
#SBATCH --time=03:55:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1GB

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
srun python exp2.py {n} {k} > out_{mkey}.log
"""

def gen_job(n, k):
    mkey = f"N{n:03}_K{k:03}"
    with open(f'job_{mkey}.sh', 'w') as f:
        f.write(template.format(mkey=mkey, n=n, k=k))

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
            





