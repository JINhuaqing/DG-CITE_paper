#!/bin/bash
#SBATCH --mem=200gb                          # Job memory request
#SBATCH --partition=long # Run on partition "dgx" (e.g. not the default partition called "long")
#SBATCH --output=scs/logs/DDPM-SIMU-%x-%j.out
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00
#SBATCH --chdir=/home/hujin/jin/MyResearch/DG-CITE_paper/bash_scripts/



echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"

singularity exec ~/jin/singularity_containers/dg-cite_latest.sif python -u ../python_scripts/simu_realY1.py --nblk ${3} --nfeat ${4} --epoch 4000 --n_T ${1} --lr ${2}

