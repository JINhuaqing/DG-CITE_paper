#!/bin/bash
#SBATCH --mem=200gb                          # Job memory request
#SBATCH --partition=long # Run on partition "dgx" (e.g. not the default partition called "long")
#SBATCH --output=scs/logs/DDPM-ABL-%x-%j.out
#SBATCH -J S2ne_d2_n3k
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00
#SBATCH --chdir=/home/hujin/jin/MyResearch/DG-CITE_paper/bash_scripts/

echo "Running prime number generator program on $SLURM_CPUS_ON_NODE CPU cores"

singularity exec ~/jin/singularity_containers/dg-cite_latest.sif python -u ../python_scripts/run_ddpm_ablation_PCP.py --setting setting2 --d 100 --n 3000 --epoch 3000 --early_stop 0
