#!/bin/bash
#### The job script, run it as qsub xxx.sh 

#### the shell language when run via the job scheduler [IMPORTANT]
#$ -S /bin/bash
#### job should run in the current working directory
###$ -cwd
##### set job working directory
#$ -wd  /wynton/home/rajlab/hjin/MyResearch/DG-CITE_paper/bash_scripts/
#### Specify job name
#$ -N S1_d1_nT1
#### Output file
#$ -o wynton/logs/DDPM-SIMU-$JOB_NAME_$JOB_ID.out
#### Error file
#$ -e wynton/logs/DDPM-SIMU-$JOB_NAME_$JOB_ID.err
#### memory per core
#$ -l mem_free=2G
#### number of cores 
#$ -pe smp 40
#### Maximum run time 
#$ -l h_rt=72:00:00
#### job requires up to 2 GB local space
#$ -l scratch=2G
#### Specify queue
###  gpu.q for using gpu
###  if not gpu.q, do not need to specify it
###$ -q gpu.q 
#### The GPU memory required, in MiB
### #$ -l gpu_mem=12000M

echo "Starting running"

singularity exec ~/MyResearch/dg-cite_latest.sif python -u ../python_scripts/simu_lei.py --setting setting${1} --d ${2} --n 3000 --epoch 3000 --early_stop 0 --n_T ${3}

[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
