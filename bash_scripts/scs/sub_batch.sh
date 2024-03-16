for nT in 100 200 400; do
    for Snum in 1 2; do 
        for d in 10 100; do
            job_name=${1}S${Snum}_d${d}
            sbatch --job-name=$job_name simu_${1}.sh $Snum $d $nT
        done
    done
done
