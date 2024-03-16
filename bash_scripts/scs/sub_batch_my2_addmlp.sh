for lr in 1e-1 1e-2 ; do
    for d in 300 1000; do
        for setting in 5 6 7 8; do
            job_name=mlpmy2S${setting}_lr${lr}_d${d}
            sbatch --job-name=$job_name simu_my2_addmlp.sh $lr $setting $d
        done
    done
done
