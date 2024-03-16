for lr in 1e-1 1e-2 ; do
    for d in 10 100; do
        for setting in 1 2 3 4; do
            job_name=mlpleiS${setting}_lr${lr}_d${d}
            sbatch --job-name=$job_name simu_lei_addmlp.sh $lr $setting $d
        done
    done
done
