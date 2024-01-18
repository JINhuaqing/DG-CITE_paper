for nT in 400 200 100; do
    for lr in 1e-1 1e-2 ; do
        for d in 10 100; do
            setting=4
            job_name=leiS${setting}_lr${lr}_nT${nT}_d${d}
            qsub -N $job_name simu_lei.sh $nT $lr $setting $d
        done
    done
done

