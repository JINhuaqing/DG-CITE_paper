for nT in 400 200 100; do
    for lr in 1e-1 1e-2 ; do
        for d in 300 1000; do
            setting=8
            job_name=my2S${setting}_lr${lr}_nT${nT}_d${d}
            qsub -N $job_name simu_my2.sh $nT $lr $setting $d
        done
    done
done

