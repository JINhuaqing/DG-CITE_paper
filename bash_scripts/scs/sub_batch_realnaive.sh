for nT in 100 200 400; do
    for lr in 1e-1 1e-2; do 
        for blk in 1 3; do
            for nfeat in 128 256; do
                job_name=nvreal${nT}S${lr}_d${blk}_nT${nfeat}
                sbatch --job-name=$job_name simu_real_naive.sh $nT $lr $blk $nfeat
            done
        done
    done
done
