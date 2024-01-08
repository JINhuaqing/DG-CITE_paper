for Snum in 1 2 3 4 5 6 7 8; do 
    for early in 0 1; do 
        for d in 1 2; do
            job_name=S${Snum}fne${early}_d${d}_n3k
            nep=4000
            if [ $early -eq 0 ]; 
            then 
            nep=3000
            fi
            sbatch --job-name=$job_name run_ddpm_ablation_PCP_fixed_batch.sh $Snum $d $early $nep
        done
    done
done

