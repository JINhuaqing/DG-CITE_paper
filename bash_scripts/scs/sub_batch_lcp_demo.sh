for d in 10 100 300 1000 ; do
for setting in 1 2; do
    for c in 0.0 0.01 0.05 0.08 0.1 0.2 0.3 0.4 0.5 0.8 1.0 1.5 2.0 ; do
    #for h in 0.2 0.4 0.6 0.8 1.0 1.2 ; do
        job_name=my4S${setting}_c${c}_d${d}_demo
        sbatch --job-name=$job_name simu_my4_lcp_demo.sh  $setting $c $d
    done
done
done

