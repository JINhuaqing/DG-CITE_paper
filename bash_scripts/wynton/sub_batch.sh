for nT in 100 200 400; do
    for Snum in 3 4 ; do 
    #for Snum in 1 2 3 4 5 6 7 8; do 
        for d in 10 100 300 1000; do
            job_name=${1}S${Snum}_d${d}_nT${nT}
            qsub -N $job_name simu_${1}.sh $Snum $d $nT
        done
    done
done

