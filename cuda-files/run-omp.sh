gridsizes=(512 1024 2048 4096)
windowsizes=(3 5 15 21)

rm omp/results.csv
touch omp/results.csv

for w in "${windowsizes[@]}"
do
    printf ",$w" >> omp/results.csv
done

printf "\n" >> omp/results.csv

for g in "${gridsizes[@]}"
do
    printf "$g" >> omp/results.csv
    for w in "${windowsizes[@]}"
    do
        # make
        printf "," >> omp/results.csv

        # execute and record
        (TIMEFORMAT="%R"; time ./hpc.omp $g $w) |& tail -n 1 | tr -d '\n' >> omp/results.csv

        # # create histograms
        # python ../histogram.py output.csv $g $w
        # mv histogram.png omp/histograms/

        # clean up
        mv output.csv omp/results/$g-$w.csv
    done
    printf "\n" >> omp/results.csv
done
