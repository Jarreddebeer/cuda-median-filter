gridsizes=(4096)
windowsizes=(3 5 15 21)

rm shared-grid/results.csv
touch shared-grid/results.csv

for w in "${windowsizes[@]}"
do
    printf ",$w" >> shared-grid/results.csv
done

printf "\n" >> shared-grid/results.csv

for g in "${gridsizes[@]}"
do
    printf "$g" >> shared-grid/results.csv
    for w in "${windowsizes[@]}"
    do
        # make
        mv shared-grid/$g/hpc.cuda-$g-$w.cu .
        make cuda
        printf "," >> shared-grid/results.csv

        # execute and record
        (TIMEFORMAT="%R"; time ./cuda $g $w) |& tail -n 1 | tr -d '\n' >> shared-grid/results.csv

        # # create histograms
        # python ../histogram.py output.csv $g $w
        # mv histogram.png shared-grid/histograms/

        # clean up
        mv output.csv shared-grid/results/$g-$w.csv
        rm cuda
        rm *.o
        mv hpc.cuda-$g-$w.cu shared-grid/$g/hpc.cuda-$g-$w.cu
    done
    printf "\n" >> shared-grid/results.csv
done
