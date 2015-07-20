gridsizes=(512 1024 2048 4096)
windowsizes=(3 5 15 21)

rm results.csv
touch results.csv

for w in "${windowsizes[@]}"
do
    printf ",$w" >> results.csv
done

printf "\n" >> results.csv

for g in "${gridsizes[@]}"
do
    printf "$g" >> results.csv
    for w in "${windowsizes[@]}"
    do
        # make
        mv $g/hpc.cuda-$g-$w.cu .
        make cuda
        printf "," >> results.csv

        # execute and record
        (TIMEFORMAT="%R"; time ./cuda $g $w) |& xargs printf >> results.csv

        # # create histograms
        python ../histogram.py output.csv $g $w

        # clean up
        mv output.csv results/$g-$w.csv
        rm cuda
        rm *.o
        mv hpc.cuda-$g-$w.cu $g/hpc.cuda-$g-$w.cu
    done
    printf "\n" >> results.csv
done
