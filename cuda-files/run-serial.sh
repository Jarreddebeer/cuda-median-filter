gridsizes=(512 1024 2048 4096)
windowsizes=(3 5 15 21)

rm serial/results.csv
touch serial/results.csv

for w in "${windowsizes[@]}"
do
    printf ",$w" >> serial/results.csv
done

printf "\n" >> serial/results.csv

for g in "${gridsizes[@]}"
do
    printf "$g" >> serial/results.csv
    for w in "${windowsizes[@]}"
    do
        # make
        printf "," >> serial/results.csv

        # execute and record
        (TIMEFORMAT="%R"; time ./hpc.serial $g $w) |& tail -n 1 | tr -d '\n' >> serial/results.csv

        # # create histograms
        # python ../histogram.py output.csv $g $w
        # mv histogram.png serial/histograms/

        # clean up
        mv output.csv serial/results/$g-$w.csv
    done
    printf "\n" >> serial/results.csv
done
