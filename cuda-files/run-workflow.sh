gridsizes=(512, 1024, 2048, 4096)
windowsizes=(3, 5, 15, 21)

rm results.csv
touch results.csv

for w in "${windowsizes[@]}"
do
    echo ",$w" >> results.csv
done

echo "\n" >> results.csv

for g in "${gridsizes[@]}"
do
    echo "$g" >> results.csv
    for w in "${windowsizes[@]}"
    do
        # make
        echo "$w $g" >> output.csv
        mv $g/hpc.cuda-$g-$w.cu .
        make cuda
        echo "," >> results.csv

        # execute
        time -p ./cuda $g $w | head -n 1 | cut -d " " -f 2 >> results.csv

        # create histograms
        python ../histogram.py output.csv $g $w

        # clean up
        mv output.csv results/$g-$w.csv
        rm cuda
        rm *.o
        mv hpc.cuda-$g-$w.cu $g/hpc.cuda-$g-$w.cu
    done
    echo "\n" >> results.csv
done
