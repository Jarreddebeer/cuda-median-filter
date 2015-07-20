gridsizes=(512, 1024, 2048, 4096)
windowsizes=(3, 5, 15, 21)

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
        echo "$w $g" >> output.csv
        mv $g/hpc.cuda-$g-$w.cu .
        make cuda
        echo "," >> results.csv
        time -p ./cuda $g $w | head -n 1 | cut -d " " -f 2 >> results.csv
        mv output.csv results/$g-$w.csv
        rm cuda
        rm *.o
        mv hpc.cuda-$g-$w.cu $g/hpc.cuda-$g-$w.cu
    done
    echo "\n" >> results.csv
done
