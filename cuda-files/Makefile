include make/makepre.mk
include make/makebulds.mk

serial:
	gcc-4.8 -std=c99 hpc.serial.c -o hpc.serial
	# time ./hpc.serial 1024 17

omp:
	gcc-4.8 -std=c99 -fopenmp hpc.omp.c -o hpc.omp
	# time ./hpc.omp 512 9
	
