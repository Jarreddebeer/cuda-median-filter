#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define BLOCKSIZE 16

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__device__ int sort(long values[], int start, int stop) {
    long pivot = values[start];
    int i = start;
    int j = stop+1;
    while (1) {
        while (values[++i] < pivot) if (i == stop) break;
        while (values[--j] >= pivot) if (j == start) break;
        if (i >= j) break;
        long tmp = values[i];
        values[i] = values[j];
        values[j] = tmp;
    }
    // place the pivot back
    long tmp = values[j];
    values[j] = pivot;
    values[start] = tmp;

    return j;
}

__device__ long getMedian(long values[], int size) {
    int start = 0;
    int stop = size - 1;
    int middle = (start + stop) / 2;
    int pivot = sort(values, start, stop);
    while (pivot != middle) {
        if (pivot > middle) {
            // median is in left half
            stop = pivot-1;
            pivot = sort(values, start, stop);
        } else {
            // median is in right half
            start = pivot+1;
            pivot = sort(values, start, stop);
        }
    }
    return values[pivot];
}

__global__ void medianFilterGPU(long* d_in, long* d_out, int histSize, int windSize) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockDim.x * blockIdx.x + tx;
    int y = blockDim.y * blockIdx.y + ty;

    // are we at a boundary?
    bool is_x_top = (tx == 0), is_x_bot = (tx == BLOCKSIZE-1);
    bool is_y_top = (ty == 0), is_y_bot = (ty == BLOCKSIZE-1);

    // window length is 9. so it overflows by 4 on each side.
    __shared__ long smem[BLOCKSIZE+8][BLOCKSIZE+8];

    // populate shared memory from histogram (histogram is padded with zeros)
    if (is_x_top) {
        for (int i = 0; i < 4; i++) smem[ty][i] = d_in[(y+4) * (histSize+8) + (x+4-(i+1))];

    } else if (is_x_bot) {
        for (int i = 0; i < 4; i++) smem[ty][BLOCKSIZE+i] = d_in[(y+4) * (histSize+8) + (x+4+(i+1))];
    }
    if (is_y_top) {
        for (int i = 0; i < 4; i++) smem[i][tx] = d_in[(y+4-(i+1)) * (histSize+8) + (x+4)];

        // corner cases
        if (is_x_top) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    smem[i][j] = d_in[(y+4-(i+1)) * (histSize+8) + (x+4-(j+1))];
                }
            }
        }

        else if (is_x_bot) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    smem[i][j] = d_in[(y+4-(i+1)) * (histSize+8) + (x+4+(j+1))];
                }
            }
        }

    } else if (is_y_bot) {
        for (int i = 0; i < 4; i++) smem[BLOCKSIZE+i][tx] = d_in[(y+4+(i+1)) * (histSize+8) + (x+4)];

        // corner cases
        if (is_x_top) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    smem[i][j] = d_in[(y+4+(i+1)) * (histSize+8) + (x+4-(j+1))];
                }
            }
        }

        else if (is_x_bot) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    smem[i][j] = d_in[(y+4+(i+1)) * (histSize+8) + (x+4+(j+1))];
                }
            }
        }
    }

    /*
    // first copy the data into the histograph

    hist[y * histSize + x] = histIn[y * histSize + x];

    __syncthreads();

    w_idx = 0;
    for (int dy = -windSize / 2; dy <= windSize / 2; dy++) {
        for (int dx = -windSize / 2; dx <= windSize / 2; dx++) {
            // gather the values in the window
            if (y + dy < 0) {gy = 0;}
            else if (y + dy > histSize - 1) {gy = histSize - 1;}
            else {gy = y + dy;}
            if (x + dx < 0) {gx = 0;}
            else if (x + dx > histSize - 1) {gx = histSize - 1;}
            else {gx = x + dx;}

            window[w_idx] = histIn[gy * histSize + gx];
            w_idx++;
        }
    }

    long median = getMedian(window, windSize * windSize);
    histOut[y * histSize + x] = median;

    __syncthreads();
    */

}

// read the binary file and perform binning
int readBinaryFile(const char* filename, long* grid, int histSize, int windSize) {
    printf("reading file...\n");
    int bloat = windSize / 2;
    FILE *dataFile = fopen(filename, "rb");
    if (!dataFile) {
        printf("Unable to open data file.");
        return -1;
    }
    while(!feof(dataFile)) {
        float x;
        float y;
        fread(&x, 1, sizeof(float), dataFile);
        fread(&y, 1, sizeof(float), dataFile);
        // get bins
        int xpos = (int) (x * (histSize - 1));
        int ypos = (int) (y * (histSize - 1));
        //
        grid[(ypos+bloat) * (histSize+windSize-1) + (xpos+bloat)] += 1;
    }
    fclose(dataFile);
    return 1;
}

// read the already written CSV histogram
int readHistogramCsvFile(const char* filename, long* grid, int histSize, int windSize) {
    printf("Reading histogram file...\n");
    int bloat = windSize / 2;
    char buffer[10240];
    FILE *dataFile = fopen(filename, "r");
    if (dataFile == NULL) {
         printf("Failed to open Histogram file.");
         return -1;
    }
    char* line;
    char* value;
    int col;
    int row = 0;
    while ((line = fgets(buffer, sizeof(buffer), dataFile)) != NULL) {
        // ignore the first row, which is a header.
        if (row > 0) {
            col = 0;
            value = strtok(line, ",");
            while (value != NULL) {
                // ignore first column, which is a header
                if (col > 0) {
                    grid[((row-1)+bloat) * (histSize+windSize-1) + ((col-1)+bloat)] = atol(value);
                }
                value = strtok(NULL, ",");
                col++;
            }
        }
        row++;
    }
    return 1;
}

int main(int argc, char **argv) {

    if (argc != 3) {
        printf("Incorrect number of arguments: %d\n", argc);
        return -1;
    }

    int histSize;
    int windSize;
    sscanf(argv[1], "%d", &histSize);
    sscanf(argv[2], "%d", &windSize);

    // window size must be odd.
    if (windSize % 2 == 0) windSize++;

    // initialise the grid
    long* grid = (long*) malloc( (histSize+windSize-1)*(histSize+windSize-1) * sizeof(long));
    long* grid2 = (long*) malloc((histSize+windSize-1)*(histSize+windSize-1) * sizeof(long));
    for (int i = 0; i < (histSize+windSize-1)*(histSize+windSize-1); i++) {
        grid[i] = 0;
        grid2[i] = 0;
    }

    double binSize = 1.0 / histSize;

    // readBinaryFile("points_noise_normal.bin", grid, histSize);
    readHistogramCsvFile("gridHistogram-512.csv", grid, histSize, windSize);

    printf("-----\n");
    printf("%lu %lu\n", grid[200000], grid[268314]);
    printf("-----\n");

    /*
    // allocate histograms to device memory
    long* d_histIn  = NULL;
    long* d_histOut = NULL;
    cudaMalloc(&d_histIn,  histSize * histSize * sizeof(long));
    cudaMalloc(&d_histOut, histSize * histSize * sizeof(long));

    // copy memory into device histograms
    cudaMemcpy(d_histIn, grid, histSize * histSize * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_histOut, grid2, histSize * histSize * sizeof(long), cudaMemcpyHostToDevice);

    dim3 dimBlock = dim3(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 dimGrid = dim3(
        ceil(histSize / (float) dimBlock.x),
        ceil(histSize / (float) dimBlock.y),
        1
    );

    printf("calling kernel...\n");
    //medianFilterGPU<<<dimGrid, dimBlock, BLOCKSIZE*BLOCKSIZE*windSize*windSize*sizeof(long)>>>(d_histIn, d_histOut, histSize, windSize);
    // medianFilterGPU<<<dimGrid, dimBlock>>>(d_histIn, d_histOut, histSize, windSize);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // cudaDeviceSynchronize();

    printf("completed kernel call.\n");
    cudaMemcpy(grid2, d_histOut, histSize * histSize * sizeof(long), cudaMemcpyDeviceToHost);

    printf("--------\n");
    printf("%lu %lu\n", grid[200], grid[201]);
    printf("%lu %lu\n", grid2[200], grid2[201]);
    printf("--------\n");

    // write results to csv file
    printf("generating output...\n");
    FILE *f = fopen("output.csv", "w");
    if (f == NULL) {
        return -1;
    }

    // print column headers
    for (int x = 0; x < histSize-1; x++) {
        fprintf(f, "%f,", binSize * x);
    }
    fprintf(f, "%f\n", binSize * (histSize-1));
    // print the columns
    for (int y = 0; y < histSize; y++) {
        fprintf(f, "%f", binSize * y);
        for (int x = 0; x < histSize-1; x++) {
            fprintf(f, "%lu,", grid2[y * histSize + x]);
        }
        fprintf(f, "%lu\n", grid2[y * histSize + histSize-1]);
    }
    fclose(f);
    printf("generated output\n");
    */

    free(grid);
    free(grid2);


}