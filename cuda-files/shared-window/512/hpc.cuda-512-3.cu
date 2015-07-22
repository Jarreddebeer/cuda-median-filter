#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define BLOCKSIZE 4

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__device__ int sort(int values[], int start, int stop) {
    int pivot = values[start];
    int i = start;
    int j = stop+1;
    while (1) {
        while (values[++i] < pivot) if (i == stop) break;
        while (values[--j] >= pivot) if (j == start) break;
        if (i >= j) break;
        int tmp = values[i];
        values[i] = values[j];
        values[j] = tmp;
    }
    // place the pivot back
    int tmp = values[j];
    values[j] = pivot;
    values[start] = tmp;

    return j;
}

__device__ int getMedian(int values[], int start, int end) {
    // int start = 0;
    // int stop = size - 1;
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

__global__ void medianFilterGPU(int* d_in, int* d_out, int histSize, int windSize) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockDim.x * blockIdx.x + tx;
    int y = blockDim.y * blockIdx.y + ty;

    __shared__ int v[4*4*21*21];

    if (x >= histSize && y >= histSize) return;

    int x_left = x - (windSize - 1) / 2;
    int x_right = x + (windSize - 1) / 2;
    int y_top = y - (windSize - 1) / 2;
    int y_bot = y + (windSize - 1) / 2;

    if (x_left < 0) x_left = 0;
    if (y_top < 0)  y_top = 0;
    if (x_right >= histSize) x_right = histSize - 1;
    if (y_bot   >= histSize) y_bot   = histSize - 1;

    int idx = (blockDim.x * blockIdx.y + blockIdx.x) * 21 * 21;
    int pos = 0;
    for (int i = y_top; i <= y_bot; i++) {
        for (int j = x_left; j <= x_right; j++) {
            v[idx + pos] = d_in[blockDim.x * i + j];
            pos++;
        }
    }

    pos--;
    int med = getMedian(v, idx, idx + pos);
    d_out[y * histSize + x] = med;

}

// read the binary file and perform binning
int readBinaryFile(const char* filename, int* grid, int histSize, int windSize) {
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

int outputResultsToFile(const char* filename, int* grid2, int histSize) {

    double binSize = 1.0 / histSize;

    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        return -1;
    }

    // print column bucket headers
    fprintf(f, ",");
    for (int x = 0; x < histSize; x++) {
        float val = binSize * x;
        if (x < histSize-1) fprintf(f, "%f,",  val);
        else                fprintf(f, "%f\n", val);
    }

    // print each row
    for (int y = 0; y < histSize; y++) {
        // first column is a bucket
        fprintf(f, "%f,", binSize * y);
        // values
        for (int x = 0; x < histSize; x++) {
            int val = grid2[y * histSize + x];
            if (x < histSize-1) fprintf(f, "%lu,",  val);
            else                fprintf(f, "%lu\n", val);
        }
    }
    fclose(f);
    return 1;
}

// read the already written CSV histogram
int readHistogramCsvFile(const char* filename, int* grid, int histSize, int windSize) {
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
    int* grid = (int*) malloc( (histSize+windSize-1)*(histSize+windSize-1) * sizeof(int));
    int* grid2 = (int*) malloc(histSize * histSize * sizeof(int));
    for (int i = 0; i < (histSize+windSize-1)*(histSize+windSize-1); i++) {
        grid[i] = 0;
    }

    // readBinaryFile("points_noise_normal.bin", grid, histSize);
    readHistogramCsvFile("grid-512.csv", grid, histSize, windSize);

    // allocate histograms to device memory
    int* d_histIn  = NULL;
    int* d_histOut = NULL;
    cudaMalloc(&d_histIn,  (histSize+windSize-1) * (histSize+windSize-1) * sizeof(int));
    cudaMalloc(&d_histOut, histSize * histSize * sizeof(int));

    // copy memory into device histograms
    cudaMemcpy(d_histIn, grid, (histSize+windSize-1) * (histSize+windSize-1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_histOut, grid2, histSize * histSize * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock = dim3(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 dimGrid = dim3(
        (int) ceil(histSize / (float) dimBlock.x),
        (int) ceil(histSize / (float) dimBlock.y),
        1
    );

    printf("calling kernel...\n");
    medianFilterGPU<<<dimGrid, dimBlock>>>(d_histIn, d_histOut, histSize, windSize);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    printf("completed kernel call.\n");
    cudaMemcpy(grid2, d_histOut, histSize * histSize * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_histIn);
    cudaFree(d_histOut);

    // write results to csv file
    printf("generating output...\n");
    outputResultsToFile("output.csv", grid2, histSize);
    printf("generated output\n");

    free(grid);
    free(grid2);

}
