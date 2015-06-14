#include <stdio.h>
#include <stdlib.h>

int cmpfunction(const void* a, const void* b) {
    return ( *(int*)a - *(int*)b );
}

int sort(long* values, int start, int stop) {
    int bSize = sizeof(long);
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

int getMedian(long* values, int size) {
    int middle = size / 2;
    int start = 0;
    int stop = size-1;
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

int main(int argc, char **argv) {

    if (argc != 3) {
        printf("Incorrect number of arguments: %d\n", argc);
        return -1;
    }

    int gridSize;
    int windSize;
    sscanf(argv[1], "%d", &gridSize);
    sscanf(argv[2], "%d", &windSize);

    // window size must be odd.
    if (windSize % 2 == 0) windSize++;

    // initialise the grid
    long* grid = (long*) malloc(gridSize * gridSize * sizeof(long));
    long* grid2 = (long*) malloc(gridSize * gridSize * sizeof(long));
    for (int i = 0; i < gridSize * gridSize; i++) {
        grid[i] = 0;
        grid2[i] = 0;
    }

    double binSize = 1.0 / gridSize;

    // read the binary file and perform binning
    FILE *dataFile = fopen("points_noise_normal.bin", "rb");
    if (!dataFile) {
        printf("Unable to open data file.");
        return -1;
    }
    int count = 0;
    while(!feof(dataFile) && count < 1000000) {
        float x;
        float y;
        fread(&x, sizeof(float), 1, dataFile);
        fread(&y, sizeof(float), 1, dataFile);
        // get bins
        int xpos = (int) (x / binSize);
        int ypos = (int) (y / binSize);
        //
        grid[ypos * gridSize + xpos] += 1;
        count++;
    }
    fclose(dataFile);

    // initialize the window
    long* window = (long*) malloc(windSize * windSize * sizeof(long));

    // perform smoothing
    for (int y = 0; y < gridSize; y++) {
        for (int x = 0; x < gridSize; x++) {

            int w_idx = 0;
            for (int dy = -windSize / 2; dy <= windSize / 2; dy++) {
                for (int dx = -windSize / 2; dx <= windSize / 2; dx++) {
                    // gather the values in the window
                    int gy, gx;
                    if (y + dy < 0) {gy = 0;}
                    else if (y + dy > gridSize - 1) {gy = gridSize - 1;}
                    else {gy = y + dy;}
                    if (x + dx < 0) {gx = 0;}
                    else if (x + dx > gridSize - 1) {gx = gridSize - 1;}
                    else {gx = x + dx;}
                    window[w_idx] = grid[gy * gridSize + gx];
                    //printf("(%d, %d): %lu, %lu\n", gx, gy, window[w_idx], grid[gy*gridSize + gx]);
                    w_idx++;
                }
            }

            long median = getMedian(window, windSize * windSize);
            grid2[y * gridSize + x] = median;

        }
    }

    free(window);


    // write results to csv file
    FILE *f = fopen("output.csv", "w");
    if (f == NULL) {
        return -1;
    }

    // print column headers
    for (int x = 0; x < gridSize-1; x++) {
        fprintf(f, "%f,", binSize * x);
    }
    fprintf(f, "%f\n", binSize * (gridSize-1));
    // print the columns
    for (int y = 0; y < gridSize; y++) {
        fprintf(f, "%f", binSize * y);
        for (int x = 0; x < gridSize-1; x++) {
            fprintf(f, "%lu,", grid2[y * gridSize + x]);
        }
        fprintf(f, "%lu\n", grid2[y * gridSize + gridSize-1]);
    }
    fclose(f);

    free(grid);
    free(grid2);

}
