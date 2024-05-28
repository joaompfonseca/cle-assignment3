/**
 *  \file cudaBitonicCols.cu (implementation file)
 *
 *  \brief Assignment 3.2: cuda-based bitonic sort by columns.
 *
 *  This file contains the definition of the cuda-based bitonic sort by columns algorithm.
 *
 *  \author João Fonseca
 *  \author Rafael Gonçalves
 */

#include <assert.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common.h"
#include "const.h"

/**
 *  \brief Prints the usage of the program.
 *
 *  \param cmd_name name of the program file
 */
void printUsage(char *cmd_name) {
    fprintf(stderr,
            "Usage: %s REQUIRED OPTIONAL\n"
            "REQUIRED\n"
            "-f input_file_path     : input file with numbers\n"
            "OPTIONAL\n"
            "-k number_of_threads   : number of threads per block (range 1 to 1024, must be power of 2, default is 1024)"
            "-h                     : shows how to use the program\n",
            cmd_name);
}

/**
 *  \brief Gets the time elapsed since the last call to this function.
 *
 *  \return time elapsed in seconds
 */
static double get_delta_time(void) {
    static struct timespec t0, t1;

    // t0 is assigned the value of t1 from the previous call. if there is no previous call, t0 = t1 = 0
    t0 = t1;

    if (clock_gettime(CLOCK_MONOTONIC, &t1) != 0) {
        fprintf(stderr, "[TIME] Could not get the time\n");
        exit(EXIT_FAILURE);
    }
    return (double)(t1.tv_sec - t0.tv_sec) + 1.0e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
}

/**
 *  \brief CUDA kernel to perform bitonic sort by columns.
 *
 *  \param d_arr array to be sorted
 *  \param size number of elements in the array
 *  \param direction 0 for descending order, 1 for ascending order
 *  \param k number of threads per block
 */
__global__ void bitonic_sort_gpu(int *d_arr, int size, int direction, int k) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = blockDim.x * gridDim.x * y + x;
    
    int iter = 0;

    /* divide the array into k parts
       make each thread bitonic sort one part */

    int sub_size = size / k;
    int sub_direction = (idx % 2 == 0) == direction;

    // for each sub sort size in the sub array
    for (int sort_size = 2; sort_size <= sub_size; sort_size *= 2) {
        // for all the sub sorts needed in the sub array
        for (int s = 0; s < sub_size; s += sort_size) {
            int sort_direction = ((s / sort_size) % 2 == 0) == sub_direction;
            // for each sub merge size in the sub sort array
            for (int merge_size = sort_size; merge_size >= 2; merge_size /= 2) {
                int half = merge_size / 2;
                // for all the sub merges needed in the sub sort array
                for (int m = 0; m < sort_size; m += merge_size) {
                    // move the numbers to the correct half
                    for (int i = 0; i < half; i++) {
                        int i1 = N_COLS / k * (1 << iter) * idx + N_COLS * ((s + m + i) % N_COLS) + ((s + m + i) / N_COLS);
                        int i2 = N_COLS / k * (1 << iter) * idx + N_COLS * ((s + m + i + half) % N_COLS) + ((s + m + i + half) / N_COLS);
                        if (sort_direction == (d_arr[i1] > d_arr[i2])) {
                            int temp = d_arr[i1];
                            d_arr[i1] = d_arr[i2];
                            d_arr[i2] = temp;
                        }
                    }
                }
            }
        }
    }

    __syncthreads(); // wait for all threads in the block to finish

    /* perform a bitonic merge of the sorted parts
       make each thread bitonic merge one pair of parts */
    
    for (sub_size *= 2; sub_size <= size; sub_size *= 2) {

        iter++;

        // terminate threads no longer involved
        if (idx >= (k >> iter)) return;

        sub_size = size / (k >> iter);
        sub_direction = (idx % 2 == 0) == direction;
        
        // for each sub merge size in the sub array
        for (int merge_size = sub_size; merge_size >= 2; merge_size /= 2) {
            int half = merge_size / 2;
            // for all the sub merges needed in the sub array
            for (int m = 0; m < sub_size; m += merge_size) {
                // move the numbers to the correct half
                for (int i = 0; i < half; i++) {
                    int i1 = N_COLS / k * (1 << iter) * idx + N_COLS * ((m + i) % N_COLS) + ((m + i) / N_COLS);
                    int i2 = N_COLS / k * (1 << iter) * idx + N_COLS * ((m + i + half) % N_COLS) + ((m + i + half) / N_COLS);
                    if (sub_direction == (d_arr[i1] > d_arr[i2])) {
                        int temp = d_arr[i1];
                        d_arr[i1] = d_arr[i2];
                        d_arr[i2] = temp;
                    }
                }
            }
        }

        __syncthreads(); // wait for all threads in the block to finish
    }
}

/**
 *  \brief Main function of the program.
 *
 *  Lifecycle:
 *  - process program arguments
 *  - read the array from the file
 *  - set up the device
 *  - reserve memory for the array on the device
 *  - copy the array to the device
 *  - run the computational kernel
 *  - copy the result back to the host
 *  - free memory
 *  - reset the device
 *  - check if the array is sorted
 *
 *  \param argc number of command line arguments
 *  \param argv array of command line arguments
 *
 *  \return EXIT_SUCCESS if the array is sorted, EXIT_FAILURE otherwise
 */
int main(int argc, char *argv[]) {
    // program arguments
    char *cmd_name = argv[0];
    char *file_path = NULL;

    int direction = DESCENDING;
    int *h_arr = NULL, size;
    int k = N_THREADS;

    // process program arguments
    int opt;
    do {
        switch ((opt = getopt(argc, argv, "f:k:h"))) {
            case 'f':
                file_path = optarg;
                if (file_path == NULL) {
                    fprintf(stderr, "Invalid file name (-f)\n");
                    printUsage(cmd_name);
                    return EXIT_FAILURE;
                }
                break;
            case 'k':
                k = atoi(optarg);
                if (k < 0 || k > 1024 || (k & (k - 1)) != 0) {
                    fprintf(stderr, "Invalid number of threads per block (-k)\n");
                    printUsage(cmd_name);
                    return EXIT_FAILURE;
                }
                break;
            case 'h':
                printUsage(cmd_name);
                return EXIT_FAILURE;
            case '?':
                fprintf(stderr, "Invalid option -%c\n", optopt);
                printUsage(cmd_name);
                return EXIT_FAILURE;
            case -1:
                break;
        }
    } while (opt != -1);
    if (file_path == NULL) {
        fprintf(stderr, "Input file not specified\n");
        printUsage(cmd_name);
        return EXIT_FAILURE;
    }
    
    // open the file
    FILE *file = fopen(file_path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Could not open file %s\n", file_path);
        return EXIT_FAILURE;
    }
    // read the size of the array
    if (fread(&size, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Could not read the size of the array\n");
        fclose(file);
        return EXIT_FAILURE;
    }
    // size must be 1M (1,048,576 elements)
    if (size != (1 << 20)) {
        fprintf(stderr, "The size of the array must be 1M (1,048,576 elements)");
        fclose(file);
        return EXIT_FAILURE;
    }
    // allocate memory for the array
    h_arr = (int *)malloc(size * sizeof(int));
    if (h_arr == NULL) {
        fprintf(stderr, "Could not allocate memory for the array\n");
        fclose(file);
        return EXIT_FAILURE;
    }
    // load array into memory
    int num, ni = 0;
    while (fread(&num, sizeof(int), 1, file) == 1 && ni < size) {
        h_arr[ni++] = num;
    }
    // close the file
    fclose(file);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaSetDevice(dev));
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    // print program configuration
    fprintf(stdout, "--- CONFIGURATION\n");
    fprintf(stdout, "%-24s : %s\n", "Input file", file_path);
    fprintf(stdout, "%-24s : %d\n", "Threads per block", k);
    fprintf(stdout, "%-24s : %d\n", "Array size", size);
    fprintf(stdout, "%-24s : %d - %s\n", "Using device", dev, deviceProp.name);

    fprintf(stdout, "--- MEASURING TIMES\n");

    // reserve memory for gpu
    int *d_arr;
    CHECK(cudaMalloc((void **)&d_arr, size * sizeof(int)));

    // copy array to gpu
    get_delta_time();
    CHECK(cudaMemcpy(d_arr, h_arr, size * sizeof(int), cudaMemcpyHostToDevice));
    fprintf(stdout, "Transfer from host to device (%ld bytes) : %.3e seconds\n",
           size * sizeof(int), get_delta_time());

    // run the computational kernel
    unsigned int gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ;

    blockDimX = k;
    blockDimY = 1;
    blockDimZ = 1;
    gridDimX = 1;
    gridDimY = 1;
    gridDimZ = 1;

    dim3 grid(gridDimX, gridDimY, gridDimZ);
    dim3 block(blockDimX, blockDimY, blockDimZ);

    // START TIME
    get_delta_time();

    bitonic_sort_gpu<<<grid, block>>>(d_arr, size, direction, k);
    CHECK(cudaDeviceSynchronize ());
    CHECK(cudaGetLastError ());  

    // END TIME
    fprintf(stdout, "CUDA kernel <<<(%d,%d,%d), (%d,%d,%d)>>> executed : %.3e seconds\n",
           gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, get_delta_time());

    // copy kernel result back to host side
    CHECK(cudaMemcpy(h_arr, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost));
    fprintf(stdout, "Transfer from device to host (%ld bytes) : %.3e seconds\n",
           size * sizeof(int), get_delta_time());

    // free memory
    CHECK(cudaFree(d_arr));

    // reset device
    CHECK(cudaDeviceReset());

    fprintf(stdout, "--- CHECKING IF ARRAY IS SORTED\n");

    // check if the array is sorted
    for (int i = 0; i < size - 1; i++) {
        int i1 = N_COLS * (i % N_COLS) + (i / N_COLS);
        int i2 = N_COLS * ((i + 1) % N_COLS) + ((i + 1) / N_COLS);
        if ((h_arr[i1] < h_arr[i2] && direction == DESCENDING) || (h_arr[i1] > h_arr[i2] && direction == ASCENDING)) {
            fprintf(stderr, "Error in position %d between element %d and %d\n", i, h_arr[i1], h_arr[i2]);
            free(h_arr);
            return EXIT_FAILURE;
        }
    }

    fprintf(stdout, "The array is sorted, everything is OK! :)\n");

    return EXIT_SUCCESS;
}
