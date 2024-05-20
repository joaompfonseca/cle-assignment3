/**
 *  \file cudaBitonicRows.cu (implementation file)
 *
 *  \brief Assignment 3.1: cuda-based bitonic sort by rows.
 *
 *  This file contains the definition of the cuda-based bitonic sort by rows algorithm.
 *
 *  \author João Fonseca
 *  \author Rafael Gonçalves
 */

#include <assert.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common.h"
#include "const.h"
#include "sortUtils.h"

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
 *  \brief CUDA kernel to perform bitonic sort by rows.
 *
 *  \param d_arr device array to be sorted
 *  \param size size of the array
 *  \param k number of bits to be sorted
 *
 *  \return EXIT_SUCCESS if the array is sorted, EXIT_FAILURE otherwise
 */
__global__ void bitonic_sort_gpu(int *d_arr, int size, int k, int direction) {

}


/**
 *  \brief Main function of the program.
 *
 *  Lifecycle:
 *  - ...
 *
 *  \param argc number of command line arguments
 *  \param argv array of command line arguments
 *
 *  \return EXIT_SUCCESS if the array is sorted, EXIT_FAILURE otherwise
 */
int main(int argc, char *argv[]) {

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;

    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));  // a gpu que vou utilizar

    // program arguments
    char *cmd_name = argv[0];
    char *file_path = NULL;

    int direction = DESCENDING;
    int *h_arr = NULL, size;
    int k = 0;

    // process program arguments
    int opt;
    do {
        switch ((opt = getopt(argc, argv, "k:f:h"))) {
            case 'f':
                file_path = optarg;
                if (file_path == NULL) {
                    fprintf(stderr, "Invalid file name\n");
                    printUsage(cmd_name);
                    return EXIT_FAILURE;
                }
                break;
            case 'k':
                k = atoi(optarg);
                if (k < 0) {
                    fprintf(stderr, "Invalid k value\n");
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
    // print program arguments
    fprintf(stdout, "%-16s : %s\n", "Input file", file_path);
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
    fprintf(stdout, "%-16s : %d\n", "Array size", size);
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

    // reserve memory for gpu
    int *d_arr;
    CHECK(cudaMalloc((void **)&d_arr, size * sizeof(int)));

    // copy array to gpu
    get_delta_time();
    CHECK(cudaMemcpy(d_arr, h_arr, size * sizeof(int), cudaMemcpyHostToDevice));
    printf("The transfer of %ld bytes from the host to the device took %.3e seconds\n",
           size * sizeof(int), get_delta_time());

    // run the computational kernel
    unsigned int gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ;

    blockDimX = 1024;
    blockDimY = 1024;
    blockDimZ = 1;
    gridDimX = 1;
    gridDimY = 1;
    gridDimZ = 1;

    dim3 grid(gridDimX, gridDimY, gridDimZ);
    dim3 block(blockDimX, blockDimY, blockDimZ);

    if ((gridDimX * gridDimY * gridDimZ * blockDimX * blockDimY * blockDimZ) != size) {
        fprintf(stderr, "Wrong configuration!\n");
        return EXIT_FAILURE;
    }

    get_delta_time();
    bitonic_sort_gpu<<<grid, block>>>(d_arr, size, k, direction);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("The CUDA kernel <<<(%d,%d,%d), (%d,%d,%d)>>> took %.3e seconds to run\n",
           gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, get_delta_time());

    // copy kernel result back to host side
    CHECK(cudaMemcpy(h_arr, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost));
    printf("The transfer of %ld bytes from the device to the host took %.3e seconds\n",
           size * sizeof(int), get_delta_time());

    // free memory
    CHECK(cudaFree(d_arr));

    // reset device
    CHECK(cudaDeviceReset());

    // check if the array is sorted
    for (int i = 0; i < size - 1; i++) {
        if ((h_arr[i] < h_arr[i + 1] && direction == DESCENDING) || (h_arr[i] > h_arr[i + 1] && direction == ASCENDING)) {
            fprintf(stderr, "Error in position %d between element %d and %d\n", i, h_arr[i], h_arr[i + 1]);
            free(h_arr);
            return EXIT_FAILURE;
        }
    }

    fprintf(stdout, "The array is sorted, everything is OK! :)\n");

    return EXIT_SUCCESS;
}
