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
    // program arguments
    char *cmd_name = argv[0];
    char *file_path = NULL;

    int direction = DESCENDING;
    int *arr = NULL, size;

    // process program arguments
    int opt;
    do {
        switch ((opt = getopt(argc, argv, "f:h"))) {
            case 'f':
                file_path = optarg;
                if (file_path == NULL) {
                    fprintf(stderr, "Invalid file name\n");
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
    arr = (int *)malloc(size * sizeof(int));
    if (arr == NULL) {
        fprintf(stderr, "Could not allocate memory for the array\n");
        fclose(file);
        return EXIT_FAILURE;
    }
    // load array into memory
    int num, ni = 0;
    while (fread(&num, sizeof(int), 1, file) == 1 && ni < size) {
        arr[ni++] = num;
    }
    // close the file
    fclose(file);

    // START TIME
    get_delta_time();

    //if (size > 1) {
    //    int count = size / mpi_size;
    //
    //    // allocate memory for the sub-array
    //    int *sub_arr = (int *)malloc(count * sizeof(int));
    //    if (sub_arr == NULL) {
    //        fprintf(stderr, "[PROC-%d] Could not allocate memory for the sub-array\n", mpi_rank);
    //        if (mpi_rank == 0) free(arr);
    //        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    //    }
    //
    //    /* divide the array into mpi_size parts
    //       make each process bitonic sort one part */
    //
    //    // scatter the array into mpi_size parts
    //    MPI_Scatter(arr, count, MPI_INT, sub_arr, count, MPI_INT, 0, curr_comm);
    //
    //    // direction of the sub-sort
    //    int sub_direction = (mpi_rank % 2 == 0) == direction;
    //
    //    // make each process bitonic sort one part
    //    bitonic_sort(sub_arr, 0, count, sub_direction);
    //
    //    // gather the sorted parts
    //    MPI_Gather(sub_arr, count, MPI_INT, arr, count, MPI_INT, 0, curr_comm);
    //
    //    /* perform a bitonic merge of the sorted parts
    //       make each process bitonic merge one part */
    //
    //    for (count *= 2; count <= size; count *= 2) {
    //        int n_merge_tasks = size / count;
    //
    //        // reallocate memory for the sub-array
    //        sub_arr = (int *)realloc(sub_arr, count * sizeof(int));
    //        if (sub_arr == NULL) {
    //            fprintf(stderr, "[PROC-%d] Could not reallocate memory for the sub-array\n", mpi_rank);
    //            free(sub_arr);
    //            if (mpi_rank == 0) free(arr);
    //            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    //        }
    //
    //        // group processes involved in merge tasks
    //        MPI_Group_incl(curr_group, n_merge_tasks, group_members, &next_group);
    //        MPI_Comm_create(curr_comm, next_group, &next_comm);
    //        curr_group = next_group;
    //        curr_comm = next_comm;
    //
    //        // terminate processes not involved
    //        if (mpi_rank >= n_merge_tasks) {
    //            break;
    //        }
    //
    //        // set communicator size
    //        MPI_Comm_size(curr_comm, &n_merge_tasks);
    //
    //        if (n_merge_tasks > 1) {
    //            // scatter the array into n_merge_tasks parts
    //            MPI_Scatter(arr, count, MPI_INT, sub_arr, count, MPI_INT, 0, curr_comm);
    //
    //            // direction of the sub-merge
    //            int sub_direction = (mpi_rank % 2 == 0) == direction;
    //
    //            // make each worker process bitonic merge one part
    //            bitonic_merge(sub_arr, 0, count, sub_direction);
    //
    //            // gather the merged parts
    //            MPI_Gather(sub_arr, count, MPI_INT, arr, count, MPI_INT, 0, curr_comm);
    //        }
    //        else {
    //            // direction of the sub-merge
    //            int sub_direction = (mpi_rank % 2 == 0) == direction;
    //
    //            // make each worker process bitonic merge one part
    //            bitonic_merge(arr, 0, count, sub_direction);
    //        }
    //    }
    //
    //    free(sub_arr);
    //}

    // END TIME
    fprintf(stdout, "%-16s : %.9f seconds\n", "Time elapsed", get_delta_time());

    // check if the array is sorted
    for (int i = 0; i < size - 1; i++) {
        if ((arr[i] < arr[i + 1] && direction == DESCENDING) || (arr[i] > arr[i + 1] && direction == ASCENDING)) {
            fprintf(stderr, "Error in position %d between element %d and %d\n", i, arr[i], arr[i + 1]);
            free(arr);
            return EXIT_FAILURE;
        }
    }
    fprintf(stdout, "The array is sorted, everything is OK! :)\n");

    free(arr);

    return EXIT_SUCCESS;
}
