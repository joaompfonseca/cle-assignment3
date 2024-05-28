/**
 *  \file const.h (interface file)
 *
 *  \brief Assignment 3.2: cuda-based bitonic sort by columns.
 *
 *  This file contains some constants used in the program.
 *
 *  \author João Fonseca
 *  \author Rafael Gonçalves
 */

#ifndef CONST_H
#define CONST_H

/** \brief Default number of threads per block */
#define N_THREADS 1024

/** \brief Number of columns in the matrix */
#define N_COLS 1024

/** \brief Descending sort direction */
#define DESCENDING 0

/** \brief Ascending sort direction */
#define ASCENDING 1

#endif /* CONST_H */
