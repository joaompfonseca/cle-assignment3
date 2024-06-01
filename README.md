# CLE Assignment 3

## Description

The assignment goal was to develop a cuda-based implementation of the second general problem given in the practical classes, the bitonic sort of an integer array. We considered the array integers to be stored in a 1024 by 1024 matrix. In the first program `prog1`, we sort the array by the rows of the matrix, and in the second program `prog2`, we sort the array by the columns. Sorting by rows was more efficient, since we are accessing contiguous integers of the array more often than by columns, where continuous integers in the array are stored 1024 positions apart.

**Course:** Large Scale Computing (2023/2024).

## 1. Bitonic sort by rows with CUDA

### Compile and execute

- Run `cd prog1` in root to change to the program's directory.
- Run `make` to compile the program.
- Run `./prog1 REQUIRED OPTIONAL` to execute the program.

### Required arguments

- `-f input_file_path`: path to the input file with numbers (string).

### Optional arguments

- `-k number_of_threads`: number of threads per block (range 1 to 1024, must be power of 2, default is 1024).
- `-h`: shows how to use the program.

### Example

`./prog1 -f data/datSeq1M.bin -k 8`

## 2. Bitonic sort by columns with CUDA

### Compile and execute

- Run `cd prog2` in root to change to the program's directory.
- Run `make` to compile the program.
- Run `./prog2 REQUIRED OPTIONAL` to execute the program.

### Required arguments

- `-f input_file_path`: path to the input file with numbers (string).

### Optional arguments

- `-k number_of_threads`: number of threads per block (range 1 to 1024, must be power of 2, default is 1024).
- `-h`: shows how to use the program.

### Example

`./prog2 -f data/datSeq1M.bin -k 8`

## Authors

- João Fonseca, 103154
- Rafael Gonçalves, 102534
