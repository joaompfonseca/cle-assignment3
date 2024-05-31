# Usage: ./benchmark.sh
# Description: Compiles the source code, runs the cuda-based bitonic sort by columns program, and outputs the results in a
#              "results.csv" file, for each configuration of threads (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024).
# Example: ./benchmark.sh

OUTPUT_FILE="results.csv"
NUMBERS_FOLDER="data"

N_THREADS="1 2 4 8 16 32 64 128 256 512 1024"
N_ITERATIONS=15

# Create the output file
rm -f $OUTPUT_FILE
touch $OUTPUT_FILE

echo "k,t1,t2,t3" > $OUTPUT_FILE

# Compile the source code
nvcc -O2 -Wno-deprecated-gpu-targets -o bmprog2 cudaBitonicCols.cu

# Run the program for each configuration of processes and array sizes
for threads in $N_THREADS; do
  echo "Running program $N_ITERATIONS times for $threads threads..."
  # Run the program and save the results
  for i in $(seq 1 $N_ITERATIONS); do
    times=$(./bmprog2 -f $NUMBERS_FOLDER/datSeq1M.bin -k $threads | grep -Po '[0-9]+\.[0-9]+e[+-][0-9]+(?= seconds)' | paste -sd ',')
    echo "$threads,$times" >> $OUTPUT_FILE
  done
done

# Clean-up
rm -f bmprog2
