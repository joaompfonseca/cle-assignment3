# Usage: ./benchmark.sh
# Description: Compiles the source code, runs the cuda-based bitonic sort by rows program, and outputs the results in a
#              "results" folders, for each configuration of threads (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024).
# Example: ./benchmark.sh

OUTPUT_FOLDER="results"
NUMBERS_FOLDER="data"

N_THREADS="1 2 4 8 16 32 64 128 256 512 1024"
N_ITERATIONS=1

# Create the output folder
mkdir -p $OUTPUT_FOLDER
rm -f $OUTPUT_FOLDER/all.txt
touch $OUTPUT_FOLDER/all.txt

# Compile the source code
nvcc -O2 -Wno-deprecated-gpu-targets -o bmprog1 cudaBitonicRows.cu

# Run the program for each configuration of processes and array sizes
for threads in $N_THREADS; do
  echo "Running program $N_ITERATIONS times for $threads threads..."
  # Create the output file
  OUTPUT_FILE="$OUTPUT_FOLDER/t$threads.txt"
  rm -f $OUTPUT_FILE
  touch $OUTPUT_FILE
  echo '--- Results for' $threads 'threads' > $OUTPUT_FILE
  # Run the program and save the results
  for i in $(seq 1 $N_ITERATIONS); do
    ./bmprog1 -f $NUMBERS_FOLDER/datSeq1M.bin -k $threads | grep -Po '[0-9]+\.[0-9]+e[+-][0-9]+(?= seconds)' >> $OUTPUT_FILE
  done
  # Combine all the results into a single file
  cat $OUTPUT_FILE >> $OUTPUT_FOLDER/all.txt
done

# Clean-up
rm -f bmprog1
