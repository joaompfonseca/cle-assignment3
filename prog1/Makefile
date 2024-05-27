.DEFAULT := compile

compile:
	@echo "Compiling..."
	nvcc -O2 -Wno-deprecated-gpu-targets -o prog1 cudaBitonicRows.cu

test: compile
	@echo "Testing..."
	./prog1 -f data/datSeq1M.bin -k 8

clean:
	@echo "Cleaning..."
	rm -f prog1