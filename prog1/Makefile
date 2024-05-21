.DEFAULT := compile

compile:
	@echo "Compiling..."
	nvcc -O2 -Wno-deprecated-gpu-targets -rdc=true -o prog1 cudaBitonicRows.cu sortUtils.cu

test: compile
	@echo "Testing..."
	./prog1 -k 1024 -f data/datSeq1M.bin

clean:
	@echo "Cleaning..."
	rm -f prog1