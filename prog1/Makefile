.DEFAULT := compile

compile:
	@echo "Compiling..."
	nvcc -O2 -Wno-deprecated-gpu-targets -o prog1 cudaBitonicRows.cu sortUtils.c

test: compile
	@echo "Testing..."
	./prog1 -f data/datSeq1M.bin

clean:
	@echo "Cleaning..."
	rm -f prog1