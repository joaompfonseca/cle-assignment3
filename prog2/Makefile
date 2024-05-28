.DEFAULT := compile

compile:
	@echo "Compiling..."
	nvcc -O2 -Wno-deprecated-gpu-targets -o prog2 cudaBitonicCols.cu

test: compile
	@echo "Testing..."
	./prog2 -f data/datSeq1M.bin -k 8

clean:
	@echo "Cleaning..."
	rm -f prog2