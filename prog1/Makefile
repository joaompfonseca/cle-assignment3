.DEFAULT := compile

compile:
	@echo "Compiling..."
	nvcc -O2 -Wno-deprecated-gpu-targets -rdc=true -o prog1 cudaBitonicRows.cu sortUtils.cu 2>/dev/null

test: compile
	@echo "Testing..."
	./prog1 -f data/datSeq1M.bin -k 8

clean:
	@echo "Cleaning..."
	rm -f prog1