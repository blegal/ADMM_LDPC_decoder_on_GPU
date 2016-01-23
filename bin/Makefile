CC=nvcc
CFLAGS=-O3 -maxrregcount 32  --compile --use_fast_math -std=c++11 -gencode arch=compute_30,code=sm_30
LDFLAGS=--cudart static -std=c++11 --relocatable-device-code=false -gencode arch=compute_30,code=sm_30 -lcurand

EXEC=main

SRC= \
 ../src/ldpc/ADMM_GPU_16b.cu \
 ../src/ldpc/ADMM_GPU_Decoder.cu \
 ../src/ldpc/ADMM_GPU_decoder_16b.cu \
 ../src/gpu/ADMM_GPU_16b.cu \
 ../src/gpu/ADMM_GPU_32b.cu \
 ../src/gpu/ADMM_shared.cu \
 ../src/custom/custom_cuda.cu \
 ../src/cpp_src/CTrame/CTrame.cu \
 ../src/cpp_src/CTerminal/CTerminal.cu \
 ../src/cpp_src/CErrorAnalyzer/CErrorAnalyzer.cu \
 ../src/cpp_src/CEncoder/CFakeEncoder.cu \
 ../src/cpp_src/CEncoder/Encoder.cu \
 ../src/cpp_src/CChanel/CChanel.cu \
 ../src/cpp_src/CChanel/CChanelAWGN2.cu  \
 ../src/cpp_src/CTimer/CTimer.cu  \
 ../src/main.cu

OBJ= $(SRC:.cu=.o)

all: $(EXEC)

main: $(OBJ)
	@echo "[LINKING] $@"
	$(CC) -link -o $@ $^ $(LDFLAGS)

%.o: %.cu
	@echo "[C++] $<"
	@$(CC) $(CFLAGS) -o $@ -c $<

%.o: %.cu
	@echo "[C++] $<"
	@$(CC) $(CFLAGS) -o $@ -c $<

.PHONY: clean mrproper

clean:
	find ../src/ -name "*.o" -exec rm {} \;

mrproper: clean
	rm $(EXEC)
