################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/ldpc/ADMM_GPU_16b.cu \
../src/ldpc/ADMM_GPU_Decoder.cu \
../src/ldpc/ADMM_GPU_decoder_16b.cu 

OBJS += \
./src/ldpc/ADMM_GPU_16b.o \
./src/ldpc/ADMM_GPU_Decoder.o \
./src/ldpc/ADMM_GPU_decoder_16b.o 

CU_DEPS += \
./src/ldpc/ADMM_GPU_16b.d \
./src/ldpc/ADMM_GPU_Decoder.d \
./src/ldpc/ADMM_GPU_decoder_16b.d 


# Each subdirectory must supply rules for building sources it contributes
src/ldpc/%.o: ../src/ldpc/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -O3 -maxrregcount 32 --use_fast_math -std=c++11 -gencode arch=compute_30,code=sm_30  -odir "src/ldpc" -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -O3 -maxrregcount 32 --use_fast_math -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


