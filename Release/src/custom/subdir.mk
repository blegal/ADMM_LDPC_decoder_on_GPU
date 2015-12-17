################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/custom/custom_cuda.cu 

OBJS += \
./src/custom/custom_cuda.o 

CU_DEPS += \
./src/custom/custom_cuda.d 


# Each subdirectory must supply rules for building sources it contributes
src/custom/%.o: ../src/custom/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -O3 -maxrregcount 32 --use_fast_math -std=c++11 -gencode arch=compute_30,code=sm_30  -odir "src/custom" -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -O3 -maxrregcount 32 --use_fast_math -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


