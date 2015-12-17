################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/shared_old/GPU_Scheduled_functions.cu 

OBJS += \
./src/shared_old/GPU_Scheduled_functions.o 

CU_DEPS += \
./src/shared_old/GPU_Scheduled_functions.d 


# Each subdirectory must supply rules for building sources it contributes
src/shared_old/%.o: ../src/shared_old/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-7.0/bin/nvcc -O3 --use_fast_math -v -gencode arch=compute_30,code=sm_30  -odir "src/shared_old" -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-7.0/bin/nvcc -O3 --use_fast_math -v --compile --relocatable-device-code=false -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


