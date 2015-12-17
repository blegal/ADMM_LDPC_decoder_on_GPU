################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/cpp_src/CFirFilter/CFirFilter.cpp 

OBJS += \
./src/cpp_src/CFirFilter/CFirFilter.o 

CPP_DEPS += \
./src/cpp_src/CFirFilter/CFirFilter.d 


# Each subdirectory must supply rules for building sources it contributes
src/cpp_src/CFirFilter/%.o: ../src/cpp_src/CFirFilter/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-7.0/bin/nvcc -O3 --use_fast_math -v -gencode arch=compute_30,code=sm_30  -odir "src/cpp_src/CFirFilter" -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-7.0/bin/nvcc -O3 --use_fast_math -v --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


