################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/cpp_src/CTimer/CTimer.cpp 

OBJS += \
./src/cpp_src/CTimer/CTimer.o 

CPP_DEPS += \
./src/cpp_src/CTimer/CTimer.d 


# Each subdirectory must supply rules for building sources it contributes
src/cpp_src/CTimer/%.o: ../src/cpp_src/CTimer/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -O3 -maxrregcount 32 --use_fast_math -std=c++11 -gencode arch=compute_30,code=sm_30  -odir "src/cpp_src/CTimer" -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -O3 -maxrregcount 32 --use_fast_math -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

