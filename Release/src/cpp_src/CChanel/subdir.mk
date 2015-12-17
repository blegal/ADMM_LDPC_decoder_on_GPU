################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/cpp_src/CChanel/CChanelAWGN2.cu 

CPP_SRCS += \
../src/cpp_src/CChanel/CChanel.cpp 

OBJS += \
./src/cpp_src/CChanel/CChanel.o \
./src/cpp_src/CChanel/CChanelAWGN2.o 

CU_DEPS += \
./src/cpp_src/CChanel/CChanelAWGN2.d 

CPP_DEPS += \
./src/cpp_src/CChanel/CChanel.d 


# Each subdirectory must supply rules for building sources it contributes
src/cpp_src/CChanel/%.o: ../src/cpp_src/CChanel/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -O3 -maxrregcount 32 --use_fast_math -std=c++11 -gencode arch=compute_30,code=sm_30  -odir "src/cpp_src/CChanel" -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -O3 -maxrregcount 32 --use_fast_math -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/cpp_src/CChanel/%.o: ../src/cpp_src/CChanel/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -O3 -maxrregcount 32 --use_fast_math -std=c++11 -gencode arch=compute_30,code=sm_30  -odir "src/cpp_src/CChanel" -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-7.5/bin/nvcc -O3 -maxrregcount 32 --use_fast_math -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


