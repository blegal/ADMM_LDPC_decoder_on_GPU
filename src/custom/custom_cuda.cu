#include "custom_cuda.h"

#define DEBUG 0

bool ERROR_CHECK(cudaError_t Status, const char * file, int line)
{
    if(Status != cudaSuccess)
    {
        printf("(EE) \n");
        printf("(EE) Error detected in the LDPC decoder (%s : %d)\n", file, line);
        printf("(EE) MSG: %s\n", cudaGetErrorString(Status));
        printf("(EE) \n");
        exit( 0 );
        return false;
    }
    return true;
}

void CUDA_MALLOC_HOST(float** ptr, int nbElements){
    cudaError_t Status;
    int nbytes = nbElements * sizeof(float);
    Status     = cudaMallocHost(ptr, nbytes);
#if DEBUG == 1
	printf("(II)    + Allocating   Host Memory, %d elements (%d bytes) adr [0x%8.8X, 0x%8.8X]\n", nbElements, nbytes, *ptr, *ptr+nbElements-1);
#endif
    ERROR_CHECK(Status, __FILE__, __LINE__);
}

void CUDA_MALLOC_HOST(int** ptr, int nbElements){
    cudaError_t Status;
    int nbytes = nbElements * sizeof(int);
    Status     = cudaMallocHost(ptr, nbytes);
#if DEBUG == 1
	printf("(II)    + Allocating   Host Memory, %d elements (%d bytes) adr [0x%8.8X, 0x%8.8X]\n", nbElements, nbytes, *ptr, *ptr+nbElements-1);
#endif
    ERROR_CHECK(Status, __FILE__, __LINE__);
}

void CUDA_MALLOC_HOST(unsigned int** ptr, int nbElements){
    cudaError_t Status;
    int nbytes = nbElements * sizeof(unsigned int);
    Status     = cudaMallocHost(ptr, nbytes);
#if DEBUG == 1
	printf("(II)    + Allocating   Host Memory, %d elements (%d bytes) adr [0x%8.8X, 0x%8.8X]\n", nbElements, nbytes, *ptr, *ptr+nbElements-1);
#endif
    ERROR_CHECK(Status, __FILE__, __LINE__);
}

void CUDA_MALLOC_HOST(char** ptr, int nbElements){
    cudaError_t Status;
    int nbytes = nbElements * sizeof(char);
    Status     = cudaMallocHost(ptr, nbytes);
#if DEBUG == 1
	printf("(II)    + Allocating   Host Memory, %d elements (%d bytes) adr [0x%8.8X, 0x%8.8X]\n", nbElements, nbytes, *ptr, *ptr+nbElements-1);
#endif
    ERROR_CHECK(Status, __FILE__, __LINE__);
}

void CUDA_MALLOC_DEVICE(float** ptr, int nbElements){
    cudaError_t Status;
    int nbytes = nbElements * sizeof(float);
    Status     = cudaMalloc(ptr, nbytes);
#if DEBUG == 1
	printf("(II)    + Allocating Device Memory, %d elements (%d bytes) adr [0x%8.8X, 0x%8.8X]\n", nbElements, nbytes, *ptr, *ptr+nbElements-1);
#endif
    ERROR_CHECK(Status, __FILE__, __LINE__);
}

void CUDA_MALLOC_DEVICE(int** ptr, int nbElements){
    cudaError_t Status;
    int nbytes = nbElements * sizeof(int);
    Status     = cudaMalloc(ptr, nbytes);
#if DEBUG == 1
	printf("(II)    + Allocating Device Memory, %d elements (%d bytes) adr [0x%8.8X, 0x%8.8X]\n", nbElements, nbytes, *ptr, *ptr+nbElements-1);
#endif
    ERROR_CHECK(Status, __FILE__, __LINE__);
}

void CUDA_MALLOC_DEVICE(unsigned int** ptr, int nbElements){
    cudaError_t Status;
    int nbytes = nbElements * sizeof(unsigned int);
    Status     = cudaMalloc(ptr, nbytes);
#if DEBUG == 1
	printf("(II)    + Allocating Device Memory, %d elements (%d bytes) adr [0x%8.8X, 0x%8.8X]\n", nbElements, nbytes, *ptr, *ptr+nbElements-1);
#endif
    ERROR_CHECK(Status, __FILE__, __LINE__);
}

void CUDA_MALLOC_DEVICE(char** ptr, int nbElements){
    cudaError_t Status;
    int nbytes = nbElements * sizeof(char);
    Status     = cudaMalloc(ptr, nbytes);
#if DEBUG == 1
	printf("(II)    + Allocating Device Memory, %d elements (%d bytes) adr [0x%8.8X, 0x%8.8X]\n", nbElements, nbytes, *ptr, *ptr+nbElements-1);
#endif
    ERROR_CHECK(Status, __FILE__, __LINE__);
}
