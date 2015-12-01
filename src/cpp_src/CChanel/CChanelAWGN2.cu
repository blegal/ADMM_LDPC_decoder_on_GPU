#include "CChanelAWGN2.h"

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);     \
      exit(0);}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
      printf("Error (%d) at %s:%d\n", x, __FILE__,__LINE__);            \
      exit(0);}} while(0)


__global__ void vectNoise(const int *IN, const float *A, const float *B, float *C, float SigB, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
    {
        float x  = sqrt(-2.0 * log( A[i] ));
        float y  = B[i];
        float Ph = x * sin(_2pi * y);
        float Qu = x * cos(_2pi * y);
#if 0
        C[i]     = ((float)(2 * IN[i]  ) + Ph * SigB) * (2.0f / (SigB * SigB));
        C[i+N]   = ((float)(2 * IN[i+N]) + Qu * SigB) * (2.0f / (SigB * SigB));
#else
        const float s2 = (2.0f * SigB * SigB);
        const float b1 = (float)(2 * IN[i]  ) - 1.0f + Ph * SigB;
        const float b2 = (float)(2 * IN[i+N]) - 1.0f + Qu * SigB;
        C[i]     = (( (b1 - 1.0f) * (b1 - 1.0f) ) - ( (b1 + 1.0f) * (b1 + 1.0f) )) / s2;
        C[i+N]   = (( (b2 - 1.0f) * (b2 - 1.0f) ) - ( (b2 + 1.0f) * (b2 + 1.0f) )) / s2;
#endif
    }
}


CChanelAWGN2::CChanelAWGN2(CTrame *t, int _BITS_LLR, bool QPSK, bool Es_N0) : CChanel(t, _BITS_LLR, QPSK, Es_N0){

	curandStatus_t Status;
	Status = curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
	CURAND_CALL(Status);

    Status = curandSetPseudoRandomGeneratorSeed(generator, 1234ULL);
	CURAND_CALL(Status);
	CUDA_MALLOC_DEVICE(&d_IN,     _data);
	CUDA_MALLOC_DEVICE(&device_A, _data);
    CUDA_MALLOC_DEVICE(&device_B, _data);
    CUDA_MALLOC_DEVICE(&device_R, _data);
}

CChanelAWGN2::~CChanelAWGN2(){
	cudaError_t Status;

	Status = cudaFree(d_IN);
	ERROR_CHECK(Status, (char*)__FILE__, __LINE__);

	Status = cudaFree(device_A);
	ERROR_CHECK(Status, (char*)__FILE__, __LINE__);
	Status = cudaFree(device_B);
	ERROR_CHECK(Status, (char*)__FILE__, __LINE__);
	Status = cudaFree(device_R);
	ERROR_CHECK(Status, (char*)__FILE__, __LINE__);

	curandStatus_t eStatus;
    eStatus = curandDestroyGenerator(generator);
	CURAND_CALL(eStatus);
}

void CChanelAWGN2::configure(double _Eb_N0) {
    rendement = (float) (_vars) / (float) (_data);
    if (es_n0) {
        Eb_N0 = _Eb_N0 - 10.0 * log10(2 * rendement);
    } else {
        Eb_N0 = _Eb_N0;
    }
    double interm = 10.0 * log10(rendement);
    interm        = -0.1*((double)Eb_N0+interm);
    SigB          = sqrt(pow(10.0,interm)/2);
}

#include <limits.h>
#define MAX_RANDOM LONG_MAX    /* Maximum value of random() */


double CChanelAWGN2::awgn(double amp)
{
    return 0.00;
}

#define QPSK 0.707106781
#define BPSK 1.0


void CChanelAWGN2::generate( )
{
	curandStatus_t Status;
	cudaError_t eStatus;

	eStatus = cudaMemcpy(d_IN, t_coded_bits, _data * sizeof(int), cudaMemcpyHostToDevice);

    Status = curandGenerateUniform( generator, device_A, _data );
	CURAND_CALL(Status);

    Status = curandGenerateUniform( generator, device_B, _data );
	CURAND_CALL(Status);

	int threadsPerBlock = 256;
    int blocksPerGrid   = (_data  + threadsPerBlock - 1) / threadsPerBlock;

    vectNoise<<<blocksPerGrid, threadsPerBlock>>>(d_IN, device_A, device_B, device_R, (float)SigB, _data);

    eStatus = cudaMemcpy(t_noise_data, device_R, _data * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CALL(eStatus);
}
