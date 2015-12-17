/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 3
 * of the programming guide with some additions like error checking.
 *
 */

#include <stdio.h>
#include <cuda_fp16.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void ADMM_ScaleLLRs(float* LLRs, int N)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
    	const float mu = 3.0f;
    	LLRs[i] = LLRs[i] / mu;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void ADMM_HardDecision(
		float* OutputFromDecoder, int* HardDecision, int N
		)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        HardDecision[i] = floorf(OutputFromDecoder[i] + 0.50f);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__shared__ int sdata[128*6]; // > 512

__global__ void reduce(int *g_idata, unsigned int n)
{
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid      =                               threadIdx.x;
    unsigned int i        = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;

    int mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];
        // ensure we don't read out of bounds
        if (i + blockDim.x < n)
            mySum += g_idata[i+blockDim.x];
        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (blockDim.x >= 1024) { if (tid < 512) { sdata[tid] = mySum = mySum + sdata[tid + 512]; } __syncthreads(); }
    if (blockDim.x >=  512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockDim.x >=  256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockDim.x >=  128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }

    // avoid bank conflict
    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile int* smem = sdata;
        if (blockDim.x >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }
        if (blockDim.x >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
        if (blockDim.x >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
        if (blockDim.x >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
        if (blockDim.x >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
        if (blockDim.x >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
    }

    // write result for this block to global mem
    if (tid == 0)
    	g_idata[blockIdx.x] = sdata[0];
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
