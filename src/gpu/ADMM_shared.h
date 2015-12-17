/*
 * GPU_functions.h
 *
 *  Created on: 8 avr. 2013
 *      Author: legal
 */

#ifndef GPU_FUNCTIONS_H_
#define GPU_FUNCTIONS_H_

// Includes
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cstring>

// includes, project
// includes, CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>

using namespace std;

extern __global__ void ADMM_ScaleLLRs     (float *LLRs, int N);

extern  __global__ void ADMM_HardDecision(
				float *OutputFromDecoder,
				int   *HardDecision,
			int N);

extern __global__ void reduce(int *g_idata, unsigned int n);

#endif /* GPU_FUNCTIONS_H_ */
