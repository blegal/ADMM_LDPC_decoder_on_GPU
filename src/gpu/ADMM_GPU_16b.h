/*
 * GPU_functions.h
 *
 *  Created on: 8 avr. 2013
 *      Author: legal
 */

#ifndef GPU_FUNCTIONS_16b_H_
#define GPU_FUNCTIONS_16b_H_

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

extern __global__ void ADMM_ScaleLLRs (float *LLRs, int N);

extern __global__ void ADMM_InitArrays(float *LZr,  int N);

extern __global__ void ADMM_InitArrays_16b(float *LZr,  int N);

extern __global__ void ADMM_VN_kernel_deg3(
	const float *_LogLikelihoodRatio,
	float *OutputFromDecoder,
	float *LZr,
	const unsigned int *t_row,
	int N);

extern __global__ void ADMM_VN_kernel_deg3_16b(
	const float *_LogLikelihoodRatio,
	float *OutputFromDecoder,
	float *LZr,
	const unsigned int *t_row,
	int N);

extern  __global__ void ADMM_CN_kernel_deg6(
	const float *OutputFromDecoder,
	float *LZr,
	const unsigned int *t_col1,
	int *cn_synrome,
	int N);

extern  __global__ void ADMM_CN_kernel_deg6_16b(
	const float *OutputFromDecoder,
	float *LZr,
	const unsigned int *t_col1,
	int *cn_synrome,
	int N);

extern __global__ void ADMM_VN_kernel_deg3_16b_mod(
	const float *_LogLikelihoodRatio,
	float *OutputFromDecoder,
	float *LZr,
	const unsigned int *t_row,
	int N);


extern  __global__ void ADMM_CN_kernel_deg6_16b_mod(
	const float *OutputFromDecoder,
	float *LZr,
	const unsigned int *t_col1,
	int *cn_synrome,
	int N);

extern  __global__ void ADMM_HardDecision(
				float *OutputFromDecoder,
				int   *HardDecision,
			int N);

extern __global__ void reduce(int *g_idata, unsigned int n);

#endif /* GPU_FUNCTIONS_H_ */
