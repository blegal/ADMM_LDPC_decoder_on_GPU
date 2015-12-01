/*
 *  ldcp_decoder.h
 *  ldpc3
 *
 *  Created by legal on 02/04/11.
 *  Copyright 2011 ENSEIRB. All rights reserved.
 *
 */

/*----------------------------------------------------------------------------*/

#include "../custom/custom_cuda.h"

#define NUMBER_OF_FRAMES     1

#if 0
	#define NOEUD       4000
	#define PARITE      2000
	#define MESSAGES    12000
#else
	#define NOEUD       2640
	#define PARITE      1320
	#define MESSAGES    7920
#endif

//#define nb_Node         NOEUD
//#define nb_Check       PARITE
//#define nb_Msg       MESSAGES

//#define sizeNode  (NUMBER_OF_FRAMES * nb_Node)
//#define sizeCheck (NUMBER_OF_FRAMES * nb_Check)
//#define sizeMsgs  (NUMBER_OF_FRAMES * nb_Msg)


class ADMM_GPU_Decoder{

private:
    float* h_iLLR;
    float* d_iLLR;

    float* d_oLLR;

    int*   h_hDecision;
    int*   d_hDecision;

    unsigned int* d_t_row;
    unsigned int* d_t_col;//1;

    float* LZr;

    unsigned int* d_degVNs;
    unsigned int* d_degCNs;

    unsigned int frames;
    unsigned int VNs_per_frame;
    unsigned int CNs_per_frame;
    unsigned int MSGs_per_frame;
    unsigned int VNs_per_load;
    unsigned int CNs_per_load;
    unsigned int MSGs_per_load;

public:
    ADMM_GPU_Decoder( int frames );
    ~ADMM_GPU_Decoder();
    void initialize();
    void decode(float* llrs, int* bits, int nb_iters);
};

