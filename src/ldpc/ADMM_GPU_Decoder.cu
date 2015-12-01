/*
 *  ldcp_decoder.h
 *  ldpc3
 *
 *  Created by legal on 02/04/11.
 *  Copyright 2011 ENSEIRB. All rights reserved.
 *
 */

/*----------------------------------------------------------------------------*/

#include "ADMM_GPU_Decoder.h"

#include "../gpu/ADMM_GPU_functions.h"

#if 0
	#include "../codes/Constantes_4000x2000.h"
#else
	#include "./admm/admm_2640x1320.h"
#endif


ADMM_GPU_Decoder::ADMM_GPU_Decoder( int _frames )
{
    cudaError_t Status;

    frames         = _frames;
    VNs_per_frame  = NOEUD;
    CNs_per_frame  = PARITE;
    MSGs_per_frame = MESSAGES;
    VNs_per_load   = frames *  VNs_per_frame;
    CNs_per_load   = frames *  CNs_per_frame;
    MSGs_per_load  = frames * MSGs_per_frame;

    //
    // LLRs entrant dans le decodeur
    //
    CUDA_MALLOC_HOST  (&h_iLLR, VNs_per_load);
    CUDA_MALLOC_DEVICE(&d_iLLR, VNs_per_load);

    //
    // LLRs interne au decodeur
    //
    CUDA_MALLOC_DEVICE(&d_oLLR, VNs_per_load);

    //
    // LLRs (decision dure) sortant du le decodeur
    //
    CUDA_MALLOC_HOST  (&h_hDecision, VNs_per_load);
    CUDA_MALLOC_DEVICE(&d_hDecision, VNs_per_load);

    // Le tableau fournissant le degree des noeuds VNs
    CUDA_MALLOC_DEVICE(&d_degVNs, VNs_per_frame);
//    Status = cudaMemcpy(d_degVNs, t_degVN, nb_Node * sizeof(unsigned int), cudaMemcpyHostToDevice);
//    ERROR_CHECK(Status, (char*)__FILE__, __LINE__);

    // Le tableau fournissant le degree des noeuds CNs
    CUDA_MALLOC_DEVICE(&d_degCNs, CNs_per_frame);
//    Status = cudaMemcpy(d_degCNs, t_degCN, nb_Check * sizeof(unsigned int), cudaMemcpyHostToDevice);
//    ERROR_CHECK(Status, (char*)__FILE__, __LINE__);

#if 0
    CUDA_MALLOC_DEVICE(&d_t_row, nb_Msg);
    Status = cudaMemcpy(d_t_row, t_row, nb_Msg * sizeof(unsigned int), cudaMemcpyHostToDevice);
    ERROR_CHECK(Status, (char*)__FILE__, __LINE__);
#else
    CUDA_MALLOC_DEVICE(&d_t_row, MSGs_per_frame);
    Status = cudaMemcpy(d_t_row, t_row_pad_4, MSGs_per_frame * sizeof(unsigned int), cudaMemcpyHostToDevice);
    ERROR_CHECK(Status, (char*)__FILE__, __LINE__);
#endif

    CUDA_MALLOC_DEVICE(&d_t_col, MSGs_per_frame);
    Status = cudaMemcpy(d_t_col, t_col, MSGs_per_frame * sizeof(unsigned int), cudaMemcpyHostToDevice);
    ERROR_CHECK(Status, (char*)__FILE__, __LINE__);

//    cudaMemcpyToSymbol (cst_t_row, t_row, nb_Msg * sizeof(unsigned int));
//    cudaMemcpyToSymbol (cst_t_col, t_col, nb_Msg * sizeof(unsigned int));

//    CUDA_MALLOC_DEVICE(&d_MSG_C_2_V,  nb_Msg   + 512);
//    CUDA_MALLOC_DEVICE(&d_MSG_V_2_C,  nb_Msg   + 512);

    // Espace memoire pour l'échange de messages dans le decodeur
    CUDA_MALLOC_DEVICE(&LZr, 2 * MSGs_per_load);
//    exit( 0 );
}


ADMM_GPU_Decoder::~ADMM_GPU_Decoder()
{
	cudaError_t Status;
	Status = cudaFreeHost(h_iLLR);		ERROR_CHECK(Status, (char*)__FILE__, __LINE__);
	Status = cudaFree(d_iLLR);			ERROR_CHECK(Status, (char*)__FILE__, __LINE__);
	Status = cudaFree(d_oLLR);			ERROR_CHECK(Status, (char*)__FILE__, __LINE__);

	Status = cudaFreeHost(h_hDecision);	ERROR_CHECK(Status, (char*)__FILE__, __LINE__);
	Status = cudaFree(d_hDecision);		ERROR_CHECK(Status, (char*)__FILE__, __LINE__);

	Status = cudaFree(d_degCNs);		ERROR_CHECK(Status, (char*)__FILE__, __LINE__);
	Status = cudaFree(d_degVNs);		ERROR_CHECK(Status, (char*)__FILE__, __LINE__);

	Status = cudaFree(d_t_row);			ERROR_CHECK(Status, (char*)__FILE__, __LINE__);
	Status = cudaFree(d_t_col);			ERROR_CHECK(Status, (char*)__FILE__, __LINE__);

	Status = cudaFree(LZr);				ERROR_CHECK(Status, (char*)__FILE__, __LINE__);
}

void ADMM_GPU_Decoder::decode(float* llrs, int* bits, int nb_iters)
{
    cudaError_t Status;

//	for(int k=1; k<frames; k++){
//	    for(int i=0; i<VNs_per_frame; i++){
//	    	llrs[VNs_per_frame * k + i] = llrs[i];
//	    }
//	}

/*
    VNs_per_frame  = NOEUD;
    CNs_per_frame  = PARITE;
    MSGs_per_frame = MESSAGES;
    VNs_per_load   = frames *  VNs_per_frame;
    CNs_per_load   = frames *  CNs_per_frame;
    MSGs_per_load  = frames * MSGs_per_frame;
*/
	int threadsPerBlock     = 128;
    int blocksPerGridNode   = (VNs_per_load  + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridCheck  = (CNs_per_load  + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridMsgs   = (MSGs_per_load + threadsPerBlock - 1) / threadsPerBlock;

    /* On copie les donnees d'entree du decodeur */
    cudaMemcpyAsync(d_iLLR, llrs, VNs_per_load * sizeof(float), cudaMemcpyHostToDevice);

    /* INITIALISATION DU DECODEUR LDPC SUR GPU */
    ADMM_InitArrays<<<blocksPerGridMsgs, threadsPerBlock>>>(LZr, MSGs_per_load);
    ERROR_CHECK(cudaGetLastError( ), __FILE__, __LINE__);

    ADMM_ScaleLLRs<<<blocksPerGridNode, threadsPerBlock>>>(d_iLLR, VNs_per_load);
    ERROR_CHECK(cudaGetLastError( ), __FILE__, __LINE__);

    // LANCEMENT DU PROCESSUS DE DECODAGE SUR n ITERATIONS
    for(int k = 0; k < 200; k++)
    {
    	ADMM_VN_kernel_deg3<<<blocksPerGridNode,  threadsPerBlock>>>
    			(d_iLLR, d_oLLR, LZr, d_t_row, VNs_per_load);
        ERROR_CHECK(cudaGetLastError( ), __FILE__, __LINE__);

        ADMM_CN_kernel_deg6<<<blocksPerGridCheck, threadsPerBlock>>>
        		(d_oLLR, LZr, d_t_col, d_hDecision, CNs_per_load);
        ERROR_CHECK(cudaGetLastError( ), __FILE__, __LINE__);

        // GESTION DU CRITERE D'ARRET DES CODEWORDS
        if( (k%5) == 0 )
        {
            reduce<<<blocksPerGridCheck, threadsPerBlock>>>(d_hDecision, CNs_per_load);
            ERROR_CHECK(cudaGetLastError( ), __FILE__, __LINE__);

            Status = cudaMemcpy(h_hDecision, d_hDecision, blocksPerGridCheck * sizeof(int), cudaMemcpyDeviceToHost);
            ERROR_CHECK(Status, __FILE__, __LINE__);

            int sum = 0;
            for(int p=0; p<blocksPerGridCheck; p++){
            	sum += h_hDecision[p];
            }
            if( sum == 0 ) break;
        }
    }

    // LANCEMENT DU PROCESSUS DE DECODAGE SUR n ITERATIONS
//    printf("ADMM_HardDecision(%d)\n", VNs_per_load);
    ADMM_HardDecision<<<blocksPerGridNode, threadsPerBlock>>>(d_oLLR, d_hDecision, VNs_per_load);
    ERROR_CHECK(cudaGetLastError(), __FILE__, __LINE__);

    // LANCEMENT DU PROCESSUS DE DECODAGE SUR n ITERATIONS
//    printf("h_hDecision = %p, d_hDecision = %p, VNs_per_load = %d\n", h_hDecision, d_hDecision, VNs_per_load);
    Status = cudaMemcpy(bits, d_hDecision, VNs_per_load * sizeof(int), cudaMemcpyDeviceToHost);
    ERROR_CHECK(Status, __FILE__, __LINE__);

//	for (int i=0; i<VNs_per_load; i++){
//		bits[i] = h_hDecision[i];
//	}
/*
	for(int k=1; k<frames; k++){
		bool error = false;
		for(int i=0; i<VNs_per_frame; i++){
	    	if( bits[VNs_per_frame * k + i] != bits[i] )
	    	{
	    		int off = VNs_per_frame * k;
	    		printf("frame %d : bit %4d : value mismatch (%d != %d | %d != %d)\n", k, i, bits[i], bits[off + i], h_hDecision[i], h_hDecision[off + i]);
	    		error = true;
	    	}
	    }
		if( error ) exit( 0 );
	}
*/
}

