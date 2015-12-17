/*
 *  ldcp_decoder.h
 *  ldpc3
 *
 *  Created by legal on 02/04/11.
 *  Copyright 2011 ENSEIRB. All rights reserved.
 *
 */

/*----------------------------------------------------------------------------*/

#include "ADMM_GPU_16b.h"

#include "../gpu/ADMM_GPU_functions.h"

#if 0
	#include "../codes/Constantes_4000x2000.h"
#else
	#include "./admm/admm_2640x1320.h"
#endif


ADMM_GPU_16b::ADMM_GPU_16b( int _frames )
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

    // ON TRANSMET LES DONNEES POUR L'ADRESSAGE DES OutputValues LORS
    // DES CALCULS DE CNs
    CUDA_MALLOC_DEVICE(&d_t_col, MSGs_per_frame);
    unsigned short* t_col_m = new unsigned short[MSGs_per_frame];
    for(int i=0; i<MSGs_per_frame; i++)
    	 t_col_m[i] = t_col[i];
    Status = cudaMemcpy(d_t_col, (int*)t_col_m, MSGs_per_frame * sizeof(unsigned int), cudaMemcpyHostToDevice);
    ERROR_CHECK(Status, (char*)__FILE__, __LINE__);
    delete t_col_m;

    unsigned short t_cn_pos[MSGs_per_frame];
    FILE* f1 = fopen("reorder.txt", "w");
    for(int i=0; i<MSGs_per_frame; i++){
    	int x = (i/6)%128;
    	int y = (128*i)%768;
    	int b = 768 * (i/768);
    	t_cn_pos[i] = y + x + b;
//    	printf("thread %4d : %4d => t_cn_pos[%4d] = %4d (%4d, %4d, %4d)\n", i/6, i, i, t_cn_pos[i], x, y, b);
		if( (i % 6) == 0 ) fprintf(f1, "\n CN %4d : ", i/6);
		fprintf(f1, "%4d ", t_cn_pos[i]);
//    	if( i == 800 ) exit ( 0 );
    }
    fclose( f1 );

    FILE* fg = fopen("reorder_gpu.txt", "w");
    for(int i=0; i<1320; i++)
    {
		fprintf(fg, "\nCN %4d : ", i);
		for(int k=0; k<6; k++)
		{
		    const int ind = 768 * (i/128) + 128 * k + i%128;
			fprintf(fg, "%4d ", ind);
		}
    }
    fclose( fg );

    FILE* f2 = fopen("reorder_vn.txt", "w");
    unsigned short t_row_mod[MSGs_per_frame];
    for(int i=0; i<MSGs_per_frame; i++)
    {
    	int value = t_row[i];
		t_row_mod[i] = t_cn_pos[value];
		if( (i % 3) == 0 ){
			fprintf(f2, "\n VN %4d : ", i/3);
			for(int j=0; j<3; j++)
				fprintf(f2, "%4d ", t_row[i+j]);
			fprintf(f2, "=> ");
		}
		fprintf(f2, "%4d ", t_row_mod[i]);
    }
    fclose( f2 );

    unsigned short t_row_ready[4 * MSGs_per_frame / 3];
    unsigned short* in  = t_row_mod;
    unsigned short* out = t_row_ready;
    int size = MSGs_per_frame / 3;
    while( size-- )
    {
    	*out++ = *in++;
    	*out++ = *in++;
    	*out++ = *in++;
    	*out++ = 0;
    }

    FILE* f3 = fopen("reorder_vn_pad.txt", "w");
    for(int i=0; i<4*MSGs_per_frame/3; i++)
    {
		if( (i % 4) == 0 ){
			fprintf(f3, "\n VN %4d : ", i/4);
		}
		fprintf(f3, "%4d ", t_row_ready[i]);
    }
    fclose( f3 );

    CUDA_MALLOC_DEVICE(&d_t_row, MSGs_per_frame);
#if 1
    Status = cudaMemcpy(d_t_row, t_row_ready, MSGs_per_frame * sizeof(unsigned int), cudaMemcpyHostToDevice);
#else
    Status = cudaMemcpy(d_t_row, t_row_pad_4, MSGs_per_frame * sizeof(unsigned int), cudaMemcpyHostToDevice);
#endif
    ERROR_CHECK(Status, (char*)__FILE__, __LINE__);

    // A REVOIR !
    // Espace memoire pour l'échange de messages dans le decodeur
    CUDA_MALLOC_DEVICE(&LZr, 2 * MSGs_per_load);
}


ADMM_GPU_16b::~ADMM_GPU_16b()
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

	Status = cudaFree(LZr);				ERROR_CHECK(Status, (char*)__FILE__, __LINE__);
}

//#define CHECK_ERRORS

void ADMM_GPU_16b::decode(float* llrs, int* bits, int nb_iters)
{
//	#define CHECK_ERRORS
    cudaError_t Status;
	int threadsPerBlock     = 128;
    int blocksPerGridNode   = (VNs_per_load  + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridCheck  = (CNs_per_load  + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridMsgs   = (MSGs_per_load + threadsPerBlock - 1) / threadsPerBlock;

    /* INITIALISATION DU DECODEUR LDPC SUR GPU */
    ADMM_InitArrays_16b<<<2*blocksPerGridMsgs, threadsPerBlock>>>(LZr, frames * 8448);
#ifdef CHECK_ERRORS
    ERROR_CHECK(cudaGetLastError( ), __FILE__, __LINE__);
#endif

    /* On copie les donnees d'entree du decodeur */
    cudaMemcpyAsync(d_iLLR, llrs, VNs_per_load * sizeof(float), cudaMemcpyHostToDevice);

    ADMM_ScaleLLRs<<<blocksPerGridNode, threadsPerBlock>>>(d_iLLR, VNs_per_load);
#ifdef CHECK_ERRORS
    ERROR_CHECK(cudaGetLastError( ), __FILE__, __LINE__);
#endif

    // LANCEMENT DU PROCESSUS DE DECODAGE SUR n ITERATIONS
    for(int k = 0; k < 200; k++)
    {
    	ADMM_VN_kernel_deg3_16b_mod<<<blocksPerGridNode,  threadsPerBlock>>>
    			(d_iLLR, d_oLLR, LZr, d_t_row, VNs_per_load);
#ifdef CHECK_ERRORS
        ERROR_CHECK(cudaGetLastError( ), __FILE__, __LINE__);
#endif

        ADMM_CN_kernel_deg6_16b_mod<<<blocksPerGridCheck, threadsPerBlock>>>
        		(d_oLLR, LZr, d_t_col, d_hDecision, CNs_per_load);
#ifdef CHECK_ERRORS
        ERROR_CHECK(cudaGetLastError( ), __FILE__, __LINE__);
#endif
//        Status = cudaMemcpy(h_hDecision, d_hDecision, blocksPerGridCheck * sizeof(int), cudaMemcpyDeviceToHost);
//        exit( 0 );

        // GESTION DU CRITERE D'ARRET DES CODEWORDS
        if( (k>=6) && ((k%2) == 0) )
        {
            reduce<<<blocksPerGridCheck, threadsPerBlock>>>(d_hDecision, CNs_per_load);
#ifdef CHECK_ERRORS
            ERROR_CHECK(cudaGetLastError( ), __FILE__, __LINE__);
#endif

            Status = cudaMemcpy(h_hDecision, d_hDecision, blocksPerGridCheck * sizeof(int), cudaMemcpyDeviceToHost);
#ifdef CHECK_ERRORS
            ERROR_CHECK(Status, __FILE__, __LINE__);
#endif

            int sum = 0;
            for(int p=0; p<blocksPerGridCheck; p++){
            	sum += h_hDecision[p];
            }
            if( sum == 0 ) break;
        }
    }

    // LANCEMENT DU PROCESSUS DE DECODAGE SUR n ITERATIONS
    ADMM_HardDecision<<<blocksPerGridNode, threadsPerBlock>>>(d_oLLR, d_hDecision, VNs_per_load);
#ifdef CHECK_ERRORS
    ERROR_CHECK(cudaGetLastError(), __FILE__, __LINE__);
#endif

    // LANCEMENT DU PROCESSUS DE DECODAGE SUR n ITERATIONS
    Status = cudaMemcpy(bits, d_hDecision, VNs_per_load * sizeof(int), cudaMemcpyDeviceToHost);
#ifdef CHECK_ERRORS
    ERROR_CHECK(Status, __FILE__, __LINE__);
#endif
}

