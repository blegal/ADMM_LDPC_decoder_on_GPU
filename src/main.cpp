#include  <stdio.h>
#include  <stdlib.h>
#include  <iostream>
#include  <cstring>
#include  <math.h>
#include  <time.h>
#include  <string.h>
#include  <limits.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>

using namespace std;

//#include "./ldpc/CFloodingGpuDecoder.h"
#include "./ldpc/ADMM_GPU_Decoder.h"
#include "./ldpc/ADMM_GPU_Decoder_16b.h"

#define pi  3.1415926536

#include "./cpp_src/CTimer/CTimer.h"
#include "./cpp_src/CTrame/CTrame.h"
#include "./cpp_src/CChanel/ChanelLibrary.h"
#include "./cpp_src/CEncoder/CFakeEncoder.h"
#include "./cpp_src/CEncoder/GenericEncoder.h"
#include "./cpp_src/CErrorAnalyzer/CErrorAnalyzer.h"
#include "./cpp_src/CTerminal/CTerminal.h"

#define SINGLE_THREAD 1

#if 0
	#define NOEUD       4000
	#define PARITE      2000
	#define MESSAGE		12000
#else
	#define NOEUD       2640
	#define PARITE      1320
	#define MESSAGE 	7920
#endif

int    QUICK_STOP           =  true;
int    FRAME_ERROR_LIMIT    =  50;
bool   BER_SIMULATION_LIMIT =  false;
bool   NORMALIZED_MIN_SUM   =  false;
double BIT_ERROR_LIMIT      =  1e-7;

int technique          = 0;
int sChannel           = 1; // 1 = CHANNEL ON GPU (works only for 4000x2000 LDPC code yet

////////////////////////////////////////////////////////////////////////////////////

double rendement;
double Eb_N0;

////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	int p;
    srand( 0 );
	printf("(II) LDPC DECODER - Flooding scheduled decoder\n");
	printf("(II) MANIPULATION DE DONNEES (IEEE-754 - %ld bits)\n", 8*sizeof(int));
	printf("(II) GENEREE : %s - %s\n", __DATE__, __TIME__);

	double MinSignalSurBruit = 0.50;
	double MaxSignalSurBruit = 4.51;
	double PasSignalSurBruit = 0.50;
    int    NOMBRE_ITERATIONS = 200;
	int    REAL_ENCODER      =  0;
	int    STOP_TIMER_SECOND = -1;
	bool   QPSK_CHANNEL      = false;
    bool   Es_N0             = false; // FALSE => MODE Eb_N0
    int NB_FRAMES_IN_PARALLEL = 1;
    int algo = 0;

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaThreadSynchronize();


	//
	// ON VA PARSER LES ARGUMENTS DE LIGNE DE COMMANDE
	//
	for (p=1; p<argc; p++) {
		if( strcmp(argv[p], "-min") == 0 ){
			MinSignalSurBruit = atof( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-frames") == 0 ){
			NB_FRAMES_IN_PARALLEL = atoi( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-timer") == 0 ){
			STOP_TIMER_SECOND = atoi( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-enc") == 0 ){
			REAL_ENCODER = 2;

		}else if( strcmp(argv[p], "-encoder") == 0 ){
			REAL_ENCODER = 1;

		}else if( strcmp(argv[p], "-random") == 0 ){
            srand( time(NULL) );

		}else if( strcmp(argv[p], "-max") == 0 ){
			MaxSignalSurBruit = atof( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-pas") == 0 ){
			PasSignalSurBruit = atof( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-stop") == 0 ){
			QUICK_STOP = 1;

		}else if( strcmp(argv[p], "-iter") == 0 ){
			NOMBRE_ITERATIONS = atoi( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-spa") == 0 ){
			algo = 0;

		}else if( strcmp(argv[p], "-oms") == 0 ){
			algo = 1;

		}else if( strcmp(argv[p], "-admm") == 0 ){
			algo = 2;

		}else if( strcmp(argv[p], "-fer") == 0 ){
			FRAME_ERROR_LIMIT = atoi( argv[p+1] );
			p += 1;

		}else if( strcmp(argv[p], "-qef") == 0 ){
			BER_SIMULATION_LIMIT =  true;
			BIT_ERROR_LIMIT      = ( atof( argv[p+1] ) );
			p += 1;

		}else if( strcmp(argv[p], "-norm-min-sum") == 0 ){
			NORMALIZED_MIN_SUM = true;

		}else if( strcmp(argv[p], "-bpsk") == 0 ){
			QPSK_CHANNEL = false;

		}else if( strcmp(argv[p], "-qpsk") == 0 ){
			QPSK_CHANNEL = true;

		}else if( strcmp(argv[p], "-Eb/N0") == 0 ){
			Es_N0 = false;

		}else if( strcmp(argv[p], "-Es/N0") == 0 ){
			Es_N0 = true;

        }else{
			printf("(EE) Unknown argument (%d) => [%s]\n", p, argv[p]);
			exit(0);
		}
	}

	rendement = (float)(1320)/(float)(NOEUD);
	printf("(II) Code LDPC (N, K)     : (%d,%d)\n", NOEUD, PARITE);
	printf("(II) Rendement du code    : %.3f\n", rendement);
	printf("(II) # ITERATIONs du CODE : %d\n", NOMBRE_ITERATIONS);
    printf("(II) FER LIMIT FOR SIMU   : %d\n", FRAME_ERROR_LIMIT);
	printf("(II) SIMULATION  RANGE    : [%.2f, %.2f], STEP = %.2f\n", MinSignalSurBruit,  MaxSignalSurBruit, PasSignalSurBruit);
	printf("(II) MODE EVALUATION      : %s\n", ((Es_N0)?"Es/N0":"Eb/N0") );
	printf("(II) MIN-SUM ALGORITHM    : %s\n", ((NORMALIZED_MIN_SUM)?"NORMALIZED":"OFFSET") );
	printf("(II) FAST STOP MODE       : %d\n", QUICK_STOP);

	CTimer simu_timer(true);
	CTrame simu_data_1(NOEUD, PARITE, NB_FRAMES_IN_PARALLEL);

	//LDPC_GPU_Decoder decoder_1( algo );
//	ADMM_GPU_Decoder decoder_1( NB_FRAMES_IN_PARALLEL );
	ADMM_GPU_decoder_16b decoder_1( NB_FRAMES_IN_PARALLEL );

	Eb_N0 = MinSignalSurBruit;

	#pragma omp parallel sections
	while (Eb_N0 <= MaxSignalSurBruit){

        //
        // ON CREE UN OBJET POUR LA MESURE DU TEMPS DE SIMULATION (REMISE A ZERO POUR CHAQUE Eb/N0)
        //
        CTimer temps_ecoule(true);
        CTimer refresh(true);

        //
        // ALLOCATION DYNAMIQUE DES DONNESS NECESSAIRES A LA SIMULATION DU SYSTEME
        //
		Encoder *encoder_1 = NULL;
		if( REAL_ENCODER == 1 ){
			encoder_1 = new GenericEncoder(&simu_data_1);
		}else{
			encoder_1 = new CFakeEncoder(&simu_data_1);
		}

		//
		// ON CREE LE CANAL DE COMMUNICATION (BRUIT GAUSSIEN)
		//
		CChanel *noise_1 = CreateChannel(sChannel, &simu_data_1, QPSK_CHANNEL, Es_N0);
		noise_1->configure( Eb_N0 );

        CErrorAnalyzer errCounter(&simu_data_1, FRAME_ERROR_LIMIT);

        //
        // ON CREE L'OBJET EN CHARGE DES INFORMATIONS DANS LE TERMINAL UTILISATEUR
        //
		CTerminal terminal(&errCounter, &temps_ecoule, Eb_N0);

        // ON GENERE LA PREMIERE TRAME BRUITEE

        double time = 0.0f;
		while( 1 ){

	        encoder_1->encode();

	        noise_1->generate();

	        errCounter.store_enc_bits();

			int mExeTime = 0;
		    auto start   = chrono::steady_clock::now();

		    float *Intrinsic_fix = simu_data_1.get_t_noise_data ();
			int *Rprime_fix      = simu_data_1.get_t_decode_data();
		    decoder_1.decode( Intrinsic_fix, Rprime_fix, NOMBRE_ITERATIONS );

		    auto end     = chrono::steady_clock::now();
		    auto diff    = end - start;
		    time        += chrono::duration <double, milli> (diff).count();

            errCounter.generate();

            //
            // ON compare le Frame Error avec la limite imposee par l'utilisateur. Si on depasse
            // alors on affiche les resultats sur Eb/N0 courant.
            //
			if ( errCounter.fe_limit_achieved() == true ){
                break;
            }

            //
            // AFFICHAGE A L'ECRAN DE L'EVOLUTION DE LA SIMULATION SI NECESSAIRE
            //

			if( (refresh.get_time_sec()) >= 2 )
            {
				refresh.reset();
            	terminal.temp_report();
			}
		}

		terminal.final_report();
	    double debit = (1000.0f / (time/errCounter.nb_processed_frames())) * NOEUD / 1000.0f / 1000.0f;
	    printf("debit %1.3f\n", debit);

		Eb_N0 = Eb_N0 + PasSignalSurBruit;

		// ON FAIT LE MENAGE PARMIS TOUS LES OBJETS CREES DYNAMIQUEMENT...
        delete noise_1;
		delete encoder_1;
		// FIN DU MENAGE

        if( (simu_timer.get_time_sec() >= STOP_TIMER_SECOND) && (STOP_TIMER_SECOND != -1) ){
        	printf("(II) THE SIMULATION HAS STOP DUE TO THE (USER) TIME CONTRAINT.\n");
        	break;
        }

        if( BER_SIMULATION_LIMIT == true ){
        	if( errCounter.ber_value() < BIT_ERROR_LIMIT ){
        		printf("(II) THE SIMULATION HAS STOP DUE TO THE (USER) QUASI-ERROR FREE CONTRAINT.\n");
        		break;
        	}
        }
	}
	return 0;
}
