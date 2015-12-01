#include "CChanel.h"

double CChanel::get_R(){
    return R;
}


CChanel::~CChanel(){
}


CChanel::CChanel(CTrame *t, int _BITS_LLR, bool QPSK, bool ES_N0)
{
    qbeta        = 0.0;
    R            = 0.0;
    _vars        = t->nb_info()   * t->nb_frames();
    _data        = t->nb_data()   * t->nb_frames();
    _checks      = t->nb_checks() * t->nb_frames();
    t_coded_bits = t->get_t_coded_bits();
    t_noise_data = t->get_t_noise_data();
    BITS_LLR     = _BITS_LLR;
    qpsk         = QPSK;
    es_n0        = ES_N0;
}


//void CChanel::configure(double _Eb_N0){
//}

#define QPSK 0.707106781
#define BPSK 1.0

//void CChanel::generate()
//{
//    printf("(EE) OUPS A PROBLEM HAS OCCURED: WE ARE IN A VIRTUAL METHOD !\n");
//    printf("%s @ %d\n", __FILE__, __LINE__);
//    exit( 0 );
//}
