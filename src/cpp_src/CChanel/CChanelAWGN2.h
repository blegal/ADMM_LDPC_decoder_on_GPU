#ifndef CLASS_CChanelAWGN2
#define CLASS_CChanelAWGN2

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "CChanel.h"

#include "../../custom/custom_cuda.h"
#include <curand.h>


class CChanelAWGN2 : public CChanel
{
private:
    double awgn(double amp);

    int   *d_IN;
    float *device_A;
    float *device_B;
    float *device_R;
//    float *host_R;

	curandGenerator_t generator;

public:
    CChanelAWGN2(CTrame *t, int _BITS_LLR, bool QPSK, bool Es_N0);
    ~CChanelAWGN2();
    virtual void configure(double _Eb_N0);
    virtual void generate();
};

#endif

