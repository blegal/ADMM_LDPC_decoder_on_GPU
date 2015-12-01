#ifndef CLASS_CFakeEncoder
#define CLASS_CFakeEncoder

#include <stdlib.h>
#include <stdio.h>
#include "Encoder.h"

class CFakeEncoder : public Encoder
{    
protected:
    int  _vars;
    int  _data;
    int*  t_in_bits;      // taille (var)
    int*  t_coded_bits;   // taille (data)
    
public:
    
    CFakeEncoder(CTrame *t);
    
    virtual void encode();
};

#endif
