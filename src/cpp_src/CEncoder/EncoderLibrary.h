

#ifndef CLASS_ENCODER_LIBRARY
#define CLASS_ENCODER_LIBRARY

#include "./CEncoder/CFakeEncoder.h"

Encoder* EncoderLibrary(bool REAL_ENCODER, CTrame *trame)
{
	return new CFakeEncoder(trame);
}

#endif

