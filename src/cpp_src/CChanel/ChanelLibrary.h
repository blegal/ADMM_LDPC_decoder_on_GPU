#ifndef CLASS_ChanelLibrary
#define CLASS_ChanelLibrary

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "CChanelAWGN2.h"

CChanel* CreateChannel(int channelType, CTrame *trame, bool QPSK_CHANNEL, bool Es_N0)
{
	return new CChanelAWGN2(trame, 4, QPSK_CHANNEL, Es_N0);
}


#endif

