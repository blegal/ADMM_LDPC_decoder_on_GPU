#include <time.h>
#include <sys/time.h>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

#ifndef CLASS_CTimer
#define CLASS_CTimer

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;

class CTimer
{
    
protected:
    timespec t_start;
    timespec t_stop;
    bool isRunning;
    
    long diff_ns(timespec start, timespec end);
    long diff_sec(timespec start, timespec end);

public:
    
    CTimer(bool _start);

    void start();

    void stop();

    void reset();

    long get_time_ns();

    long get_time_us();

    long get_time_ms();

    long get_time_sec();

};

#endif
