#include "CTimer.h"


//
// OS X does not have clock_gettime, use clock_get_time
//
#ifdef __MACH__
#define CLOCK_REALTIME  0
#define CLOCK_MONOTONIC 1
void clock_gettime(int useless,  timespec *ts){    
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    ts->tv_sec = mts.tv_sec;
    ts->tv_nsec = mts.tv_nsec;
}
#endif
//
// END OF MACOS X SPECIAL DEFINTIION
//


long CTimer::diff_ns(timespec start, timespec end)
{
    timespec temp;
    if ((end.tv_nsec-start.tv_nsec)<0) {
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    } else {
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return (temp.tv_nsec);
}

long CTimer::diff_sec(timespec start, timespec end)
{
    timespec temp;
    if ((end.tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec  = end.tv_sec-start.tv_sec-1;
    } else {
        temp.tv_sec  = end.tv_sec-start.tv_sec;
    }
    return (temp.tv_sec);
}


CTimer::CTimer(bool _start){
    if(_start == true){
        clock_gettime(CLOCK_MONOTONIC, &t_start);
        isRunning = true;
    }else{
        isRunning = false;
    }
}

void CTimer::start(){
    if( isRunning == true ){
        cout << "(EE) CTimer :: trying to start a CTimer object that is already running !" << endl;
    }
    clock_gettime(CLOCK_MONOTONIC, &t_start);
}

void CTimer::stop(){
    if( isRunning == false ){
        cout << "(EE) CTimer :: trying to stop a CTimer object that is not running !" << endl;
    }
    clock_gettime(CLOCK_MONOTONIC, &t_stop);
}

void CTimer::reset(){
    clock_gettime(CLOCK_MONOTONIC, &t_start);
}

long CTimer::get_time_ns(){
    if( isRunning == true ){
        clock_gettime(CLOCK_MONOTONIC, &t_stop);
    }
    return diff_ns( t_start, t_stop );
}

long CTimer::get_time_us(){
    return get_time_ns() / 1000;
}

long CTimer::get_time_ms(){
    return get_time_us() / 1000;
}

long CTimer::get_time_sec(){
    if( isRunning == true ){
        clock_gettime(CLOCK_MONOTONIC, &t_stop);
    }
    return diff_sec( t_start, t_stop );
}
