/************************************************
 * @file Timer.h
 * 
 * @brief Definition file for the Timer class
 * 
 * @details Specifies types as well as the
 *          Timer class
 *                                   
 ************************************************/
#ifndef TIMER_H
#define TIMER_H

#include <chrono>

typedef std::chrono::high_resolution_clock hresClock; 
typedef hresClock::time_point hresClockTimePoint; 

class Timer {
public:
    //class constructor
    Timer();

    //retrieve relative time
    void start(); //sets internal time point 
    long long getMicrosecondsElapsed(); //returns microseconds since timer was created, or last start() call
    static long long getMicrosecondsSince(hresClockTimePoint t1); //returns microseconds since provided time point

    //wait
    bool compareElapsedMicroseconds(const long long& microseconds); //returns true if specified time has elapsed;
    static void sleepMicroseconds(const long long& microseconds); //waits for specified time period

    //retrieve current time
    static hresClockTimePoint getCurrentTime(); //returns current time as hresClockTimePoint

private:
    hresClockTimePoint m_initialTime;
};

#endif 

