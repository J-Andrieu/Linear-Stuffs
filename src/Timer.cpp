/**************************************************
 * @file Timer.cpp
 * 
 * @brief Implementation file for Timer class
 * 
 * @details Implements all member methods of the 
 *          Timer class
 *
 * @note Requires Timer.h 
 **************************************************/
#include <chrono>

#include "../include/Timer.h"

Timer::Timer( ) {
    m_initialTime = hresClock::now( );
}

void Timer::start( ) {
    m_initialTime = hresClock::now( );
}

long long Timer::getMicrosecondsElapsed( ) {
    return std::chrono::duration_cast< std::chrono::microseconds >( hresClock::now( ) - m_initialTime ).count( );
}

long long Timer::getMicrosecondsSince( hresClockTimePoint t1 ) {
    return std::chrono::duration_cast< std::chrono::microseconds >( hresClock::now( ) - t1 ).count( );
}

bool Timer::compareElapsedMicroseconds( const long long& microseconds ) {
    return getMicrosecondsElapsed( ) >= microseconds;
}

void Timer::sleepMicroseconds( const long long& microseconds ) {
    hresClockTimePoint t1 = hresClock::now( );
    while ( getMicrosecondsSince( t1 ) < microseconds );
}

hresClockTimePoint Timer::getCurrentTime( ) {
    return hresClock::now( );
}
