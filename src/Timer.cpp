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

void Timers::logNamedTimers() {
    if (out_file == "") {
        for (auto i = named_timers.begin(); i < named_timers.end(); i++) {
            std::cout << i->m_scopeName << ": " << *(i->m_executionTime) << "us" << std::endl;
        }
    } else {
        std::ofstream out(out_file, std::ofstream::out | std::ofstream::app);
        for (auto i = named_timers.begin(); i < named_timers.end(); i++) {
            std::cout << i->m_scopeName << ": " << *(i->m_executionTime) << "us" << std::endl;
            out << i->m_scopeName << ": " << *(i->m_executionTime) << "us" << std::endl;
        }
    }
}

void Timers::setLogFile(std::string filename) {
    Timers::out_file = filename;
}

Timer::Timer( ) {
    m_initialTime = hresClock::now( );
    m_out = nullptr;
}

Timer::Timer(long long& out) {
    m_initialTime = hresClock::now( );
    m_out = &out;
}

Timer::Timer(std::string scopeName) {
    m_initialTime = hresClock::now( );
    m_out = new long long;
    Timers::named_timers.push_back({scopeName, m_out});
}

Timer::~Timer() {
    if (m_out != nullptr)
        *m_out = std::chrono::duration_cast< std::chrono::microseconds > ( hresClock::now( ) - m_initialTime ).count( );
}

void Timer::start( ) {
    m_initialTime = hresClock::now( );
}

long long Timer::getMicrosecondsElapsed( ) {
    return std::chrono::duration_cast< std::chrono::microseconds > ( hresClock::now( ) - m_initialTime ).count( );
}

long long Timer::getMicrosecondsSince ( hresClockTimePoint t1 ) {
    return std::chrono::duration_cast< std::chrono::microseconds > ( hresClock::now( ) - t1 ).count( );
}

bool Timer::compareElapsedMicroseconds ( const long long& microseconds ) {
    return getMicrosecondsElapsed( ) >= microseconds;
}

void Timer::sleepMicroseconds ( const long long& microseconds ) {
    hresClockTimePoint t1 = hresClock::now( );
    while ( getMicrosecondsSince ( t1 ) < microseconds );
}

hresClockTimePoint Timer::getCurrentTime( ) {
    return hresClock::now( );
}
