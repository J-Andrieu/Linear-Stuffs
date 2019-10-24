/**
* @file accuracy.cpp
*
* @brief Tests accuracy of LinAlgo namespace
*
* @notes Requires LinAlgo.h and Timer.h
*/

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <climits>
#include <chrono>

#include "../../include/LinAlgo.hpp"
#include "../Timer/include/Timer.h"
#include "utilities.h"

typedef struct {
    bool char_test;
    bool short_test;
    bool int_test;
    bool long_test;
    bool float_test;
    bool double_test;
} test_out;
bool passFail(test_out& test);

#ifndef DONT_USE_GPU
test_out testGPUAddition(size_t height, size_t width, bool verbose);
test_out testGPUSubtraction(size_t height, size_t width, bool verbose);
test_out testGPUMultiplication(size_t height, size_t width, bool verbose);
test_out testGPUDivision(size_t height, size_t width, bool verbose);
#endif

void CheckAccuracy(size_t height, size_t width, std::string logfile, bool verbose) {
    std::srand((unsigned) std::time(NULL));
    Timer t_accuracy("Full Accuracy Suite");
    std::ofstream log(logfile, std::ofstream::out | std::ofstream::app);

    log << std::endl;
    log << "Beginning accuracy tests" << std::endl;

#ifndef DONT_USE_GPU
    test_out addition_result = testGPUAddition(height, width, verbose);
    bool addition_test = passFail(addition_result);
    log << "GPU addition tests: " << (addition_test ? "passed" : "failed") << std::endl;
    //log << "GPU subtraction tests: " << testGPUSubtraction(height, width) ? "passed" : "failed" << std::endl;
    //log << "GPU multiplication tests: " << testGPUMultiplication(height, width) ? "passed" : "failed" << std::endl;
    //log << "GPU division tests: " << testGPUDivision(height, width) ? "passed" : "failed" << std::endl;
#endif

}

bool passFail(test_out& test) {
    if (!test.char_test)
        return false;
    if (!test.short_test)
        return false;
    if (!test.int_test)
        return false;
    if (!test.long_test)
        return false;
    if (!test.float_test)
        return false;
    if (!test.double_test)
        return false;
    return true;
}

test_out testGPUAddition(size_t height, size_t width, bool verbose) {
    if (verbose) {
        std::cout << "Beginning GPU Accuracy test: addition" << std::endl;
    }
    bool char_test, short_test, int_test, long_test, float_test, double_test;
    char_test = short_test = int_test = long_test = float_test = double_test = true;

    {
        Timer t_char("Accuracy: GPU Addition: character");
        LinAlgo::matrix<char> m1(height, width);
        LinAlgo::matrix<char> m2(height, width);
        for (auto i = m1.begin(), j = m2.begin(); i <= m1.end(); i++, j++) {
            *i = rand() % CHAR_MAX;
            *j = rand() % CHAR_MAX;
        }
        if (verbose) {
            std::cout << "M1 character test:" << std::endl;
            print_matrix<char>(m1);
            std::cout << "M2 character test:" << std::endl;
            print_matrix<char>(m2);
        }
        LinAlgo::matrix<char> result_CPU = m1 + m2;
        m1.useGPU(true);
        m2.useGPU(true);
        LinAlgo::matrix<char> result_GPU = m1 + m2;
        if (verbose) {
            std::cout << "GPU Addition results:" << std::endl;
            print_matrix<char>(result_GPU);
        }
        char_test = result_GPU == result_CPU;
    }

    {
        Timer t_short("Accuracy: GPU Addition: short");
        LinAlgo::matrix<short> m1(height, width);
        LinAlgo::matrix<short> m2(height, width);
        for (auto i = m1.begin(), j = m2.begin(); i <= m1.end(); i++, j++) {
            *i = rand() % SHRT_MAX;
            *j = rand() % SHRT_MAX;
        }
        if (verbose) {
            std::cout << "M1 short test:" << std::endl;
            print_matrix<short>(m1);
            std::cout << "M2 short test:" << std::endl;
            print_matrix<short>(m2);
        }
        LinAlgo::matrix<short> result_CPU = m1 + m2;
        m1.useGPU(true);
        m2.useGPU(true);
        LinAlgo::matrix<short> result_GPU = m1 + m2;
        if (verbose) {
            std::cout << "GPU Addition results:" << std::endl;
            print_matrix<short>(result_GPU);
        }
        short_test = result_GPU == result_CPU;
    }

    {
        Timer t_int("Accuracy: GPU Addition: integer");
        LinAlgo::matrix<int> m1(height, width);
        LinAlgo::matrix<int> m2(height, width);
        for (auto i = m1.begin(), j = m2.begin(); i <= m1.end(); i++, j++) {
            *i = rand() % INT_MAX;
            *j = rand() % INT_MAX;
        }
        if (verbose) {
            std::cout << "M1 integer test:" << std::endl;
            print_matrix<int>(m1, -13);
            std::cout << "M2 integer test:" << std::endl;
            print_matrix<int>(m2, -13);
        }
        LinAlgo::matrix<int> result_CPU = m1 + m2;
        m1.useGPU(true);
        m2.useGPU(true);
        LinAlgo::matrix<int> result_GPU = m1 + m2;
        if (verbose) {
            std::cout << "GPU Addition results:" << std::endl;
            print_matrix<int>(result_GPU, -13);
        }
        int_test = result_GPU == result_CPU;
    }

    {
        Timer t_long("Accuracy: GPU Addition: long integer");
        LinAlgo::matrix<long> m1(height, width);
        LinAlgo::matrix<long> m2(height, width);
        for (auto i = m1.begin(), j = m2.begin(); i <= m1.end(); i++, j++) {
            *i = rand() % LONG_MAX;
            *j = rand() % LONG_MAX;
        }
        if (verbose) {
            std::cout << "M1 long integer test:" << std::endl;
            print_matrix<long>(m1, -13);
            std::cout << "M2 long integer test:" << std::endl;
            print_matrix<long>(m2, -13);
        }
        LinAlgo::matrix<long> result_CPU = m1 + m2;
        m1.useGPU(true);
        m2.useGPU(true);
        LinAlgo::matrix<long> result_GPU = m1 + m2;
        if (verbose) {
            std::cout << "GPU Addition results:" << std::endl;
            print_matrix<long>(result_GPU, -13);
        }
        long_test = result_GPU == result_CPU;
    }

    {
        Timer t_float("Accuracy: GPU Addition: float");
        LinAlgo::matrix<float> m1(height, width);
        LinAlgo::matrix<float> m2(height, width);
        for (auto i = m1.begin(), j = m2.begin(); i <= m1.end(); i++, j++) {
            *i = static_cast<float>(rand() % INT_MAX);
            *j = static_cast<float>(rand() % INT_MAX);;
        }
        if (verbose) {
            std::cout << "M1 float test:" << std::endl;
            print_matrix<float>(m1, -15);
            std::cout << "M2 float test:" << std::endl;
            print_matrix<float>(m2, -15);
        }
        LinAlgo::matrix<float> result_CPU = m1 + m2;
        m1.useGPU(true);
        m2.useGPU(true);
        LinAlgo::matrix<float> result_GPU = m1 + m2;
        if (verbose) {
            std::cout << "GPU Addition results:" << std::endl;
            print_matrix<float>(result_GPU, -15);
        }
        float_test = result_GPU == result_CPU;
    }

    {
        Timer t_double("Accuracy: GPU Addition: double");
        LinAlgo::matrix<double> m1(height, width);
        LinAlgo::matrix<double> m2(height, width);
        for (auto i = m1.begin(), j = m2.begin(); i <= m1.end(); i++, j++) {
            *i = static_cast<double>(rand() % LONG_MAX);
            *j = static_cast<double>(rand() % LONG_MAX);;
        }
        if (verbose) {
            std::cout << "M1 double test:" << std::endl;
            print_matrix<double>(m1, -13);
            std::cout << "M2 double test:" << std::endl;
            print_matrix<double>(m2, -13);
        }
        LinAlgo::matrix<double> result_CPU = m1 + m2;
        m1.useGPU(true);
        m2.useGPU(true);
        LinAlgo::matrix<double> result_GPU = m1 + m2;
        if (verbose) {
            std::cout << "GPU Addition results:" << std::endl;
            print_matrix<double>(result_GPU, -13);
        }
        double_test = result_GPU == result_CPU;
    }

    return {char_test, short_test, int_test, long_test, float_test, double_test};
}


