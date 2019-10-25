/**
* @file accuracy.cpp
*
* @brief Tests accuracy of LinAlgo namespace
*
* @notes Requires LinAlgo.h and Timer.h
*/

#ifndef TESTING_UTILITIES
#define TESTING_UTILITIES

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>
#include <tuple>
#include <algorithm>
#include <climits>
#include <chrono>

#include "../include/LinAlgo.hpp"
#include "Timer/include/Timer.h"
#include "CMDParser/include/CMDParser.h"

template <class ItemType>
void print_matrix (const LinAlgo::matrix<ItemType>& M, int padding = -10) {
    if (M.getWidth() == 0) {
        std::cout << "Null matrix" << std::endl;
    }
    if (M.getWidth() <= 15 && M.getHeight() <= 20) {
        std::string fstring = std::string("% ") + std::to_string(padding) + std::string("s");
        for (int i = 0; i < M.getHeight(); i++) {
            for (int j = 0; j < M.getWidth(); j++) {
                ItemType temp = M[i][j];
                std::string valStr;
                if (std::abs(temp) < 0.0009) {
                    valStr = std::string(" ") + std::to_string(0);
                } else {
                    std::string sign = temp < ItemType(0) ? "-" : " ";
                    long double val = std::abs((long double) temp * 1000);
                    long int rounded = (long int)(val + .5);
                    long int modulo = rounded % 1000;
                    std::string moduloStr = std::to_string(modulo);
                    for (auto k = moduloStr.end() - 1; k >= moduloStr.begin(); k--) {
                        if (*k != '0') {
                            break;
                        } else {
                            *k = ' ';
                        }
                    }
                    rounded /= 1000;
                    valStr = sign + std::to_string(rounded) + (modulo == 0 ? "" : std::string(".") + moduloStr);
                }

                printf(fstring.c_str(), valStr.c_str());
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << "Too Large" << std::endl;
    }
}

#ifndef DONT_USE_GPU
template <class ItemType>
std::tuple<size_t, size_t, ItemType, ItemType> locateError(const LinAlgo::matrix<ItemType>& M1, const LinAlgo::matrix<ItemType>& M2) {
    for (size_t i = 0; i < M1.getHeight(); i++) {
        for (size_t j = 0; j < M1.getWidth(); j++) {
            if (M1[i][j] != M2[i][j]) {
                return {i, j, M1[i][j], M2[i][j]};
            }
        }
    }
    return {0, 0, 0, 0};
}
#endif
#endif // TESTING_UTILITIES
