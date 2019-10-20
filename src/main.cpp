/**
* @file main.cpp
*
* @brief Test driver for matrix class
*
* @notes Requires matrix.h and Timer.h
*/

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <tuple>
#include <algorithm>
#include <climits>

#include "../include/LinAlgo.hpp"
#include "../include/Timer.h"

//less than 25x25 causes bsod with the destructor uncommented for some reason
int HEIGHT = 15;
int WIDTH = 15;
typedef float type;

template <class ItemType>
void print_matrix (const LinAlgo::matrix<ItemType>& M, int padding = -10);
void checkReturn (cl_int ret);
template <class ItemType>
std::tuple<size_t, size_t, ItemType, ItemType> locateError(const LinAlgo::matrix<ItemType>& M1, const LinAlgo::matrix<ItemType>& M2);

std::tuple<size_t, size_t> getDimensions (matrix<type> &M) {
    return {M.getHeight(), M.getWidth()};
}

/*MATRIX TESTING*/
int main (int argc, char* argv[]) {

    if (argc > 1) {
        try {
            HEIGHT = std::stoi (argv[1]);
            if (argc > 2) {
                WIDTH = std::stoi (argv[2]);
            } else {
                WIDTH = HEIGHT;
            }
        } catch (...) {
            std::cout << "accepted usage: matrix_test <height <width>>" << std::endl;
            return -1;
        }
    }

    checkReturn (LinAlgo::InitGPU());

    std::cout << (LinAlgo::IsGPUInitialized() ? "The GPU is initialized" : "The GPU is not initialized") << std::endl;

    Timer t;

    LinAlgo::matrix<type> m1 (HEIGHT, WIDTH);
    auto[height, width] = getDimensions (m1);
    std::cout << "The dimensions of m1 are height: " << height << ", width: " << width << std::endl;
    LinAlgo::matrix<type> m2 (WIDTH, HEIGHT);
    //LinAlgo::AllUseGPU(true);

    for (size_t i = 0; i < HEIGHT; i++) {
        for (size_t j = 0; j < WIDTH; j++) {
            type val1 = (std::pow((type) i, j) - j) / WIDTH;
            m1.set (i, j, (type) val1 < SHRT_MAX ? (val1 > UCHAR_MAX ? val1 / UCHAR_MAX : val1) : (int) val1 % UCHAR_MAX);
            m2.set (j, i, (type) i + j);
        }
    }
    std::cout << "A small identity:" << std::endl;
    print_matrix<int> (LinAlgo::matrix<int>::identity (5));
    std::cout << std::endl;
    std::cout << "M1:" << std::endl;
    print_matrix<type> (m1);
    std::cout << std::endl;
    std::cout << "M2:" << std::endl;
    print_matrix<type> (m2);
    std::cout << std::endl;
    std::cout << "Transposed M1:" << std::endl;
    LinAlgo::matrix<type> m1T (LinAlgo::transpose<type> (m1));
    print_matrix<type> (m1T);
    std::cout << std::endl;
    std::cout << "Row Echelon M1:" << std::endl;
    print_matrix<type> (LinAlgo::re (m1));
    std::cout << std::endl;
    std::cout << "Reduced Row Echelon M1:" << std::endl;
    print_matrix<type> (LinAlgo::rre (m1));
    std::cout << std::endl;
    matrix<type> m1gj(HEIGHT, WIDTH + 1);
    m1gj.copy(0, 0, m1);
    for (size_t i = 0; i < HEIGHT; i++) {
        m1gj[i][WIDTH] = i + 1;
    }
    std::cout << "Gauss-Jordan Elimination on M1 with solution vector (1, ..., " << HEIGHT << "):\nBefore:" << std::endl;
    print_matrix<type> (m1gj);
    std::cout << "After:" << std::endl;
    print_matrix<type> (LinAlgo::gj (m1gj));
    std::cout << std::endl;
    std::cout << "Gauss-Jordan Elimination on test linear system:" << std::endl;
    LinAlgo::matrix<type> linsys ({
        {1, 2, 3, 4},
        {19, 22, 0, 6},
        {12, -14, 7, 3}
    });
    std::cout << "Original system: " << std::endl;
    print_matrix<type> (linsys);
    std::cout << "Solved system: " << std::endl;
    print_matrix<type> (LinAlgo::gj (linsys));
    std::cout << std::endl;
    std::cout << "Inverse of 2x2 matrix:" << std::endl;
    LinAlgo::matrix<type> invert1 ({
        {1, 2},
        {3, 4}
    });
    std::cout << "Original matrix: " << std::endl;
    print_matrix<type> (invert1);
    std::cout << "Inverted matrix: " << std::endl;
    print_matrix<type> (LinAlgo::inverse (invert1));
    std::cout << std::endl;
    std::cout << "Inverse of 5x5 matrix:" << std::endl;
    LinAlgo::matrix<type> invert2 ({
        {1, 8, -9, 7, 5},
        {0, 1, 0, 4, 4},
        {0, 0, 1, 2, 5},
        {0, 0, 0, 1, -5},
        {0, 0, 0, 0, 1}
    });
    std::cout << "Original matrix: " << std::endl;
    print_matrix<type> (invert2);
    std::cout << "Inverted matrix: " << std::endl;
    print_matrix<type> (LinAlgo::inverse (invert2));
    std::cout << std::endl;
    std::cout << "Inverse of M1:" << std::endl;
    print_matrix<type> (LinAlgo::inverse (m1));
    std::cout << "The determinant of M1 is: " << m1.getDeterminant() << std::endl << std::endl;
    LinAlgo::matrix<type> detMat({
                                 {2, 1, 2},
                                 {1, 1, 1},
                                 {2, 2, 5}});
    std::cout << "The determinant of the following matrix should be 3: " << std::endl;
    print_matrix<type>(detMat);
    std::cout << "Determinant is: " << detMat.getDeterminant() << std::endl << std::endl;
    std::cout << "Matrix division: \nNumerator:" << std::endl;
    //LinAlgo::matrix<type> numerator({
    //                                {1, 2, 3, 4, 5},
    //                                {1, 2, 3, 4, 5},
    //                                {1, 2, 3, 4, 5},
    //                                {1, 2, 3, 4, 5},
    //                                {1, 2, 3, 4, 5}
    //                                });
    //print_matrix<type>(numerator);
    print_matrix<type>(invert2);
    std::cout << "Denominator: " << std::endl;
    print_matrix<type>(invert2);
    std::cout << "Result: " << std::endl;
    print_matrix(invert2.divide(invert2));
    std::cout << std::endl;
    LinAlgo::matrix<type> m1Sorted(m1);
    std::sort(m1Sorted.begin(), m1Sorted.end(), [](auto a, auto b) {
                return a < b;
              });
    std::cout << "M1 sorted:" << std::endl;
    print_matrix<type>(m1Sorted);
    std::cout << std::endl;
    //for (auto e : m1) {
    //    std::cout << e << '\t';
    //}
    //for (auto i = m1Sorted.begin(), j = m1Sorted.end(); i < j; i++, j--) {
    //    auto temp = *i;
    //    i = *j;
    //    j = temp;
    //}
    //print_matrix<type>(m1Sorted);
    //std::cout << std::endl;
    matrix<type> Q(0, 0), R(0, 0);
    matrix<type> orthogMe({{1,-1, 0},
                           {2, 0, 0},
                           {2, 2, 1}});
    qr(orthogMe, Q, R);
    std::cout << "Matrix before QR decomposition: " << std::endl;
    print_matrix<type>(orthogMe);
    std::cout << std::endl;
    std::cout << "Q: " << std::endl;
    print_matrix<type>(Q);
    std::cout << std::endl;
    std::cout << "R: " << std::endl;
    print_matrix<type>(R);
    std::cout << std::endl;
    std::cout << "Q*R: " << std::endl;
    print_matrix<type>(Q * R);
    std::cout << std::endl;
    qr(m1, Q, R);
    std::cout << "Matrix before QR decomposition: " << std::endl;
    print_matrix<type>(m1);
    std::cout << std::endl;
    std::cout << "Q: " << std::endl;
    print_matrix<type>(Q);
    std::cout << std::endl;
    std::cout << "R: " << std::endl;
    print_matrix<type>(R);
    std::cout << std::endl;
    std::cout << "Q*R: " << std::endl;
    print_matrix<type>(Q * R);
    std::cout << std::endl;

    m1.useGPU (true);
    m2.useGPU (true);
    t.start();
    LinAlgo::matrix<type> m3 = m1.add (m2);
    long long int t1 = t.getMicrosecondsElapsed();
    t.start();
    LinAlgo::matrix<type> m4 = m1.multiply (m2);
    long long int t2 = t.getMicrosecondsElapsed();
    t.start();
    LinAlgo::matrix<type> m7 = m1.subtract (m2);
    long long int t5 = t.getMicrosecondsElapsed();

    std::cout << "Using the GPU:" << std::endl;
    std::cout << "Addition:" << std::endl;
    print_matrix<type> (m3);
    std::cout << std::endl;
    std::cout << "Subtraction:" << std::endl;
    print_matrix<type> (m7);
    std::cout << std::endl;
    std::cout << "Multiplication:" << std::endl;
    print_matrix<type> (m4);
    std::cout << std::endl;

    //LinAlgo::AllUseGPU(false);
    m1.useGPU (false);
    m2.useGPU (false);
    t.start();
    LinAlgo::matrix<type> m5 = m1.add (m2);
    long long int t3 = t.getMicrosecondsElapsed();
    t.start();
    LinAlgo::matrix<type> m6 = m1.multiply (m2);
    long long int t4 = t.getMicrosecondsElapsed();
    t.start();
    LinAlgo::matrix<type> m8 = m1 - m2;
    long long int t6 = t.getMicrosecondsElapsed();

    std::cout << "Using the CPU:" << std::endl;
    std::cout << "Addition:" << std::endl;
    print_matrix<type> (m5);
    std::cout << std::endl;
    std::cout << "Subtraction:" << std::endl;
    print_matrix<type> (m8);
    std::cout << std::endl;
    std::cout << "Multiplication:" << std::endl;
    print_matrix<type> (m6);
    std::cout << std::endl;

    m1 = m1.subMatrix(0, 0, m1.getHeight() < m1.getWidth() ? m1.getHeight() : m1.getWidth(), m1.getHeight() < m1.getWidth() ? m1.getHeight() : m1.getWidth());
    m1.leaveDataOnGPU(true);
    m1.useGPU(true);
    LinAlgo::matrix<type> id = LinAlgo::identityMatrix<type>(m1.getHeight());
    id.leaveDataOnGPU(true);
    id.useGPU(true);
    t.start();
    id = id * id * id * id * m1;
    long long int t7 = t.getMicrosecondsElapsed();
    id.pullData();

    std::cout << "Microseconds for addition with GPU: " << t1 << std::endl;
    std::cout << "Microseconds for subtraction with GPU: " << t5 << std::endl;
    std::cout << "Microseconds for multiplication with GPU: " << t2 << std::endl;

    std::cout << "Microseconds for addition with CPU: " << t3 << std::endl;
    std::cout << "Microseconds for subtraction with CPU: " << t6 << std::endl;
    std::cout << "Microseconds for multiplication with CPU: " << t4 << std::endl;

    std::cout << "\nMicroseconds required to execute 5 chained multiplications of m1 with identity matrices" << std::endl;
    std::cout << "with leaveDataOnGPU active: " << t7 << std::endl;
    std::cout << std::endl;

    bool accuracy = true;
    bool overall_accuracy = true;
    size_t locX, locY;
    type V1, V2;
    accuracy = m5 == m3;
    std::cout << "The addition kernel is " << (accuracy ? "accurate" : "not accurate") << std::endl;
    if (!accuracy) {
        overall_accuracy = false;
        std::tie(locY, locX, V1, V2) = locateError<type>(m5, m3);
        std::cout << "\tThe offending location is: " << locY << ", " << locX << std::endl;
        std::cout << "\tThe CPU provided: " << V1 << std::endl;
        std::cout << "\tThe GPU provided: " << V2 << std::endl;
    }
    accuracy = m7 == m8;
    std::cout << "The subtraction kernel is " << (accuracy ? "accurate" : "not accurate") << std::endl;
    if (!accuracy) {
        overall_accuracy = false;
        std::tie(locY, locX, V1, V2) = locateError<type>(m7, m8);
        std::cout << "\tThe offending location is: " << locY << ", " << locX << std::endl;
        std::cout << "\tThe CPU provided: " << V1 << std::endl;
        std::cout << "\tThe GPU provided: " << V2 << std::endl;
    }
    accuracy = m6 == m4;
    std::cout << "The multiplication kernel is " << (accuracy ? "accurate" : "not accurate") << std::endl;
    if (!accuracy) {
        overall_accuracy = false;
        std::tie(locY, locX, V1, V2) = locateError<type>(m6, m4);
        std::cout << "\tThe offending location is: " << locY << ", " << locX << std::endl;
        std::cout << "\tThe CPU provided: " << V1 << std::endl;
        std::cout << "\tThe GPU provided: " << V2 << std::endl;
    }
    if (!overall_accuracy) {
        std::cout << std::endl;
        std::cout << "Note: Since the comparison is between CPU and GPU computation, floating point error may occur." << std::endl;
        std::cout << "If the offending values are reasonably close then the kernel is most likely still accurate." << std::endl;
        //std::cout << "Also, Location (0, 0) with values 0, and 0 mean that the error could not be located a second time." << std::endl;
    }
    std::cout << std::endl;

    LinAlgo::BreakDownGPU();

    return 0;
}

template <class ItemType>
void print_matrix (const LinAlgo::matrix<ItemType>& M, int padding) {
    if (M.getWidth() == 0) {
        std::cout << "Null matrix" << std::endl;
    }
    if (M.getWidth() <= 20) {
        std::string fstring = std::string("% ") + std::to_string(padding) + std::string("s");
        for (int i = 0; i < M.getHeight(); i++) {
            for (int j = 0; j < M.getWidth(); j++) {
                ItemType temp = M[i][j];
                std::string valStr;
                if (std::abs(temp) < 0.0009) {
                    valStr = std::string(" ") + std::to_string(0);
                } else {
                    std::string sign = temp < 0.0 ? "-" : " ";
                    ItemType val = std::abs(temp * 1000);
                    long int rounded = (int)(val + .5);
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
    }
}

const char* getErrorString (cl_int error) {
    switch (error) {
        // run-time and JIT compiler errors
        case 0:
            return "CL_SUCCESS";
        case -1:
            return "CL_DEVICE_NOT_FOUND";
        case -2:
            return "CL_DEVICE_NOT_AVAILABLE";
        case -3:
            return "CL_COMPILER_NOT_AVAILABLE";
        case -4:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5:
            return "CL_OUT_OF_RESOURCES";
        case -6:
            return "CL_OUT_OF_HOST_MEMORY";
        case -7:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8:
            return "CL_MEM_COPY_OVERLAP";
        case -9:
            return "CL_IMAGE_FORMAT_MISMATCH";
        case -10:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11:
            return "CL_BUILD_PROGRAM_FAILURE";
        case -12:
            return "CL_MAP_FAILURE";
        case -13:
            return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14:
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15:
            return "CL_COMPILE_PROGRAM_FAILURE";
        case -16:
            return "CL_LINKER_NOT_AVAILABLE";
        case -17:
            return "CL_LINK_PROGRAM_FAILURE";
        case -18:
            return "CL_DEVICE_PARTITION_FAILED";
        case -19:
            return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30:
            return "CL_INVALID_VALUE";
        case -31:
            return "CL_INVALID_DEVICE_TYPE";
        case -32:
            return "CL_INVALID_PLATFORM";
        case -33:
            return "CL_INVALID_DEVICE";
        case -34:
            return "CL_INVALID_CONTEXT";
        case -35:
            return "CL_INVALID_QUEUE_PROPERTIES";
        case -36:
            return "CL_INVALID_COMMAND_QUEUE";
        case -37:
            return "CL_INVALID_HOST_PTR";
        case -38:
            return "CL_INVALID_MEM_OBJECT";
        case -39:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40:
            return "CL_INVALID_IMAGE_SIZE";
        case -41:
            return "CL_INVALID_SAMPLER";
        case -42:
            return "CL_INVALID_BINARY";
        case -43:
            return "CL_INVALID_BUILD_OPTIONS";
        case -44:
            return "CL_INVALID_PROGRAM";
        case -45:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46:
            return "CL_INVALID_KERNEL_NAME";
        case -47:
            return "CL_INVALID_KERNEL_DEFINITION";
        case -48:
            return "CL_INVALID_KERNEL";
        case -49:
            return "CL_INVALID_ARG_INDEX";
        case -50:
            return "CL_INVALID_ARG_VALUE";
        case -51:
            return "CL_INVALID_ARG_SIZE";
        case -52:
            return "CL_INVALID_KERNEL_ARGS";
        case -53:
            return "CL_INVALID_WORK_DIMENSION";
        case -54:
            return "CL_INVALID_WORK_GROUP_SIZE";
        case -55:
            return "CL_INVALID_WORK_ITEM_SIZE";
        case -56:
            return "CL_INVALID_GLOBAL_OFFSET";
        case -57:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case -58:
            return "CL_INVALID_EVENT";
        case -59:
            return "CL_INVALID_OPERATION";
        case -60:
            return "CL_INVALID_GL_OBJECT";
        case -61:
            return "CL_INVALID_BUFFER_SIZE";
        case -62:
            return "CL_INVALID_MIP_LEVEL";
        case -63:
            return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64:
            return "CL_INVALID_PROPERTY";
        case -65:
            return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66:
            return "CL_INVALID_COMPILER_OPTIONS";
        case -67:
            return "CL_INVALID_LINKER_OPTIONS";
        case -68:
            return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000:
            return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001:
            return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002:
            return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003:
            return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004:
            return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005:
            return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default:
            return "Unknown OpenCL error";
    }
}

void checkReturn (cl_int ret) {
    if (ret != CL_SUCCESS)
    { std::cerr << getErrorString (ret) << std::endl; }
}

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
