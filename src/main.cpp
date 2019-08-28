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

#include "../include/LinAlgo.hpp"
#include "../include/Timer.h"

//less than 25x25 causes bsod with the destructor uncommented for some reason
const int HEIGHT = 15;
const int WIDTH = 5;
typedef float type;

template <class ItemType>
void print_matrix (const LinAlgo::matrix<ItemType>& M);
void checkReturn (cl_int ret);
//template <class ItemType>
//bool checkAccuracy(LinAlgo::matrix<ItemType>& gpu, LinAlgo::matrix<ItemType>& cpu);

std::tuple<size_t, size_t> getDimensions (matrix<type> &M) {
    return {M.getHeight(), M.getWidth()};
}

/*MATRIX TESTING*/
int main (void) {
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
            m1.set (i, j, (type) (i + j));
            m2.set (j, i, (type) (i * j));
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
    std::cout << "Gauss-Jordan Elimination M1:" << std::endl;
    print_matrix<type> (LinAlgo::gj (m1));
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
    LinAlgo::matrix<type> m8 = m1.subtract (m2);
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

    std::cout << "Microseconds for addition with GPU: " << t1 << std::endl;
    std::cout << "Microseconds for subtraction with GPU: " << t5 << std::endl;
    std::cout << "Microseconds for multiplication with GPU: " << t2 << std::endl;

    std::cout << "Microseconds for addition with CPU: " << t3 << std::endl;
    std::cout << "Microseconds for subtraction with CPU: " << t6 << std::endl;
    std::cout << "Microseconds for multiplication with CPU: " << t4 << std::endl;
    std::cout << std::endl;

    std::cout << "The addition kernel is " << (m5 == m3 ? "accurate" : "not accurate") << std::endl;
    std::cout << "The subtraction kernel is " << (m7 == m8 ? "accurate" : "not accurate") << std::endl;
    std::cout << "The multiplication kernel is " << (m6 == m4 ? "accurate" : "not accurate") << std::endl;


//  if (HEIGHT < 15) {
//      std::cout << "Now to chain multiply all those matrices together just for the fun of it and see what pops out..." << std::endl;
//
//      AllUseGPU(true);
//
//      t.start();
//      matrix<int> finalmat = m1.multiply(m2).multiply(m3).multiply(m4).multiply(m5).multiply(m6);
//      long long int t5 = t.getMicrosecondsElapsed();
//      std::cout << "Well, that took " << t5 << " microseconds..." << std::endl;
//      std::cout << "Anyways, this is what the result is:" << std::endl;
//      print_matrix<int>(finalmat);
//  }


//  print_matrix<int>((matrix<int> &) m1.multiply((matrix<int>&) m1.transpose()));
//  std::cout << std::endl;
//  print_matrix<int>((matrix<int> &) m1.transpose().multiply(m1));

    LinAlgo::BreakDownGPU();

//  int x;
//  std::cin >> x;

    return 0;
}

template <class ItemType>
void print_matrix (const LinAlgo::matrix<ItemType>& M) {
    if (M.getWidth() == 0) {
        std::cout << "Null matrix" << std::endl;
    }
    if (M.getWidth() < 20) {
        for (int i = 0; i < M.getHeight(); i++) {
            for (int j = 0; j < M.getWidth(); j++) {
                //that size check is cuz floating point error is annoying to look at
                //also -9.25596e+61 still gets printed... really dumb number
                std::cout << (std::abs (M[i][j]) > 0.000001f ? M[i][j] : 0) << '\t';
            }
            std::cout << std::endl;
        }
    }
}

/*
template <class ItemType>
bool checkAccuracy(LinAlgo::matrix<ItemType>& gpu, LinAlgo::matrix<ItemType>& cpu) {
    if (gpu.getHeight() != cpu.getHeight() || gpu.getWidth() != gpu.getWidth()) {
        return false;
    }
    for (size_t i = 0; i < gpu.getHeight(); i++) {
        for (size_t j = 0; j < gpu.getWidth(); j++) {
            if (gpu[i][j] != cpu[i][j]) {
                return false;
            }
        }
    }
    return true;
}
*/

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
