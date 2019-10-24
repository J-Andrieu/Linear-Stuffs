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
#include "tests/utilities.h"

int HEIGHT = 15;
int WIDTH = 15;
typedef float type;

void checkReturn (cl_int ret);

void CheckAccuracy(size_t height, size_t width, std::string logfile, bool verbose);

typedef struct {
    int dimensions[2];
    bool accuracy_test;
    bool speed_test;
    bool operator_test;
    bool methods_test;
    bool matrix_test;
    bool all;
    bool verbose;
    std::string log_file;
} parameters;

bool validateParameters(parameters& params);//will set defaults if necessary

/*MATRIX TESTING*/
int main (int argc, char* argv[]) {
    parameters params;
    params.dimensions[0] = 0;
    params.dimensions[1] = 0;
    params.accuracy_test = false;
    params.speed_test = false;
    params.operator_test = false;
    params.methods_test = false;
    params.matrix_test = false;
    params.all = false;
    params.log_file = "";

    CMDParser parser;
    parser.bindVar<int[2]>("-d", params.dimensions, 2, "Defines the dimensions of matrices for certain tests");
    parser.bindVar<bool>("-accuracy", params.accuracy_test, 0, "Tests the accuracy of matrix operations with known results");
    parser.bindVar<bool>("-speed", params.speed_test, 0, "Tests the speed of matrix operations on and off the GPU");
    parser.bindVar<bool>("-operators", params.operator_test, 0, "Tests vector and matrix operator overloads");
    parser.bindVar<bool>("-methods", params.methods_test, 0, "Tests methods and procedures against know outcomes");
    parser.bindVar<bool>("-matrix", params.matrix_test, 0, "IDK man, as i need types of tests i'll add them :P");
    parser.bindVar<bool>("-a", params.all, 0, "Run all test");
    parser.bindVar<bool>("-v", params.verbose, 0, "Display matrices and test data to console");
    parser.bindVar<std::string>("-log", params.log_file, 1, "Sets the location to log test results");
    if (!parser.parse(argc, argv)) {
      return -1;
    }


    if (params.dimensions[0] == 0) {
        if (params.dimensions[1] == 0) {
            HEIGHT = 15;
            WIDTH = 15;
        } else {
            HEIGHT = params.dimensions[1];
            WIDTH = params.dimensions[1];
        }
    } else {
        HEIGHT = params.dimensions[0];
        if (params.dimensions[1] == 0) {
            WIDTH = HEIGHT;
        } else {
            WIDTH = params.dimensions[1];
        }
    }

#ifndef DONT_USE_GPU
    checkReturn (LinAlgo::InitGPU());

    if (params.verbose) {
        std::cout << (LinAlgo::IsGPUInitialized() ? "The GPU is initialized" : "The GPU is not initialized") << std::endl;
    }
#endif // DONT_USE_GPU

    if (params.log_file != "") {
        std::ofstream log(params.log_file, std::ofstream::out);
        time_t currentTime = std::chrono::system_clock::to_time_t (std::chrono::system_clock::now());
        log << "============================================" << std::endl;
        log << "Beginning LinAlgo Tests: " << std::put_time(std::localtime(&currentTime), "%Y-%m-%d %X") << std::endl;
        log << "============================================" << std::endl;
#ifndef DONT_USE_GPU
        log << (LinAlgo::IsGPUInitialized() ? "The GPU is initialized" : "The GPU is not initialized") << std::endl;
#endif // DONT_USE_GPU
        log.close();
    }

    {
        Timer t_test("Full testing suite");
        if(params.accuracy_test || params.all) {
            CheckAccuracy(HEIGHT, WIDTH, params.log_file, params.verbose);
        }
    }

    if (params.log_file != "") {

        std::ofstream log(params.log_file, std::ofstream::out | std::ofstream::app);
        log << std::endl;
        log.close();

        if (params.verbose) {
            Timers::toScreen = true;
        }
        Timers::setLogFile(params.log_file);
        Timers::logNamedTimers();
    }

    return 0;
}

#ifndef DONT_USE_GPU
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
#endif
