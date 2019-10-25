/**
* @file main.cpp
*
* @brief Test driver for matrix class
*
* @notes Requires matrix.h and Timer.h
*/

#include "utilities.h"

//Default height and width
int HEIGHT = 15;
int WIDTH = 15;

void checkReturn (cl_int ret);

void CheckSpeed(std::string logfile, bool verbose);
void CheckAccuracy(size_t height, size_t width, std::string logfile, bool verbose);
void CheckMethods(size_t height, size_t width, std::string logfile, bool verbose);
void CheckOperators(size_t height, size_t width, std::string logfile, bool verbose);

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
    params.log_file = "log.txt";

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
/*
        if(params.methods_test || params.all) {
            CheckMethods(HEIGHT, WIDTH, params.log_file, params.verbose);
        }

        if(params.operator_test || params.all) {
            CheckOperators(HEIGHT, WIDTH, params.log_file, params.verbose);
        }
*/
        if(params.speed_test || params.all) {
            CheckSpeed(params.log_file, params.verbose);
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

#ifndef DONT_USE_GPU
size_t testGPUvsCPUSpeed(std::ofstream& log, bool verbose, std::string test);
#endif // DONT_USE_GPU

void CheckSpeed(std::string logfile, bool verbose) {
    std::srand((unsigned) std::time(NULL));
    Timer t_accuracy("Full speed suite");
    std::ofstream log(logfile, std::ofstream::out | std::ofstream::app);

    log << std::endl;
    log << "Beginning speed tests" << std::endl;
#ifdef DONT_USE_GPU
    log << "No gpu, nothing to test" << std::endl;
#else
    std::vector<std::string> tests = {"addition", "multiplication"};
    for (auto t : tests) {
        size_t num_elements = testGPUvsCPUSpeed(log, verbose, t);
        log << "GPU outpaces CPU in " << t << " starting at " << num_elements << " elements" << std::endl;
    }
#endif
}

#ifndef DONT_USE_GPU
size_t testGPUvsCPUSpeed(std::ofstream& log, bool verbose, std::string test) {
    Timer t_test(std::string("Speed: ") + test);
    if (verbose) {
        std::cout << "Beginning " << test << " test" << std::endl;
    }
    size_t dim = 100;
    long long gpuTime, cpuTime;
    LinAlgo::matrix<double> gpuMatrix(dim, dim);
    LinAlgo::matrix<double> cpuMatrix(dim, dim);
    do {
        if (verbose) {
            std::cout << "Trying " << dim << "x" << dim << " matrix..." << std::endl;
        }
        gpuMatrix = LinAlgo::matrix<double>(dim, dim, 0, true);
        cpuMatrix = LinAlgo::matrix<double>(dim, dim, 0, false);
        LinAlgo::matrix<double> junkResult(0, 0);
        for (auto i = gpuMatrix.begin(), j = cpuMatrix.begin(); i <= gpuMatrix.end(); i++, j++) {
            *i = static_cast<double>(rand());
            *j = static_cast<double>(rand());
        }
        gpuMatrix.pushData();
        if (test == "addition") {
            Timer cpu;
            junkResult = cpuMatrix + cpuMatrix;
            cpuTime = cpu.getMicrosecondsElapsed();
            Timer gpu;
            junkResult = gpuMatrix + gpuMatrix;
            gpuTime = gpu.getMicrosecondsElapsed();
        }
        if (test == "multiplication") {
            Timer cpu;
            junkResult = cpuMatrix * cpuMatrix;
            cpuTime = cpu.getMicrosecondsElapsed();
            Timer gpu;
            junkResult = gpuMatrix * gpuMatrix;
            gpuTime = gpu.getMicrosecondsElapsed();
        }
        if (verbose) {
            std::cout << "GPU time: " << gpuTime << std::endl;
            std::cout << "CPU time: " << cpuTime << std::endl;
        }
        if (gpuTime <= cpuTime) {
            log << "GPU time: " << gpuTime << std::endl;
            log << "CPU time: " << cpuTime << std::endl;
        }
        dim += 10;
    } while (gpuTime > cpuTime && dim < 3000);

    if (dim >= 1000) {
        if (verbose) {
            std::cout << "GPU unable to outpace CPU" << std::endl;
        }
        log << "GPU was unable to match CPU by 9000000 elements (failed)" << std::endl;
        return 0;
    }

    return dim * dim;
}
#endif // DONT_USE_GPU

typedef struct {
    bool char_test;
    bool short_test;
    bool int_test;
    bool long_test;
    bool float_test;
    bool double_test;
} test_out;
bool passFail(test_out& test);
std::vector<std::string> whichFailed(test_out& test);
void logFailure(std::ofstream& log, test_out& test, bool verbose);

#ifndef DONT_USE_GPU
test_out testGPUMatrixArithmetic(size_t height, size_t width, bool verbose, std::string test);
test_out testGPUMatrixScalarArithmetic(size_t height, size_t width, bool verbose, std::string test);
#endif

void CheckAccuracy(size_t height, size_t width, std::string logfile, bool verbose) {
    std::srand((unsigned) std::time(NULL));
    Timer t_accuracy("Full accuracy suite");
    std::ofstream log(logfile, std::ofstream::out | std::ofstream::app);

    log << std::endl;
    log << "Beginning accuracy tests" << std::endl;
#ifdef DONT_USE_GPU
    log << "No gpu, nothing to test" << std::endl;
#endif // DONT_USE_GPU

#ifndef DONT_USE_GPU
    std::vector<std::string> tests = {"addition", "subtraction", "multiplication"};
    for (auto t : tests) {
        test_out result = testGPUMatrixArithmetic(height, width, verbose, t);
        bool test_passed = passFail(result);
        log << "GPU matrix " << t << " tests: " << (test_passed ? "passed" : "failed") << std::endl;
        if (!test_passed) {
            logFailure(log, result, verbose);
        }
    }

    tests.push_back("division");
    for (auto t : tests) {
        test_out result = testGPUMatrixScalarArithmetic(height, width, verbose, t);
        bool test_passed = passFail(result);
        log << "GPU matrix/scalar " << t << " tests: " << (test_passed ? "passed" : "failed") << std::endl;
        if (!test_passed) {
            logFailure(log, result, verbose);
        }
    }
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

std::vector<std::string> whichFailed(test_out& test) {
    std::vector<std::string> failed;
    if (!test.char_test)
        failed.push_back("char");
    if (!test.short_test)
        failed.push_back("short");
    if (!test.int_test)
        failed.push_back("int");
    if (!test.long_test)
        failed.push_back("long");
    if (!test.float_test)
        failed.push_back("float");
    if (!test.double_test)
        failed.push_back("double");
    return failed;
}

void logFailure(std::ofstream& log, test_out& test, bool verbose) {
    std::vector<std::string> failedTests = whichFailed(test);
    if (verbose) {
        std::cout << "Failed tests: " << std::flush;
        log << "Failed tests: " << std::flush;
        for (auto s : failedTests) {
            std::cout << s << " " << std::flush;
            log << s << " " << std::flush;
        }
        std::cout << std::endl;
        log << std::endl;
    } else {
        log << "Failed tests: " << std::flush;
        for (auto s : failedTests) {
            log << s << " " << std::flush;
        }
        log << std::endl;
    }
}

#ifndef DONT_USE_GPU
test_out testGPUMatrixArithmetic(size_t height, size_t width, bool verbose, std::string test) {
    if (verbose) {
        std::cout << "Beginning GPU Accuracy test: matrix " << test << std::endl;
    }
    bool char_test, short_test, int_test, long_test, float_test, double_test;
    char_test = short_test = int_test = long_test = float_test = double_test = false;

    {
        Timer t_char(std::string("Accuracy: GPU matrix ") + test + std::string(": character"));
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
        LinAlgo::matrix<char> result_CPU(0, 0);
        LinAlgo::matrix<char> result_GPU(0, 0);
        if (test == "subtraction") {
            result_CPU = m1 - m2;
            m1.useGPU(true);
            m2.useGPU(true);
            result_GPU = m1 - m2;
        } else if (test == "addition") {
            result_CPU = m1 + m2;
            m1.useGPU(true);
            m2.useGPU(true);
            result_GPU = m1 + m2;
        } else if (test == "multiplication") {
            result_CPU = m1 * m2;
            m1.useGPU(true);
            m2.useGPU(true);
            result_GPU = m1 * m2;
        }

        if (verbose) {
            std::cout << "GPU " << test << " results:" << std::endl;
            print_matrix<char>(result_GPU);
        }
        char_test = result_GPU == result_CPU;
    }

    {
        Timer t_char(std::string("Accuracy: GPU matrix ") + test + std::string(": short"));
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
        LinAlgo::matrix<short> result_CPU(0, 0);
        LinAlgo::matrix<short> result_GPU(0, 0);
        if (test == "subtraction") {
            result_CPU = m1 - m2;
            m1.useGPU(true);
            m2.useGPU(true);
            result_GPU = m1 - m2;
        } else if (test == "addition") {
            result_CPU = m1 + m2;
            m1.useGPU(true);
            m2.useGPU(true);
            result_GPU = m1 + m2;
        } else if (test == "multiplication") {
            result_CPU = m1 * m2;
            m1.useGPU(true);
            m2.useGPU(true);
            result_GPU = m1 * m2;
        }

        if (verbose) {
            std::cout << "GPU " << test << " results:" << std::endl;
            print_matrix<short>(result_GPU);
        }
        short_test = result_GPU == result_CPU;
    }

    {
        Timer t_char(std::string("Accuracy: GPU matrix ") + test + std::string(": integer"));
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
        LinAlgo::matrix<int> result_CPU(0, 0);
        LinAlgo::matrix<int> result_GPU(0, 0);
        if (test == "subtraction") {
            result_CPU = m1 - m2;
            m1.useGPU(true);
            m2.useGPU(true);
            result_GPU = m1 - m2;
        } else if (test == "addition") {
            result_CPU = m1 + m2;
            m1.useGPU(true);
            m2.useGPU(true);
            result_GPU = m1 + m2;
        } else if (test == "multiplication") {
            result_CPU = m1 * m2;
            m1.useGPU(true);
            m2.useGPU(true);
            result_GPU = m1 * m2;
        }

        if (verbose) {
            std::cout << "GPU " << test << " results:" << std::endl;
            print_matrix<int>(result_GPU, -13);
        }
        int_test = result_GPU == result_CPU;
    }

    {
        Timer t_char(std::string("Accuracy: GPU matrix ") + test + std::string(": long"));
        LinAlgo::matrix<long> m1(height, width);
        LinAlgo::matrix<long> m2(height, width);
        for (auto i = m1.begin(), j = m2.begin(); i <= m1.end(); i++, j++) {
            *i = rand() % LONG_MAX;
            *j = rand() % LONG_MAX;
        }
        if (verbose) {
            std::cout << "M1 long test:" << std::endl;
            print_matrix<long>(m1, -13);
            std::cout << "M2 long test:" << std::endl;
            print_matrix<long>(m2, -13);
        }
        LinAlgo::matrix<long> result_CPU(0, 0);
        LinAlgo::matrix<long> result_GPU(0, 0);
        if (test == "subtraction") {
            result_CPU = m1 - m2;
            m1.useGPU(true);
            m2.useGPU(true);
            result_GPU = m1 - m2;
        } else if (test == "addition") {
            result_CPU = m1 + m2;
            m1.useGPU(true);
            m2.useGPU(true);
            result_GPU = m1 + m2;
        } else if (test == "multiplication") {
            result_CPU = m1 * m2;
            m1.useGPU(true);
            m2.useGPU(true);
            result_GPU = m1 * m2;
        }

        if (verbose) {
            std::cout << "GPU " << test << " results:" << std::endl;
            print_matrix<long>(result_GPU, -13);
        }
        long_test = result_GPU == result_CPU;
    }

    {
        Timer t_char(std::string("Accuracy: GPU matrix ") + test + std::string(": float"));
        LinAlgo::matrix<float> m1(height, width);
        LinAlgo::matrix<float> m2(height, width);
        for (auto i = m1.begin(), j = m2.begin(); i <= m1.end(); i++, j++) {
            *i = static_cast<float>(rand());
            *j = static_cast<float>(rand());
        }
        if (verbose) {
            std::cout << "M1 float test:" << std::endl;
            print_matrix<float>(m1, -15);
            std::cout << "M2 float test:" << std::endl;
            print_matrix<float>(m2, -15);
        }
        LinAlgo::matrix<float> result_CPU(0, 0);
        LinAlgo::matrix<float> result_GPU(0, 0);
        if (test == "subtraction") {
            result_CPU = m1 - m2;
            m1.useGPU(true);
            m2.useGPU(true);
            result_GPU = m1 - m2;
        } else if (test == "addition") {
            result_CPU = m1 + m2;
            m1.useGPU(true);
            m2.useGPU(true);
            result_GPU = m1 + m2;
        } else if (test == "multiplication") {
            result_CPU = m1 * m2;
            m1.useGPU(true);
            m2.useGPU(true);
            result_GPU = m1 * m2;
        }

        if (verbose) {
            std::cout << "GPU " << test << " results:" << std::endl;
            print_matrix<float>(result_GPU, -15);
        }
        float_test = result_GPU == result_CPU;
    }

    {
        Timer t_char(std::string("Accuracy: GPU matrix ") + test + std::string(": double"));
        LinAlgo::matrix<double> m1(height, width);
        LinAlgo::matrix<double> m2(height, width);
        for (auto i = m1.begin(), j = m2.begin(); i <= m1.end(); i++, j++) {
            *i = static_cast<double>(rand());
            *j = static_cast<double>(rand());
        }
        if (verbose) {
            std::cout << "M1 double test:" << std::endl;
            print_matrix<double>(m1, -13);
            std::cout << "M2 double test:" << std::endl;
            print_matrix<double>(m2, -13);
        }
        LinAlgo::matrix<double> result_CPU(0, 0);
        LinAlgo::matrix<double> result_GPU(0, 0);
        if (test == "subtraction") {
            result_CPU = m1 - m2;
            m1.useGPU(true);
            m2.useGPU(true);
            result_GPU = m1 - m2;
        } else if (test == "addition") {
            result_CPU = m1 + m2;
            m1.useGPU(true);
            m2.useGPU(true);
            result_GPU = m1 + m2;
        } else if (test == "multiplication") {
            result_CPU = m1 * m2;
            m1.useGPU(true);
            m2.useGPU(true);
            result_GPU = m1 * m2;
        }

        if (verbose) {
            std::cout << "GPU " << test << " results:" << std::endl;
            print_matrix<double>(result_GPU, -13);
        }
        double_test = result_GPU == result_CPU;
    }

    return {char_test, short_test, int_test, long_test, float_test, double_test};
}

test_out testGPUMatrixScalarArithmetic(size_t height, size_t width, bool verbose, std::string test) {
    if (verbose) {
        std::cout << "Beginning GPU Accuracy test: scalar " << test << std::endl;
    }
    bool char_test, short_test, int_test, long_test, float_test, double_test;
    char_test = short_test = int_test = long_test = float_test = double_test = false;

    {
        Timer t_char(std::string("Accuracy: GPU scalar ") + test + std::string(": character"));
        LinAlgo::matrix<char> m1(height, width);
        for (auto i = m1.begin(); i <= m1.end(); i++) {
            *i = rand() % CHAR_MAX;
        }
        char scalar = rand() % CHAR_MAX;
        if (verbose) {
            std::cout << "M1 character test:" << std::endl;
            print_matrix<char>(m1);
            std::cout << "Scalar1 character test: " << scalar << std::endl;
        }
        LinAlgo::matrix<char> result_CPU(0, 0);
        LinAlgo::matrix<char> result_GPU(0, 0);
        if (test == "subtraction") {
            result_CPU = m1 - scalar;
            m1.useGPU(true);
            result_GPU = m1 - scalar;
        } else if (test == "addition") {
            result_CPU = m1 + scalar;
            m1.useGPU(true);
            result_GPU = m1 + scalar;
        } else if (test == "multiplication") {
            result_CPU = m1 * scalar;
            m1.useGPU(true);
            result_GPU = m1 * scalar;
        } else if (test == "division") {
            result_CPU = m1 / scalar;
            m1.useGPU(true);
            result_GPU = m1 / scalar;
        }

        if (verbose) {
            std::cout << "GPU " << test << " results:" << std::endl;
            print_matrix<char>(result_GPU);
        }
        char_test = result_GPU == result_CPU;
    }

    {
        Timer t_char(std::string("Accuracy: GPU scalar ") + test + std::string(": short"));
        LinAlgo::matrix<short> m1(height, width);
        for (auto i = m1.begin(); i <= m1.end(); i++) {
            *i = rand() % SHRT_MAX;
        }
        short scalar = rand() % SHRT_MAX;
        if (verbose) {
            std::cout << "M1 short test:" << std::endl;
            print_matrix<short>(m1);
            std::cout << "Scalar1 short test: " << scalar << std::endl;
        }
        LinAlgo::matrix<short> result_CPU(0, 0);
        LinAlgo::matrix<short> result_GPU(0, 0);
        if (test == "subtraction") {
            result_CPU = m1 - scalar;
            m1.useGPU(true);
            result_GPU = m1 - scalar;
        } else if (test == "addition") {
            result_CPU = m1 + scalar;
            m1.useGPU(true);
            result_GPU = m1 + scalar;
        } else if (test == "multiplication") {
            result_CPU = m1 * scalar;
            m1.useGPU(true);
            result_GPU = m1 * scalar;
        } else if (test == "division") {
            result_CPU = m1 / scalar;
            m1.useGPU(true);
            result_GPU = m1 / scalar;
        }

        if (verbose) {
            std::cout << "GPU " << test << " results:" << std::endl;
            print_matrix<short>(result_GPU);
        }
        short_test = result_GPU == result_CPU;
    }

    {
        Timer t_char(std::string("Accuracy: GPU scalar ") + test + std::string(": integer"));
        LinAlgo::matrix<int> m1(height, width);
        for (auto i = m1.begin(); i <= m1.end(); i++) {
            *i = rand() % INT_MAX;
        }
        int scalar = rand() % INT_MAX;
        if (verbose) {
            std::cout << "M1 integer test:" << std::endl;
            print_matrix<int>(m1, -13);
            std::cout << "Scalar1 integer test: " << scalar << std::endl;
        }
        LinAlgo::matrix<int> result_CPU(0, 0);
        LinAlgo::matrix<int> result_GPU(0, 0);
        if (test == "subtraction") {
            result_CPU = m1 - scalar;
            m1.useGPU(true);
            result_GPU = m1 - scalar;
        } else if (test == "addition") {
            result_CPU = m1 + scalar;
            m1.useGPU(true);
            result_GPU = m1 + scalar;
        } else if (test == "multiplication") {
            result_CPU = m1 * scalar;
            m1.useGPU(true);
            result_GPU = m1 * scalar;
        } else if (test == "division") {
            result_CPU = m1 / scalar;
            m1.useGPU(true);
            result_GPU = m1 / scalar;
        }

        if (verbose) {
            std::cout << "GPU " << test << " results:" << std::endl;
            print_matrix<int>(result_GPU, -13);
        }
        int_test = result_GPU == result_CPU;
    }

    {
        Timer t_char(std::string("Accuracy: GPU scalar ") + test + std::string(": long"));
        LinAlgo::matrix<long> m1(height, width);
        for (auto i = m1.begin(); i <= m1.end(); i++) {
            *i = rand() % LONG_MAX;
        }
        long scalar = rand() % LONG_MAX;
        if (verbose) {
            std::cout << "M1 long test:" << std::endl;
            print_matrix<long>(m1, -13);
            std::cout << "Scalar1 long test: " << scalar << std::endl;
        }
        LinAlgo::matrix<long> result_CPU(0, 0);
        LinAlgo::matrix<long> result_GPU(0, 0);
        if (test == "subtraction") {
            result_CPU = m1 - scalar;
            m1.useGPU(true);
            result_GPU = m1 - scalar;
        } else if (test == "addition") {
            result_CPU = m1 + scalar;
            m1.useGPU(true);
            result_GPU = m1 + scalar;
        } else if (test == "multiplication") {
            result_CPU = m1 * scalar;
            m1.useGPU(true);
            result_GPU = m1 * scalar;
        } else if (test == "division") {
            result_CPU = m1 / scalar;
            m1.useGPU(true);
            result_GPU = m1 / scalar;
        }

        if (verbose) {
            std::cout << "GPU " << test << " results:" << std::endl;
            print_matrix<long>(result_GPU, -13);
        }
        long_test = result_GPU == result_CPU;
    }

    {
        Timer t_char(std::string("Accuracy: GPU scalar ") + test + std::string(": float"));
        LinAlgo::matrix<float> m1(height, width);
        for (auto i = m1.begin(); i <= m1.end(); i++) {
            *i = static_cast<float>(rand());
        }
        float scalar = static_cast<float>(rand());
        if (verbose) {
            std::cout << "M1 float test:" << std::endl;
            print_matrix<float>(m1, -15);
            std::cout << "Scalar1 float test: " << scalar << std::endl;
        }
        LinAlgo::matrix<float> result_CPU(0, 0);
        LinAlgo::matrix<float> result_GPU(0, 0);
        if (test == "subtraction") {
            result_CPU = m1 - scalar;
            m1.useGPU(true);
            result_GPU = m1 - scalar;
        } else if (test == "addition") {
            result_CPU = m1 + scalar;
            m1.useGPU(true);
            result_GPU = m1 + scalar;
        } else if (test == "multiplication") {
            result_CPU = m1 * scalar;
            m1.useGPU(true);
            result_GPU = m1 * scalar;
        } else if (test == "division") {
            result_CPU = m1 / scalar;
            m1.useGPU(true);
            result_GPU = m1 / scalar;
        }

        if (verbose) {
            std::cout << "GPU " << test << " results:" << std::endl;
            print_matrix<float>(result_GPU, -15);
        }
        float_test = result_GPU == result_CPU;
    }

    {
        Timer t_char(std::string("Accuracy: GPU scalar ") + test + std::string(": double"));
        LinAlgo::matrix<double> m1(height, width);
        for (auto i = m1.begin(); i <= m1.end(); i++) {
            *i = static_cast<double>(rand());
        }
        double scalar = static_cast<double>(rand());
        if (verbose) {
            std::cout << "M1 double test:" << std::endl;
            print_matrix<double>(m1, -13);
            std::cout << "Scalar1 double test: " << scalar << std::endl;
        }
        LinAlgo::matrix<double> result_CPU(0, 0);
        LinAlgo::matrix<double> result_GPU(0, 0);
        if (test == "subtraction") {
            result_CPU = m1 - scalar;
            m1.useGPU(true);
            result_GPU = m1 - scalar;
        } else if (test == "addition") {
            result_CPU = m1 + scalar;
            m1.useGPU(true);
            result_GPU = m1 + scalar;
        } else if (test == "multiplication") {
            result_CPU = m1 * scalar;
            m1.useGPU(true);
            result_GPU = m1 * scalar;
        } else if (test == "division") {
            result_CPU = m1 / scalar;
            m1.useGPU(true);
            result_GPU = m1 / scalar;
        }

        if (verbose) {
            std::cout << "GPU " << test << " results:" << std::endl;
            print_matrix<double>(result_GPU, -13);
        }
        double_test = result_GPU == result_CPU;
    }

    return {char_test, short_test, int_test, long_test, float_test, double_test};
}
#endif

