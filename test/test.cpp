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

#ifndef DONT_USE_GPU
void checkReturn (cl_int ret);
#endif

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
    params.verbose = false;
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
    checkReturn (LinAlgo::InitGPU( ));

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

        //LinAlgo::matrix<double> m1 = generateShiftedPrimesMatrix<double>(10);
        //LinAlgo::matrix<double> m2 = LinAlgo::identityMatrix(10);
        //print_matrix(m1);
        //m1 = m1 + 5 + LinAlgo::inverse(m1) + generateShiftedPrimesMatrix<double>(10) + m2;
        //m1 = 5 + m1;
        //print_matrix(m1);

/*
        LinAlgo::matrix<float> M = LinAlgo::identityMatrix(10);
        M.useGPU(true);
        print_matrix(M);
        LinAlgo::matrix<float> N = LinAlgo::mapGPU(M, "__kernel void func(__global const float* A, __global float* B) { int i = get_global_id(0); B[i] = A[i] + 1; }");
        cl_int err;
        M.mapGPU("__kernel void func(__global float* A) { int i = get_global_id(0); A[i] += 1; }");
        M.pullData();
        print_matrix(M);
        printf("Both map functions executed %scorrectly\n", (M == N ? "" : "in"));
*/
/*
        LinAlgo::matrix<int> M1(10, 10);
        LinAlgo::matrix<int> M2 = LinAlgo::identityMatrix(M1.getWidth());

        for (auto iter = M1.begin(); iter < M1.end(); iter++) {
            *iter = rand();
        }
        LinAlgo::matrix<int> M3(M1);

        printf("The non-gpu matrix before multiplication: \n");
        print_matrix(M1);
        printf("\nThe gpu matrix before multiplication: \n");
        print_matrix(M3);
        printf("\n");

        for (int i = 0; i < 60; i++) {
            M1 = M1 * M2;
        }
        printf("The non-gpu matrix after multiplication: \n");
        print_matrix(M1);

        M3.useGPU(true);
        M3.leaveDataOnGPU(true);
        M2.useGPU(true);
        M2.leaveDataOnGPU(true);
        M3.pushData();
        M2.pushData();
        for (int i = 0; i < 60; i++) {
            M3 = M3 * M2;
        }
        M3.pullData();

        printf("\nThe gpu matrix after multiplication: \n");
        print_matrix(M3);
        printf("\nThe chained multiplication is %saccurate\n", (M3 == M1 ? "" : "not "));
*/
/*
        printf("Attempting QR Algorithm for finding eigenvalues of a matrix\n");
        printf("The matrix is: \n");
        LinAlgo::matrix<float> eigenattempt = generateShiftedPrimesMatrix<float>(5);
        print_matrix<float>(eigenattempt);
        eigenattempt.getDeterminant();
        LinAlgo::matrix<float> eigenvecs;
        Timer eigen;
        std::vector<float> eigenvals = LinAlgo::eigenvalues(eigenattempt, eigenvecs);
        printf("The calculated eigenvalues after %d microseconds are: ", eigen.getMicrosecondsElapsed());
        {
            auto v = eigenvals.begin();
            std::cout << *v;
            for (v++; v < eigenvals.end(); v++) {
                std::cout << ", " << *v;
            }
            std::cout << std::endl;
        }
        for (int i = 0; i < eigenvals.size(); i++) {
            printf("The eigenvector for [lambda]%d = %f is:\n", i + 1, eigenvals[i]);
            LinAlgo::matrix<float> eigenvec = LinAlgo::columnVector(eigenvecs.getColumn(i));
            print_matrix(eigenvec);
            printf("The matrix acting on the vector is:\n");
            print_matrix(eigenattempt * eigenvec);
            printf("The vector multiplied by [lambda]%d is:\n", i + 1);
            print_matrix(eigenvec * eigenvals[i]);
        }
*/

/*
        printf("Testing asynchronous map function\n");
        char (*threading_test)(char&) = [](char& doot) -> char {
                        std::this_thread::sleep_for(std::chrono::microseconds(200));
                        return (char) doot * 2;
                    };
        LinAlgo::matrix<char> testmap1(10, 10, 5);
        LinAlgo::matrix<char> testmap2(10, 10, 5);
        Timer t1;
        testmap1.map(threading_test, false);
        printf("Finished control after %d milliseconds\n", (int) t1.getMicrosecondsElapsed() / 1000);
        Timer t2;
        testmap2.map(threading_test, true);
        printf("Finished threaded after %d milliseconds\n", (int) t2.getMicrosecondsElapsed() / 1000);
        std::cout << std::endl;
*/
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

    LinAlgo::BreakDownGPU();

    return 0;
}

#ifndef DONT_USE_GPU
void checkReturn (cl_int ret) {
    if (ret != CL_SUCCESS)
    { std::cerr << LinAlgo::getErrorString(ret) << std::endl; }
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
    std::vector<std::string> tests = {"multiplication"};//addition test should never succeed, but meh
    for (auto t : tests) {
        size_t num_elements = testGPUvsCPUSpeed(log, verbose, t);
        if (num_elements != 0) {
            log << "GPU outpaces CPU in " << t << " starting at " << num_elements << " elements" << std::endl;
        }
    }

    LinAlgo::BreakDownGPU();
    LinAlgo::InitGPU();
    log << "Testing chained multiplications" << std::endl;
    if (verbose) {
        std::cout << "Testing chained multiplications..." << std::endl;
    }
    LinAlgo::matrix<int> ID = LinAlgo::identityMatrix(500);
    LinAlgo::matrix<int> M1(500, 500, 1);
    LinAlgo::matrix<int> M2(500, 500, 1);
    M2.useGPU(true);
    M2.leaveDataOnGPU(false);
    LinAlgo::matrix<int> M3(500, 500, 1);
    M3.useGPU(true);
    M3.leaveDataOnGPU(true);
    Timer chain_tests;
    {
        Timer t("Chained CPU multiplications");
        for (int i = 0; i < 60; i++) {
            M1 = M1 * ID;
        }
    }
    if (verbose) {
        printf("Finished CPU only chained multiplication after %d milliseconds\n", (int) chain_tests.getMicrosecondsElapsed() / 1000);
    }
    log << "Finished CPU only chained multiplication after " << chain_tests.getMicrosecondsElapsed() / 1000 << " milliseconds" << std::endl;
    ID.useGPU(true);
    chain_tests.start();
    {
        Timer t("Chained GPU multiplications (data pull each round)");
        for (int i = 0; i < 60; i++) {
            M2 = M2 * ID;
        }
    }
    int chained_t1 = (int) chain_tests.getMicrosecondsElapsed() / 1000;
    if (verbose) {
        printf("Finished GPU chained multiplication with data pull after %d milliseconds\n", chained_t1);
    }
    log << "Finished GPU chained multiplication with data pull after " << chained_t1 << " milliseconds" << std::endl;
    ID.leaveDataOnGPU(true);
    chain_tests.start();
    {
        Timer t("Chained GPU multiplications (leave data on GPU)");
        //it looks like clCreateCommandQueue has a memory leak on amd drivers, and /that's/ why this crashes... not sure tho.
        for (int i = 0; i < 60; i++) {
            M3 = M3 * ID;
        }
        M3.pullData();
    }
    int chained_t2 = (int) chain_tests.getMicrosecondsElapsed() / 1000;
    if (verbose) {
        printf("Finished GPU chained multiplication with leaveData active after %d milliseconds\n", chained_t2);
    }
    log << "Finished GPU chained multiplication with leaveData active after " << chained_t2 << " milliseconds" << std::endl;
    log << "Chained multiplication tests: " << (chained_t1 <= chained_t2 ? "failed" : "success") << std::endl;
    if (verbose) {
        std::cout << "Chained multiplication tests: " << (chained_t1 <= chained_t2 ? "failed" : "success") << std::endl;
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
    bool keepTrying = true;
    do {
        if (verbose) {
            std::cout << "Trying " << dim << "x" << dim << " matrix..." << std::endl;
        }
        gpuMatrix = LinAlgo::matrix<double>(dim, dim);
        gpuMatrix.useGPU(true);
        cpuMatrix = LinAlgo::matrix<double>(dim, dim);
        LinAlgo::matrix<double> junkResult(0, 0);
        for (auto i = gpuMatrix.begin(), j = cpuMatrix.begin(); i < gpuMatrix.end(); i++, j++) {
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
    } while (gpuTime > cpuTime && dim < 1500);

    if (!keepTrying) {
        if (verbose) {
            std::cout << "GPU unable to outpace CPU" << std::endl;
        }
        log << "GPU was unable to match CPU before GPU allocation error at " << (dim * dim) << " elements (failed)" << std::endl;
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
        for (auto i = m1.begin(), j = m2.begin(); i < m1.end(); i++, j++) {
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
        for (auto i = m1.begin(), j = m2.begin(); i < m1.end(); i++, j++) {
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
        for (auto i = m1.begin(), j = m2.begin(); i < m1.end(); i++, j++) {
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
        for (auto i = m1.begin(), j = m2.begin(); i < m1.end(); i++, j++) {
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
        for (auto i = m1.begin(), j = m2.begin(); i < m1.end(); i++, j++) {
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
        for (auto i = m1.begin(), j = m2.begin(); i < m1.end(); i++, j++) {
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
        for (auto i = m1.begin(); i < m1.end(); i++) {
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
        for (auto i = m1.begin(); i < m1.end(); i++) {
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
        for (auto i = m1.begin(); i < m1.end(); i++) {
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
        for (auto i = m1.begin(); i < m1.end(); i++) {
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
        for (auto i = m1.begin(); i < m1.end(); i++) {
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
        for (auto i = m1.begin(); i < m1.end(); i++) {
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

