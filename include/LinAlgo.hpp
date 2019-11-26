/**
* @file LinAlgo.hpp
*
* @brief contains the LinAlgo namespace
*/

#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

#include <cmath>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <cassert>
#include <thread>

#ifndef DONT_USE_GPU
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS //just in case somone has OpenCL 1.2
//#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#ifdef _WIN32
#include <CL\cl2.hpp>
#else
#include <CL/cl2.hpp>
#endif
#endif

/**
* @namespace LinAlgo
*
* @brief Contains everything necessary to get some linear algebra done
*
* @detail This namespace contains the matrix definition as well as all of the
* global functions and variables required for initializing the gpu. The namespace
* also has many functions with the same name as functions in the matrix class.
* The difference between these functions is that the matrix version of the
* functions overwrites the calling object, while the LinAlgo versions does not.
*/
namespace LinAlgo {
//the all important matrix class
    template <class ItemType>
    class matrix;

    template <class ItemType = double>
    matrix<ItemType> identityMatrix(size_t N);
    template <class ItemType>
    matrix<ItemType> columnVector(std::vector<ItemType> vec);
    //these aren't going to be terribly efficient, especially for the small vectors one would normally use these on
    //glm is a much better option for creating rotation/translation/scale matrices
    matrix<double> rotationMatrix(size_t N, std::vector<double> angles);
    matrix<double> translationMatrix(size_t N, std::vector<double> direction);
    matrix<double> scaleMatrix(size_t N, std::vector<double> scale);

//linear algebra functions? how fun :P
    template <class ItemType>
    matrix<ItemType> transpose (const matrix<ItemType>& M);
    template <class ItemType>
    matrix<ItemType> inverse (matrix<ItemType>& M); //requires rre to work efficiently, not const due to determinant

    template <class ArgType, class ItemType = ArgType>
    matrix<ArgType> map (const matrix<ItemType>& M, ArgType (*function) (ItemType));
#ifndef DONT_USE_GPU
//due to updating gpu data counting as cahnging the matrix, I can't
//actually have these be const :'(
    template <class ItemType>
    matrix<ItemType> map (const matrix<ItemType>& M, std::string kernel, cl_int& error_ret);
    template <class ItemType>
    matrix<ItemType> map (const matrix<ItemType>& M, cl_kernel kernel, cl_int& error_ret);
#endif


    template <class ItemType>
    bool qr (const matrix<ItemType>& M, matrix<ItemType>& Q, matrix<ItemType>& R); //returns if it was successful i guess
    template <class ItemType>
    std::vector<ItemType> eigenvalues(matrix<ItemType>& M);//should these actually modify the matrix object by saving the vals/vecs?
    template <class ItemType>
    std::vector<ItemType> eigenvalues(matrix<ItemType>& M, matrix<ItemType>& eigenvecs_out);

    template <class ItemType>
    matrix<ItemType> gj (const matrix<ItemType>&); //Gauss-Jordan elimination
    template <class ItemType>
    matrix<ItemType> re (const matrix<ItemType>& M); //Row Echelon form
    template <class ItemType>
    matrix<ItemType> re (const matrix<ItemType>& M, int& row_swaps);
    template <class ItemType>
    matrix<ItemType> rre (const matrix<ItemType>&); //Reduced Row Echelon form

//these should have options for using the gpu... probably if ALL_USE_GPU is true
//vector operations
    template <class ItemType>
    ItemType operator*(const std::vector<ItemType>& A, const std::vector<ItemType>& B);//define the inner product for vectors
    template <class ItemType>
    std::vector<ItemType> operator*(const std::vector<ItemType>& A, const ItemType& B);
    template <class ItemType>
    std::vector<ItemType> operator*(const ItemType& A, const std::vector<ItemType>& B);
    template <class ItemType>
    std::vector<ItemType> operator/(const std::vector<ItemType>& A, const ItemType& B);
    template <class ItemType>
    std::vector<ItemType> operator-(const std::vector<ItemType>& A, const std::vector<ItemType>& B);
    template <class ItemType>
    std::vector<ItemType> operator+(const std::vector<ItemType>& A, const std::vector<ItemType>& B);

//vector functions
    template <class ItemType, class FuncRet>
    ItemType normalize(const ItemType& val, FuncRet(*innerProduct)(const ItemType&, const ItemType&) = [](const ItemType& A, const ItemType& B) {return A * B;});//default inner product is for real numbers and vectors
    template <class ItemType, class FuncRet>
    std::vector<ItemType> gs(const std::vector<ItemType>& vals, FuncRet(*innerProduct)(const ItemType&, const ItemType&) = [](const ItemType& A, const ItemType& B) {return A * B;}, bool normaliz_output = true);//gram-schmidt process on a set of (vectors)

#ifndef DONT_USE_GPU
//functions and such for dealing with the gpu
    static cl_int InitGPU();
    static bool BreakDownGPU();
    static bool AllUseGPU (bool use_it);
    static bool IsGPUInitialized();

//these will allow users to generate kernels
    static const cl_platform_id retrievePlatformID();
    static const cl_device_id retrieveDeviceID();
    static const cl_context retrieveContext();

#include <exception>
    class gpu_exception : public std::exception {
    public:
        gpu_exception(std::string msg, const char* _file, int _line, cl_int err) : m_what(msg), m_file(_file), m_error(err) {
            m_line = std::to_string(_line);
        }

        void setLog(std::string log) {
            m_log = log;
        }

        const char* what() const throw() {
            return m_what.c_str();
        }

        const char* file() {
            return m_file.c_str();
        }

        const char* line() {
            return m_line.c_str();
        }

        cl_int errorCode() {
            return m_error;
        }

        const char* getLog() {
            return m_log.c_str();
        }

    private:
        std::string m_what;
        std::string m_file;
        std::string m_line;
        std::string m_log;
        cl_int m_error;
    };

    static const char* getErrorString (cl_int error);
#endif // DONT_USE_GPU
    namespace {
        double EPSILON = 0.000001;

        template <class ItemType>
        bool epsilon_equal(const LinAlgo::matrix<ItemType>& M1, const LinAlgo::matrix<ItemType>& M2, double epsilon) {
            ItemType temp;
            epsilon = std::abs(epsilon);
            if (M1.getWidth() != M2.getHeight()) {
                return false;
            }
            for (size_t i = 0; i < M1.getHeight(); i++) {
                temp = std::abs(M1[i][i] - M2[i][i]);
                if (temp > epsilon) {
                    return false;
                }
            }
            return true;
        };

#ifndef DONT_USE_GPU
//private functions
        //auxillary function for loading kernel files and returning a program
        cl::Program create_program (std::string filename, cl::Context* context, cl_int* errcode_ret) {
            //first load the source code
            std::ifstream file;
            file.open (filename.c_str(), std::ios::in);
            if (file.bad()) {
                throw(gpu_exception(std::string("Failed to load ") + filename, __FILE__, __LINE__, -100));
            }
            std::string line;
            std::string kernel_src;
            while (std::getline (file, line)) {
                kernel_src += line;
                kernel_src += '\n';
            }
            file.close();

            //cretea Sources obect
            cl::Program::Sources sources;
            sources.push_back({kernel_src.c_str(), kernel_src.length()});

            //create and return the program
            return cl::Program(*context, sources, errcode_ret);
        }

//private variables
        bool GPU_INITIALIZED = false;
        bool ALL_USE_GPU = false;
        float OPENCL_VERSION = 0.0f;

        typedef enum {
            ADD,
            ADD_SCALAR,
            SUB,
            SUB_SCALAR,
            MULTIPLY,
            MULTIPLY_SCALAR,
            MULTIPLY_ELEMENT,
            DIVIDE_SCALAR,
            DIVIDE_ELEMENT,
            NUM_KERNELS
        } Kernel;

        typedef enum {
            CHAR,
            SHORT,
            INT,
            LONG,
            FLOAT,
            DOUBLE,
            NUM_TYPES
        } KernelType;

        cl::Kernel* m_charKernels;
        cl::Kernel* m_shortKernels;
        cl::Kernel* m_intKernels;
        cl::Kernel* m_longKernels;
        cl::Kernel* m_floatKernels;
        cl::Kernel* m_doubleKernels;
        cl::Kernel** m_kernels;

        std::vector<cl::Platform> m_platforms;
        cl::Platform m_default_platform;
        std::vector<cl::Device> m_devices;
        cl::Device m_default_device;
        cl::Context m_context;
        cl::CommandQueue m_command_queue;
#endif // DONT_USE_GPU
    }
}

#ifndef DONT_USE_GPU
#pragma region GPU Functions
/**
* @brief Returns a string to describe an error code from opencl or a gpu_exception
*/
const char* LinAlgo::getErrorString (cl_int error) {
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

        //something else
        case -99:
            return "LINAGLO_INVALID_MATRIX_CONDITION";
        case -100:
            return "LINALGO_INVALID_INITIALIZATION";

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

// <editor-fold desc="GPU Functions">
/**
* @brief InitGPU() initilizes the gpu for linear algebra
*
* @detail InitGPU() loads in the matrix_kernels_TYPE.cl OpenCL files,
* loads the context and devices for gpu usege, and generates the kernels
* for basic arithmetic types. These types currently include int and double.
*
* @return cl_int This function returns CL_SUCCESS if the initialization is successful, otherwise it returns an error code.
*/
static cl_int LinAlgo::InitGPU() {

    if (GPU_INITIALIZED)
    { return CL_SUCCESS; }

    m_charKernels = new cl::Kernel[Kernel::NUM_KERNELS];
    m_shortKernels = new cl::Kernel[Kernel::NUM_KERNELS];
    m_intKernels = new cl::Kernel[Kernel::NUM_KERNELS];
    m_longKernels = new cl::Kernel[Kernel::NUM_KERNELS];
    m_floatKernels = new cl::Kernel[Kernel::NUM_KERNELS];
    m_doubleKernels = new cl::Kernel[Kernel::NUM_KERNELS];
    m_kernels = new cl::Kernel*[KernelType::NUM_TYPES];
    m_kernels[KernelType::CHAR] = m_charKernels;
    m_kernels[KernelType::SHORT] = m_shortKernels;
    m_kernels[KernelType::INT] = m_intKernels;
    m_kernels[KernelType::LONG] = m_longKernels;
    m_kernels[KernelType::FLOAT] = m_floatKernels;
    m_kernels[KernelType::DOUBLE] = m_doubleKernels;

    cl_int ret;

    //get the platforms
    ret = cl::Platform::get(&m_platforms);
    if (ret != CL_SUCCESS) {
        throw(gpu_exception(std::string("Unable to load platform IDs") + LinAlgo::getErrorString(ret), __FILE__, __LINE__, ret));
    }
    m_default_platform = m_platforms[0];

    std::string version;
    m_default_platform.getInfo(CL_PLATFORM_VERSION, &version);
    printf("Using OpenCL version: %s\n", version.c_str());

    //get the devices
    m_default_platform.getDevices(CL_DEVICE_TYPE_GPU, &m_devices);
    if (m_devices.size() == 0) {
        ret = m_default_platform.getDevices(CL_DEVICE_TYPE_CPU, &m_devices);
    }
    if (ret != CL_SUCCESS) {
        throw(gpu_exception(std::string("Unable to get device IDs: ") + LinAlgo::getErrorString(ret), __FILE__, __LINE__, ret));
        return ret;
    }
    m_default_device = m_devices[0];

    //create the context
    m_context = cl::Context({m_default_device}, NULL, NULL, NULL, &ret);
    if (ret != CL_SUCCESS) {
        throw(gpu_exception(std::string("Unable to create context: ") + LinAlgo::getErrorString(ret), __FILE__, __LINE__, ret));
    }

    /*
    Once kernels are completed the template will be stored as part of the source code to remove the need to track files
    */
    //create kernels
    //first get the programs set up
    #ifdef MATRIX_KERNEL_DIR
    std::string kernel_directory = MATRIX_KERNEL_DIR ;
    #else
    std::string kernel_directory = "../kernels/";
    #endif
    std::vector<cl::Program> programs(KernelType::NUM_TYPES);
    std::vector<std::string> type_str ({ "char", "short", "int", "long", "float", "double" });
    for (size_t type = 0; type < KernelType::NUM_TYPES; type++) {
        //printf("The kernel file name is: %s\n", std::string(kernel_directory + std::string("matrix_kernels_") + type_str[type] + std::string(".cl")).c_str());
        programs[type] = create_program (kernel_directory +
                                         std::string ("matrix_kernels_") +
                                         type_str[type] +
                                         std::string (".cl"),
                                         &m_context,
                                         &ret);
        if (ret != CL_SUCCESS) {
            throw(gpu_exception("Unable to create program", __FILE__, __LINE__, ret));
        }
    }

    //now build the programs
    for (size_t type = 0; type < KernelType::NUM_TYPES; type++) {
        ret = programs[type].build({m_default_device});
        if (ret != CL_SUCCESS) {
            gpu_exception ouch(std::string("program ") + type_str[type] + std::string(" build unsuccessful"), __FILE__, __LINE__, ret);
            if (ret == CL_BUILD_PROGRAM_FAILURE) {
                ouch.setLog(programs[type].getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_default_device));
            }
            throw(ouch);
        }
    }

    //and finally, create the kernels
    std::vector<std::string> kernelNames ({ "add", "addScalar", "subtract", "subtractScalar", "multiply", "multiplyScalar", "elementMultiply", "divideScalar", "elementDivide" });
    for (size_t type = 0; type < KernelType::NUM_TYPES; type++) {
        //printf("Generating kernels for type: %s\n", type_str[type].c_str());
        for (size_t kernel = 0; kernel < Kernel::NUM_KERNELS; kernel++) {
            m_kernels[type][kernel] = cl::Kernel(programs[type], kernelNames[kernel].c_str(), &ret);
            if (ret != CL_SUCCESS) {
                throw(gpu_exception(std::string("Could not create kernel. Kernel name: ") + kernelNames[kernel] + std::string(", type: ") + type_str[type].c_str(), __FILE__, __LINE__, ret));
            }
        }
    }

    //create namespace command queue
    m_command_queue = cl::CommandQueue(m_context, m_default_device, 0, &ret);
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to create command queue, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }

    //le done :P
    GPU_INITIALIZED = true;
    return ret;
}

/**
* @brief BreakDownGPU() uninitilizes the gpu and frees all resources
*
* @return True if successful, false if there was nothing to do in the first place
*/
static bool LinAlgo::BreakDownGPU() {
    if (!GPU_INITIALIZED) {
        return false;//why did u even try to break something that wasn't even there?
    }
    GPU_INITIALIZED = false;
    ALL_USE_GPU = false;

    //delete all the kernels. Everything else can clean itself up.
    for (size_t type = 0; type < KernelType::NUM_TYPES; type++) {
        delete[] m_kernels[type];
    }
    delete[] m_kernels;
    m_kernels = NULL;

    return true;
}

/**
* @brief Tells all matrices to use the GPU.
*
* @return True if successful, false if the GPU is not yet initialized
*
* @note If a matrix stores a non-basic type, then the operation will not be performed
*/
static bool LinAlgo::AllUseGPU (bool use_it) {
    if (GPU_INITIALIZED) {
        ALL_USE_GPU = use_it;
        return true;
    }
    return false;
}

/**
* @brief returns if the GPU is initialized
*/
static bool LinAlgo::IsGPUInitialized() {
    return GPU_INITIALIZED;
}
// </editor-fold>
#pragma endregion
#endif

#include "matrix.h"

#pragma region Linear Algebra Functions
// <editor-fold desc="Linear Algebra Functions">

/**
* @brief Creates an identity matrix of the specified size
*/
template <class ItemType = double>
LinAlgo::matrix<ItemType> LinAlgo::identityMatrix(size_t N) {
    matrix<ItemType> M(N, N, 0);
    for (size_t i = 0; i < N; i++) {
        M[i][i] = 1;
    }
    return M;
}

/**
* @brief Creates a column vector from a provided vector
*/
template <class ItemType>
LinAlgo::matrix<ItemType> LinAlgo::columnVector(std::vector<ItemType> vec) {
    matrix<ItemType> M(vec.size(), 1);
    for (size_t i = 0; i < vec.size(); i++) {
        M[i][0] = vec[i];
    }
    return M;
}

/**
* @brief Non-overwriting matrix transposition
*/
template <class ItemType>
LinAlgo::matrix<ItemType> LinAlgo::transpose (const LinAlgo::matrix<ItemType>& M) {
    matrix<ItemType> result (M.m_width, M.m_height);
    if (false) {
        //gpu copy transpose
    } else {
        for (size_t i = 0; i < M.m_height; i++) {
            for (size_t j = 0; j < M.m_width; j++) {
                result.m_data[j][i] = M.m_data[i][j];
            }
        }
    }
    return result;
}

/**
* @brief Non-overwriting matrix inverse
*
* @detail Uses gauss-jordan elimination to calculate the inverse
*/
template <class ItemType>
LinAlgo::matrix<ItemType> LinAlgo::inverse (matrix<ItemType>& M) { //would be const, but matrix::getDeterminant() sets the determinant if it isn't already
    if (M.m_height != M.m_width) {
        return matrix<ItemType> (0, 0);
    } else if (M.m_height == 1) {
        matrix<ItemType> ret (M);
        if (M.m_data[0][0] == 0) {
            return matrix<ItemType> (0, 0);
        }
        ret.m_data[0][0] = 1 / ret.m_data[0][0];
        return ret;
    } else if (M.m_height == 2) {
        matrix<ItemType> ret (M.m_height, M.m_height);
        ItemType det = M.getDeterminant();
        if (det == 0) {
            return matrix<ItemType> (0, 0);
        }
        ret.m_data[0][0] = M.m_data[1][1];
        ret.m_data[1][1] = M.m_data[0][0];
        ret.m_data[0][1] = -1 * M.m_data[0][1];
        ret.m_data[1][0] = -1 * M.m_data[1][0];
        return ret / det;
    } else {
        if (M.getDeterminant() == 0) {
            return matrix<ItemType> (0, 0);
        }
        matrix<ItemType> augmented (M.m_height, M.m_width * 2);
#ifndef DONT_USE_GPU
        augmented.useGPU(M.useGPU());
#endif
        for (size_t i = 0; i < augmented.m_height; i++) {
            augmented.m_data[i][i + M.m_width] = 1;
            for (size_t j = 0; j < M.m_width; j++) {
                augmented.m_data[i][j] = M.m_data[i][j];
            }
        }
        return gj (augmented).subMatrix (0, M.m_width, M.m_height, M.m_width);
    }
}

/**
* @brief Maps a provided function to all values in a matrix
*
* @param M The matrix containing the input
*
* @param func The function to map to all values in the matrix
*
* @param asynchronous Set to true in order to execute with a thread pool
*
* @param num_threads Use this to specify the number of threads (if threading).
* Defaults to double the hardware concurrency
*
* @return A matrix containing the result of executing func on every value in the matrix
*/
template <class ItemType, class ArgType>
LinAlgo::matrix<ArgType> LinAlgo::map(LinAlgo::matrix<ItemType> &M, ArgType (*func)(ItemType&), bool asynchronous, int thread_count) {
	matrix<ArgType> ret(M.getHeight(), M.getWidth());
	ret.useGPU(M.useGPU());
	bool leave_data = false;
	if (M.leaveOnGPU()) {
        M.pullData();
        leave_data = true;
	}
	if (!asynchronous) {
        for (size_t i = 0; i < M.getHeight(); i++) {
            for (size_t j = 0; j < M.getWidth(); j++) {
                ret[i][j] = func(M[i][j]);
            }
        }
    } else {
        auto thread_func = [&](ItemType* dest, ItemType& val){
            *dest = func(val);
        };
        std::vector<std::thread> pool;
        int default_pool_size;
        if (thread_count == 0) {
            default_pool_size = std::thread::hardware_concurrency() * 2;
        } else {
            default_pool_size = thread_count;
        }
        if (default_pool_size == 0) {
            default_pool_size = 4;
        }
        int num_vals = M.getHeight() * M.getWidth();
        if (num_vals < default_pool_size) {
            default_pool_size = num_vals;
            pool.reserve(default_pool_size);
        } else {
            pool.reserve(default_pool_size);
        }
        int start_index = 0;
        for (auto iter1 = ret.begin(), iter2 = M.begin(); start_index < default_pool_size; start_index++) {
            std::thread th (thread_func, &iter1[start_index], iter2[start_index]);
            pool.push_back (std::move (th));
        }
        if (num_vals > default_pool_size) {
            bool first = true;
            for (size_t i = start_index / M.getWidth(); i < M.getHeight(); i++) {
                for (size_t j = first ? start_index % M.getWidth() : 0; j < M.getWidth(); j++) {
                    bool waiting = true;
                    while (waiting) {
                        for (auto t1 = pool.begin(); t1 < pool.end(); t1++) {
                            if (t1->joinable()) {
                                std::thread t2 (thread_func, &ret[i][j], M[i][j]);
                                t1->swap (t2);
                                t2.detach();
                                waiting = false;
                                break;
                            }
                        }
                    }
                }
                first = false;
            }
        }
        for (auto t = pool.begin(); t < pool.end(); t++) {
            if (t->joinable()) {
                t->join();
            }
        }
    }
    if (leave_data) {
        ret.pushData();
        ret.leaveDataOnGPU(true);
    }
    return ret;
}

/**
* @brief QR decomposition
*
* @note V1 uses the Graham-Schmidt method for QR factorization
*/
template <class ItemType>
bool LinAlgo::qr (const LinAlgo::matrix<ItemType>& M, matrix<ItemType>& Q, matrix<ItemType>& R) {
    matrix<ItemType> M_trans (transpose (M)); //can use for "vertical slices" of M
    Q = matrix<ItemType>(M.m_height, M.m_width);
    R = matrix<ItemType>(M.m_height, M.m_width, 0);
    if (false) {//use gpu

    } else {
        std::vector<std::vector<ItemType>> data(M_trans.m_height);
        for (size_t i = 0; i < M_trans.m_height; i++) {
            data[i] = M_trans.m_data[i];
        }
        data = gs<std::vector<ItemType>, ItemType>( data, [](const std::vector<ItemType>& A, const std::vector<ItemType>& B){return A * B;});
        for (size_t i = 0; i < data.size(); i++) {
            for (size_t j = 0; j < data[i].size(); j++) {
                Q[j][i] = data[i][j];
            }
        }
        for (size_t i = 0; i < R.m_width; i++) {
            for (size_t j = i; j < R.m_width; j++) {
                R[i][j] = data[i] * M_trans[j];
            }
        }
    }

    return true;
}

/**
* @brief Calculates the eigenvalues of a given matrix
*
* @note returns an empty std::vector if the matrix isn't invertible.
* This function uses the QR Algorithm to calculate eigenvalues.
*/
template <class ItemType>
std::vector<ItemType> LinAlgo::eigenvalues(LinAlgo::matrix<ItemType>& M) {
    if (M.getDeterminant() == 0) {
        return std::vector<ItemType>(0);
    }
    matrix<ItemType> Q(0, 0), R(0, 0);
    matrix<ItemType> values_matrix(M);
    matrix<ItemType> temp(0, 0);
    int i = 0;
    do {
        qr(values_matrix, Q, R);
        temp = std::move(R * Q);
        if (epsilon_equal(values_matrix, temp, EPSILON)) {
            break;
        }
        values_matrix = std::move(temp);
        i++;
    } while (i < 500);
    std::vector<ItemType> vals;
    vals.reserve(M.getWidth());
    for (i = 0; i < vals.size(); i++) {
        vals.push_back(values_matrix[i][i]);
    }
    return vals;
}

/**
* @brief Calculates the eigenvalues and eigenvectors of a given matrix
*
* @note returns an empty std::vector if the matrix isn't invertible.
* This function uses the mildly extended QR Algorithm to calculate
* eigenvalues and eigenvectors.
*/
template <class ItemType>
std::vector<ItemType> LinAlgo::eigenvalues(LinAlgo::matrix<ItemType>& M, LinAlgo::matrix<ItemType>& eigenvecs_out) {
    if (M.getDeterminant() == 0) {
        return std::vector<ItemType>(0);
    }
    eigenvecs_out = std::move(identityMatrix(M.getWidth()));
    matrix<ItemType> Q(0, 0), R(0, 0);
    matrix<ItemType> values_matrix(M);
    matrix<ItemType> temp(0, 0);
    int i = 0;
    do {
        qr(values_matrix, Q, R);
        temp = std::move(R * Q);
        eigenvecs_out = std::move(eigenvecs_out * Q);
        if (epsilon_equal(values_matrix, temp, EPSILON)) {
            break;
        }
        values_matrix = std::move(temp);
        i++;
    } while (i < 500);
    std::vector<ItemType> vals;
    vals.reserve(values_matrix.getWidth());
    for (i = 0; i < values_matrix.getWidth(); i++) {
        vals.push_back(values_matrix[i][i]);
    }
    return vals;
}

/**
* @brief Non-overwriting row-echelon
*
* @details returns the row-echelon form of a matrix without overwriting the original
*/
template <class ItemType>
LinAlgo::matrix<ItemType> LinAlgo::re (const LinAlgo::matrix<ItemType>& M) {
    matrix<ItemType> result (M);
    if (false) {
        //use gpu vector addition
    } else {
        bool pivotFound = false;
        bool nonZeroExists = false;

        size_t numPivots = result.m_height > result.m_width ? result.m_width : result.m_height;
        size_t pivotColumn = 0, pivotRow = 0;
        ItemType pivot;
        ItemType scalar;

        //time to get it to row echelon form
        for (size_t i = 0; i < numPivots; i++, pivotColumn++, pivotRow++) {
            if (pivotRow > result.m_height - 1 || pivotColumn > result.m_width - 1) {
                break;//break if pivot location is outside matrix, or in the last column
            }

            //find or create the pivot
            pivotFound = true;
            if (result.m_data[pivotRow][pivotColumn] == ItemType(0)) {
                pivotFound = false;
                for (size_t j = pivotRow; j < result.m_height; j++) {
                    if (result.m_data[j][pivotColumn] != ItemType(0)) {
                        pivotFound = true;
                        if (j != pivotRow) {
                            //swap the pivot row with the current row
                            result.m_data[pivotRow].swap(result.m_data[j]);
                        }
                    }
                }
            }
            if (!pivotFound) {
                //advance pivot column. go to start of loop
                pivotRow--;
                continue;
            }

            //eliminate below pivot
            pivot = result.m_data[pivotRow][pivotColumn];
            for (size_t j = pivotRow + 1; j < result.m_height; j++) {
                scalar = result.m_data[j][pivotColumn] / pivot;
                for (size_t k = pivotColumn; k < result.m_width; k++) {
                    result.m_data[j][k] -= result.m_data[pivotRow][k] * scalar;
                }
            }
        }
    }
    return result;
}

/**
* @brief Non-overwriting row-echelon
*
* @details returns the row-echelon form of a matrix without overwriting the original, this one also counts row swaps for calculating the determinant
*/
template <class ItemType>
LinAlgo::matrix<ItemType> LinAlgo::re (const LinAlgo::matrix<ItemType>& M, int& row_swaps) {
    matrix<ItemType> result (M);
    row_swaps = 0;
    if (false) {
        //use gpu vector addition
    } else {
        bool pivotFound = false;
        bool nonZeroExists = false;

        size_t numPivots = result.m_height > result.m_width ? result.m_width : result.m_height;
        size_t pivotColumn = 0, pivotRow = 0;
        ItemType pivot;
        ItemType scalar;

        //time to get it to row echelon form
        for (size_t i = 0; i < numPivots; i++, pivotColumn++, pivotRow++) {
            if (pivotRow > result.m_height - 1 || pivotColumn > result.m_width - 1) {
                break;//break if pivot location is outside matrix, or in the last column
            }

            //find or create the pivot
            pivotFound = true;
            if (result.m_data[pivotRow][pivotColumn] == ItemType(0)) {
                pivotFound = false;
                for (size_t j = pivotRow; j < result.m_height; j++) {
                    if (result.m_data[j][pivotColumn] != ItemType(0)) {
                        pivotFound = true;
                        if (j != pivotRow) {
                            //swap the pivot row with the current row
                            result.m_data[pivotRow].swap(result.m_data[j]);
                            row_swaps++;
                        }
                    }
                }
            }
            if (!pivotFound) {
                //advance pivot column. go to start of loop
                pivotRow--;
                continue;
            }

            //eliminate below pivot
            pivot = result.m_data[pivotRow][pivotColumn];
            for (size_t j = pivotRow + 1; j < result.m_height; j++) {
                scalar = result.m_data[j][pivotColumn] / pivot;
                for (size_t k = pivotColumn; k < result.m_width; k++) {
                    result.m_data[j][k] -= result.m_data[pivotRow][k] * scalar;
                }
            }
        }
    }
    return result;
}

/**
* @brief Non-overwriting reduced row-echelon
*/
template <class ItemType>
LinAlgo::matrix<ItemType> LinAlgo::rre (const LinAlgo::matrix<ItemType>& M) {
    matrix<ItemType> result (re (M));
    if (false) {//use gpu

    } else {
        ItemType scalar;
        for (size_t i = 0, j = 0; i < result.m_height && j < result.m_width; i++, j++) {
            if (result.m_data[i][j] != ItemType(0)) {
                scalar = result.m_data[i][j];
                for (size_t k = j; k < M.m_width; k++) {
                    result.m_data[i][k] /= scalar;
                }
            } else {
                i--;
            }
        }
    }
    return result;
}

/**
* @brief Non-overwriting Gauss-Jordan Elimination
*/
template <class ItemType>
LinAlgo::matrix<ItemType> LinAlgo::gj (const LinAlgo::matrix<ItemType>& M) {
    matrix<ItemType> result (LinAlgo::rre (M));
    if (false) {//use gpu

    } else {
        ItemType scalar;
        //stars at 1,1 b/c can't eliminate above row 0
        for (size_t pivotRow = 1, pivotColumn = 1; pivotRow < result.m_height && pivotColumn < result.m_width; pivotRow++, pivotColumn++) {
            if (result.m_data[pivotRow][pivotColumn] == ItemType(1)) {
                for (size_t i = pivotRow - 1; i >= 0 && i < (((size_t) 0) - 1); i--) {
                    scalar = result.m_data[i][pivotColumn];
                    for (size_t j = pivotColumn; j < result.m_width; j++) {
                        result.m_data[i][j] -= result.m_data[pivotRow][j] * scalar;
                    }
                }
            } else {
                pivotRow--;
            }
        }
    }

    return result;
}

#pragma region Vector Functions
// <editor-fold desc="Vector Functions">

/**
* @brief Vector inner product
*/
template <class ItemType>
ItemType LinAlgo::operator*(const std::vector<ItemType>& A, const std::vector<ItemType>& B) {
    ItemType sum = 0;
    for (size_t i = 0; i < A.size() && i < B.size(); i++) {
        sum += A[i] * B[i];
    }
    return sum;
}

/**
* @brief Vector scalar multiplication
*/
template <class ItemType>
std::vector<ItemType> LinAlgo::operator*(const std::vector<ItemType>& A, const ItemType& B) {
    std::vector<ItemType> ret(A.size());
    for (size_t i = 0; i < ret.size(); i++) {
        ret[i] = A[i] * B;
    }
    return ret;
}

/**
* @brief Vector scalar multiplication
*/
template <class ItemType>
std::vector<ItemType> LinAlgo::operator*(const ItemType& A, const std::vector<ItemType>& B) {
    std::vector<ItemType> ret(B.size());
    for (size_t i = 0; i < ret.size(); i++) {
        ret[i] = B[i] * A;
    }
    return ret;
}

/**
* @brief Vector scalar division
*/
template <class ItemType>
std::vector<ItemType> LinAlgo::operator/(const std::vector<ItemType>& A, const ItemType& B) {
    std::vector<ItemType> ret(A.size());
    for (size_t i = 0; i < ret.size(); i++) {
        ret[i] = A[i] / B;
    }
    return ret;
}

/**
* @brief Vector addition
*/
template <class ItemType>
std::vector<ItemType> LinAlgo::operator+(const std::vector<ItemType>& A, const std::vector<ItemType>& B) {
    std::vector<ItemType> ret(A.size());
    for (size_t i = 0; i < A.size() && i < B.size(); i++) {
        ret[i] = A[i] + B[i];
    }
    return ret;
}

/**
* @brief Vector subtraction
*/
template <class ItemType>
std::vector<ItemType> LinAlgo::operator-(const std::vector<ItemType>& A, const std::vector<ItemType>& B) {
    std::vector<ItemType> ret(A.size());
    for (size_t i = 0; i < A.size() && i < B.size(); i++) {
        ret[i] = A[i] - B[i];
    }
    return ret;
}

/**
* @brief Normalizes a vector
*
* @note Innder product defaults to operator*() which is defined for reals and vectors containing arithmetic values
*/
template <class ItemType, class FuncRet>
ItemType LinAlgo::normalize(const ItemType& val, FuncRet(*innerProduct)(const ItemType&, const ItemType&)) {
    FuncRet norm = innerProduct(val, val);
    norm = std::sqrt(norm);
    return val / norm;
}

/**
* @brief Performs the Gram-Schmidt procedure to orthonormalize a set of vectors
*
* @note Inner product defaults to operator*() which is defined for reals and vectors containing arithmetic values
*/
template <class ItemType, class FuncRet>
std::vector<ItemType> LinAlgo::gs(const std::vector<ItemType>& vals, FuncRet(*innerProduct)(const ItemType&, const ItemType&), bool normalize_output) {
    std::vector<ItemType> ret(vals.size());
    ret[0] = vals[0];
    for (size_t i = 1; i < ret.size(); i++) {
        ret[i] = vals[i];
        for (size_t j = i - 1; ; j--) {
            FuncRet tempScalar = innerProduct(vals[i], ret[j]) / innerProduct(ret[j], ret[j]);
            ret[i] = ret[i] - (tempScalar * ret[j]);
            if (j == 0) {
                break;
            }
        }
    }
    if (normalize_output) {
        for (size_t i = 0; i < ret.size(); i++) {
            ret[i] = normalize(ret[i], innerProduct);
        }
    }
    return ret;
}


// </editor-fold>
#pragma endregion

// </editor-fold>
#pragma endregion

#endif
