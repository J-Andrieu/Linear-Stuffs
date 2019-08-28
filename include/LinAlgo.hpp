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

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef _WIN32
#include <CL\cl.h>
#else
#include <CL/cl.h>
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
	//the all importan matrix class
	template <class ItemType>
	class matrix;

	//linear algebra functions? how fun :P
	template <class ItemType>
	matrix<ItemType> transpose(const matrix<ItemType>& M);
	template <class ItemType>
	matrix<ItemType> inverse(matrix<ItemType>& M); //requires rre to work efficiently, not const due to determinant

	template <class ArgType, class ItemType = ArgType>
	matrix<ArgType> map(const matrix<ItemType>& M, ArgType(*function)(ItemType));
	//due to updating gpu data counting as cahnging the matrix, I can't
	//actually have these be const :'(
	template <class ItemType>
	matrix<ItemType> map(const matrix<ItemType>& M, std::string kernel, cl_int& error_ret);
	template <class ItemType>
	matrix<ItemType> map(const matrix<ItemType>& M, cl_kernel kernel, cl_int& error_ret);


	template <class ItemType>
	bool qr(const matrix<ItemType>& M, matrix<ItemType>& Q, matrix<ItemType>& R);//returns if it was successful i guess

	template <class ItemType>
	matrix<ItemType> solve(const matrix<ItemType>&);//solve a system of linear equations
	template <class ItemType>
	matrix<ItemType> gj(const matrix<ItemType>&);//Gauss-Jordan elimination
	template <class ItemType>
	matrix<ItemType> re(const matrix<ItemType>& M);//Row Echelon form
	template <class ItemType>
	matrix<ItemType> rre(const matrix<ItemType>&);//Reduced Row Echelon form


	//functions and such for dealing with the gpu
	static cl_int InitGPU();
	static bool BreakDownGPU();
	static bool AllUseGPU(bool use_it);
	static bool IsGPUInitialized();

	//these will allow users to generate kernels
	static const cl_platform_id retrievePlatformID();
	static const cl_device_id retrieveDeviceID();
	static const cl_context retrieveContext();

	namespace {
		//private functions
		//auxillary function for loading kernel files and returning a program
		cl_program create_program(std::string filename, cl_context context, cl_int* errcode_ret) {
			//first load the source code
			std::ifstream file;
			file.open(filename.c_str(), std::ios::in);
			if (file.bad()) {
				printf("Failed to load kernel.cl!\n");
			}
			std::string line;
			std::string kernel_src;
			while (std::getline(file, line)) {
				kernel_src += line;
				kernel_src += '\n';
			}
			//printf("Kernel source: \n%s", kernel_src.c_str());
			const size_t* kernel_size = new size_t(kernel_src.size());
			//if (kernel_size == 0) {
			//	printf("Did not load kernel!\n");
			//}
			const char* kernel_str = kernel_src.c_str();
			file.close();

			//then create the program
			cl_program program = clCreateProgramWithSource(context, 1, &kernel_str, kernel_size, errcode_ret);
			return program;
		}

		float getPlatformVersion(cl_platform_id platform_id) {
            //evaluate opencl version
#define VERSION_LENGTH 64
            char complete_version[VERSION_LENGTH];
            size_t realSize = 0;
            clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, VERSION_LENGTH,
                              &complete_version, &realSize);
            char version[4];
            version[3] = 0;
            //printf("%s\n", complete_version);
            std::copy(complete_version + 7, complete_version + 11, version);
            return atof(version);
            //printf("V %s %f\n", version, OPENCL_VERSION);
		}

		int choosePlatform(cl_platform_id* platform_id, size_t numPlatforms, float preferredOCLVersion) {
		    float versions[numPlatforms];
            for (size_t i = 0; i < numPlatforms; i++) {
                versions[i] = getPlatformVersion(platform_id[i]);
                if (versions[i] == preferredOCLVersion) {
                    return i;
                }
            }
            for (size_t i = 0; i < numPlatforms; i++) {
                if (std::floor(versions[i]) == std::floor(preferredOCLVersion)) {
                    return i;
                }
            }
            return 0;
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

		cl_kernel m_charKernels[Kernel::NUM_KERNELS] = { NULL };
		cl_kernel m_shortKernels[Kernel::NUM_KERNELS] = { NULL };
		cl_kernel m_intKernels[Kernel::NUM_KERNELS] = { NULL };
		cl_kernel m_longKernels[Kernel::NUM_KERNELS] = { NULL };
		cl_kernel m_floatKernels[Kernel::NUM_KERNELS] = { NULL };
		cl_kernel m_doubleKernels[Kernel::NUM_KERNELS] = { NULL };
		cl_kernel* m_Kernels[KernelType::NUM_TYPES] = { m_charKernels, m_shortKernels, m_intKernels, m_longKernels, m_floatKernels, m_doubleKernels };

		cl_platform_id* m_platform_id = NULL;
		cl_device_id m_device_id = NULL;
		cl_context m_context = NULL;
	}
}

#pragma region GPU Functions
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
		return CL_SUCCESS;

	//get platform and device information
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret = clGetPlatformIDs(0, m_platform_id, &ret_num_platforms);
	m_platform_id = new cl_platform_id[ret_num_platforms];
	ret = clGetPlatformIDs(ret_num_devices, m_platform_id, &ret_num_platforms);
	if (ret != CL_SUCCESS) {
		printf("Could not get platform IDs.\n");
		return ret;
	}
	int platform_index = choosePlatform(m_platform_id, ret_num_platforms, 2.2);
	OPENCL_VERSION = getPlatformVersion(m_platform_id[platform_index]);
	printf("The chosen platform version is %f\n", OPENCL_VERSION);
	if (OPENCL_VERSION < 2.0) {
        printf("Warning: Opencl 1.x functions are not required to be implemented,\nFunctions such as clCreateQueue() which only changed names between versions may cause issues.\n");
	}
	ret = clGetDeviceIDs(m_platform_id[platform_index], CL_DEVICE_TYPE_GPU, 0, &m_device_id, &ret_num_devices);
	if (ret_num_devices == 0) {
        ret = clGetDeviceIDs(m_platform_id[platform_index], CL_DEVICE_TYPE_CPU, 0, &m_device_id, &ret_num_devices);
        if (ret_num_devices == 0) {
            printf("Warning: No OpenCL device available\n");
            return ret;
        } else {
            printf("CPU device chosen\n");
            ret = clGetDeviceIDs(m_platform_id[platform_index], CL_DEVICE_TYPE_CPU, 1, &m_device_id, &ret_num_devices);
        }
	} else {
	    printf("GPU device chosen\n");
        ret = clGetDeviceIDs(m_platform_id[platform_index], CL_DEVICE_TYPE_GPU, 1, &m_device_id, &ret_num_devices);
	}

	if (ret != CL_SUCCESS) {
		printf("Could not get device IDs.\n");
		return ret;
	}

	//create the context
	m_context = clCreateContext(NULL, 1, &m_device_id, NULL, NULL, &ret);
	if (ret != CL_SUCCESS) {
		printf("Could not create context.\n");
		return ret;
	}

	//create kernels
	//first get the programs set up
#ifdef MATRIX_KERNEL_DIR
	std::string kernel_directory = MATRIX_KERNEL_DIR ;
#else
	std::string kernel_directory = "../kernels/";
#endif
	cl_program* programs = new cl_program[KernelType::NUM_TYPES];
	std::vector<std::string> type_str({ "char", "short", "int", "long", "float", "double" });
	for (size_t type = 0; type < KernelType::NUM_TYPES; type++) {
        //printf("The kernel file name is: %s\n", std::string(kernel_directory + std::string("matrix_kernels_") + type_str[type] + std::string(".cl")).c_str());
		programs[type] = create_program(kernel_directory +
			std::string("matrix_kernels_") +
			type_str[type] +
			std::string(".cl"),

			m_context,
			&ret);
		if (ret != CL_SUCCESS) {
			printf("Could not create program.\n");
			return ret;
		}
	}

	//now build the programs
	for (size_t type = 0; type < KernelType::NUM_TYPES; type++) {
		ret = clBuildProgram(programs[type], 1, &m_device_id, NULL, NULL, NULL);
		if (ret != CL_SUCCESS) {
			if (ret == CL_BUILD_PROGRAM_FAILURE) {
				// Determine the size of the log
				size_t log_size;
				clGetProgramBuildInfo(programs[type], m_device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

				// Allocate memory for the log
				char* log = (char*)malloc(log_size);

				// Get the log
				clGetProgramBuildInfo(programs[type], m_device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

				// Print the log
				printf("program_%s build unsuccessful:\n%s\n", type_str[type].c_str(), log);
			}
			return ret;
		}
	}

	//and finally, create the kernels
	std::vector<std::string> kernelNames({ "add", "addScalar", "subtract", "subtractScalar", "multiply", "multiplyScalar", "elementMultiply", "divideScalar", "elementDivide" });
	for (size_t type = 0; type < KernelType::NUM_TYPES; type++) {
            //printf("Generating kernels for type: %s\n", type_str[type].c_str());
		for (size_t kernel = 0; kernel < Kernel::NUM_KERNELS; kernel++) {
			m_Kernels[type][kernel] = clCreateKernel(programs[type], kernelNames[kernel].c_str(), &ret);
			if (ret != CL_SUCCESS) {
				//printf("Could not create kernel. Kernel name: %s, type: %s\n", kernelNames[kernel].c_str(), type_str[type].c_str());
				return ret;
			}
		}
	}

	//free the programs since I don't need those anymore
	for (size_t type = 0; type < KernelType::NUM_TYPES; type++) {
		clReleaseProgram(programs[type]);
	}
    delete[] programs;

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

	for (size_t type = 0; type < KernelType::NUM_TYPES; type++) {
		for (size_t kernel = 0; kernel < Kernel::NUM_KERNELS; kernel++) {
			clReleaseKernel(m_Kernels[type][kernel]);
			m_Kernels[type][kernel] = NULL;
		}
	}

	clReleaseDevice(m_device_id);
	clReleaseContext(m_context);

	m_platform_id = NULL;
	m_device_id = NULL;
	m_context = NULL;

	return true;
}

/**
* @brief Tells all matrices to use the GPU.
*
* @return True if successful, false if the GPU is not yet initialized
*
* @notes If a matrix stores a non-basic type, then the operation will not be performed
*/
static bool LinAlgo::AllUseGPU(bool use_it) {
	if (GPU_INITIALIZED) {
		ALL_USE_GPU = use_it;
		return true;
	}
	return false;
}

static bool LinAlgo::IsGPUInitialized() {
	return GPU_INITIALIZED;
}
// </editor-fold>
#pragma endregion

#include "matrix.h"

#pragma region Linear Algebra Functions
// <editor-fold desc="Linear Algebra Functions">

/**
* @brief Non-overwriting matrix transposition
*/
template <class ItemType>
LinAlgo::matrix<ItemType> LinAlgo::transpose(const LinAlgo::matrix<ItemType>& M) {
	matrix<ItemType> result(M.m_width, M.m_height);
	if (false) {
		//gpu copy transpose
	}
	else {
		for (size_t i = 0; i < M.m_height; i++) {
			for (size_t j = 0; j < M.m_width; j++) {
				(*result.m_data[j])[i] = (*M.m_data[i])[j];
			}
		}
	}
	return result;
}

/**
* @brief Non-overwriting matrix inverse
*
* @detail Uses gauss-jordan elimination after a threshold to maintain efficiency
*/
template <class ItemType>
LinAlgo::matrix<ItemType> LinAlgo::inverse(matrix<ItemType>& M) {//would be const, but matrix::getDeterminant() sets the determinant if it isn't already
    if (M.m_height != M.m_width) {
        return matrix<ItemType>(0, 0);
    } else if (M.m_height == 1) {
        matrix<ItemType> ret(M);
        if ((*M.m_data[0])[0] == 0) {
            return matrix<ItemType>(0,0);
        }
        (*ret.m_data[0])[0] = 1 / (*ret.m_data[0])[0];
        return ret;
    } else if (M.m_height == 2) {
        matrix<ItemType> ret(M.m_height, M.m_height);
        ItemType det = M.getDeterminant();
        if (det == 0) {
            return matrix<ItemType>(0,0);
        }
        (*ret.m_data[0])[0] = (*M.m_data[1])[1];
        (*ret.m_data[1])[1] = (*M.m_data[0])[0];
        (*ret.m_data[0])[1] = -1 * (*M.m_data[0])[1];
        (*ret.m_data[1])[0] = -1 * (*M.m_data[1])[0];
        return ret / det;
    } else {
        if (M.getDeterminant() == 0) {
            printf("The determinant was 0\n");
            return matrix<ItemType>(0,0);
        }
        matrix<ItemType> augmented(M.m_height, M.m_width * 2, 0, M.m_useGPU);
        for (size_t i = 0; i < augmented.m_height; i++) {
            (*augmented.m_data[i])[i + M.m_width] = 1;
            for (size_t j = 0; j < M.m_width; j++) {
                (*augmented.m_data[i])[j] = (*M.m_data[i])[j];
            }
        }
        //augmented.gj();
        return gj(augmented).subMatrix(0, M.m_width, M.m_height, M.m_width);
    }
}

/**
* @brief QR decomposition
*
* @detail V1 uses the Graham-Schmidt method for QR factorization
*/
template <class ItemType>
bool LinAlgo::qr(const LinAlgo::matrix<ItemType>& M, matrix<ItemType>& Q, matrix<ItemType>& R) {
  if (M.m_height != M.m_width) {
    return false;
  }

  matrix<ItemType> M_trans(transpose(M));//can use for "vertical slices" of M
  if (false) {//use gpu

  } else {

  }

  return true;
}

/**
* @brief Non-overwriting row-echelon
*
* @details returns the row-echelon form of a matrix without overwriting the original
*/
template <class ItemType>
LinAlgo::matrix<ItemType> LinAlgo::re(const LinAlgo::matrix<ItemType>& M) {
	matrix<ItemType> result(M);
	if (false) {
		//use gpu vector addition
	}
	else {
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
			if ((*result.m_data[pivotRow])[pivotColumn] == 0) {
				pivotFound = false;
				for (size_t j = pivotRow; j < result.m_height; j++) {
					if ((*result.m_data[j])[pivotColumn] != 0) {
						pivotFound = true;
						if (j != pivotRow) {
							//swap the pivot row with the current row
							result.m_data[pivotRow]->swap(*result.m_data[j]);
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
			pivot = (*result.m_data[pivotRow])[pivotColumn];
			for (size_t j = pivotRow + 1; j < result.m_height; j++) {
				scalar = (*result.m_data[j])[pivotColumn] / pivot;
				for (size_t k = pivotColumn; k < result.m_width; k++) {
					(*result.m_data[j])[k] -= (*result.m_data[pivotRow])[k] * scalar;
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
LinAlgo::matrix<ItemType> LinAlgo::rre(const LinAlgo::matrix<ItemType>& M) {
	matrix<ItemType> result(re(M));
	if (false) {//use gpu

	} else {
		ItemType scalar;
		for (size_t i = 0, j = 0; i < result.m_height && j < result.m_width; i++, j++) {
			if ((*result.m_data[i])[j] != 0) {
				scalar = (*result.m_data[i])[j];
				for (size_t k = j; k < M.m_width; k++) {
					(*result.m_data[i])[k] /= scalar;
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
LinAlgo::matrix<ItemType> LinAlgo::gj(const LinAlgo::matrix<ItemType>& M) {
	matrix<ItemType> result(LinAlgo::rre(M));
	if (false) {//use gpu

	}
	else {
		ItemType scalar;
		//stars at 1,1 b/c can't eliminate above row 0
		for (size_t pivotRow = 1, pivotColumn = 1; pivotRow < result.m_height && pivotColumn < result.m_width; pivotRow++, pivotColumn++) {
			if ((*result.m_data[pivotRow])[pivotColumn] == 1) {
				for (size_t i = pivotRow - 1; i >= 0 && i < (((size_t) 0) - 1); i--) {
					scalar = (*result.m_data[i])[pivotColumn];
					for (size_t j = pivotColumn; j < result.m_width; j++) {
						(*result.m_data[i])[j] -= (*result.m_data[pivotRow])[j] * scalar;
					}
				}
			}
			else {
				pivotRow--;
			}
		}
	}

	return result;
}

// </editor-fold>
#pragma endregion

#endif
