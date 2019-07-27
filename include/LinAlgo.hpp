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
	matrix<ItemType> inverse(const matrix<ItemType>& M); //requires rre to work efficiently

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

		//private variables
		bool GPU_INITIALIZED = false;
		bool ALL_USE_GPU = false;

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

		cl_platform_id m_platform_id = NULL;
		cl_device_id m_device_id = NULL;
		cl_context m_context = NULL;
	}
}

#pragma region GPU Functions
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
	cl_int ret = clGetPlatformIDs(1, &m_platform_id, &ret_num_platforms);
	if (ret != CL_SUCCESS) {
		printf("Could not get platform IDs.\n");
		return ret;
	}
	ret = clGetDeviceIDs(m_platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &m_device_id, &ret_num_devices);
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
	std::string kernel_directory = #MATRIX_KERNEL_DIR ;
#else 
	std::string kernel_directory = "../kernels/";
#endif
	cl_program programs[KernelType::NUM_TYPES];
	std::vector<std::string> type_str({ "char", "short", "int", "long", "float", "double" });
	for (size_t type = 0; type < KernelType::NUM_TYPES; type++) {
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
				printf("program_int build unsuccessful:\n%s\n", log);
			}
			return ret;
		}
	}

	//and finally, create the kernels
	std::vector<std::string> kernelNames({ "add", "addScalar", "subtract", "subtractScalar", "multiply", "multiplyScalar", "elementMultiply", "divideScalar", "elementDivide" });
	for (size_t type = 0; type < KernelType::NUM_TYPES; type++) {
		for (size_t kernel = 0; kernel < Kernel::NUM_KERNELS; kernel++) {
			m_Kernels[type][kernel] = clCreateKernel(programs[type], kernelNames[kernel].c_str(), &ret);
			if (ret != CL_SUCCESS) {
				printf("Could not create kernel.\n");
				return ret;
			}
		}
	}

	//free the programs since I don't need those anymore
	for (size_t type = 0; type < KernelType::NUM_TYPES; type++) {
		clReleaseProgram(programs[type]);
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
#pragma endregion

#include "matrix.h"

#pragma region Linear Algebra Functions

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

//WHY IS THIS OVERWRITING THE INPUT??!?
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

#pragma endregion

#endif