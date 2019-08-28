/**
* @file matrix.cpp
*
* @brief Implements the matrix class.
*/

#ifndef _MATRIX_INCLUDED
#define _MATRIX_INCLUDED
#include "../include/matrix.h"

using namespace LinAlgo;

#pragma region Constructors and Destructors
// <editor-fold desc="Constructors and Destructors">
//{
/**
* @brief This is the constructor for the matrix class.
*
* @details Creates the desired mxn matrix of the template-specified data type.
*
* @param[height] The desired height of the matrix
*
* @param[width] The desired width of the matrix
*
* @param[enable_gpu] Whether the matrix should attempt to use the GPU to make computations. Defaults to false.
*/
template <class ItemType>
matrix<ItemType>::matrix (size_t height, size_t width, bool enable_gpu) : m_height (height), m_width (width), m_useGPU (enable_gpu),
    m_gpuUpToDate (false), m_gpuData (NULL), m_gpuHeight (NULL),
    m_gpuWidth (NULL), m_command_queue (NULL), m_gpuSlicesUpToDate (height, false) {
    m_data.resize (m_height);
    for (size_t i = 0; i < m_height; i++) {
        m_data[i] = new std::vector<ItemType> (m_width);
    }
    m_upToDate = 0;
}

//this constructor is very ambiguous and I should fix that
/**
* @brief This is the constructor for the matrix class.
*
* @details Creates the desired mxn matrix of the template-specified data type.
*
* @param[height] The desired height of the matrix
*
* @param[width] The desired width of the matrix
*
* @param[val] The value to initialize the matrix with
*
* @param[enable_gpu] Whether the matrix should attempt to use the GPU to make computations. Defaults to false.
*/
template <class ItemType>
matrix<ItemType>::matrix (size_t height, size_t width, ItemType val, bool enable_gpu) : m_height (height), m_width (width), m_useGPU (enable_gpu),
    m_gpuUpToDate (false), m_gpuData (NULL), m_gpuHeight (NULL), m_upToDate (0),
    m_gpuWidth (NULL), m_command_queue (NULL), m_gpuSlicesUpToDate (height, false) {
    m_data.resize (m_height);
    for (size_t i = 0; i < m_height; i++) {
        m_data[i] = new std::vector<ItemType> (m_width, val);
    }
}

/**
* @brief Explicit matrix constructor
*
* @detail Allows you to create a matrix from initializer lists. If the
* lists aren't all the same size, then the width will be the widest list,
* and 0 will be assigned to all empty locations
*
* @param[vals] The vector of vectors to be turned into a matrix
*
* @param[enable_gpu] Whether or not the gpu should be enabled for this matrix
*/
template <class ItemType>
matrix<ItemType>::matrix (const std::vector<std::vector<ItemType>>& vals, bool enable_gpu) : m_useGPU (enable_gpu), m_gpuUpToDate (false), m_gpuData (NULL),
    m_gpuHeight (NULL), m_gpuWidth (NULL), m_command_queue (NULL), m_leaveOnGPU (false) {
    size_t maxSize = 0;
    for (size_t i = 0; i < vals.size(); i++) {
        if (vals[i].size() > maxSize) {
            maxSize = vals[i].size();
        }
    }
    m_height = vals.size();
    m_width = maxSize;
    m_data.resize (m_height);
    for (size_t i = 0; i < m_height; i++) {
        m_data[i] = new std::vector<ItemType> (m_width, 0);
        for (size_t j = 0; j < vals[i].size(); j++) {
            (*m_data[i])[j] = vals[i][j];
        }
    }
    m_gpuSlicesUpToDate.resize (m_height, false);
    m_upToDate = 0;
}

/**
* @brief This is the (templated) copy constructor for the matrix class
*
* @details This constructor will copy the height, width, and data. It also copies whether or not
* it should use the gpu. This onstructor is only called when it needs to convert matrix types
*
* @param[M] This is the matrix to be copied
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType>::matrix (const matrix<ArgType>& M) : m_useGPU (M.m_useGPU), m_gpuUpToDate (false),
    m_gpuData (NULL), m_gpuHeight (NULL),
    m_gpuWidth (NULL), m_command_queue (NULL),
    m_gpuSlicesUpToDate (M.m_height, false), m_leaveOnGPU (M.m_leaveOnGPU) {
    m_height = M.m_height;
    m_width = M.m_width;
    m_upToDate = 0;//this will defo be different later
    m_data.resize (m_height);
    for (size_t i = 0; i < m_height; i++) {
        m_data[i] = new std::vector<ItemType> (m_width);
        for (size_t j = 0; j < m_width; j++) {
            (*m_data[i])[j] = ItemType ((*M.m_data[i])[j]);
        }
    }
}

/**
* @brief This is the copy constructor for the matrix class
*
* @details This constructor will copy the height, width, and data. It also copies whether or not
* it should use the gpu.
*
* @param[M] This is the matrix to be copied
*/
template <class ItemType>
matrix<ItemType>::matrix (const matrix<ItemType>& M) : m_useGPU (M.m_useGPU), m_gpuUpToDate (false),
    m_gpuData (NULL), m_gpuHeight (NULL),
    m_gpuWidth (NULL), m_command_queue (NULL),
    m_gpuSlicesUpToDate (M.m_height, false), m_leaveOnGPU (M.m_leaveOnGPU),
    m_data (M.m_height) {
    m_height = M.m_height;
    m_width = M.m_width;
    m_upToDate = 0;//this will defo be different later
    for (size_t i = 0; i < m_height; i++) {
        m_data[i] = new std::vector<ItemType> (m_width);
        for (size_t j = 0; j < m_width; j++) {
            (*m_data[i])[j] = (*M.m_data[i])[j];
        }
    }

}

/**
* @brief This is the move constructor for the matrix class
*
* @param[M] This is the matrix to be moved
*/
template <class ItemType>
matrix<ItemType>::matrix (matrix<ItemType>&& M) : m_useGPU (M.m_useGPU), m_gpuUpToDate (M.m_gpuUpToDate),
    m_gpuData (M.m_gpuData), m_gpuHeight (M.m_gpuHeight),
    m_gpuWidth (M.m_gpuWidth), m_command_queue (M.m_command_queue),
    m_gpuSlicesUpToDate (M.m_gpuSlicesUpToDate), m_leaveOnGPU (M.m_leaveOnGPU) {
    m_height = M.m_height;
    m_width = M.m_width;
    m_data = M.m_data;
    for (size_t i = 0; i < M.m_height; i++) {
        M.m_data[i] = NULL;
    }
    M.m_gpuData = NULL;
    M.m_command_queue = NULL;
    M.m_gpuHeight = NULL;
    M.m_gpuWidth = NULL;
}

//this is broken... somehow
/**
* @brief matrix destructor
*
* @detail Mostly just here to clean up any gpu data lyin' about
*
* @note Work in progress
*/
template <class ItemType>
matrix<ItemType>::~matrix() {
    /*
    clFinish(m_command_queue);
    clReleaseCommandQueue(m_command_queue);
    clReleaseMemObject(m_gpuData);
    clReleaseMemObject(m_gpuHeight);//causes read violations???
    clReleaseMemObject(m_gpuWidth);
    */
}
//}
// </editor-fold>
#pragma endregion

#pragma region Initializers
// <editor-fold desc="Initializers">
//{
/**
* @brief Fills the matrix with the provided value
*
* @param[val] The value to fill the matrix with
*/
template <class ItemType>
void matrix<ItemType>::fill (ItemType val) {
    for (size_t i = 0; i < m_height; i++) {
        for (size_t j = 0; j < m_width; j++) {
            (*m_data[i])[j] = val;
        }
    }
    m_gpuUpToDate = false;
    m_upToDate &= (dataFlag::GPU_HEIGHT | dataFlag::GPU_WIDTH);
    m_gpuSlicesUpToDate.clear();
    m_gpuSlicesUpToDate.resize (m_height, false);
}

/**
* @brief Clears the matrix
*/
template <class ItemType>
void matrix<ItemType>::clear() {
    for (size_t i = 0; i < m_height; i++) {
        (*m_data[i]).clear();
        (*m_data[i]).resize (m_width);
    }
    m_gpuUpToDate = false;
    m_upToDate &= (dataFlag::GPU_HEIGHT | dataFlag::GPU_WIDTH);
    m_gpuSlicesUpToDate.clear();
    m_gpuSlicesUpToDate.resize (m_height, false);
}

/**
* @brief Individually set if a matrix should use the gpu or not
*
* @details This will unallocate all data for the matrix on the gpu
* if it was previously using the gpu
*/
template <class ItemType>
bool matrix<ItemType>::useGPU (bool use_it) {
    if (GPU_INITIALIZED) {
        m_useGPU = use_it;
        if (!m_useGPU) {
            if (m_command_queue) {
                clFinish (m_command_queue);
                clReleaseCommandQueue (m_command_queue);
                m_command_queue = NULL;
            }
            if (m_gpuData) {
                clReleaseMemObject (m_gpuData);
                m_gpuData = NULL;
            }
            if (m_gpuHeight || m_gpuWidth) {
                clReleaseMemObject (m_gpuHeight);
                clReleaseMemObject (m_gpuWidth);
                m_gpuHeight = NULL;
                m_gpuWidth = NULL;
            }
            m_gpuUpToDate = false;
            m_gpuSlicesUpToDate.clear();
            m_gpuSlicesUpToDate.resize (m_height, false);
            m_upToDate |= ! (dataFlag::GPU_HEIGHT | dataFlag::GPU_WIDTH | dataFlag::GPU_DATA);
        }
        //if this is being set to true, pushing and calculating everything at once
        //could take some time, so everything should get pushed when it's first needed
        //... or i could make pushToGPU public? nah... an updateGPU() function. yes. that.
        return true;
    }
    return false;
}
//}
// </editor-fold>
#pragma endregion

#pragma region Setters and Getters
// <editor-fold desc="Setters and Getters">
//{
/**
* @brief This function returns the value or object stored at the specified position.
*
* @param[y] The height of the position.
*
* @param[x] The width of the position.
*/
template <class ItemType>
ItemType matrix<ItemType>::get (size_t y, size_t x) const  {
    return (*m_data[y])[x];
}

/**
* @brief This function sets the specified location in the matrix to the specified value.
*
* @param[y] The height of the position.
*
* @param[x] The width of the position.
*
* @param[val] The value to be inserted to the position.
*/
template <class ItemType>
void matrix<ItemType>::set (size_t y, size_t x, ItemType val) {
    (*m_data[y])[x] = val;
    m_gpuUpToDate = false;
    m_upToDate &= dataFlag::GPU_DATA;
    m_gpuSlicesUpToDate[y] = false;
}

/**
* @brief Returns the height of the matrix
*/
template <class ItemType>
size_t matrix<ItemType>::getHeight() const  {
    return m_height;
}

/**
* @brief Returns the width of the matrix
*/
template <class ItemType>
size_t matrix<ItemType>::getWidth() const  {
    return m_width;
}

/**
* @brief Returns true of the matrix is square
*/
template <class ItemType>
bool matrix<ItemType>::isSquare() const {
    return m_height == m_width;
}

/**
* @brief Returns the determinant
*
* @detail Will calculate the determinant if it hasn't been done already
*/
template <class ItemType>
ItemType matrix<ItemType>::getDeterminant() {
    if (m_upToDate & dataFlag::DETERMINANT) {
        return m_determinant;
    }
    if (m_height != m_width) {
        m_determinant = 0;//no determinant
    } else if (m_height == 1) {
        m_determinant = (*m_data[0])[0];
    } else if (m_height == 2) {
        m_determinant = (*m_data[0])[0] * (*m_data[1])[1] - (*m_data[0])[1] * (*m_data[1])[0];
    } else {
        matrix<ItemType> detMat (LinAlgo::re (*this));
        m_determinant = (ItemType) 1;
        for (size_t i = 0; i < detMat.m_height; i++) {
            m_determinant *= (*detMat.m_data[i])[i];
        }
    }
    m_upToDate |= dataFlag::DETERMINANT;
    return m_determinant;
}

/**
* @brief Resizes the calling matrix
*
* @detail This resizes the matrix to the given height and width.
*
* @param[height] The new height
*
* @param[width] The new width
*
* @param[val] If the matrix is being expanded, val will be used to fill the void
*/
template <class ItemType>
matrix<ItemType>& matrix<ItemType>::resize (size_t height, size_t width, ItemType& val) {
    if (m_height != height) {
        m_data.resize (height);
        m_upToDate &= !dataFlag::GPU_HEIGHT;
        m_gpuSlicesUpToDate.resize (height, false);
        clReleaseMemObject (m_gpuHeight);
        m_gpuHeight = NULL;
    }
    if (m_width != width) {
        for (size_t i = 0; i < height; i++) {
            (*m_data[i]).resize (width, val);
        }
        m_upToDate &= !dataFlag::GPU_WIDTH;
        clReleaseMemObject (m_gpuWidth);
    }
    m_height = height;
    m_width = width;
    clReleaseMemObject (m_gpuData);
    m_gpuData = NULL;
    m_upToDate &= !dataFlag::GPU_DATA;
    m_gpuSlicesUpToDate.clear();
    m_gpuSlicesUpToDate.resize (m_height, false);
    m_gpuUpToDate = false;
    return *this;
}

/**
* @brief Creates a new matrix from a subregion of the calling matrix
*
* @param[y] The row to start the slice
*
* @param[x] The column to start the slice
*
* @param[h] The height of the new matrix
*
* @param[w] The width of the new matrix
*
* @note If the requested region is out of bounds it will return a 0x0 matrix
*/
template <class ItemType>
matrix<ItemType> matrix<ItemType>::subMatrix (size_t y, size_t x, size_t h, size_t w) {
    if (y >= m_height || x >= m_width || y + h > m_height || x + w > m_width) {
        return matrix<ItemType> (0, 0);
    }
    matrix<ItemType> result (h, w);
    for (size_t i = 0; i < h; i++) {
        for (size_t j = 0; j < w; j++) {
            (*result.m_data[i])[j] = (*m_data[i + y])[j + x];
        }
    }
    return result;
}

/**
* @brief Returns an Identity matrix of the given size
*/
template <class ItemType>
matrix<ItemType> matrix<ItemType>::identity (size_t height, size_t width) {
    if (width == 0) {
        width = height;
    }
    matrix<ItemType> id (height, width, ItemType (0), false);
    for (size_t i = 0; i < width && i < height; i++) {
        id[i][i] = 1;
    }
    return id;
}
//}
// </editor-fold>
#pragma endregion

#pragma region Addition Functions
// <editor-fold desc="Addition Functions">
//{
/**
* @brief Adds two matrices.
*
* @details If m_useGPU is true, then this function will attempt to use one of the available kernels to add the two matrices together.
* GPU mode only works if they are the same type. GPU usage is determined by calling matrix and preserved through the returned matrix.
*
* @param[M] The matrix to be added to the calling matrix.
*
* @return returns a matrix of the type of the calling matrix with the dimensions of the overlap between the matrices.
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::add (matrix<ArgType>& M) {
    //should I be concatenating the matrix ike this? probably not, a submatrix
    //can be made easily enough if they want concatenation...
    //yeah, i'll change this later
    matrix<ItemType> result (m_height < M.m_height ? m_height : M.m_height, m_width < M.m_width ? m_width : M.m_width);
    if (! (m_useGPU || ALL_USE_GPU)) { //if don't use the gpu
        for (size_t i = 0; i < result.m_height; i++) {
            for (size_t j = 0; j < result.m_width; j++) {
                (*result.m_data[i])[j] = (*m_data[i])[j] + (*M.m_data[i])[j];
            }
        }
        result.m_useGPU = m_useGPU;
        return result;
    } else {
        if (!GPU_INITIALIZED) {
            printf ("GPU is not initialized!\n");
            return matrix<ItemType> (0, 0);
        }
        initQueue();
        pushToGPU (m_command_queue);
        M.pushToGPU (m_command_queue); //if this matrix isn't supposed to use gpu, should I unpush its data after this?
        result.createResultBuffer (m_command_queue);
        if (std::is_same<ItemType, ArgType>::value) {
            cl_int ret; //i'm not even kinda checking this rn, but maybe i will later :P
            if (std::is_same<ItemType, char>::value) {
                ret = execute_add_kernel (m_charKernels[Kernel::ADD], M, result);
            } else if (std::is_same<ItemType, short>::value) {
                ret = execute_add_kernel (m_shortKernels[Kernel::ADD], M, result);
            } else if (std::is_same<ItemType, int>::value) {
                ret = execute_add_kernel (m_intKernels[Kernel::ADD], M, result);
            } else if (std::is_same<ItemType, long>::value) {
                ret = execute_add_kernel (m_longKernels[Kernel::ADD], M, result);
            } else if (std::is_same<ItemType, float>::value) {
                ret = execute_add_kernel (m_floatKernels[Kernel::ADD], M, result);
            } else if (std::is_same<ItemType, double>::value) {
                ret = execute_add_kernel (m_doubleKernels[Kernel::ADD], M, result);
            } else {
                printf ("Can't GPU compute, unsupported item type.\n");
            }
        } else {
            printf ("Can't GPU compute, matrices are of differing types.\n");
        }
        result.pullFromGPU (m_command_queue);
        clFinish (m_command_queue);
        clReleaseCommandQueue (m_command_queue);
        m_command_queue = NULL;
        return result;
    }
}

/**
* @brief Adds a single value to all elements in matrix.
*
* @param[val] The value to be added to the calling matrix.
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::add (const ArgType& val) {
    matrix<ItemType> result (m_height, m_width);
    if (! (m_useGPU || ALL_USE_GPU)) { //if don't use the gpu
        for (size_t i = 0; i < result.m_height; i++) {
            for (size_t j = 0; j < result.m_width; j++) {
                (*result.m_data[i])[j] = (*m_data[i])[j] + val;
            }
        }
        result.m_useGPU = m_useGPU;
        return result;
    } else {
        if (!GPU_INITIALIZED) {
            printf ("GPU is not initialized!\n");
            return matrix<ItemType> (0, 0);
        }
        initQueue();
        pushToGPU (m_command_queue);
        result.createResultBuffer (m_command_queue);
        cl_int ret; //i'm not even kinda checking this rn, but maybe i will later :P
        if (std::is_same<ItemType, char>::value) {
            ret = execute_array_val_kernel (m_charKernels[Kernel::ADD_SCALAR], (ItemType &) val, result);
        } else if (std::is_same<ItemType, short>::value) {
            ret = execute_array_val_kernel (m_shortKernels[Kernel::ADD_SCALAR], (ItemType&) val, result);
        } else if (std::is_same<ItemType, int>::value) {
            ret = execute_array_val_kernel (m_intKernels[Kernel::ADD_SCALAR], (ItemType&) val, result);
        } else if (std::is_same<ItemType, long>::value) {
            ret = execute_array_val_kernel (m_longKernels[Kernel::ADD_SCALAR], (ItemType&) val, result);
        } else if (std::is_same<ItemType, float>::value) {
            ret = execute_array_val_kernel (m_floatKernels[Kernel::ADD_SCALAR], (ItemType&) val, result);
        } else if (std::is_same<ItemType, double>::value) {
            ret = execute_array_val_kernel (m_doubleKernels[Kernel::ADD_SCALAR], (ItemType&) val, result);
        } else {
            printf ("Can't GPU compute, unsupported item type.\n");
        }
        result.pullFromGPU (m_command_queue);
        clFinish (m_command_queue);
        clReleaseCommandQueue (m_command_queue);
        m_command_queue = NULL;
        return result;
    }
}

/**
* @brief Operator overload for matrix::add()
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::operator+ (matrix<ArgType>& M) {
    return this->add (M);
}

/**
* @brief Operator overload for matrix::add()
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::operator+ (const ArgType& val) {
    return this->add (val);
}
//}
// </editor-fold>
#pragma endregion

#pragma region Subtraction Functions
// <editor-fold desc="Subtraction Functions">
//{
/**
* @brief Subtracts two matrices.
*
* @param[M] The matrix to be subtracted from the calling matrix.
*
* @return returns a matrix of the type of the calling matrix with the dimensions of the overlap between the matrices.
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::subtract (matrix<ArgType>& M) {
    //should I be concatenating the matrix ike this? probably not, a submatrix
    //can be made easily enough if they want concatenation...
    //yeah, i'll change this later
    matrix<ItemType> result (m_height < M.m_height ? m_height : M.m_height, m_width < M.m_width ? m_width : M.m_width);
    if (! (m_useGPU || ALL_USE_GPU)) { //if don't use the gpu
        for (size_t i = 0; i < result.m_height; i++) {
            for (size_t j = 0; j < result.m_width; j++) {
                (*result.m_data[i])[j] = (*m_data[i])[j] - (*M.m_data[i])[j];
            }
        }
        result.m_useGPU = m_useGPU;
        return result;
    } else {
        if (!GPU_INITIALIZED) {
            printf ("GPU is not initialized!\n");
            return matrix<ItemType> (0, 0);
        }
        initQueue();
        pushToGPU (m_command_queue);
        M.pushToGPU (m_command_queue); //if this matrix isn't supposed to use gpu, should I unpush its data after this?
        result.createResultBuffer (m_command_queue);
        if (std::is_same<ItemType, ArgType>::value) {
            cl_int ret; //i'm not even kinda checking this rn, but maybe i will later :P
            if (std::is_same<ItemType, char>::value) {
                ret = execute_add_kernel (m_charKernels[Kernel::SUB], M, result);
            } else if (std::is_same<ItemType, short>::value) {
                ret = execute_add_kernel (m_shortKernels[Kernel::SUB], M, result);
            } else if (std::is_same<ItemType, int>::value) {
                ret = execute_add_kernel (m_intKernels[Kernel::SUB], M, result);
            } else if (std::is_same<ItemType, long>::value) {
                ret = execute_add_kernel (m_longKernels[Kernel::SUB], M, result);
            } else if (std::is_same<ItemType, float>::value) {
                ret = execute_add_kernel (m_floatKernels[Kernel::SUB], M, result);
            } else if (std::is_same<ItemType, double>::value) {
                ret = execute_add_kernel (m_doubleKernels[Kernel::SUB], M, result);
            } else {
                printf ("Can't GPU compute, unsupported item type.\n");
            }
        } else {
            printf ("Can't GPU compute, matrices are of differing types.\n");
        }
        result.pullFromGPU (m_command_queue);
        clFinish (m_command_queue);
        clReleaseCommandQueue (m_command_queue);
        m_command_queue = NULL;
        return result;
    }
}

/**
* @brief Subtracts a single value from all elements in matrix.
*
* @param[val] The value to be subtracted from the calling matrix.
*
* @return returns a matrix of the same type as the calling matrix.
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::subtract (const ArgType& val) {
    matrix<ItemType> result (m_height, m_width);
    if (! (m_useGPU || ALL_USE_GPU)) { //if don't use the gpu
        for (size_t i = 0; i < result.m_height; i++) {
            for (size_t j = 0; j < result.m_width; j++) {
                (*result.m_data[i])[j] = (*m_data[i])[j] - val;
            }
        }
        result.m_useGPU = m_useGPU;
        return result;
    } else {
        if (!GPU_INITIALIZED) {
            printf ("GPU is not initialized!\n");
            return matrix<ItemType> (0, 0);
        }
        initQueue();
        pushToGPU (m_command_queue);
        result.createResultBuffer (m_command_queue);
        cl_int ret; //i'm not even kinda checking this rn, but maybe i will later :P
        if (std::is_same<ItemType, char>::value) {
            ret = execute_array_val_kernel (m_charKernels[Kernel::SUB_SCALAR], (ItemType&) val, result);
        } else if (std::is_same<ItemType, short>::value) {
            ret = execute_array_val_kernel (m_shortKernels[Kernel::SUB_SCALAR], (ItemType&) val, result);
        } else if (std::is_same<ItemType, int>::value) {
            ret = execute_array_val_kernel (m_intKernels[Kernel::SUB_SCALAR], (ItemType&) val, result);
        } else if (std::is_same<ItemType, long>::value) {
            ret = execute_array_val_kernel (m_longKernels[Kernel::SUB_SCALAR], (ItemType&) val, result);
        } else if (std::is_same<ItemType, float>::value) {
            ret = execute_array_val_kernel (m_floatKernels[Kernel::SUB_SCALAR], (ItemType&) val, result);
        } else if (std::is_same<ItemType, double>::value) {
            ret = execute_array_val_kernel (m_doubleKernels[Kernel::SUB_SCALAR], (ItemType&) val, result);
        } else {
            printf ("Can't GPU compute, unsupported item type.\n");
        }
        result.pullFromGPU (m_command_queue);
        clFinish (m_command_queue);
        clReleaseCommandQueue (m_command_queue);
        m_command_queue = NULL;
        return result;
    }
}

/**
* @brief Operator overload for matrix::subtract()
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::operator- (matrix<ArgType>& M) {
    return this->subtract (M);
}

/**
* @brief Operator overload for matrix::subtract()
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::operator- (const ArgType& val) {
    return this->subtract (val);
}
//}
// </editor-fold>
#pragma endregion

#pragma region Multiplication Functions
// <editor-fold desc="Multiplication Functions">
//{
/**
* @brief Multiplies two matrices.
*
* @details If m_useGPU is true, then this function will attempt to use one of the available kernels to multiply the two matrices together.
* GPU mode only works if they are the same type. GPU usage is determined by the calling matrix and is preserved through the returned matrix.
*
* @param[M] The rhs matrix for the multiplication.
*
* @return Returns the result of multiplying the two matrices together, of the same type as the calling matrix. Returned matrix is 0x0
* if the matrices were of incompatible dimensions.
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::multiply (matrix<ArgType>& M) {
    if (m_width != M.m_height) {
        return matrix<ItemType> (0, 0); //null matrix
    }
    matrix<ItemType> result (m_height, M.m_width);
    if (! (m_useGPU || ALL_USE_GPU)) { //if don't use the gpu
        for (size_t i = 0; i < result.m_height; i++) {
            for (size_t j = 0; j < result.m_width; j++) {
                ItemType sum = ItemType (0);
                for (size_t k = 0; k < m_width; k++) {
                    sum += (*m_data[i])[k] * (*M.m_data[k])[j];
                }
                (*result.m_data[i])[j] = sum;
            }
        }
        return result;
    } else {
        if (!GPU_INITIALIZED) {
            printf ("GPU is not initialized!\n");
            return matrix<ItemType> (0, 0);
        }
        initQueue();
        pushToGPU (m_command_queue);
        M.pushToGPU (m_command_queue);
        result.createResultBuffer (m_command_queue);
        if (std::is_same<ItemType, ArgType>::value) {
            cl_int ret;

            if (std::is_same<ItemType, char>::value) {
                ret = execute_multiply_kernel (m_charKernels[Kernel::MULTIPLY], M, result);
            } else if (std::is_same<ItemType, short>::value) {
                ret = execute_multiply_kernel (m_shortKernels[Kernel::MULTIPLY], M, result);
            } else if (std::is_same<ItemType, int>::value) {
                ret = execute_multiply_kernel (m_intKernels[Kernel::MULTIPLY], M, result);
            } else if (std::is_same<ItemType, long>::value) {
                ret = execute_multiply_kernel (m_longKernels[Kernel::MULTIPLY], M, result);
            } else if (std::is_same<ItemType, float>::value) {
                ret = execute_multiply_kernel (m_floatKernels[Kernel::MULTIPLY], M, result);
            } else if (std::is_same<ItemType, double>::value) {
                ret = execute_multiply_kernel (m_doubleKernels[Kernel::MULTIPLY], M, result);
            } else {
                printf ("Can't GPU compute, unsupported item type.\n");
            }
        } else {
            printf ("Can't GPU compute, matrices are of differing types.\n");
        }
        result.pullFromGPU (m_command_queue);
        clFinish (m_command_queue);
        return result;
    }
}

/**
* @brief Multiplies each element by a single value.
*
* @param[val] The value to be multiplied through the calling matrix.
*
* @return returns a matrix of the same type as the calling matrix.
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::multiply (const ArgType& val) {
    matrix<ItemType> result (m_height, m_width);
    if (! (m_useGPU || ALL_USE_GPU)) { //if don't use the gpu
        for (size_t i = 0; i < result.m_height; i++) {
            for (size_t j = 0; j < result.m_width; j++) {
                (*result.m_data[i])[j] = (*m_data[i])[j] * val;
            }
        }
        result.m_useGPU = m_useGPU;
        return result;
    } else {
        if (!GPU_INITIALIZED) {
            printf ("GPU is not initialized!\n");
            return matrix<ItemType> (0, 0);
        }
        initQueue();
        pushToGPU (m_command_queue);
        result.createResultBuffer (m_command_queue);
        cl_int ret; //i'm not even kinda checking this rn, but maybe i will later :P
        if (std::is_same<ItemType, char>::value) {
            ret = execute_array_val_kernel (m_charKernels[Kernel::MULTIPLY_SCALAR], (ItemType&) val, result);
        } else if (std::is_same<ItemType, short>::value) {
            ret = execute_array_val_kernel (m_shortKernels[Kernel::MULTIPLY_SCALAR], (ItemType&) val, result);
        } else if (std::is_same<ItemType, int>::value) {
            ret = execute_array_val_kernel (m_intKernels[Kernel::MULTIPLY_SCALAR], (ItemType&) val, result);
        } else if (std::is_same<ItemType, long>::value) {
            ret = execute_array_val_kernel (m_longKernels[Kernel::MULTIPLY_SCALAR], (ItemType&) val, result);
        } else if (std::is_same<ItemType, float>::value) {
            ret = execute_array_val_kernel (m_floatKernels[Kernel::MULTIPLY_SCALAR], (ItemType&) val, result);
        } else if (std::is_same<ItemType, double>::value) {
            ret = execute_array_val_kernel (m_doubleKernels[Kernel::MULTIPLY_SCALAR], (ItemType&) val, result);
        } else {
            printf ("Can't GPU compute, unsupported item type.\n");
        }
        result.pullFromGPU (m_command_queue);
        clFinish (m_command_queue);
        clReleaseCommandQueue (m_command_queue);
        m_command_queue = NULL;
        return result;
    }
}

/**
* @brief Multiplies each element of the calling matrix by the corresponding element of the rhs matrix.
*
* @param[M] The rhs matrix for the multiplications.
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::elementMultiply (matrix<ArgType>& M) {
    //should I be concatenating the matrix ike this? probably not, a submatrix
    //can be made easily enough if they want concatenation...
    //yeah, i'll change this later
    matrix<ItemType> result (m_height < M.m_height ? m_height : M.m_height, m_width < M.m_width ? m_width : M.m_width);
    if (! (m_useGPU || ALL_USE_GPU)) { //if don't use the gpu
        for (size_t i = 0; i < result.m_height; i++) {
            for (size_t j = 0; j < result.m_width; j++) {
                (*result.m_data[i])[j] = (*m_data[i])[j] * (*M.m_data[i])[j];
            }
        }
        result.m_useGPU = m_useGPU;
        return result;
    } else {
        if (!GPU_INITIALIZED) {
            printf ("GPU is not initialized!\n");
            return matrix<ItemType> (0, 0);
        }
        initQueue();
        pushToGPU (m_command_queue);
        M.pushToGPU (m_command_queue);
        result.createResultBuffer (m_command_queue);
        if (std::is_same<ItemType, ArgType>::value) {
            cl_int ret;

            if (std::is_same<ItemType, char>::value) {
                ret = execute_multiply_kernel (m_charKernels[Kernel::MULTIPLY_ELEMENT], M, result);
            } else if (std::is_same<ItemType, short>::value) {
                ret = execute_multiply_kernel (m_shortKernels[Kernel::MULTIPLY_ELEMENT], M, result);
            } else if (std::is_same<ItemType, int>::value) {
                ret = execute_multiply_kernel (m_intKernels[Kernel::MULTIPLY_ELEMENT], M, result);
            } else if (std::is_same<ItemType, long>::value) {
                ret = execute_multiply_kernel (m_longKernels[Kernel::MULTIPLY_ELEMENT], M, result);
            } else if (std::is_same<ItemType, float>::value) {
                ret = execute_multiply_kernel (m_floatKernels[Kernel::MULTIPLY_ELEMENT], M, result);
            } else if (std::is_same<ItemType, double>::value) {
                ret = execute_multiply_kernel (m_doubleKernels[Kernel::MULTIPLY_ELEMENT], M, result);
            } else {
                printf ("Can't GPU compute, unsupported item type.\n");
            }
        } else {
            printf ("Can't GPU compute, matrices are of differing types.\n");
        }
        result.pullFromGPU (m_command_queue);
        clFinish (m_command_queue);
        return result;
    }
}

/**
* @brief Operator overload for matrix::multiply()
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::operator* (matrix<ArgType>& M) {
    return this->multiply (M);
}

/**
* @brief Operator overload for matrix::multiply()
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::operator* (const ArgType& val) {
    return this->multiply (val);
}
//}
// </editor-fold>
#pragma endregion

#pragma region Division Functions
// <editor-fold desc="Division Functions">
//{

/**
* @brief Divides a matrix by another
*
* @detail It isn't /exactly/ division, it's just multiplying M1 by the inverse of M2... but that's pretty dang similar
*
* @return returns a matrix of the same type as the calling matrix
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::divide(matrix<ArgType>& M) {
    if (M.m_height != M.m_width) {
        return matrix<ItemType>(0, 0);
    }
    matrix<ArgType> invM = LinAlgo::inverse(M);
    if (invM == matrix<ArgType>(0, 0)) {
        return matrix<ItemType>(0, 0);
    }
    return multiply(invM);
}

/**
* @brief Divides each element by a single value.
*
* @param[val] The value to divide the calling matrix by.
*
* @return returns a matrix of the same type as the calling matrix.
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::divide (const ArgType& val) {
    matrix<ItemType> result (m_height, m_width);
    if (! (m_useGPU || ALL_USE_GPU)) { //if don't use the gpu
        for (size_t i = 0; i < result.m_height; i++) {
            for (size_t j = 0; j < result.m_width; j++) {
                (*result.m_data[i])[j] = (*m_data[i])[j] / val;
            }
        }
        result.m_useGPU = m_useGPU;
        return result;
    } else {
        if (!GPU_INITIALIZED) {
            printf ("GPU is not initialized!\n");
            return matrix<ItemType> (0, 0);
        }
        initQueue();
        pushToGPU (m_command_queue);
        result.createResultBuffer (m_command_queue);
        cl_int ret; //i'm not even kinda checking this rn, but maybe i will later :P
        if (std::is_same<ItemType, char>::value) {
            ret = execute_array_val_kernel (m_charKernels[Kernel::DIVIDE_SCALAR], (ItemType&) val, result);
        } else if (std::is_same<ItemType, short>::value) {
            ret = execute_array_val_kernel (m_shortKernels[Kernel::DIVIDE_SCALAR], (ItemType&) val, result);
        } else if (std::is_same<ItemType, int>::value) {
            ret = execute_array_val_kernel (m_intKernels[Kernel::DIVIDE_SCALAR], (ItemType&) val, result);
        } else if (std::is_same<ItemType, long>::value) {
            ret = execute_array_val_kernel (m_longKernels[Kernel::DIVIDE_SCALAR], (ItemType&) val, result);
        } else if (std::is_same<ItemType, float>::value) {
            ret = execute_array_val_kernel (m_floatKernels[Kernel::DIVIDE_SCALAR], (ItemType&) val, result);
        } else if (std::is_same<ItemType, double>::value) {
            ret = execute_array_val_kernel (m_doubleKernels[Kernel::DIVIDE_SCALAR], (ItemType&) val, result);
        } else {
            printf ("Can't GPU compute, unsupported item type.\n");
        }
        result.pullFromGPU (m_command_queue);
        clFinish (m_command_queue);
        clReleaseCommandQueue (m_command_queue);
        m_command_queue = NULL;
        return result;
    }
}

/**
* @brief Divides each element of the calling matrix by each element of the rhs matrix.
*
* @param[M] The rhs matrix for the divisions.
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::elementDivide (matrix<ArgType>& M) {
    //should I be concatenating the matrix ike this? probably not, a submatrix
    //can be made easily enough if they want concatenation...
    //yeah, i'll change this later
    matrix<ItemType> result (m_height < M.m_height ? m_height : M.m_height, m_width < M.m_width ? m_width : M.m_width);
    if (! (m_useGPU || ALL_USE_GPU)) { //if don't use the gpu
        for (size_t i = 0; i < result.m_height; i++) {
            for (size_t j = 0; j < result.m_width; j++) {
                (*result.m_data[i])[j] = (*m_data[i])[j] / (*M.m_data[i])[j];
            }
        }
        result.m_useGPU = m_useGPU;
        return result;
    } else {
        if (!GPU_INITIALIZED) {
            printf ("GPU is not initialized!\n");
            return matrix<ItemType> (0, 0);
        }
        initQueue();
        pushToGPU (m_command_queue);
        M.pushToGPU (m_command_queue);
        result.createResultBuffer (m_command_queue);
        if (std::is_same<ItemType, ArgType>::value) {
            cl_int ret;

            if (std::is_same<ItemType, char>::value) {
                ret = execute_multiply_kernel (m_charKernels[Kernel::DIVIDE_ELEMENT], M, result);
            } else if (std::is_same<ItemType, short>::value) {
                ret = execute_multiply_kernel (m_shortKernels[Kernel::DIVIDE_ELEMENT], M, result);
            } else if (std::is_same<ItemType, int>::value) {
                ret = execute_multiply_kernel (m_intKernels[Kernel::DIVIDE_ELEMENT], M, result);
            } else if (std::is_same<ItemType, long>::value) {
                ret = execute_multiply_kernel (m_longKernels[Kernel::DIVIDE_ELEMENT], M, result);
            } else if (std::is_same<ItemType, float>::value) {
                ret = execute_multiply_kernel (m_floatKernels[Kernel::DIVIDE_ELEMENT], M, result);
            } else if (std::is_same<ItemType, double>::value) {
                ret = execute_multiply_kernel (m_doubleKernels[Kernel::DIVIDE_ELEMENT], M, result);
            } else {
                printf ("Can't GPU compute, unsupported item type.\n");
            }
        } else {
            printf ("Can't GPU compute, matrices are of differing types.\n");
        }
        result.pullFromGPU (m_command_queue);
        clFinish (m_command_queue);
        return result;
    }
}

/**
* @brief Operator overload for matrix::divide()
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::operator/ (const ArgType& val) {
    return this->divide (val);
}

/**
* @brief Operator overload for matrix::divide()
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::operator/ (matrix<ArgType>& M) {
    return this->divide (M);
}
//}
// </editor-fold>
#pragma endregion

#pragma region Linear Algebra Functions
// <editor-fold desc="Linear Algebra Functions">
//{
/**
* @brief Transposes the matrix
*
* @return resturns a reference to the calling matrix
*/
template <class ItemType>
matrix<ItemType>& matrix<ItemType>::transpose() {
    matrix<ItemType> result (m_width, m_height, m_useGPU);
    for (size_t i = 0; i < m_width; i++) {
        for (size_t j = 0; j < m_height; j++) {
            (*result.m_data[i])[j] = (*m_data[j])[i];
        }
    }
    //I could try to do an in-place transpose on the gpu
    //then pull it... hm. I'll have to look into that later
    m_data = result.m_data;
    m_height = m_width;
    m_width = result.m_width;
    m_gpuUpToDate = false;
    m_upToDate &= ! (dataFlag::GPU_HEIGHT | dataFlag::GPU_WIDTH | dataFlag::GPU_DATA);
    m_gpuSlicesUpToDate.clear();
    m_gpuSlicesUpToDate.resize (m_height, false);
    return *this;
}
//}
// </editor-fold>
#pragma endregion

#pragma region Operator Overloads
// <editor-fold desc="Operator Overloads">
//{
/**
* @brief Using the [] operator will slice a row of the matrix.
*
* @param[y] The row to be sliced.
*
* @return A reference to the std::vector storing the appropriate row of the matrix. use with caution pls.
*/
template <class ItemType>
std::vector<ItemType>& matrix<ItemType>::operator[] (size_t y) const {
    return (std::vector<ItemType>&) * m_data[y];
    //the data in ram can be changed by this...
    //how to make sure unchanged assumes gpu is safe
    //but know when it's been changed?
}

/**
* @brief Assignment operator whoot
*
* @param[M] The matrix to be copied.
*
* @return A reference to the lhs matrix.
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType>& matrix<ItemType>::operator= (const matrix<ArgType>& M) {
    if (m_height != M.m_height) {
        m_data.resize (M.m_height);
        m_height = M.m_height;
    }
    if (m_width != M.m_width) {
        for (size_t i = 0; i < m_height; i++) {
            (*m_data[i]).resize (M.m_width);
        }
        m_width = M.m_width;
    }
    if (std::is_same<ItemType, ArgType>::value) {
        if (GPU_INITIALIZED && (M.useGPU || ALL_USE_GPU) && M.gpuUpToData) {
            while (m_command_queue != NULL);
            initQueue();
            if (m_gpuData) {
                clReleaseMemObject (m_gpuData);
            }
            if (m_gpuHeight) {
                clReleaseMemObject (m_gpuHeight);
            }
            if (m_gpuWidth) {
                clReleaseMemObject (m_gpuWidth);
            }
            m_gpuData = M.m_gpuData;
            m_gpuHeight = M.m_gpuHeight;
            m_gpuWidth = M.m_gpuWidth;
            pullFromGPU (m_command_queue);
            clFinish (m_command_queue);
            clReleaseCommandQueue (m_command_queue);
            m_command_queue = NULL;
            m_gpuData = NULL;
            m_gpuHeight = NULL;
            m_gpuWidth = NULL;
        } else {
            for (size_t i = 0; i < m_height; i++) {
                for (size_t j = 0; j < m_width; j++) {
                    (*m_data[i])[j] = (*M.m_data[i])[j];
                }
            }
        }
    } else {
        for (size_t i = 0; i < m_height; i++) {
            for (size_t j = 0; j < m_width; j++) {
                (*m_data[i])[j] = ItemType ((*M.m_data[i])[j]);
            }
        }
    }
    m_gpuUpToDate = false;
    m_gpuSlicesUpToDate.resize (m_height, false);
    m_upToDate = 0;//if eigenvalues and stuff get copied over, this will be different
    m_useGPU = M.m_useGPU;
    if (m_command_queue) {
        clFinish (m_command_queue);
        m_command_queue = NULL;
    }
    if (m_gpuData) {
        clReleaseMemObject (m_gpuData);
        m_gpuData = NULL;
    }
    if (m_gpuHeight) {
        clReleaseMemObject (m_gpuHeight);
        m_gpuHeight = NULL;
    }
    if (m_gpuWidth) {
        clReleaseMemObject (m_gpuWidth);
        m_gpuWidth = NULL;
    }

    //clear everything else that's now irrelevent (or copy things that are over lol)
    return *this;
}

/**
* @brief Move assignment operator
*/
template <class ItemType>
matrix<ItemType>& matrix<ItemType>::operator= (matrix<ItemType>&& M) {
    m_height = M.m_height;
    m_width = M.m_width;
    m_data = M.m_data;
    for (size_t i = 0; i < M.m_height; i++) {
        M.m_data[i] = NULL;
    }
    m_gpuData = M.m_gpuData;
    M.m_gpuData = NULL;
    m_command_queue = M.m_command_queue;
    M.m_command_queue = NULL;
    m_gpuHeight = M.m_gpuHeight;
    M.m_gpuHeight = NULL;
    m_gpuWidth = M.m_gpuWidth;
    M.m_gpuWidth = NULL;
    m_upToDate = M.m_upToDate;
    m_useGPU = M.m_useGPU;
    m_leaveOnGPU = M.m_leaveOnGPU;
    m_gpuUpToDate = M.m_gpuUpToDate;
    m_gpuSlicesUpToDate = M.m_gpuSlicesUpToDate;
    return *this;
}

/**
* @brief Equality operator
*
* @detail Checks if non-GPU data is equivalent, checked data is only height, width, and matrix content.
* (Requires an update from GPU if set to leave data on the GPU)
*/
template <class ItemType>
template <class ArgType>
bool matrix<ItemType>::operator== (const matrix<ArgType>& M) const {
    if (m_height != M.m_height || m_width != M.m_width) {
        return false;
    }
    for (size_t i = 0; i < m_height; i++) {
        for (size_t j = 0; j < m_width; j++) {
            if ((*m_data[i])[j] != (*M.m_data[i])[j]) {
                return false;
            }
        }
    }
    return true;
}

/**
* @brief Inequality operator
*
* @detail Checks if non-GPU data is equivalent, checked data is only height, width, and matrix content.
* (Requires an update from GPU if set to leave data on the GPU)
*/
template <class ItemType>
template <class ArgType>
bool matrix<ItemType>::operator!= (const matrix<ArgType>& M) const {
    if (m_height != M.m_height || m_width != M.m_width) {
        return true;
    }
    for (size_t i = 0; i < m_height; i++) {
        for (size_t j = 0; j < m_width; j++) {
            if ((*m_data[i])[j] == (*M.m_data[i])[j]) {
                return false;
            }
        }
    }
    return true;
}
//}
// </editor-fold>
#pragma endregion

#pragma region Auxilliary Functions for Handling GPU Data
// <editor-fold desc="Auxilliary Functions for Handling GPU Data">
//{
#pragma region Initialization Functions
// <editor-fold desc="Initialization Functions">
//{
//private auxilliary function to initialize the OpenCL command queue
template <class ItemType>
cl_int matrix<ItemType>::initQueue() {
    if (m_command_queue != NULL) {
        clReleaseCommandQueue (m_command_queue);
    }
    cl_int ret;
    if (OPENCL_VERSION >= 2.0) {
        m_command_queue = clCreateCommandQueueWithProperties (m_context, m_device_id, 0, &ret); //segfaults on manjaro/radeon
    } else {
        m_command_queue = clCreateCommandQueue (m_context, m_device_id, 0, &ret); //infinite hange on manjaro/radeon
    }
    if (ret != CL_SUCCESS) {
        printf ("Unable to create command queue, error code: %d\n", ret);
    }
    return ret;
}

//private auxialliary function for creating an empty memory buffer to store result
template <class ItemType>
cl_int matrix<ItemType>::createResultBuffer (cl_command_queue& command_queue) {
    if (m_gpuData != NULL) {
        clReleaseMemObject (m_gpuData);
    }
    if (m_gpuHeight != NULL) {
        clReleaseMemObject (m_gpuHeight);
    }
    if (m_gpuWidth != NULL) {
        clReleaseMemObject (m_gpuWidth);
    }

    cl_int ret;
    m_gpuData = clCreateBuffer (m_context, CL_MEM_READ_ONLY, m_height * m_width * sizeof (ItemType), NULL, &ret);
    if (ret != CL_SUCCESS) {
        printf ("Unable to create memory buffer, error code: %d\n", ret);
        return ret;
    }

    m_gpuHeight = clCreateBuffer (m_context, CL_MEM_READ_ONLY, sizeof (ItemType), NULL, &ret);
    if (ret != CL_SUCCESS) {
        printf ("Unable to create memory buffer, error code: %d\n", ret);
        return ret;
    }

    m_gpuWidth = clCreateBuffer (m_context, CL_MEM_READ_ONLY, sizeof (ItemType), NULL, &ret);
    if (ret != CL_SUCCESS) {
        printf ("Unable to create memory buffer, error code: %d\n", ret);
        return ret;
    }

    ret = clEnqueueWriteBuffer (command_queue, m_gpuHeight, CL_TRUE, 0, sizeof (ItemType), &m_height, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf ("Unable to push data to GPU, error code: %d\n", ret);
        return ret;
    }

    ret = clEnqueueWriteBuffer (command_queue, m_gpuWidth, CL_TRUE, 0, sizeof (ItemType), &m_width, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf ("Unable to push data to GPU, error code: %d\n", ret);
        return ret;
    }

    m_upToDate |= (dataFlag::GPU_HEIGHT | dataFlag::GPU_WIDTH);

    return ret;
}

//private auxilliary function to push the matrix data to the gpu
template <class ItemType>
cl_int matrix<ItemType>::pushToGPU (cl_command_queue& command_queue) {
    if (m_gpuUpToDate) {
        return CL_SUCCESS;
    }

    cl_int ret;
    if (!m_gpuData) {
        m_gpuData = clCreateBuffer (m_context, CL_MEM_READ_ONLY, m_height * m_width * sizeof (ItemType), NULL, &ret);
        if (ret != CL_SUCCESS) {
            printf ("Unable to create memory buffer, error code: %d\n", ret);
            return ret;
        }
    }

    if (!m_gpuHeight) {
        m_gpuHeight = clCreateBuffer (m_context, CL_MEM_READ_ONLY, sizeof (ItemType), NULL, &ret);
        if (ret != CL_SUCCESS) {
            printf ("Unable to create memory buffer, error code: %d\n", ret);
            return ret;
        }
    }

    if (!m_gpuWidth) {
        m_gpuWidth = clCreateBuffer (m_context, CL_MEM_READ_ONLY, sizeof (ItemType), NULL, &ret);
        if (ret != CL_SUCCESS) {
            printf ("Unable to create memory buffer, error code: %d\n", ret);
            return ret;
        }
    }

    if (! (m_upToDate & dataFlag::GPU_DATA)) {
        for (size_t i = 0; i < m_height; i++) {
            if (!m_gpuSlicesUpToDate[i]) {
                ret = clEnqueueWriteBuffer (command_queue, m_gpuData, CL_TRUE, i * m_width * sizeof (ItemType), m_width * sizeof (ItemType), (*m_data[i]).data(), 0, NULL, NULL);
                if (ret != CL_SUCCESS) {
                    printf ("Unable to push data to GPU, error code: %d\n", ret);
                    return ret;
                }
                m_gpuSlicesUpToDate[i] = true;
            }
        }
        m_upToDate |= dataFlag::GPU_DATA;
    }

    if (! (m_upToDate & dataFlag::GPU_HEIGHT)) {
        ret = clEnqueueWriteBuffer (command_queue, m_gpuHeight, CL_TRUE, 0, sizeof (ItemType), &m_height, 0, NULL, NULL);
        if (ret != CL_SUCCESS) {
            printf ("Unable to push data to GPU, error code: %d\n", ret);
            return ret;
        }
        m_upToDate |= dataFlag::GPU_HEIGHT;
    }

    if (! (m_upToDate & dataFlag::GPU_WIDTH)) {
        ret = clEnqueueWriteBuffer (command_queue, m_gpuWidth, CL_TRUE, 0, sizeof (ItemType), &m_width, 0, NULL, NULL);
        if (ret != CL_SUCCESS) {
            printf ("Unable to push data to GPU, error code: %d\n", ret);
            return ret;
        }
        m_upToDate |= dataFlag::GPU_WIDTH;
    }

    m_gpuUpToDate = true;

    return ret;
}

//private auxilliary function for retrieving data from the gpu
template <class ItemType>
cl_int matrix<ItemType>::pullFromGPU (cl_command_queue& command_queue) {
    cl_int ret;

    for (int i = 0; i < m_height; i++) {
        ret = clEnqueueReadBuffer (command_queue, m_gpuData, CL_TRUE, i * m_width * sizeof (ItemType), m_width * sizeof (ItemType), (*m_data[i]).data(), 0, NULL, NULL);
        if (ret != CL_SUCCESS) {
            printf ("Unable to retrieve data from GPU, error code: %d\n", ret);
            return ret;
        }
        m_gpuSlicesUpToDate[i] = true;
    }
    m_gpuUpToDate = true;
    m_upToDate |= dataFlag::GPU_DATA;
    return ret;
}
//}
// </editor-fold>
#pragma endregion

#pragma region Functions for Executing Kernels
// <editor-fold desc="Functions for Executing Kernels">
//{
//Private function for executing the add kernel
template <class ItemType>
cl_int matrix<ItemType>::execute_add_kernel (cl_kernel kernel, matrix<ItemType>& rhs, matrix<ItemType>& result) {
    cl_int ret;
    //set kernel arguments
    ret = clSetKernelArg (kernel, 0, sizeof (cl_mem), (void*)& m_gpuData);
    if (ret != CL_SUCCESS) {
        printf ("Could not set kernel argument.\n");
    }
    ret = clSetKernelArg (kernel, 1, sizeof (cl_mem), (void*)& m_gpuWidth);
    if (ret != CL_SUCCESS) {
        printf ("Could not set kernel argument.\n");
    }
    ret = clSetKernelArg (kernel, 2, sizeof (cl_mem), (void*)& rhs.m_gpuData);
    if (ret != CL_SUCCESS) {
        printf ("Could not set kernel argument.\n");
    }
    ret = clSetKernelArg (kernel, 3, sizeof (cl_mem), (void*)& rhs.m_gpuWidth);
    if (ret != CL_SUCCESS) {
        printf ("Could not set kernel argument.\n");
    }
    ret = clSetKernelArg (kernel, 4, sizeof (cl_mem), (void*)& result.m_gpuData);
    if (ret != CL_SUCCESS) {
        printf ("Could not set kernel argument.\n");
    }
    ret = clSetKernelArg (kernel, 5, sizeof (cl_mem), (void*)& result.m_gpuWidth);
    if (ret != CL_SUCCESS) {
        printf ("Could not set kernel argument.\n");
    }

    //execute the kernel
    size_t global_item_size = result.m_width * result.m_height;
    //size_t local_item_size = 64;

    //removed local work group size since using it properly requires a fair amount bothering to query the cpu
    //and calculating divisibility of the broader work group space by the local. As well as determing if there
    //should be work group dimensions and splitting amoung those. So, in lieu of not wanting to modify both this
    //function and the gpu initialization function, imma leave it at NULL so that the gpu decides it on its own
    //until i decide i have a better idea on how i can handle it myself (especially the dimesnsions)
    //Oh, and changing it to null made it actually work on linux instead of throwing CL_INVALID_WORK_GROUP_SIZE
    ret = clEnqueueNDRangeKernel (m_command_queue, kernel, 1, NULL, &global_item_size, /*&local_item_size*/ NULL, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf ("Could not execute kernel. Error Code: %d\n", ret);
    }
    return ret;
}

//function for executing the multiplication kernel
template <class ItemType>
cl_int matrix<ItemType>::execute_multiply_kernel (cl_kernel kernel, matrix<ItemType>& rhs, matrix<ItemType>& result) {
    cl_int ret;
    //set kernel arguments
    ret = clSetKernelArg (kernel, 0, sizeof (cl_mem), (void*)& m_gpuData);
    if (ret != CL_SUCCESS) {
        printf ("Could not set kernel argument.\n");
    }
    ret = clSetKernelArg (kernel, 1, sizeof (cl_mem), (void*)& m_gpuWidth); //single arg
    if (ret != CL_SUCCESS) {
        printf ("Could not set kernel argument.\n");
    }
    ret = clSetKernelArg (kernel, 2, sizeof (cl_mem), (void*)& rhs.m_gpuData);
    if (ret != CL_SUCCESS) {
        printf ("Could not set kernel argument.\n");
    }
    ret = clSetKernelArg (kernel, 3, sizeof (cl_mem), (void*)& rhs.m_gpuWidth); //single arg
    if (ret != CL_SUCCESS) {
        printf ("Could not set kernel argument.\n");
    }
    ret = clSetKernelArg (kernel, 4, sizeof (cl_mem), (void*)& result.m_gpuData);
    if (ret != CL_SUCCESS) {
        printf ("Could not set kernel argument.\n");
    }
    ret = clSetKernelArg (kernel, 5, sizeof (cl_mem), (void*)& result.m_gpuWidth); //single arg
    if (ret != CL_SUCCESS) {
        printf ("Could not set kernel argument.\n");
    }

    //execute the kernel
    size_t global_item_size = result.m_width * result.m_height;
    size_t local_item_size = 64;

    ret = clEnqueueNDRangeKernel (m_command_queue, kernel, 1, NULL, &global_item_size, /*&local_item_size*/ NULL, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf ("Could not execute kernel. Error Code: %d\n", ret);
    }
    return ret;
}

//function for executing the array and single value kernels
template <class ItemType>
cl_int matrix<ItemType>::execute_array_val_kernel (cl_kernel kernel, ItemType& val, matrix<ItemType>& result) {
    cl_int ret;

    //push val to gpu
    cl_mem value = clCreateBuffer (m_context, CL_MEM_READ_ONLY, sizeof (ItemType), NULL, &ret);
    if (ret != CL_SUCCESS) {
        printf ("Unable to create memory buffer, error code: %d\n", ret);
        return ret;
    }

    ret = clEnqueueWriteBuffer (m_command_queue, value, CL_TRUE, 0, sizeof (ItemType), &val, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf ("Unable to push data to GPU, error code: %d\n", ret);
        return ret;
    }

    //set kernel arguments
    ret = clSetKernelArg (kernel, 0, sizeof (cl_mem), (void*)& m_gpuData);
    if (ret != CL_SUCCESS) {
        printf ("Could not set kernel argument.\n");
    }
    ret = clSetKernelArg (kernel, 1, sizeof (cl_mem), (void*)& result.m_gpuData);
    if (ret != CL_SUCCESS) {
        printf ("Could not set kernel argument.\n");
    }
    ret = clSetKernelArg (kernel, 2, sizeof (cl_mem), (void*)& value); //single arg
    if (ret != CL_SUCCESS) {
        printf ("Could not set kernel argument.\n");
    }

    //execute the kernel
    size_t global_item_size = result.m_width * result.m_height;

    ret = clEnqueueNDRangeKernel (m_command_queue, kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf ("Could not execute kernel. Error Code: %d\n", ret);
    }

    clFinish (m_command_queue);
    clReleaseMemObject (value);

    return ret;
}
//}
// </editor-fold>
#pragma endregion
//}
// </editor-fold>
#pragma endregion

#endif
