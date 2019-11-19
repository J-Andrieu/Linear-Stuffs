/**
* @file matrix.cpp
*
* @brief Implements the matrix class.
*/

//once kernels are finalized the template and type-specific macros will be stored in here, and the kernels will generated when the gpu is initialized

#ifndef _MATRIX_INCLUDED
#define _MATRIX_INCLUDED
#include "../include/matrix.h"

using namespace LinAlgo;

#pragma region Constructors and Destructors
// <editor-fold desc="Constructors and Destructors">
//{

/**
* @brief This is the default constructor for the matrix class.
*
* @details Creates an empty 0x0 matrix of the specified type.
*/
#ifndef DONT_USE_GPU
template <class ItemType>
matrix<ItemType>::matrix () : m_height (0), m_width (0), m_useGPU (ALL_USE_GPU),
    m_gpuUpToDate (false), m_gpuData (NULL), m_gpuHeight (NULL), m_leaveOnGPU(false),
    m_gpuWidth (NULL), m_gpuSlicesUpToDate (0) {
    m_dataInitialized = false;
    m_upToDate = 0;
}
#else
template <class ItemType>
matrix<ItemType>::matrix () : m_height (0), m_width (0) {}
#endif

/**
* @brief This is the constructor for the matrix class.
*
* @details Creates the desired mxn matrix of the template-specified data type.
*
* @param height The desired height of the matrix
*
* @param width The desired width of the matrix
*/
#ifndef DONT_USE_GPU
template <class ItemType>
matrix<ItemType>::matrix (const size_t& height, const size_t& width, const ItemType val) : m_height (height), m_width (width), m_useGPU (ALL_USE_GPU),
    m_gpuUpToDate (false), m_gpuData (NULL), m_gpuHeight (NULL), m_leaveOnGPU(false),
    m_gpuWidth (NULL), m_gpuSlicesUpToDate (height, false) {
    if (m_height != 0 && m_width != 0) {
        m_data.resize (m_height);
        for (size_t i = 0; i < m_height; i++) {
            m_data[i].resize(m_width, val);
        }
        m_dataInitialized = true;
    } else {
        m_dataInitialized = false;
    }
    m_upToDate = 0;
    if (m_useGPU) {
        initQueue();
    }
}
#else
template <class ItemType>
matrix<ItemType>::matrix (const size_t& height, const size_t& width, const ItemType val) : m_height (height), m_width (width) {
    m_data.resize (m_height);
    for (size_t i = 0; i < m_height; i++) {
        m_data[i].resize(m_width, val);
    }
}
#endif

/**
* @brief Explicit matrix constructor
*
* @detail Allows you to create a matrix from initializer lists. If the
* lists aren't all the same size, then the width will be the widest list,
* and 0 will be assigned to all empty locations
*
* @param vals The vector of vectors to be turned into a matrix
*/
#ifndef DONT_USE_GPU
template <class ItemType>
matrix<ItemType>::matrix (const std::vector<std::vector<ItemType>>& vals) : m_useGPU (ALL_USE_GPU), m_gpuUpToDate (false), m_gpuData (NULL),
    m_gpuHeight (NULL), m_gpuWidth (NULL), m_leaveOnGPU (false) {
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
        m_data[i].resize(m_width, 0);
        for (size_t j = 0; j < vals[i].size(); j++) {
            m_data[i][j] = vals[i][j];
        }
    }
    m_gpuSlicesUpToDate.resize (m_height, false);
    m_upToDate = 0;
    m_dataInitialized = true;
    if (m_useGPU) {
        initQueue();
    }
}
#else
template <class ItemType>
matrix<ItemType>::matrix (const std::vector<std::vector<ItemType>>& vals) {
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
        m_data[i].resize(m_width, 0);
        for (size_t j = 0; j < vals[i].size(); j++) {
            m_data[i][j] = vals[i][j];
        }
    }
}
#endif

/**
* @brief Explicit matrix constructor
*
* @detail Allows you to create a matrix from a pointer to a 2D array of values
*
* @param vals The 2D array of values to be turned into a matrix
*
* @param height Height of the provided data
*
* @param width Width of the provided data
*/
#ifndef DONT_USE_GPU
template <class ItemType>
matrix<ItemType>::matrix (const ItemType** vals, const size_t& height, const size_t& width) : m_useGPU (ALL_USE_GPU), m_gpuUpToDate (false), m_gpuData (NULL),
    m_gpuHeight (NULL), m_gpuWidth (NULL), m_leaveOnGPU (false), m_height(height), m_width(width) {
    m_data.resize (m_height);
    for (size_t i = 0; i < m_height; i++) {
        m_data[i].resize(m_width, 0);
        for (size_t j = 0; j < vals[i].size(); j++) {
            m_data[i][j] = vals[i][j];
        }
    }
    m_gpuSlicesUpToDate.resize (m_height, false);
    m_upToDate = 0;
    m_dataInitialized = true;
    if (m_useGPU) {
        initQueue();
    }
}
#else
template <class ItemType>
matrix<ItemType>::matrix (const ItemType** vals, const size_t& height, const size_t& width) : m_height(height), m_width(width) {
    m_data.resize (m_height);
    for (size_t i = 0; i < m_height; i++) {
        m_data[i].resize(m_width, 0);
        for (size_t j = 0; j < vals[i].size(); j++) {
            m_data[i][j] = vals[i][j];
        }
    }
}
#endif

/**
* @brief This is the (templated) copy constructor for the matrix class
*
* @details This constructor will copy the height, width, and data. It also copies whether or not
* it should use the gpu. This onstructor is only called when it needs to convert matrix types
*
* @param M This is the matrix to be copied
*/
#ifndef DONT_USE_GPU
template <class ItemType>
template <class ArgType>
matrix<ItemType>::matrix (const matrix<ArgType>& M) : m_useGPU (M.useGPU()), m_gpuUpToDate (false),
    m_gpuData (NULL), m_gpuHeight (NULL),
    m_gpuWidth (NULL),
    m_gpuSlicesUpToDate (M.getHeight(), false), m_leaveOnGPU (M.leaveDataOnGPU()) {
    m_height = M.getHeight();
    m_width = M.getWidth();
    m_upToDate = 0;//this will defo be different later
    m_data.resize (m_height);
    //it's up to the programmer to remember to pull data if they want it left on the gpu
    //if (M.leaveDataOnGPU()) {
    //    M.pullData();
    //}
    for (size_t i = 0; i < m_height; i++) {
        m_data[i].resize(m_width);
        for (size_t j = 0; j < m_width; j++) {
            m_data[i][j] = ItemType (M[i][j]);
        }
    }
    if (m_height != 0 && m_width != 0) {
        m_dataInitialized = true;
    } else {
        m_dataInitialized = false;
    }
    if (m_useGPU) {
        initQueue();
    }
}
#else
template <class ItemType>
template <class ArgType>
matrix<ItemType>::matrix (const matrix<ArgType>& M) {
    m_height = M.getHeight();
    m_width = M.getWidth();
    m_data.resize (m_height);
    for (size_t i = 0; i < m_height; i++) {
        m_data[i].resize(m_width);
        for (size_t j = 0; j < m_width; j++) {
            m_data[i][j] = ItemType (M[i][j]);
        }
    }
}
#endif

/**
* @brief This is the copy constructor for the matrix class
*
* @details This constructor will copy the height, width, and data. It also copies whether or not
* it should use the gpu.
*
* @param M This is the matrix to be copied
*/
#ifndef DONT_USE_GPU
template <class ItemType>
matrix<ItemType>::matrix (const matrix<ItemType>& M) : m_useGPU (M.m_useGPU), m_gpuUpToDate (false),
    m_gpuData (NULL), m_gpuHeight (NULL),
    m_gpuWidth (NULL),
    m_gpuSlicesUpToDate (M.m_height, false), m_leaveOnGPU (M.m_leaveOnGPU),
    m_data (M.m_height) {
    m_height = M.m_height;
    m_width = M.m_width;
    m_upToDate = 0;//this will defo be different later
    for (size_t i = 0; i < m_height; i++) {
        m_data[i].resize(m_width);
        for (size_t j = 0; j < m_width; j++) {
            m_data[i][j] = M.m_data[i][j];
        }
    }
    m_dataInitialized = M.m_dataInitialized;
    if (m_useGPU) {
        initQueue();
    }
}
#else
template <class ItemType>
matrix<ItemType>::matrix (const matrix<ItemType>& M) : m_data (M.m_height) {
    m_height = M.m_height;
    m_width = M.m_width;
    for (size_t i = 0; i < m_height; i++) {
        m_data[i].resize(m_width);
        for (size_t j = 0; j < m_width; j++) {
            m_data[i][j] = M.m_data[i][j];
        }
    }

}
#endif

/**
* @brief This is the move constructor for the matrix class
*
* @param M This is the matrix to be moved
*/
#ifndef DONT_USE_GPU
template <class ItemType>
matrix<ItemType>::matrix (matrix<ItemType>&& M) : m_useGPU (M.m_useGPU), m_gpuUpToDate (M.m_gpuUpToDate),
    m_gpuData (M.m_gpuData), m_gpuHeight (M.m_gpuHeight),
    m_gpuWidth (M.m_gpuWidth),
    m_gpuSlicesUpToDate (M.m_gpuSlicesUpToDate), m_leaveOnGPU (M.m_leaveOnGPU) {
    m_height = M.m_height;
    m_width = M.m_width;
    m_data = std::move(M.m_data);
    m_dataInitialized = M.m_dataInitialized;
    M.m_gpuData = NULL;
    //M.m_command_queue = NULL;
    M.m_gpuHeight = NULL;
    M.m_gpuWidth = NULL;
}
#else
template <class ItemType>
matrix<ItemType>::matrix (matrix<ItemType>&& M) {
    m_height = M.m_height;
    m_width = M.m_width;
    m_data = std::move(M.m_data);
}
#endif

//this is broken... somehow
/**
* @brief matrix destructor
*
* @detail Mostly just here to clean up any gpu data lyin' about
*/
template <class ItemType>
matrix<ItemType>::~matrix() {
#ifndef DONT_USE_GPU
    if (m_command_queue != NULL) {
        //clReleaseCommandQueue(m_command_queue);
    }
    if (m_gpuData != NULL) {
        clReleaseMemObject(m_gpuData);
    }
    if (m_gpuHeight != NULL) {
        clReleaseMemObject(m_gpuHeight);
    }
    if (m_gpuWidth != NULL) {
        clReleaseMemObject(m_gpuWidth);
    }
#endif
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
* @param val The value to fill the matrix with
*/
template <class ItemType>
void matrix<ItemType>::fill (ItemType val) {
    for (size_t i = 0; i < m_height; i++) {
        for (size_t j = 0; j < m_width; j++) {
            m_data[i][j] = val;
        }
    }
#ifndef DONT_USE_GPU
    m_gpuUpToDate = false;
    m_upToDate &= (!(dataFlag::GPU_HEIGHT | dataFlag::GPU_WIDTH));
    m_gpuSlicesUpToDate.clear();
    m_gpuSlicesUpToDate.resize (m_height, false);
#endif
}

/**
* @brief Clears the matrix
*/
template <class ItemType>
void matrix<ItemType>::clear() {
    m_data.clear();
    m_height = 0;
    m_width = 0;
#ifndef DONT_USE_GPU
    m_dataInitialized = false;
    m_gpuUpToDate = false;
    m_upToDate &= (dataFlag::GPU_HEIGHT | dataFlag::GPU_WIDTH);
    m_gpuSlicesUpToDate.clear();
    m_gpuSlicesUpToDate.resize (m_height, false);
#endif
}

#ifndef DONT_USE_GPU
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
            if (m_leaveOnGPU) {
                pullFromGPU(m_command_queue);
                m_leaveOnGPU = false;
            }
            //maybe this shouldn't just erase the gpu data
            //if (m_command_queue) {
            //    clReleaseCommandQueue (m_command_queue);
            //    m_command_queue = NULL;
            //}
            //if (m_gpuData) {
            //    clReleaseMemObject (m_gpuData);
            //    m_gpuData = NULL;
            //}
            //if (m_gpuHeight || m_gpuWidth) {
            //    clReleaseMemObject (m_gpuHeight);
            //    clReleaseMemObject (m_gpuWidth);
            //    m_gpuHeight = NULL;
            //    m_gpuWidth = NULL;
            //}
            //m_gpuUpToDate = false;
            //m_gpuSlicesUpToDate.clear();
            //m_gpuSlicesUpToDate.resize (m_height, false);
            //m_upToDate |= ! (dataFlag::GPU_HEIGHT | dataFlag::GPU_WIDTH | dataFlag::GPU_DATA);
        } else {
            if (m_command_queue == NULL) {
                //initQueue();
            }
        }
        return true;
    }
    return false;
}

/**
* @brief Return if the matrix is using the GPU
*/
template <class ItemType>
bool matrix<ItemType>::useGPU() const {
    return m_useGPU;
}
#endif
//}
// </editor-fold>
#pragma endregion

#pragma region Setters and Getters
// <editor-fold desc="Setters and Getters">
//{
/**
* @brief This function returns the value or object stored at the specified position.
*
* @param y The height of the position.
*
* @param x The width of the position.
*/
template <class ItemType>
ItemType matrix<ItemType>::get (size_t y, size_t x) const  {
    return m_data[y][x];
}

/**
* @brief This function returns an std::vector of the requested row
*
* @param r The row to slice
*/
template <class ItemType>
std::vector<ItemType> matrix<ItemType>::getRow (size_t r) const  {
    return m_data[r];
}

/**
* @brief This function returns an std::vector of the requested column
*
* @param c The column to slice
*/
template <class ItemType>
std::vector<ItemType> matrix<ItemType>::getColumn (size_t c) const  {
    std::vector<ItemType> vec(m_height);
    for (size_t i = 0; i < m_height; i++) {
        vec[i] = m_data[i][c];
    }
    return vec;
}

/**
* @brief This function sets the specified location in the matrix to the specified value.
*
* @param y The height of the position.
*
* @param x The width of the position.
*
* @param val The value to be inserted to the position.
*/
template <class ItemType>
void matrix<ItemType>::set (size_t y, size_t x, ItemType val) {
    m_data[y][x] = val;
#ifndef DONT_USE_GPU
    m_gpuUpToDate = false;
    m_upToDate &= dataFlag::GPU_DATA;
    m_gpuSlicesUpToDate[y] = false;
#endif
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
        m_determinant = m_data[0][0];
    } else if (m_height == 2) {
        m_determinant = m_data[0][0] * m_data[1][1] - m_data[0][1] * m_data[1][0];
    } else {
        int row_swaps;
        matrix<ItemType> detMat (LinAlgo::re (*this, row_swaps));
        m_determinant = (ItemType) 1;
        for (size_t i = 0; i < detMat.m_height; i++) {
            m_determinant *= detMat.m_data[i][i];
        }
        if (row_swaps % 2 != 0) {
            m_determinant *= -1;
        }
    }
    m_upToDate |= dataFlag::DETERMINANT;
    return m_determinant;
}

/**
* @brief Returns the trace of a square matrix
*
* @detail Will calculate the determinant if it hasn't been done already
*/
template <class ItemType>
ItemType matrix<ItemType>::trace() {
    //should trace be stored? it's easy enough to track, but it's also fast anyways
    if (m_height != m_width) {
        return 0;//no trace
    } else {
        int t = 0;
        for (int i = 0; i < m_height; i++) {
            t += m_data[i][i];
        }
        return t;
    }
}

/**
* @brief Resizes the calling matrix
*
* @detail This resizes the matrix to the given height and width.
*
* @param height The new height
*
* @param width The new width
*
* @param val If the matrix is being expanded, val will be used to fill the void
*/
template <class ItemType>
matrix<ItemType>& matrix<ItemType>::resize (size_t height, size_t width, ItemType val) {
    if (m_height != height || m_data.size() == 0) {
        m_data.resize (height);
#ifndef DONT_USE_GPU
        m_upToDate &= !dataFlag::GPU_HEIGHT;
        m_gpuSlicesUpToDate.resize (height, false);
        clReleaseMemObject (m_gpuHeight);
        m_gpuHeight = NULL;
#endif
    }
    if (m_width != width || m_data[0].size() == 0) {
        for (size_t i = 0; i < height; i++) {
            m_data[i].resize (width, val);
        }
#ifndef DONT_USE_GPU
        m_upToDate &= !dataFlag::GPU_WIDTH;
        clReleaseMemObject (m_gpuWidth);
        m_gpuWidth = NULL;
#endif
    }
    m_height = height;
    m_width = width;
#ifndef DONT_USE_GPU
    if (m_height == 0 || m_width == 0) {
        m_dataInitialized == false;
    } else {
        m_dataInitialized = true;
    }
    clReleaseMemObject (m_gpuData);
    m_gpuData = NULL;
    m_upToDate &= !dataFlag::GPU_DATA;
    m_gpuSlicesUpToDate.clear();
    m_gpuSlicesUpToDate.resize (m_height, false);
    m_gpuUpToDate = false;
#endif
    return *this;
}

/**
* @brief Copies provided matrix into the calling matrix starting at the specified index
*
* @param y The starting height
*
* @param x The starting width
*
* @param M The matrix to copy
*/
template <class ItemType>
matrix<ItemType>& matrix<ItemType>::copy (size_t y, size_t x, matrix<ItemType> M) {
    for (size_t i = 0; i < M.m_height && ((i + y) < m_height); i++) {
        for (size_t j = 0; j < M.m_width && ((j + x) < m_width); j++) {
            m_data[i+y][j+x] = M.m_data[i][j];
        }
    }
#ifndef DONT_USE_GPU
    clReleaseMemObject (m_gpuData);
    m_gpuData = NULL;
    m_upToDate &= !dataFlag::GPU_DATA;
    m_gpuSlicesUpToDate.clear();
    m_gpuSlicesUpToDate.resize (m_height, false);
    m_gpuUpToDate = false;
#endif
    return *this;
}

/**
* @brief Creates a new matrix from a subregion of the calling matrix
*
* @param y The row to start the slice
*
* @param x The column to start the slice
*
* @param h The height of the new matrix
*
* @param w The width of the new matrix
*
* @note If the requested region is out of bounds it will throw a runtime error
*/
template <class ItemType>
matrix<ItemType> matrix<ItemType>::subMatrix (size_t y, size_t x, size_t h, size_t w) {
    if (y >= m_height || x >= m_width || y + h > m_height || x + w > m_width) {
        throw(std::runtime_error("Requested submatrix out of bounds!"));
        //return matrix<ItemType> (0, 0);
    }
    matrix<ItemType> result (h, w);
    for (size_t i = 0; i < h; i++) {
        for (size_t j = 0; j < w; j++) {
            result.m_data[i][j] = m_data[i + y][j + x];
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
    matrix<ItemType> id (height, width, ItemType (0));
    for (size_t i = 0; i < width && i < height; i++) {
        id.m_data[i][i] = 1;
    }
    return id;
}

/**
* @brief Returns a const interator at the frst element of the matrix
*/
template <class ItemType>
matrix<ItemType>::iterator<const ItemType> matrix<ItemType>::cbegin() const {
    return iterator<const ItemType>(0, 0, m_width * m_height - 1, m_width, &m_data);
}

/**
* @brief Returns a const interator at the last element of the matrix
*/
template <class ItemType>
matrix<ItemType>::iterator<const ItemType> matrix<ItemType>::cend() const {
    return iterator<const ItemType>(m_width * m_height - 1, 0, m_width * m_height, m_width, &m_data);
}

/**
* @brief Returns an interator at the frst element of the matrix
*/
template <class ItemType>
matrix<ItemType>::iterator<ItemType> matrix<ItemType>::begin() {
    return iterator<ItemType>(0, 0, m_width * m_height - 1, m_width, &m_data);
}

/**
* @brief Returns an interator at the last element of the matrix
*/
template <class ItemType>
matrix<ItemType>::iterator<ItemType> matrix<ItemType>::end() {
    return iterator<ItemType>(m_width * m_height - 1, 0, m_width * m_height, m_width, &m_data);
}

#ifndef DONT_USE_GPU
/**
* @brief Sets whether or not data should be left on the GPU to increase performance
*/
template <class ItemType>
bool matrix<ItemType>::leaveDataOnGPU(bool val) {
    //shoudl this straight up clear the matrix?
    if (val) {
        m_gpuUpToDate = false;
        m_gpuSlicesUpToDate.clear();
        m_gpuSlicesUpToDate.resize(m_height, false);
        m_upToDate &= !(dataFlag::GPU_DATA | dataFlag::GPU_HEIGHT | dataFlag::GPU_WIDTH);
    }
    return m_leaveOnGPU = val;
}

/**
* @brief Returns whether or not data is being left on the GPU
*/
template <class ItemType>
bool matrix<ItemType>::leaveDataOnGPU() const {
    return m_leaveOnGPU;
}

/**
* @brief Manually pulls data from the GPU
*/
template <class ItemType>
bool matrix<ItemType>::pullData() {
    if (!m_useGPU) {
        return false;
    }
    cl_int ret = pullFromGPU(m_command_queue);
    return ret == CL_SUCCESS;
}

/**
* @brief Manually pushes data to the GPU
*/
template <class ItemType>
bool matrix<ItemType>::pushData() {
    if (!m_useGPU) {
        return false;
    }
    cl_int ret = pushToGPU(m_command_queue);
    return ret == CL_SUCCESS;
}
#endif

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
* @param M The matrix to be added to the calling matrix.
*
* @return returns a matrix of the type of the calling matrix with the dimensions of the overlap between the matrices.
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::add (matrix<ArgType>& M) {
    //should I be concatenating the matrix ike this? probably not, a submatrix
    //can be made easily enough if they want concatenation...
    //yeah, i'll change this later
#ifndef DONT_USE_GPU
    matrix<ItemType> result(0, 0);
    if (m_leaveOnGPU && m_useGPU) {
        result.m_leaveOnGPU = m_leaveOnGPU;
        result.m_height = m_height < M.m_height ? m_height : M.m_height;
        result.m_width = m_width < M.m_width ? m_width : M.m_width;
        result.m_gpuSlicesUpToDate.resize(result.m_height, false);
    } else {
        result.resize (m_height < M.m_height ? m_height : M.m_height, m_width < M.m_width ? m_width : M.m_width);
    }
    if (! (m_useGPU || ALL_USE_GPU)) { //if don't use the gpu
        if (M.m_leaveOnGPU && M.m_useGPU) {
            M.pullFromGPU(m_command_queue);
        }
#else
        matrix<ItemType> result (m_height < M.m_height ? m_height : M.m_height, m_width < M.m_width ? m_width : M.m_width);
#endif
        for (size_t i = 0; i < result.m_height; i++) {
            for (size_t j = 0; j < result.m_width; j++) {
                result.m_data[i][j] = m_data[i][j] + M.m_data[i][j];
            }
        }
        return result;
#ifndef DONT_USE_GPU
    } else {
        if (!GPU_INITIALIZED) {
            throw(LinAlgo::gpu_exception("GPU is not initialized", __FILE__, __LINE__, -99));
            return matrix<ItemType> (0, 0);
        }
        if (!m_leaveOnGPU) {
            pushToGPU (m_command_queue);
        }
        if (!M.m_leaveOnGPU){
            M.pushToGPU (m_command_queue);
        }
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
                throw(LinAlgo::gpu_exception("Can't GPU compute, unsupported item type", __FILE__, __LINE__, -99));
            }
        } else {
            throw(LinAlgo::gpu_exception("Can't GPU compute, matrices are of differing types", __FILE__, __LINE__, -99));
        }
        if (!result.m_leaveOnGPU) {
            result.pullFromGPU (m_command_queue);
        } else {
            result.m_gpuUpToDate = true;
            result.m_upToDate |= (dataFlag::GPU_DATA | dataFlag::GPU_HEIGHT | dataFlag::GPU_WIDTH);
        }
        result.useGPU(true);
        return result;
    }
#endif
}

/**
* @brief Adds a single value to all elements in matrix.
*
* @param val The value to be added to the calling matrix.
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::add (const ArgType& val) {
#ifndef DONT_USE_GPU
    matrix<ItemType> result(0, 0);
    if (m_leaveOnGPU) {
        result.m_height = m_height;
        result.m_width = m_width;
        result.m_gpuSlicesUpToDate.resize(m_height, false);
        result.m_leaveOnGPU = true;
    } else {
        result.resize (m_height, m_width);
    }
    if (! (m_useGPU || ALL_USE_GPU)) { //if don't use the gpu
#else
    matrix<ItemType> result (m_height, m_width);
#endif
        for (size_t i = 0; i < result.m_height; i++) {
            for (size_t j = 0; j < result.m_width; j++) {
                result.m_data[i][j] = m_data[i][j] + val;
            }
        }
        return result;
#ifndef DONT_USE_GPU
    } else {
        if (!GPU_INITIALIZED) {
            throw(LinAlgo::gpu_exception("GPU is not initialized", __FILE__, __LINE__, -99));
            return matrix<ItemType> (0, 0);
        }
        if (!m_leaveOnGPU) {
            pushToGPU (m_command_queue);
        }
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
            throw(LinAlgo::gpu_exception("Can't GPU compute, unsupported item type", __FILE__, __LINE__, -99));
        }
        if (!result.m_leaveOnGPU) {
            result.pullFromGPU (m_command_queue);
        } else {
            result.m_gpuUpToDate = true;
            result.m_upToDate |= (dataFlag::GPU_DATA | dataFlag::GPU_HEIGHT | dataFlag::GPU_WIDTH);
        }
        result.useGPU(true);
        return result;
    }
#endif
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
* @param M The matrix to be subtracted from the calling matrix.
*
* @return returns a matrix of the type of the calling matrix with the dimensions of the overlap between the matrices.
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::subtract (matrix<ArgType>& M) {
    //should I be concatenating the matrix ike this? probably not, a submatrix
    //can be made easily enough if they want concatenation...
    //yeah, i'll change this later
#ifndef DONT_USE_GPU
    matrix<ItemType> result(0, 0);
    if (m_leaveOnGPU) {
        result.m_height = m_height < M.m_height ? m_height : M.m_height;
        result.m_width = m_width < M.m_width ? m_width : M.m_width;
        result.m_gpuSlicesUpToDate.resize(result.m_height, false);
        result.m_leaveOnGPU = true;
    } else {
        result.resize (m_height < M.m_height ? m_height : M.m_height, m_width < M.m_width ? m_width : M.m_width);
    }
    if (! (m_useGPU || ALL_USE_GPU)) { //if don't use the gpu
        if (M.m_leaveOnGPU && M.m_useGPU) {
            M.pullFromGPU(m_command_queue);
        }
#else
    matrix<ItemType> result (m_height < M.m_height ? m_height : M.m_height, m_width < M.m_width ? m_width : M.m_width);
#endif
        for (size_t i = 0; i < result.m_height; i++) {
            for (size_t j = 0; j < result.m_width; j++) {
                result.m_data[i][j] = m_data[i][j] - M.m_data[i][j];
            }
        }
        return result;
#ifndef DONT_USE_GPU
    } else {
        if (!GPU_INITIALIZED) {
            throw(LinAlgo::gpu_exception("GPU is not initialized", __FILE__, __LINE__, -99));
            return matrix<ItemType> (0, 0);
        }
        if (!m_leaveOnGPU) {
            pushToGPU (m_command_queue);
        }
        if (!M.m_leaveOnGPU) {
            M.pushToGPU (m_command_queue); //if this matrix isn't supposed to use gpu, should I unpush its data after this?
        }
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
                throw(LinAlgo::gpu_exception("Can't GPU compute, unsupported item type", __FILE__, __LINE__, -99));
            }
        } else {
            throw(LinAlgo::gpu_exception("Can't GPU compute, matrices are of differing types", __FILE__, __LINE__, -99));
        }
        if (!result.m_leaveOnGPU) {
            result.pullFromGPU (m_command_queue);
        } else {
            result.m_gpuUpToDate = true;
            result.m_upToDate |= (dataFlag::GPU_DATA | dataFlag::GPU_HEIGHT | dataFlag::GPU_WIDTH);
        }
        result.useGPU(true);
        return result;
    }
#endif
}

/**
* @brief Subtracts a single value from all elements in matrix.
*
* @param val The value to be subtracted from the calling matrix.
*
* @return returns a matrix of the same type as the calling matrix.
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::subtract (const ArgType& val) {
#ifndef DONT_USE_GPU
    matrix<ItemType> result(0, 0);
    if (m_leaveOnGPU) {
        result.m_height = m_height;
        result.m_width = m_width;
        result.m_gpuSlicesUpToDate.resize(m_height, false);
        result.m_leaveOnGPU = true;
    } else {
        result.resize(m_height, m_width);
    }
    if (! (m_useGPU || ALL_USE_GPU)) { //if don't use the gpu
#else
    matrix<ItemType> result (m_height, m_width);
#endif
        for (size_t i = 0; i < result.m_height; i++) {
            for (size_t j = 0; j < result.m_width; j++) {
                result.m_data[i][j] = m_data[i][j] - val;
            }
        }
        return result;
#ifndef DONT_USE_GPU
    } else {
        if (!GPU_INITIALIZED) {
            throw(LinAlgo::gpu_exception("GPU is not initialized", __FILE__, __LINE__, -99));
            return matrix<ItemType> (0, 0);
        }
        if (!m_leaveOnGPU) {
            pushToGPU (m_command_queue);
        }
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
            throw(LinAlgo::gpu_exception("Can't GPU compute, unsupported item type", __FILE__, __LINE__, -99));
        }
        if (!result.m_leaveOnGPU) {
            result.pullFromGPU (m_command_queue);
        } else {
            result.m_gpuUpToDate = true;
            result.m_upToDate |= dataFlag::GPU_DATA;
        }
        result.useGPU(true);
        return result;
    }
#endif
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
* @param M The rhs matrix for the multiplication.
*
* @return Returns the result of multiplying the two matrices together, of the same type as the calling matrix. Returned matrix is 0x0
* if the matrices were of incompatible dimensions.
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::multiply (matrix<ArgType>& M) {
    if (m_width != M.m_height) {
        throw(std::runtime_error(std::string("The dimensions ") + std::to_string(m_height) + std::string("x") + std::to_string(m_width) + std::string(" and ") +
                                 std::to_string(M.m_height) + std::string("x") + std::to_string(M.m_width) + std::string(" are mismatched.")));
        return matrix<ItemType> (0, 0); //null matrix
    }
#ifndef DONT_USE_GPU
    matrix<ItemType> result;
    if (m_leaveOnGPU) {
        result.m_height = m_height;
        result.m_width = M.m_width;
        result.m_gpuSlicesUpToDate.resize(m_height, false);
        result.m_leaveOnGPU = true;
    } else {
        result.resize(m_height, M.m_width);
    }
    if (! (m_useGPU || ALL_USE_GPU)) { //if don't use the gpu
        if (M.m_leaveOnGPU && M.m_useGPU) {
            M.pullFromGPU(m_command_queue);
        }
#else
    matrix<ItemType> result (m_height, M.m_width);
#endif
        for (size_t i = 0; i < result.m_height; i++) {
            for (size_t j = 0; j < result.m_width; j++) {
                ItemType sum = ItemType (0);
                for (size_t k = 0; k < m_width; k++) {
                    sum += m_data[i][k] * M.m_data[k][j];
                }
                result.m_data[i][j] = sum;
            }
        }
        return result;
#ifndef DONT_USE_GPU
    } else {
        if (!GPU_INITIALIZED) {
            throw(LinAlgo::gpu_exception("GPU is not initialized", __FILE__, __LINE__, -99));
            return matrix<ItemType> (0, 0);
        }
        if (!m_leaveOnGPU) {
            pushToGPU (m_command_queue);
        }
        if (!M.m_leaveOnGPU) {
            M.pushToGPU (m_command_queue);
        }
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
                throw(LinAlgo::gpu_exception("Can't GPU compute, unsupported item type", __FILE__, __LINE__, -99));
            }
        } else {
            throw(LinAlgo::gpu_exception("Can't GPU compute, matrices are of differing types", __FILE__, __LINE__, -99));
        }
        if (!result.m_leaveOnGPU) {
            result.pullFromGPU (m_command_queue);
        } else {
            result.m_gpuUpToDate = true;
            result.m_upToDate |= (dataFlag::GPU_DATA | dataFlag::GPU_HEIGHT | dataFlag::GPU_WIDTH);
        }
        result.useGPU(true);
        return result;
    }
#endif
}

/**
* @brief Multiplies each element by a single value.
*
* @param val The value to be multiplied through the calling matrix.
*
* @return returns a matrix of the same type as the calling matrix.
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::multiply (const ArgType& val) {
#ifndef DONT_USE_GPU
    matrix<ItemType> result(0, 0);
    if (m_leaveOnGPU) {
        result.m_height = m_height;
        result.m_width = m_width;
        result.m_gpuSlicesUpToDate.resize(m_height, false);
        result.m_leaveOnGPU = true;
    } else {
        result.resize(m_height, m_width);
    }
    if (! (m_useGPU || ALL_USE_GPU)) { //if don't use the gpu
#else
    matrix<ItemType> result (m_height, m_width);
#endif
        for (size_t i = 0; i < result.m_height; i++) {
            for (size_t j = 0; j < result.m_width; j++) {
                result.m_data[i][j] = m_data[i][j] * val;
            }
        }
        return result;
#ifndef DONT_USE_GPU
    } else {
        if (!GPU_INITIALIZED) {
            throw(LinAlgo::gpu_exception("GPU is not initialized", __FILE__, __LINE__, -99));
            return matrix<ItemType> (0, 0);
        }
        if (!m_leaveOnGPU) {
            pushToGPU (m_command_queue);
        }
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
            throw(LinAlgo::gpu_exception("Can't GPU compute, unsupported item type", __FILE__, __LINE__, -99));
        }
        if (!result.m_leaveOnGPU) {
            result.pullFromGPU (m_command_queue);
        } else {
            result.m_gpuUpToDate = true;
            result.m_upToDate |= (dataFlag::GPU_DATA | dataFlag::GPU_HEIGHT | dataFlag::GPU_WIDTH);
        }
        result.useGPU(true);
        return result;
    }
#endif
}

/**
* @brief Multiplies each element of the calling matrix by the corresponding element of the rhs matrix.
*
* @param M The rhs matrix for the multiplications.
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::elementMultiply (matrix<ArgType>& M) {
    //should I be concatenating the matrix ike this? probably not, a submatrix
    //can be made easily enough if they want concatenation...
    //yeah, i'll change this later
#ifndef DONT_USE_GPU
    matrix<ItemType> result(0, 0);
    if (m_leaveOnGPU) {
        result.m_height = m_height < M.m_height ? m_height : M.m_height;
        result.m_width = m_width < M.m_width ? m_width : M.m_width;
        result.m_gpuSlicesUpToDate.resize(result.m_height, false);
        result.m_leaveOnGPU = true;
    } else {
        result.resize (m_height < M.m_height ? m_height : M.m_height, m_width < M.m_width ? m_width : M.m_width);
    }
    if (! (m_useGPU || ALL_USE_GPU)) { //if don't use the gpu
        if (M.m_leaveOnGPU && M.m_useGPU) {
            M.pullFromGPU(m_command_queue);
        }
#else
    matrix<ItemType> result (m_height < M.m_height ? m_height : M.m_height, m_width < M.m_width ? m_width : M.m_width);
#endif
        for (size_t i = 0; i < result.m_height; i++) {
            for (size_t j = 0; j < result.m_width; j++) {
                result.m_data[i][j] = m_data[i][j] * M.m_data[i][j];
            }
        }
        return result;
#ifndef DONT_USE_GPU
    } else {
        if (!GPU_INITIALIZED) {
            throw(LinAlgo::gpu_exception("GPU is not initialized", __FILE__, __LINE__, -99));
            return matrix<ItemType> (0, 0);
        }
        if (!m_leaveOnGPU) {
            pushToGPU (m_command_queue);
        }
        if (!M.m_leaveOnGPU) {
            M.pushToGPU (m_command_queue);
        }
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
                throw(LinAlgo::gpu_exception("Can't GPU compute, unsupported item type", __FILE__, __LINE__, -99));
            }
        } else {
            throw(LinAlgo::gpu_exception("Can't GPU compute, matrices are of differing types", __FILE__, __LINE__, -99));
        }
        if (!result.m_leaveOnGPU) {
            result.pullFromGPU (m_command_queue);
        } else {
            result.m_gpuUpToDate = true;
            result.m_upToDate |= (dataFlag::GPU_DATA | dataFlag::GPU_HEIGHT | dataFlag::GPU_WIDTH);
        }
        result.useGPU(true);
        return result;
    }
#endif
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
        throw(std::runtime_error(std::string("The dimensions ") + std::to_string(m_height) + std::string("x") + std::to_string(m_width) + std::string(" and ") +
                                 std::to_string(M.m_height) + std::string("x") + std::to_string(M.m_width) + std::string(" are mismatched.")));
        return matrix<ItemType>(0, 0);
    }
    matrix<ArgType> invM = LinAlgo::inverse(M);
    if (invM == matrix<ArgType>(0, 0)) {
        throw(std::runtime_error("Cannot divide by singular matrix"));
        return matrix<ItemType>(0, 0);
    }
    return multiply(invM);
}

/**
* @brief Divides each element by a single value.
*
* @param val The value to divide the calling matrix by.
*
* @return returns a matrix of the same type as the calling matrix.
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::divide (const ArgType& val) {
#ifndef DONT_USE_GPU
    matrix<ItemType> result(0, 0);
    if (m_leaveOnGPU) {
        result.m_height = m_height;
        result.m_width = m_width;
        result.m_gpuSlicesUpToDate.resize(m_height, false);
        result.m_leaveOnGPU = true;
    } else {
        result.resize(m_height, m_width);
    }
    if (! (m_useGPU || ALL_USE_GPU)) { //if don't use the gpu
#else
    matrix<ItemType> result (m_height, m_width);
#endif
        for (size_t i = 0; i < result.m_height; i++) {
            for (size_t j = 0; j < result.m_width; j++) {
                result.m_data[i][j] = m_data[i][j] / val;
            }
        }
        return result;
#ifndef DONT_USE_GPU
    } else {
        if (!GPU_INITIALIZED) {
            throw(LinAlgo::gpu_exception("GPU is not initialized", __FILE__, __LINE__, -99));
            return matrix<ItemType> (0, 0);
        }
        if (!m_leaveOnGPU) {
            pushToGPU (m_command_queue);
        }
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
            throw(LinAlgo::gpu_exception("Can't GPU compute, unsupported item type", __FILE__, __LINE__, -99));
        }
        if (!result.m_leaveOnGPU) {
            result.pullFromGPU (m_command_queue);
        } else {
            result.m_gpuUpToDate = true;
            result.m_upToDate |= (dataFlag::GPU_DATA | dataFlag::GPU_HEIGHT | dataFlag::GPU_WIDTH);
        }
        result.useGPU(true);
        return result;
    }
#endif
}

/**
* @brief Divides each element of the calling matrix by each element of the rhs matrix.
*
* @param M The rhs matrix for the divisions.
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType> matrix<ItemType>::elementDivide (matrix<ArgType>& M) {
    //should I be concatenating the matrix ike this? probably not, a submatrix
    //can be made easily enough if they want concatenation...
    //yeah, i'll change this later
#ifndef DONT_USE_GPU
    matrix<ItemType> result(0, 0);
    if (m_leaveOnGPU) {
        result.m_height = m_height < M.m_height ? m_height : M.m_height;
        result.m_width = m_width < M.m_width ? m_width : M.m_width;
        result.m_gpuSlicesUpToDate.resize(result.m_height, false);
        result.m_leaveOnGPU = true;
    } else {
        result.resize (m_height < M.m_height ? m_height : M.m_height, m_width < M.m_width ? m_width : M.m_width);
    }
    if (! (m_useGPU || ALL_USE_GPU)) { //if don't use the gpu
        if (M.m_leaveOnGPU && M.m_useGPU) {
            M.pullFromGPU(m_command_queue);
        }
#else
    matrix<ItemType> result (m_height < M.m_height ? m_height : M.m_height, m_width < M.m_width ? m_width : M.m_width);
#endif
        for (size_t i = 0; i < result.m_height; i++) {
            for (size_t j = 0; j < result.m_width; j++) {
                result.m_data[i][j] = m_data[i][j] / M.m_data[i][j];
            }
        }
        return result;
#ifndef DONT_USE_GPU
    } else {
        if (!GPU_INITIALIZED) {
            throw(LinAlgo::gpu_exception("GPU is not initialized", __FILE__, __LINE__, -99));
            return matrix<ItemType> (0, 0);
        }
        if (!m_leaveOnGPU) {
            pushToGPU (m_command_queue);
        }
        if (!M.m_leaveOnGPU) {
            M.pushToGPU (m_command_queue);
        }
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
                throw(LinAlgo::gpu_exception("Can't GPU compute, unsupported item type", __FILE__, __LINE__, -99));
            }
        } else {
            throw(LinAlgo::gpu_exception("Can't GPU compute, matrices are of differing types", __FILE__, __LINE__, -99));
        }
        if (!result.m_leaveOnGPU) {
            result.pullFromGPU (m_command_queue);
        } else {
            result.m_gpuUpToDate = true;
            result.m_upToDate |= (dataFlag::GPU_DATA | dataFlag::GPU_HEIGHT | dataFlag::GPU_WIDTH);
        }
        result.useGPU(true);
        return result;
    }
#endif
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
    matrix<ItemType> result (m_width, m_height);
    for (size_t i = 0; i < m_width; i++) {
        for (size_t j = 0; j < m_height; j++) {
            result.m_data[i][j] = m_data[j][i];
        }
    }
    //I could try to do an in-place transpose on the gpu
    //then pull it... hm. I'll have to look into that later
    m_data = std::move(result.m_data);
    m_height = m_width;
    m_width = result.m_width;
#ifndef DONT_USE_GPU
    m_gpuUpToDate = false;
    m_upToDate &= ! (dataFlag::GPU_HEIGHT | dataFlag::GPU_WIDTH | dataFlag::GPU_DATA);
    m_gpuSlicesUpToDate.clear();
    m_gpuSlicesUpToDate.resize (m_height, false);
#endif
    return *this;
}
//}
// </editor-fold>
#pragma endregion

/**
* @brief Maps a function to every value in the matrix
*
* @param func The function to map to every value of the matrix
*
* @param asynchronous If true, executes func using a thread pool. Defaults to false
*
* @param thread_count The number of threads to use if threading
*
* @note The thread count defaults to double the hardware concurrency. This will
* overwrite the data in the matrix
*/
template <class ItemType>
matrix<ItemType>& matrix<ItemType>::map(ItemType (*func)(ItemType&), bool asynchronous, size_t thread_count) {//maybe change async to int so user can  specify thread count for pool... or just add a default param
    if (!asynchronous) {
        for (size_t i = 0; i < m_height; i++) {
            for (size_t j = 0; j < m_width; j++) {
                m_data[i][j] = func(m_data[i][j]);
            }
        }
    } else {
        auto thread_func = [&](ItemType* item){
            *item = func(*item);
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
        int num_vals = m_height * m_width;
        if (num_vals < default_pool_size) {
            default_pool_size = num_vals;
            pool.reserve(default_pool_size);
        } else {
            pool.reserve(default_pool_size);
        }
        int start_index = 0;
        for (auto iter = this->begin(); start_index < default_pool_size; start_index++) {
            std::thread th (thread_func, &iter[start_index]);
            pool.push_back (std::move (th));
        }
        if (num_vals > default_pool_size) {
            bool first = true;
            for (size_t i = start_index / m_width; i < m_height; i++) {
                for (size_t j = first ? start_index % m_width : 0; j < m_width; j++) {
                    bool waiting = true;
                    while (waiting) {
                        for (auto t1 = pool.begin(); t1 < pool.end(); t1++) {
                            if (t1->joinable()) {
                                std::thread t2 (thread_func, &m_data[i][j]);
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
    return *this;
}

#pragma region Operator Overloads
// <editor-fold desc="Operator Overloads">
//{
/**
* @brief Using the [] operator will slice a row of the matrix.
*
* @param y The row to be sliced.
*
* @return A const reference to the std::vector storing the appropriate row of the matrix.
*
* @note If m_leaveOnGPU is set, don't forget to retrieve data first and/or push data after
*/
template <class ItemType>
const std::vector<ItemType>& matrix<ItemType>::operator[] (size_t y) const {
    return (const std::vector<ItemType>&) m_data[y];
}

/**
* @brief Using the [] operator will slice a row of the matrix.
*
* @param y The row to be sliced.
*
* @return A reference to the std::vector storing the appropriate row of the matrix.
*
* @note If m_leaveOnGPU is set, don't forget to retrieve data first and/or push data after
*/
template <class ItemType>
std::vector<ItemType>& matrix<ItemType>::operator[] (size_t y) {
#ifndef DONT_USE_GPU
    m_gpuSlicesUpToDate[y] = false;
    m_gpuUpToDate = false;
    m_upToDate &= !(dataFlag::GPU_DATA);
#endif // DONT_USE_GPU
    return (std::vector<ItemType>&) m_data[y];
}

/**
* @brief Assignment operator whoot
*
* @param M The matrix to be copied.
*
* @return A reference to the lhs matrix.
*/
template <class ItemType>
matrix<ItemType>& matrix<ItemType>::operator= (const matrix<ItemType>& M) {
    if (m_height != M.m_height) {
        m_data.resize (M.m_height);
        m_height = M.m_height;
    }
    if (m_width != M.m_width) {
        for (size_t i = 0; i < m_height; i++) {
            m_data[i].resize (M.m_width);
        }
        m_width = M.m_width;
    }
#ifndef DONT_USE_GPU
    if (GPU_INITIALIZED && (M.m_useGPU || ALL_USE_GPU) && (M.m_upToDate & dataFlag::GPU_DATA)) {
        if (!M.m_leaveOnGPU) {
            //while (m_command_queue != NULL);
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
            //clReleaseCommandQueue (m_command_queue);
            //m_command_queue = NULL;
            m_gpuData = NULL;
            m_gpuHeight = NULL;
            m_gpuWidth = NULL;
        } else {
            //To Do:
            //copy kernel
        }
    } else {
#endif
        for (size_t i = 0; i < m_height; i++) {
            for (size_t j = 0; j < m_width; j++) {
                m_data[i][j] = M.m_data[i][j];
            }
        }
#ifndef DONT_USE_GPU
    }

    m_gpuUpToDate = false;
    m_gpuSlicesUpToDate.resize (m_height, false);
    m_upToDate = 0;//if eigenvalues and stuff get copied over, this will be different
    m_useGPU = M.m_useGPU;
    if (m_command_queue) {
        //clFinish (m_command_queue);
        //m_command_queue = NULL;
    }
    if (m_gpuData) {
        clReleaseMemObject (m_gpuData);
        m_gpuData = NULL;
    }
    m_dataInitialized = M.m_dataInitialized;
    if (m_useGPU) {
        initQueue();
    }
    m_leaveOnGPU = M.m_leaveOnGPU;
    m_dataInitialized = M.m_dataInitialized;
#endif

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
    m_data = std::move(M.m_data);
#ifndef DONT_USE_GPU
    if (m_gpuData) {
        clReleaseMemObject(m_gpuData);
    }
    m_gpuData = M.m_gpuData;
    M.m_gpuData = NULL;
    //m_command_queue = M.m_command_queue;
    //M.m_command_queue = NULL;
    if (m_gpuHeight) {
        clReleaseMemObject(m_gpuHeight);
    }
    m_gpuHeight = M.m_gpuHeight;
    M.m_gpuHeight = NULL;
    if (m_gpuWidth) {
        clReleaseMemObject(m_gpuWidth);
    }
    m_gpuWidth = M.m_gpuWidth;
    M.m_gpuWidth = NULL;

    m_upToDate = M.m_upToDate;
    m_useGPU = M.m_useGPU;
    m_leaveOnGPU = M.m_leaveOnGPU;
    m_gpuUpToDate = M.m_gpuUpToDate;
    m_gpuSlicesUpToDate = std::move(M.m_gpuSlicesUpToDate);
    m_dataInitialized = M.m_dataInitialized;
#endif
    return *this;
}

/**
* @brief Assignment operator whoot
*
* @param M The matrix to be copied.
*
* @return A reference to the lhs matrix.
*/
template <class ItemType>
template <class ArgType>
matrix<ItemType>& matrix<ItemType>::operator= (const matrix<ArgType>& M) {
    if (m_height != M.getHeight()) {
        m_data.resize (M.getHeight());
        m_height = M.getHeight();
    }
    if (m_width != M.getWidth()) {
        for (size_t i = 0; i < m_height; i++) {
            m_data[i].resize (M.getWidth());
        }
        m_width = M.getWidth();
    }
#ifndef DONT_USE_GPU
    if (M.leaveDataOnGPU()) {
        m_leaveOnGPU = true;
    }
#endif
    for (size_t i = 0; i < m_height; i++) {
        for (size_t j = 0; j < m_width; j++) {
            m_data[i][j] = ItemType (M[i][j]);
        }
    }

#ifndef DONT_USE_GPU
    m_gpuUpToDate = false;
    m_gpuSlicesUpToDate.resize (m_height, false);
    m_upToDate = 0;//if eigenvalues and stuff get copied over, this will be different
    m_useGPU = M.useGPU();
    //if (m_command_queue) {
    //    clFinish (m_command_queue);
    //    m_command_queue = NULL;
    //}
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
    if (m_width != 0 && m_height != 0) {
        m_dataInitialized = true;
    } else {
        m_dataInitialized = false;
    }
    if (m_useGPU) {
        initQueue();
    }
#endif

    //clear everything else that's now irrelevent (or copy things that are over lol)
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
            if (m_data[i][j] != M.m_data[i][j]) {
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
            if (m_data[i][j] == M.m_data[i][j]) {
                return false;
            }
        }
    }
    return true;
}
//}
// </editor-fold>
#pragma endregion

#ifndef DONT_USE_GPU
#pragma region Auxilliary Functions for Handling GPU Data
// <editor-fold desc="Auxilliary Functions for Handling GPU Data">
//{
#pragma region Initialization Functions
// <editor-fold desc="Initialization Functions">
//{
//private auxilliary function to initialize the OpenCL command queue

/**
* @brief This helper function initializes the OpenCL command queue for this matrix
*
* @note This function uses the OpenCL version foun during InitGPU() to determine
* the appropriate command to create the queue
*/
template <class ItemType>
cl_int matrix<ItemType>::initQueue() {
    if (m_command_queue != NULL) {
        return CL_SUCCESS;
    }
    //cl_int ret;
    //if (OPENCL_VERSION >= 2.0) {
    //    m_command_queue = clCreateCommandQueueWithProperties (m_context, m_device_id, 0, &ret);
    //} else {
    //    m_command_queue = clCreateCommandQueue (m_context, m_device_id, 0, &ret);
    //}
    //if (ret != CL_SUCCESS) {
    //    throw(LinAlgo::gpu_exception(std::string("Unable to create command queue, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    //}
    //ret = clRetainCommandQueue(m_command_queue);//idk abt thi, but whatevs
    //return ret;
    return CL_SUCCESS;
}

/**
* @brief Private auxialliary function for creating an empty memory buffer to store result
*
* @param command_queue A command queue to use for writing to the GPU
*/
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
    m_gpuData = clCreateBuffer (m_context, CL_MEM_READ_WRITE, m_height * m_width * sizeof (ItemType), NULL, &ret);
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to create memory buffer, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }

    m_gpuHeight = clCreateBuffer (m_context, CL_MEM_READ_ONLY, sizeof (ItemType), NULL, &ret);
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to create memory buffer, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }

    m_gpuWidth = clCreateBuffer (m_context, CL_MEM_READ_ONLY, sizeof (ItemType), NULL, &ret);
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to create memory buffer, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }

    ret = clEnqueueWriteBuffer (command_queue, m_gpuHeight, CL_FALSE, 0, sizeof (ItemType), &m_height, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to push data to GPU, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }

    ret = clEnqueueWriteBuffer (command_queue, m_gpuWidth, CL_TRUE, 0, sizeof (ItemType), &m_width, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to push data to GPU, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }

    m_upToDate |= (dataFlag::GPU_HEIGHT | dataFlag::GPU_WIDTH);
    return ret;
}

/**
* @brief private auxilliary function to push the matrix data to the gpu
*
* @param command_queue The command queue to use for writing to the GPU
*/
template <class ItemType>
cl_int matrix<ItemType>::pushToGPU (cl_command_queue& command_queue) {
    if (m_gpuUpToDate && !m_leaveOnGPU) {
        return CL_SUCCESS;
    }

    //throw an error if someone tries to push when there's literally no data
    if (!m_dataInitialized) {
        throw(LinAlgo::gpu_exception(std::string("Matrix data not initialized for push"), __FILE__, __LINE__, -99));
    }

    //create storage buffers if they for some reason don't exist
    cl_int ret;
    if (!m_gpuData || m_leaveOnGPU) {
        if (m_leaveOnGPU) {
            clReleaseMemObject(m_gpuData);
        }
        m_gpuData = clCreateBuffer (m_context, CL_MEM_READ_WRITE, m_height * m_width * sizeof (ItemType), NULL, &ret);
        if (ret != CL_SUCCESS) {
            throw(LinAlgo::gpu_exception(std::string("Unable to create memory buffer, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
        }
    }

    if (!m_gpuHeight || m_leaveOnGPU) {
        if (m_leaveOnGPU) {
            clReleaseMemObject(m_gpuHeight);
        }
        m_gpuHeight = clCreateBuffer (m_context, CL_MEM_READ_ONLY, sizeof (ItemType), NULL, &ret);
        if (ret != CL_SUCCESS) {
            throw(LinAlgo::gpu_exception(std::string("Unable to create memory buffer, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
        }
    }

    if (!m_gpuWidth || m_leaveOnGPU) {
        if (m_leaveOnGPU) {
            clReleaseMemObject(m_gpuWidth);
        }
        m_gpuWidth = clCreateBuffer (m_context, CL_MEM_READ_ONLY, sizeof (ItemType), NULL, &ret);
        if (ret != CL_SUCCESS) {
            throw(LinAlgo::gpu_exception(std::string("Unable to create memory buffer, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
        }
    }

    //update matrix data
    if (! (m_upToDate & dataFlag::GPU_DATA) || m_leaveOnGPU) {
        for (size_t i = 0; i < m_height; i++) {
            if (!m_gpuSlicesUpToDate[i] || m_leaveOnGPU) {
                ret = clEnqueueWriteBuffer (command_queue, m_gpuData, CL_FALSE, i * m_width * sizeof (ItemType), m_width * sizeof (ItemType), m_data[i].data(), 0, NULL, NULL);
                if (ret != CL_SUCCESS) {
                    throw(LinAlgo::gpu_exception(std::string("Unable to push data to GPU, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
                }
                m_gpuSlicesUpToDate[i] = true;
            }
        }
        m_upToDate |= dataFlag::GPU_DATA;
    }

    //update height
    if (! (m_upToDate & dataFlag::GPU_HEIGHT) || m_leaveOnGPU) {
        ret = clEnqueueWriteBuffer (command_queue, m_gpuHeight, CL_FALSE, 0, sizeof (ItemType), &m_height, 0, NULL, NULL);
        if (ret != CL_SUCCESS) {
            throw(LinAlgo::gpu_exception(std::string("Unable to push data to GPU, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
        }
        m_upToDate |= dataFlag::GPU_HEIGHT;
    }

    if (! (m_upToDate & dataFlag::GPU_WIDTH) || m_leaveOnGPU) {
        ret = clEnqueueWriteBuffer (command_queue, m_gpuWidth, CL_FALSE, 0, sizeof (ItemType), &m_width, 0, NULL, NULL);
        if (ret != CL_SUCCESS) {
            throw(LinAlgo::gpu_exception(std::string("Unable to push data to GPU, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
        }
        m_upToDate |= dataFlag::GPU_WIDTH;
    }

    m_gpuUpToDate = true;
    clFinish(command_queue);//make sure everything is written to gpu... if anything
    return ret;
}

/**
* @brief private auxilliary function for retrieving data from the gpu
*
* @param command_queue The command queue to use for reading the buffers
*/
template <class ItemType>
cl_int matrix<ItemType>::pullFromGPU (cl_command_queue& command_queue) {
    cl_int ret;
    if (!m_dataInitialized) {
        //created empty to increase speed when leaving data on gpu
        m_data.resize(m_height);
        for (size_t i = 0; i < m_height; i++) {
            m_data[i].resize(m_width);
        }
        if (m_gpuSlicesUpToDate.size() != m_height) {
            m_gpuSlicesUpToDate.resize(m_height, false);
        }
        m_dataInitialized = true;
    }

    for (int i = 0; i < m_height; i++) {
        ret = clEnqueueReadBuffer (command_queue, m_gpuData, CL_FALSE, i * m_width * sizeof (ItemType), m_width * sizeof (ItemType), m_data[i].data(), 0, NULL, NULL);
        if (ret != CL_SUCCESS) {
            throw(LinAlgo::gpu_exception(std::string("Unable to retrieve data from GPU, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
        }
        m_gpuSlicesUpToDate[i] = true;
    }
    m_gpuUpToDate = true;
    m_upToDate |= dataFlag::GPU_DATA;

    clFinish(command_queue);//make sure everything gets read
    return ret;
}
//}
// </editor-fold>
#pragma endregion

#pragma region Functions for Executing Kernels
// <editor-fold desc="Functions for Executing Kernels">
//{
/**
* @brief Private function for executing the add kernel
*
* @note Can be used for any kernel acting on two matrices w/ the same number of elements
*/
template <class ItemType>
cl_int matrix<ItemType>::execute_add_kernel (cl_kernel kernel, matrix<ItemType>& rhs, matrix<ItemType>& result) {
    cl_int ret;
    //set kernel arguments
    ret = clSetKernelArg (kernel, 0, sizeof (cl_mem), (void*)& m_gpuData);
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to set kernel argument, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }
    ret = clSetKernelArg (kernel, 1, sizeof (cl_mem), (void*)& m_gpuWidth);
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to set kernel argument, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }
    ret = clSetKernelArg (kernel, 2, sizeof (cl_mem), (void*)& rhs.m_gpuData);
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to set kernel argument, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }
    ret = clSetKernelArg (kernel, 3, sizeof (cl_mem), (void*)& rhs.m_gpuWidth);
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to set kernel argument, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }
    ret = clSetKernelArg (kernel, 4, sizeof (cl_mem), (void*)& result.m_gpuData);
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to set kernel argument, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }
    ret = clSetKernelArg (kernel, 5, sizeof (cl_mem), (void*)& result.m_gpuWidth);
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to set kernel argument, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
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
        throw(LinAlgo::gpu_exception(std::string("Unable to execute kernel, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }

    ret = clFlush(m_command_queue);
    return ret;
}

/**
* @brief function for executing the multiplication kernel
*
* @note This could just be the execute_add_kernel tbh
*/
template <class ItemType>
cl_int matrix<ItemType>::execute_multiply_kernel (cl_kernel kernel, matrix<ItemType>& rhs, matrix<ItemType>& result) {
    cl_int ret;
    //set kernel arguments
    ret = clSetKernelArg (kernel, 0, sizeof (cl_mem), (void*)& m_gpuData);
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to set kernel argument, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }
    ret = clSetKernelArg (kernel, 1, sizeof (cl_mem), (void*)& m_gpuWidth); //single arg
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to set kernel argument, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }
    ret = clSetKernelArg (kernel, 2, sizeof (cl_mem), (void*)& rhs.m_gpuData);
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to set kernel argument, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }
    ret = clSetKernelArg (kernel, 3, sizeof (cl_mem), (void*)& rhs.m_gpuWidth); //single arg
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to set kernel argument, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }
    ret = clSetKernelArg (kernel, 4, sizeof (cl_mem), (void*)& result.m_gpuData);
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to set kernel argument, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }
    ret = clSetKernelArg (kernel, 5, sizeof (cl_mem), (void*)& result.m_gpuWidth); //single arg
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to set kernel argument, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }

    //execute the kernel
    size_t global_item_size = result.m_width * result.m_height;
    size_t local_item_size = 64;

    ret = clEnqueueNDRangeKernel (m_command_queue, kernel, 1, NULL, &global_item_size, /*&local_item_size*/ NULL, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to execute kernel, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }

    ret = clFlush(m_command_queue);
    return ret;
}

/**
* @brief function for executing the array and single value kernels
*/
template <class ItemType>
cl_int matrix<ItemType>::execute_array_val_kernel (cl_kernel kernel, ItemType& val, matrix<ItemType>& result) {
    cl_int ret;

    //push val to gpu
    cl_mem value = clCreateBuffer (m_context, CL_MEM_READ_ONLY, sizeof (ItemType), NULL, &ret);
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to create memory buffer, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }

    ret = clEnqueueWriteBuffer (m_command_queue, value, CL_TRUE, 0, sizeof (ItemType), &val, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to push data to GPU, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }

    //set kernel arguments
    ret = clSetKernelArg (kernel, 0, sizeof (cl_mem), (void*)& m_gpuData);
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to set kernel argument, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }
    ret = clSetKernelArg (kernel, 1, sizeof (cl_mem), (void*)& result.m_gpuData);
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to set kernel argument, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }
    ret = clSetKernelArg (kernel, 2, sizeof (cl_mem), (void*)& value); //single arg
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to set kernel argument, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }

    //execute the kernel
    size_t global_item_size = result.m_width * result.m_height;

    ret = clEnqueueNDRangeKernel (m_command_queue, kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        throw(LinAlgo::gpu_exception(std::string("Unable to execute kernel, error code: ") + std::string(LinAlgo::getErrorString(ret)), __FILE__, __LINE__, ret));
    }

    ret = clFinish (m_command_queue);
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

#endif
