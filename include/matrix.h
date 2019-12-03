/**
* @file matrix.h
*
* @brief Specifies the matrix class.
*
* @notes Requires OpenCL to work
*/

#ifndef _MATRIXH_INCLUDED
#define _MATRIXH_INCLUDED
#include "LinAlgo.hpp"

//I need to find a way to allow functions that update GPU data to still be const

/**
* @class matrix
*
* @brief The matrix class is a matrix-based storage type
*
* @details The matrix class is a templated class which allows the storage of data in
* a 2D matrix, this type also allows for general linear algebra operations on arithmetic
* types, as well as gpu support for basic arithmetic types
*
* @note In order to use the gpu, LinAlgo::InitGPU() must first be called, then either
* LinAlgo::AllUseGPU for all matrices to use the gpu, or individually call m.useGPU() to
* allow select matrices
*/
template <class ItemType = double>
class LinAlgo::matrix {
public:
    //constructors
#ifndef DONT_USE_GPU
    //maybe have one that doesn't initialize m_data. Like a "leave on gpu" from the very start? just to make initializing a little faster in situations that need it
    matrix ();//no allocating space or copying, meant for transferring over gpu data only
    matrix (const size_t& height, const size_t& width, const ItemType val = ItemType(0));
    matrix (const std::vector<std::vector<ItemType>>& vals);
    matrix (const ItemType** vals, const size_t& height, const size_t& width);
#else
    matrix();//matrix(0, 0)
    matrix (const size_t& height, const size_t& width, const ItemType val = ItemType(0));
    matrix (const std::vector<std::vector<ItemType>>& vals);
    matrix (const ItemType** vals, const size_t& height, const size_t& width);
#endif
    template <class ParamType>
    matrix (const matrix<ParamType>& M);
    matrix (const matrix<ItemType>& M);
    matrix (matrix<ItemType>&& M);

    //destructor
    ~matrix();//doesn't work, should fix :'(

    //initializers
    void fill (ItemType val);
    void clear();
#ifndef DONT_USE_GPU
    bool useGPU (bool);
    bool useGPU() const;
    bool leaveDataOnGPU (bool); //don't pull data for chained operations
    bool leaveDataOnGPU () const; //will return if the data is being left on the GPU
    bool pushData();//manually move data to and from the CPU
    bool pullData();
#endif

    //setters and getters
    ItemType get (size_t y, size_t x) const;
    void set (size_t y, size_t x, ItemType val);

    size_t getHeight() const;
    size_t getWidth() const;
    bool isSquare() const;

    matrix<ItemType>& resize (size_t height, size_t width, ItemType val = ItemType(0));
    matrix<ItemType> subMatrix (size_t y, size_t x, size_t h, size_t w);
    matrix<ItemType>& copy(size_t y, size_t x, matrix<ItemType>);//coordinates are position to copy into in order to copy a smaller matrix into a select part

    std::vector<ItemType> getRow(size_t r) const;
    std::vector<ItemType> getColumn(size_t c) const;

    static matrix<ItemType> identity (size_t height, size_t width = 0); //can't have a width=0 matrix

    //Idk if these should count as getters and setters or not
    ItemType getDeterminant();
    ItemType trace();//this doesn't use the gpu, so be sure to pull from the gpu if m_leaveOnGPU is set

    std::vector<ItemType> getEigenValues();
    std::vector<ItemType> getEigenVector (ItemType);
    matrix<ItemType> getEigenVectors();

    //basic math
   template<class ArgType>
    matrix<ItemType> add(ArgType& val);
    template<class ArgType>
    matrix<ItemType> addMatrix(matrix<ArgType>& val);
    template <class ArgType>
    matrix<ItemType> addScalar(ArgType scalar);

    template <class argType>
    matrix<ItemType> subtract (matrix<argType>& M);
    template <class argType>
    matrix<ItemType> subtract (const argType& val);

    template <class argType>
    matrix<ItemType> multiply (matrix<argType>& M);
    template <class argType>
    matrix<ItemType> multiply (const argType& val);
    template <class argType>
    matrix<ItemType> elementMultiply (matrix<argType>& M);

    template <class argType>
    matrix<ItemType> divide (matrix<argType>& M);
    template <class argType>
    matrix<ItemType> divide (const argType& val);
    template <class argType>
    matrix<ItemType> elementDivide (matrix<argType>& M);

    //linear fun times
    //the versions in here will overwrite the matrix
    //use the LinAlgo namespace versions if u don't want it overwritten
    matrix<ItemType>& transpose();
    matrix<ItemType>& inverse();

    matrix<ItemType>& map(ItemType (*func) (ItemType&), bool asynchronus = false, size_t thread_count = 0); //map a function via the cpu

    //I don't think I can actually template the gpu maps... we'll cross that bridge later
    //spatial map, like a mask? hmmmm... probably is something i should have
#ifndef DONT_USE_GPU
    matrix<ItemType>& mapGPU (std::string kernel, cl_int* err_code = NULL); //this function will compile and run a kernel that acts on a single array pointer... be careful
    matrix<ItemType>& mapGPU (cl::Kernel& kernel, cl_int* err_code = NULL); //it will also assume that your kernel is accepting the right data from that array, so once again, be careful :P
    //should this take another one that's just a fully compiled and such kernel? probably
#endif

    bool qr (matrix<ItemType>& Q, matrix<ItemType>& R); //returns if it was successful

    matrix<ItemType> solve();
    matrix<ItemType> gaussJordan();//gj
    matrix<ItemType> rowEchelon();//re
    matrix<ItemType> reducedRowEchelon();//rre

    //friends and operators
    const std::vector<ItemType>& operator[] (size_t y) const;
    std::vector<ItemType>& operator[] (size_t y);
    template <class ArgType>
    matrix<ItemType>& operator= (const matrix<ArgType>& M);
    matrix<ItemType>& operator= (const matrix<ItemType>& M);
    matrix<ItemType>& operator= (matrix<ItemType>&& M);

    template <class ArgType>
    bool operator== (const matrix<ArgType>& M) const;
    template <class ArgType>
    bool operator!= (const matrix<ArgType>& M) const;

    template <class ArgType>
    matrix<ItemType> operator+ (ArgType&& val);
    template <class ArgType>
    matrix<ItemType>& operator+= (matrix<ArgType>& M);
    template <class ArgType>
    matrix<ItemType>& operator+= (const ArgType& val);

    template <class ArgType>
    matrix<ItemType> operator- (matrix<ArgType>& M);
    template <class ArgType>
    matrix<ItemType> operator- (const ArgType& val);
    template <class ArgType>
    matrix<ItemType>& operator-= (matrix<ArgType>& M);
    template <class ArgType>
    matrix<ItemType>& operator-= (const ArgType& val);

    template <class ArgType>
    matrix<ItemType> operator* (matrix<ArgType>& M);
    template <class ArgType>
    matrix<ItemType> operator* (const ArgType& val);
    template <class ArgType>
    matrix<ItemType>& operator*= (matrix<ArgType>& M);
    template <class ArgType>
    matrix<ItemType>& operator*= (const ArgType& val);

    template <class ArgType>
    matrix<ItemType> operator/ (matrix<ArgType>& M);
    template <class ArgType>
    matrix<ItemType> operator/ (const ArgType& val);
    template <class ArgType>
    matrix<ItemType>& operator/= (matrix<ArgType>& M);
    template <class ArgType>
    matrix<ItemType>& operator/= (const ArgType& val);

    //All the friendly LinAlgo functions
    template <class ArgType>
    friend matrix<ArgType> LinAlgo::transpose (const matrix<ArgType>& M);
    template <class ArgType>
    friend matrix<ArgType> LinAlgo::inverse (matrix<ArgType>& M);

    template <class ArgType1, class ArgType2>
    friend matrix<ArgType1> LinAlgo::map (const matrix<ArgType2>& M, ArgType1 (*function) (ArgType2), bool asynchronous, int num_threads);
#ifndef DONT_USE_GPU
    //due to updating gpu data counting as cahnging the matrix, I can't
    //actually have these be const :'(
    template <class ArgType>
    friend matrix<ArgType> LinAlgo::mapGPU (matrix<ArgType>& M, std::string kernel, cl_int* error_ret);
    template <class ArgType>
    friend matrix<ArgType> LinAlgo::mapGPU (matrix<ArgType>& M, cl::Kernel& kernel, cl_int* error_ret);
#endif

    template <class ArgType>
    friend bool LinAlgo::qr (const matrix<ArgType>& M, matrix<ArgType>& Q, matrix<ArgType>& R);
    template <class ArgType>
    friend std::vector<ArgType> eigenvalues(matrix<ArgType>& M);
    template <class ArgType>
    friend std::vector<ArgType> eigenvalues(matrix<ArgType>& M, matrix<ItemType>& eigenvecs_out);

    template <class ArgType>
    friend matrix<ArgType> LinAlgo::gj (const matrix<ArgType>& M);
    template <class ArgType>
    friend matrix<ArgType> LinAlgo::re (const matrix<ArgType>& M);
    template <class ArgType>
    friend matrix<ArgType> LinAlgo::re (const matrix<ArgType>& M, int& row_swaps);
    template <class ArgType>
    friend matrix<ArgType> LinAlgo::rre (const matrix<ArgType>& M);

    //iterators
    template <typename ptr_type>
    class iterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = ptr_type;
        using difference_type = std::ptrdiff_t;
        using pointer = ptr_type*;
        using reference = ptr_type&;

        iterator(long int index, size_t _min, size_t _max, size_t _width, const std::vector<std::vector<ItemType>>* _data) : m_index(index), m_min(_min), m_max(_max), mat_width(_width), mat_data(dynamic_cast<const std::vector<std::vector<ItemType>>*>(_data)) {
            //assert index between min and max
        }

        iterator(const iterator<ptr_type>& other) {
            m_index = other.m_index;
            m_min = other.m_min;
            m_max = other.m_max;
            mat_width = other.mat_width;
            mat_data = other.mat_data;
        }

        //increment/decrement
        iterator<ptr_type>& operator++() {
            m_index++;
            //assert in range
            return *this;
        }
        iterator<ptr_type> operator++(int) {
            iterator<ptr_type> ret = *this;
            ++(*this);
            return ret;
        }
        iterator<ptr_type>& operator--() {
            m_index--;
            //assert
            return *this;
        }
        iterator<ptr_type> operator--(int) {
            iterator<ptr_type> ret = *this;
            --(*this);
            //assert
            return ret;
        }

        iterator<ptr_type> operator+(const iterator<ptr_type>& other) {
            iterator<ptr_type> ret(m_index + other.m_index, m_min, m_max, mat_width, mat_data);
            return ret;
        }
        iterator<ptr_type> operator+(const long int& rhs) {
            return iterator<ptr_type>(m_index + rhs, m_min, m_max, mat_width, mat_data);
        }
        friend iterator<ptr_type> operator+(const long int& lhs, const iterator<ptr_type>& rhs) {
            return rhs+lhs;
        }
        difference_type operator-(const iterator<ptr_type>& other) {
            return m_index - other.m_index;
        }
        iterator<ptr_type> operator-(const long int& rhs) {
            return iterator<ptr_type>(m_index - rhs, m_min, m_max, mat_width, mat_data);
        }
        friend iterator<ptr_type> operator-(const long int& lhs, const iterator<ptr_type>& rhs) {
            return rhs-lhs;
        }

        iterator<ptr_type>& operator=(const iterator<ptr_type>& rhs) {
            m_index = rhs.m_index;
            m_min = rhs.m_min;
            m_max = rhs.m_max;
            mat_width = rhs.mat_width;
            mat_data = rhs.mat_data;
            return *this;
        }
        iterator<ptr_type>& operator+=(const iterator<ptr_type>& rhs) {
            return *this = *this + rhs;
        }
        iterator<ptr_type>& operator+=(const long int& rhs) {
            return *this = *this + rhs;
        }
        iterator<ptr_type>& operator-=(const iterator<ptr_type>& rhs) {
            return *this = *this - rhs;
        }
        iterator<ptr_type>& operator-=(const long int& rhs) {
            return *this = *this - rhs;
        }

        //evaluation
        bool operator==(const iterator<ptr_type>& other) const {
            return other.mat_data == mat_data && other.m_index == m_index;
        }
        bool operator!=(const iterator<ptr_type>& other) const {
            return other.mat_data != mat_data || other.m_index != m_index;
        }
        bool operator<(const iterator<ptr_type>& other) const {
            return m_index < other.m_index;
        }
        bool operator<=(const iterator<ptr_type>& other) const {
            return m_index <= other.m_index;
        }
        bool operator>(const iterator<ptr_type>& other) const {
            return m_index > other.m_index;
        }

        bool operator>=(const iterator<ptr_type>& other) const {
            return m_index >= other.m_index;
        }

        //access
        iterator<ptr_type>& operator=(const ItemType& item) {
            (*mat_data)[m_index/mat_width][m_index%mat_width] = item;
            return *this;
        }
        reference operator*() const {
            return (reference) (*mat_data)[m_index/mat_width][m_index%mat_width];
        }
        pointer operator->() {
            return (ptr_type) &(*mat_data)[m_index/mat_width][m_index%mat_width];
        }
        reference operator[](const size_t index) const {
            //assert m_index + index
            size_t pos = m_index + index;
            return (reference) (*mat_data)[pos/mat_width][pos%mat_width];
        }

    private:
        const std::vector<std::vector<ItemType>>* mat_data;
        size_t mat_width;
        long int m_index;//so that can check negative index
        size_t m_min;
        size_t m_max;
    };

    iterator<const ItemType> cbegin() const;
    iterator<const ItemType> cend() const;

    iterator<ItemType> begin();
    iterator<ItemType> end();



private:
//basic data
    std::vector<std::vector<ItemType>> m_data;
    size_t m_height;
    size_t m_width;
    ItemType m_determinant;
    std::vector<ItemType> m_eigenValues;//another for vectors?

#ifndef DONT_USE_GPU
    bool m_dataInitialized;
//are things up to date?
    bool m_gpuUpToDate;//this may grow to be a pain to manage later... and it may be better to track the validity of individual bits of data/rows/etc so that whole matrices down't get pushed each time
#endif
    typedef enum {
        GPU_DATA = 1 << 0,
        GPU_HEIGHT = 1 << 1,
        GPU_WIDTH = 1 << 2,
        DETERMINANT = 1 << 3,
        EIGENVALUES = 1 << 4 //things that rely on other things being up to date will need to be unticked an unallocated when other data goes out of date
    } dataFlag;
    unsigned char m_upToDate;//use the data flags to mark what is currently up to date
#ifndef DONT_USE_GPU
    std::vector<bool> m_gpuSlicesUpToDate;//vector m_height in size, true if that slice is up to date, false if a change occured

//gpu things
    //whether or not i'm allowed to use gpu, or should use cpu instead
    bool m_useGPU;
    bool m_leaveOnGPU;//do not pull data from gpu, speeds up performance for chained operations

    //aux functions for the gpu usage
    cl_int createResultBuffer ( ); //returns CL_SUCCESS if successful
    cl_int pushToGPU ( ); //returns CL_SUCCESS if successful
    cl_int pullFromGPU ( ); //returns CL_SUCCESS if successful

    //data involved in the gpu computation
    cl::Buffer* m_gpuData;
    cl::Buffer* m_gpuHeight;
    cl::Buffer* m_gpuWidth;

    //functions to execute the kernels
    cl_int execute_add_kernel (cl::Kernel kernel, matrix<ItemType>& rhs, matrix<ItemType>& result); //so far these two functions are basically exactly the same...
    cl_int execute_multiply_kernel (cl::Kernel kernel, matrix<ItemType>& rhs, matrix<ItemType>& result); //they'll change if the kernel params do, but rn....
    cl_int execute_array_val_kernel (cl::Kernel kernel, ItemType& val, matrix<ItemType>& result);
#endif
};

template <class ArgType1, class ArgType2>
LinAlgo::matrix<ArgType1> operator+(ArgType2&& scalar, LinAlgo::matrix<ArgType1>& M);

#include "../src/matrix.cpp"
#endif
