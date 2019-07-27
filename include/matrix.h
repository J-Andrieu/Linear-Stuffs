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

//I need to find a wag to allow functions that update GPU data to still be const

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
template <class ItemType>
class LinAlgo::matrix {
public:
	//constructors
	matrix(size_t height, size_t width, bool enable_gpu = false);
	//this one is a little ambiguous b/c of the default...
	//replacing above and below w/ matrix(size_t h, size_t w, ItemType& val = NULL, bool use_gpu = fase) 
	//may be better
	matrix(size_t height, size_t width, ItemType val, bool enable_gpu = false);
	template <class ParamType>
	matrix(const matrix<ParamType>& M);
	matrix(const matrix<ItemType>& M);
	matrix(matrix<ItemType>&& M);
	
	//destructor
	~matrix();//doesn't work, should fix :'(

	//initializers
	void fill(ItemType val);
	void clear();
	bool useGPU(bool);
	bool leaveDataOnGPU(bool);//don't pull data for chained operations

	//setters and getters
	ItemType get(size_t y, size_t x) const;
	void set(size_t y, size_t x, ItemType val);

	size_t getHeight() const;
	size_t getWidth() const;

	matrix<ItemType>& resize(size_t height, size_t width, ItemType& val = NULL);
	matrix<ItemType> subMatrix(size_t y, size_t x, size_t h, size_t w);

	static matrix<ItemType> identity(size_t height, size_t width = 0);//can't have a width=0 matrix
	
	//Idk if these should count as getters and setters or not
	ItemType getDeterminant();
	
	std::vector<ItemType> getEigenValues();
	std::vector<ItemType> getEigenVector(ItemType);
	matrix<ItemType> getEigenVectors();

	//basic math
	template <class argType>
	matrix<ItemType> add(matrix<argType>& M); //can't const these cuz of pushing to gpu :'(
	template <class argType>
	matrix<ItemType> add(const argType& val);

	template <class argType>
	matrix<ItemType> subtract(matrix<argType>& M);
	template <class argType>
	matrix<ItemType> subtract(const argType& val);

	template <class argType>
	matrix<ItemType> multiply(matrix<argType>& M);
	template <class argType>
	matrix<ItemType> multiply(const argType& val);
	template <class argType>
	matrix<ItemType> elementMultiply(matrix<argType>& M);
	
	//divide...? I don't quite remember how this works aside from maybe multiplying by the inverse?
	template <class argType>
	matrix<ItemType> divide(matrix<argType>& M);
	template <class argType>
	matrix<ItemType> divide(const argType& val);
	template <class argType>
	matrix<ItemType> elementDivide(matrix<argType>& M);

	//linear fun times
	//the versions in here will overwrite the matrix
	//use the LinAlgo namespace versions if u don't want it overwritten
	matrix<ItemType>& transpose();
	matrix<ItemType>& inverse();

	template <class ArgType>
	matrix<ArgType> map(ArgType(*function)(ItemType));//map a function via the cpu
	//I don't think I can actually template the gpu maps... we'll cross that bridge later
	template <class ArgType>
	matrix<ArgType> mapGPU(std::string kernel, cl_int& error_code);//this function will compile and run a kernel that acts on a single array pointer... be careful
	template <class ArgType>
	matrix<ArgType> mapGPU(cl_kernel kernel, cl_int& error_code);
	//should this take another one that's just a fully compiled and such kernel? probably

	bool qr(matrix<ItemType>& Q, matrix<ItemType>& R);//returns if it was successful

	matrix<ItemType> solve();
	matrix<ItemType> gaussJordan();//gj
	matrix<ItemType> rowEchelon();//re
	matrix<ItemType> reducedRowEchelon();//rre

	//friends and operators
	std::vector<ItemType>& operator[](size_t y) const;
	template <class ArgType>
	matrix<ItemType>& operator=(const matrix<ArgType>& M);
	matrix<ItemType>& operator=(matrix<ItemType>&& M);

	template <class ArgType>
	bool operator==(const matrix<ArgType>& M) const;
	template <class ArgType>
	bool operator!=(const matrix<ArgType>& M) const;

	template <class ArgType>
	matrix<ItemType> operator+(matrix<ArgType>& M);
	template <class ArgType>
	matrix<ItemType> operator+(const ArgType& val);
	template <class ArgType>
	matrix<ItemType>& operator+=(matrix<ArgType>& M);
	template <class ArgType>
	matrix<ItemType>& operator+=(const ArgType& val);

	template <class ArgType>
	matrix<ItemType> operator-(matrix<ArgType>& M);
	template <class ArgType>
	matrix<ItemType> operator-(const ArgType& val);
	template <class ArgType>
	matrix<ItemType>& operator-=(matrix<ArgType>& M);
	template <class ArgType>
	matrix<ItemType>& operator-=(const ArgType& val);

	template <class ArgType>
	matrix<ItemType> operator*(matrix<ArgType>& M);
	template <class ArgType>
	matrix<ItemType> operator*(const ArgType& val);
	template <class ArgType>
	matrix<ItemType>& operator*=(matrix<ArgType>& M);
	template <class ArgType>
	matrix<ItemType>& operator*=(const ArgType& val);

	template <class ArgType>
	matrix<ItemType> operator/(matrix<ArgType>& M);
	template <class ArgType>
	matrix<ItemType> operator/(const ArgType& val);
	template <class ArgType>
	matrix<ItemType>& operator/=(matrix<ArgType>& M);
	template <class ArgType>
	matrix<ItemType>& operator/=(const ArgType& val);

	friend matrix<ItemType> LinAlgo::transpose(const matrix<ItemType>& M);
	friend matrix<ItemType> LinAlgo::inverse(const matrix<ItemType>& M);

	template <class ArgType, class ItemType = ArgType>
	friend matrix<ArgType> LinAlgo::map(const matrix<ItemType>& M, ArgType(*function)(ItemType));
	//due to updating gpu data counting as cahnging the matrix, I can't
	//actually have these be const :'(
	friend matrix<ItemType> LinAlgo::map(const matrix<ItemType>& M, std::string kernel, cl_int& error_ret);
	friend matrix<ItemType> LinAlgo::map(const matrix<ItemType>& M, cl_kernel kernel, cl_int& error_ret);

	friend bool LinAlgo::qr(const matrix<ItemType>& M, matrix<ItemType>& Q, matrix<ItemType>& R);

	friend matrix<ItemType> LinAlgo::solve(const matrix<ItemType>& M);
	friend matrix<ItemType> LinAlgo::gj(const matrix<ItemType>& M);
	friend matrix<ItemType> LinAlgo::re(const matrix<ItemType>& M);
	friend matrix<ItemType> LinAlgo::rre(const matrix<ItemType>& M);

	//iterators?
private:
//basic data
	std::vector<std::vector<ItemType>*> m_data;
	size_t m_height;
	size_t m_width;
	std::vector<ItemType> m_eigenValues;//another for vectors?

//are things up to date?
	bool m_gpuUpToDate;//this may grow to be a pain to manage later... and it may be better to track the validity of individual bits of data/rows/etc so that whole matrices down't get pushed each time
	typedef enum {
		GPU_DATA = 1<<0,
		GPU_HEIGHT = 1<<1,
		GPU_WIDTH = 1<<2,
		DISCRIMINANT = 1<<3,
		EIGENVALUES = 1<<4//things that rely on other things being up to date will need to be unticked an unallocated when other data goes out of date
	} dataFlag;
	unsigned char m_upToDate;//use the data flags to mark what is currently up to date
	std::vector<bool> m_gpuSlicesUpToDate;//vector m_height in size, true if that slice is up to date, false if a change occured

//gpu things
	//whether or not i'm allowed to use gpu, or should use cpu instead
	bool m_useGPU;
	bool m_leaveOnGPU;//do not pull data from gpu, speeds up performance for chained operations
	
	//aux functions for the gpu usage
	cl_int initQueue();//returns CL_SUCCESS is successful
	cl_int createResultBuffer(cl_command_queue&);//returns CL_SUCCESS if successful
	cl_int pushToGPU(cl_command_queue&);//returns CL_SUCCESS is successful
	cl_int pullFromGPU(cl_command_queue&);//returns CL_SUCCESS is successful

	//data involved in the gpu computation
	cl_mem m_gpuData;
	cl_mem m_gpuHeight;
	cl_mem m_gpuWidth;
	cl_command_queue m_command_queue;
	
	//functions to execute the kernels
	cl_int execute_add_kernel(cl_kernel kernel, matrix<ItemType>& rhs, matrix<ItemType>& result);//so far these two functions are basically exactly the same...
	cl_int execute_multiply_kernel(cl_kernel kernel, matrix<ItemType>& rhs, matrix<ItemType>& result);//they'll change if the kernel params do, but rn....
	cl_int execute_array_val_kernel(cl_kernel kernel, ItemType& val, matrix<ItemType>& result);
};

#include "../src/matrix.cpp"
#endif 
