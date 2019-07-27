
__kernel void add(__global const double* A, __global const size_t* a_width, 
				__global const double* B, __global const size_t* b_width, 
				__global double* C, __global const size_t* c_width) {
	size_t index = get_global_id(0);
	size_t x = index % c_width[0];
	size_t y = index / c_width[0];
	C[index] = A[y * a_width[0] + x] + B[y * b_width[0] + x];
}


__kernel void elementMultiply(__global const double* A, __global const size_t* a_width, 
				__global const double* B, __global const size_t* b_width, 
				__global double* C, __global const size_t* c_width) {
	size_t index = get_global_id(0);
	size_t x = index % c_width[0];
	size_t y = index / c_width[0];
	C[index] = A[y * a_width[0] + x] * B[y * b_width[0] + x];
}


__kernel void subtract(__global const double* A, __global const size_t* a_width, 
				__global const double* B, __global const size_t* b_width, 
				__global double* C, __global const size_t* c_width) {
	size_t index = get_global_id(0);
	size_t x = index % c_width[0];
	size_t y = index / c_width[0];
	C[index] = A[y * a_width[0] + x] - B[y * b_width[0] + x];
}


__kernel void elementDivide(__global const double* A, __global const size_t* a_width, 
				__global const double* B, __global const size_t* b_width, 
				__global double* C, __global const size_t* c_width) {
	size_t index = get_global_id(0);
	size_t x = index % c_width[0];
	size_t y = index / c_width[0];
	C[index] = A[y * a_width[0] + x] / B[y * b_width[0] + x];
}


__kernel void multiply(__global const double* A, __global const size_t* a_width, 
						__global const double* B, __global const size_t* b_width, 
						__global double* C, __global const size_t* res_width) {
	size_t index = get_global_id(0);
	size_t row_index = (index / res_width[0]) * a_width[0];
	size_t column_index = index % res_width[0];

	double sum = 0;
	for (size_t i = 0; i < a_width[0]; i++, row_index++, column_index += b_width[0]) {
		sum += A[row_index] * B[column_index];
	}

	C[index] = sum;
}

__kernel void addScalar(__global const double* A, __global double* B, __global const double* val) {
	size_t index = get_global_id(0);
	B[index] = A[index] + val[0];
}

__kernel void multiplyScalar(__global const double* A, __global double* B, __global const double* val) {
	size_t index = get_global_id(0);
	B[index] = A[index] * val[0];
}

__kernel void subtractScalar(__global const double* A, __global double* B, __global const double* val) {
	size_t index = get_global_id(0);
	B[index] = A[index] - val[0];
}

__kernel void divideScalar(__global const double* A, __global double* B, __global const double* val) {
	size_t index = get_global_id(0);
	B[index] = A[index] / val[0];
}
