#include <stdio.h>

#include "GPU.cuh"
#include "Utils.h"
#include <omp.h>


__global__ void kernel(float a, float* x, int incx, float* y, int incy, int threads_num, int n)
{
	int global_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (global_index >= threads_num)
		return;
	if (global_index * incx >= n || global_index * incy >= n)
		return;

	y[global_index * incy] = y[global_index * incy] + a * x[global_index * incx];
	
}



float* saxpy_gpu(int threads_num, int block_size, int n, float a, float* x, int incx, float* y, int incy, std::vector<double>* times)
{
	int num_blocks = threads_num / block_size;
	if (threads_num % block_size != 0)
		num_blocks++;
	

	float* x_gpu;
	float* y_gpu;

	cudaError_t error;
	error = cudaMalloc((void**)&x_gpu, n * sizeof(float));
	if (error)
		printf("cudaMalloc1 error : %i \n", error);

	error = cudaMalloc((void**)&y_gpu, n * sizeof(float));
	if (error)
		printf("cudaMalloc2 error: %i \n", error);

	error = cudaMemcpy(x_gpu, x, n * sizeof(float), cudaMemcpyHostToDevice);
	if (error)
		printf("cudaMemcpy1 error: %i \n", error);

	error = cudaMemcpy(y_gpu, y, n * sizeof(float), cudaMemcpyHostToDevice);
	if (error)
		printf("cudaMemcpy2 error: %i \n", error);

	double t1, t2;
	t1 = omp_get_wtime();
	kernel << <num_blocks, block_size >> > (a, x_gpu, incx, y_gpu, incy, threads_num, n);
	
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(error));
	}

	error = cudaDeviceSynchronize();
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", error);
	}
	t2 = omp_get_wtime();
	if (times != nullptr)
	{
		(*times).push_back(t2 - t1);
	}

	error = cudaMemcpy(y, y_gpu, n * sizeof(float), cudaMemcpyDeviceToHost);
	if (error)
		printf("cudaMemcpy device to host error: %i \n", error);
	cudaFree(x_gpu);
	cudaFree(y_gpu);
	

	return y;
}


__global__ void kerneld(double a, double* x, int incx, double* y, int incy, int operations, int threads_num, int n)
{
	int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = global_index * operations;
	if (global_index >= threads_num)
		return;

	int id = 0;
	for (int i = 0; i < operations; i++)
	{
		id = idx + i;
		if (id * incx >= n || id * incy >= n)
			return;
	}

	
	for (int i = 0; i < operations; i++)
	{
		
		y[incy * id] = y[incy * id] + a * x[incx * id];
	}
}



double* daxpy_gpu(int threads_num, int block_size, int n, double a, double* x, int incx, double* y, int incy, std::vector<double>* times)
{
	int num_blocks = threads_num / block_size;
	if (threads_num % block_size != 0)
	{
		num_blocks++;
	}

	int op_per_thread = n / threads_num;
	if (n < threads_num)
	{
		op_per_thread = 1;
	}

	double* x_gpu;
	double* y_gpu;

	cudaError_t error;

	error = cudaMalloc((void**)&x_gpu, n * sizeof(double));
	/*if (error)
		printf("cudaMalloc1 error : %i \n",error);*/

	error = cudaMalloc((void**)&y_gpu, n * sizeof(double));
	if (error)
		printf("cudaMalloc2 error: %i \n", error);

	error= cudaMemcpy(x_gpu, x, n * sizeof(double), cudaMemcpyHostToDevice);
	if (error)
		printf("cudaMemcpy1 error: %i \n", error);

	error =  cudaMemcpy(y_gpu, y, n * sizeof(double), cudaMemcpyHostToDevice);
	/*if (error)
		printf("cudaMemcpy2 error: %i \n", error);*/


	double t1, t2;
	t1 = omp_get_wtime();
	kerneld << <num_blocks, block_size >> > (a, x_gpu, incx, y_gpu, incy, op_per_thread, threads_num, n);
	
	error = cudaGetLastError();
	/*if (error != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(error));
		
	}*/

	error = cudaDeviceSynchronize();
	/*if (error != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", error);

	}*/
	t2 = omp_get_wtime();
	/*if (times != nullptr)
	{
		(*times).push_back(t2 - t1);
	}*/

	error = cudaMemcpy(y, y_gpu, n * sizeof(double), cudaMemcpyDeviceToHost);
	/*if (error)
		printf("cudaMemcpy device to host error: %i \n", error);*/

	cudaFree(x_gpu);
	cudaFree(y_gpu);
	
	return y;
}

