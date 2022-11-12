#include <stdio.h>

#include "GPU.cuh"
#include "Utils.h"


//
//__global__ void kernel(float* a) {
//
//	int bloc_num = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;       //номер блока
//	int global_index = bloc_num * blockDim.x * blockDim.y * blockDim.z +                    //индекс первого потока в блоке
//		(threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x); //смещение внутри блока 
//	a[global_index] = a[global_index] + global_index;
//
//	printf("I am from  (%i, %i, %i) block, (%i, %i, %i) thread (global index: %i)\n",
//		blockIdx.x, blockIdx.y, blockIdx.z,
//		threadIdx.x, threadIdx.y, threadIdx.z,
//		global_index
//	);
//}


__global__ void kernel(float a, float* x, int incx, float* y, int incy)
{

	int bloc_num = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;       //номер блока
	int global_index = bloc_num * blockDim.x * blockDim.y * blockDim.z +                    //индекс первого потока в блоке
		(threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x); //смещение внутри блока 

	y[global_index * incy] += a * x[global_index * incx];
}



float* saxpy_gpu(int n, float a, float* x, int incx, float* y, int incy, bool print)
{
	int num_blocks = n, block_size = 1;

	int sizeX = GetBufferSize(n, incx), sizeY = GetBufferSize(n, incy);

	float* xBuff = new float[sizeX], * x_gpu;
	float* yBuff = new float[sizeY], * y_gpu;

	std::fill(xBuff, xBuff + sizeX, 0);
	std::fill(yBuff, yBuff + sizeY, 0);

	std::memcpy(xBuff, x, n * sizeof(float));
	std::memcpy(yBuff, y, n * sizeof(float));


	cudaMalloc((void**)&x_gpu, sizeX * sizeof(float));
	cudaMalloc((void**)&y_gpu, sizeY * sizeof(float));

	cudaMemcpy(x_gpu, xBuff, sizeX * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, yBuff, sizeY * sizeof(float), cudaMemcpyHostToDevice);


	kernel << <num_blocks, block_size >> > (a, x_gpu, incx, y_gpu, incy);
	cudaDeviceSynchronize();
	cudaMemcpy(yBuff, y_gpu, sizeY * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(y_gpu);
	cudaFree(x_gpu);
	return yBuff;
}



__global__ void kerneld(double a, double* x, int incx, double* y, int incy)
{

	int bloc_num = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;       //номер блока
	int global_index = bloc_num * blockDim.x * blockDim.y * blockDim.z +                    //индекс первого потока в блоке
		(threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x); //смещение внутри блока 

	y[global_index * incy] += a * x[global_index * incx];
}



double* daxpy_gpu(int n, double a, double* x, int incx, double* y, int incy, bool print)
{
	int num_blocks = n, block_size = 1;

	int sizeX = GetBufferSize(n, incx), sizeY = GetBufferSize(n, incy);

	double* xBuff = new double[sizeX], * x_gpu;
	double* yBuff = new double[sizeY], * y_gpu;

	std::fill(xBuff, xBuff + sizeX, 0);
	std::fill(yBuff, yBuff + sizeY, 0);

	std::memcpy(xBuff, x, n * sizeof(double));
	std::memcpy(yBuff, y, n * sizeof(double));


	cudaMalloc((void**)&x_gpu, sizeX * sizeof(double));
	cudaMalloc((void**)&y_gpu, sizeY * sizeof(double));

	cudaMemcpy(x_gpu, xBuff, sizeX * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, yBuff, sizeY * sizeof(double), cudaMemcpyHostToDevice);


	kerneld << <num_blocks, block_size >> > (a, x_gpu, incx, y_gpu, incy);
	cudaDeviceSynchronize();
	cudaMemcpy(yBuff, y_gpu, sizeY * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(y_gpu);
	cudaFree(x_gpu);
	return yBuff;
}