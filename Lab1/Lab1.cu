
#include <stdio.h>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void kernel(float* a) {

	int bloc_num = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;       //номер блока
	int global_index = bloc_num * blockDim.x * blockDim.y * blockDim.z +                    //индекс первого потока в блоке
		(threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x); //смещение внутри блока 
	a[global_index] = a[global_index] + global_index;

	printf("I am from  (%i, %i, %i) block, (%i, %i, %i) thread (global index: %i)\n",
		blockIdx.x, blockIdx.y, blockIdx.z,
		threadIdx.x, threadIdx.y, threadIdx.z,
		global_index
	);
}

int main() {

	int gX = 1, gY = 1, gZ = 3;
	int bX = 3, bY = 1, bZ = 2;

	const dim3* num_blocks = new dim3(gX, gY, gZ);
	const dim3* block_size = new dim3(bX, bY, bZ);

	const int n = bX * bY * bZ * // количество потоков в блоке 
		gX * gY * gZ;  // количество блоков
	float* a = new float[n], * a_gpu;
	printf("Host before device:\n");
	for (int i = 0; i < n; i++)
	{
		a[i] = i * 2;
		printf("%.2f; ", a[i]);
	}
	printf("\n--------------------\n");

	printf("Device:\n");

	cudaMalloc((void**)&a_gpu, n * sizeof(float));
	cudaMemcpy(a_gpu, a, n * sizeof(float), cudaMemcpyHostToDevice);

	kernel << <*num_blocks, *block_size >> > (a_gpu);
	cudaDeviceSynchronize();
	cudaMemcpy(a, a_gpu, n * sizeof(float), cudaMemcpyDeviceToHost);

	printf("--------------------\n");
	printf("Host after device:\n");
	for (int i = 0; i < n; i++)
	{
		printf("%.2f; ", a[i]);
	}
	printf("\n");


	delete[] a;
	cudaFree(a_gpu);
	return 0;
}
