#include <stdio.h>
#include <iostream>
#include "CPU.h"
#include "GPU.cuh"
#include "Utils.h"
#include <vector>
#include <omp.h>

#include <chrono>
#include <thread>

double average(std::vector<double> arr)
{
	double res = 0;
	for (int i = 0; i < arr.size(); i++)
	{
		res += arr[i];
	}
	return res / arr.size();
}


int findMinElementIdx(const std::vector<double> vec)
{
	double min = vec[0];
	int idx = 0;
	for (int i = 1; i < vec.size(); i++)
	{
		if (vec[i] < min)
		{
			min = vec[i];
			idx = i;
		}
	}

	return idx;
}

int findBestBlockSize(const std::vector<int> block_sizes)
{
	std::vector<double> times, avg_time;



	const int threads_num = 50000000;
	const int iterations = 10;

	int n = threads_num;
	int incx = 1, incy = 1;
	const double a = 1.25;

	printf("--------------------\n");
	for (int i = 0; i < block_sizes.size(); i++)
	{
		float* x = GenerateTestArray<float>(n);
		float* y = GenerateTestArray<float>(n);

		saxpy_gpu(threads_num, block_sizes[i], n, a, x, incx, y, incy, &times);
		std::cout << "Blocks: " << block_sizes[i] << "   Avg time: " << times[i] << std::endl;
		delete[] x;
		delete[] y;
	}
	printf("--------------------\n");
	return block_sizes[findMinElementIdx(times)];
}



void PrintTable(double tCPU_f, double tOMP_f, double tGPU_f , double tCPU_d, double tOMP_d, double tGPU_d)
{
	printf("------------------------------");
	printf("|Technology|Time|Acceleration|Data type");
	printf("|CPU       |%f|%f|float|", tCPU_f, tCPU_f/ tCPU_f);
	printf("|OMP       |%f|%f|float|",tOMP_f, tOMP_f/ tCPU_f);
	printf("|GPU       |%f|%f|float|", tGPU_f, tGPU_f / tCPU_f);
	
	printf("|CPU       |%f|%f|double|", tCPU_d, tCPU_d / tCPU_d);
	printf("|OMP       |%f|%f|double|", tOMP_d, tOMP_d / tCPU_d);
	printf("|GPU       |%f|%f|double|", tGPU_d, tGPU_d / tCPU_d);
	printf("------------------------------");

}

// 1 000 000 000 байт на видюхе
// 250 000 000 float
void test()
{
	const std::vector<int> block_sizes{ 8, 16, 32, 64, 128, 256 };
	int best_block_size = findBestBlockSize(block_sizes);
	std::cout << "Best block size = " << best_block_size << std::endl;



	const int threads_num =50000000;
	const int iterations = 10;

	int n = threads_num;
	int incx = 3, incy = 2;
	const double a = 1.25;
	printf("Array Size (float) %i\n", n);

	

	double t1, t2;
	double tCPU_f,tOMP_f,tGPU_f ;
	std::vector<double> times;

	//--------float---------//
	float* x = GenerateArray<float>(n);
	float* y = GenerateArray<float>(n);
	printf("Arrays Generated\n");
	float* res, * resOMP, * resGPU;

	
	// SAXPY CPU
	for (int i = 0; i < iterations; i++)
	{
		t1 = omp_get_wtime();
		res = saxpy(n, a, x, incx, y, incy);
		t2 = omp_get_wtime();
		times.push_back(t2 - t1);
	}
	tCPU_f = average(times);
	printf("CPU:Work took %f seconds\n", tCPU_f);

	times.clear();

	// SAXPY OMP
	for (int i = 0; i < iterations; i++)
	{
		t1 = omp_get_wtime();
		resOMP = saxpyOMP(n, a, x, incx, y, incy);
		t2 = omp_get_wtime();
		times.push_back(t2 - t1);
	}
	tOMP_f = average(times);
	printf("OMP:Work took %f seconds\n", tOMP_f);

	times.clear();

	// SAXPY GPU
	for (int i = 0; i < iterations; i++)
	{
		resGPU = saxpy_gpu(threads_num, best_block_size, n, a, x, incx, y, incy, &times);
	}	
	tGPU_f = average(times);
	printf("GPU:Work took %f seconds\n", tGPU_f);
	times.clear();

	bool OMP_correct = CompareArrays<float>(res, resOMP, n);
	bool GPU_correct = CompareArrays<float>(res, resGPU, n);

	if (OMP_correct==true && GPU_correct==true)
	{
		std::cout << "All float calculations correct!:)" << std::endl;
	}
	else
	{
		std::cout << "Something wrong with floats:(" << std::endl;
	}
	printf("---------------\n");
	//------------------------//


	//--------double---------//
	incx = 4; incy = 2;
	n = 100 ;
	printf("Array Size (double) %i\n", n);
	double tCPU_d, tOMP_d, tGPU_d;

	double* x_d = GenerateArray<double>(n);
	double* y_d = GenerateArray<double>(n);
	printf("Arrays Generated\n");
	double* res_d, * resOMP_d, * resGPU_d;


	// SAXPY CPU
	for (int i = 0; i < iterations; i++)
	{
		t1 = omp_get_wtime();
		res_d = daxpy(n, a, x_d, incx, y_d, incy);
		t2 = omp_get_wtime();
		times.push_back(t2 - t1);
	}
	tCPU_d = average(times);
	printf("CPU:Work took %f seconds\n", tCPU_d);

	times.clear();

	// SAXPY OMP
	for (int i = 0; i < iterations; i++)
	{
		t1 = omp_get_wtime();
		resOMP_d = daxpyOMP(n, a, x_d, incx, y_d, incy);
		t2 = omp_get_wtime();
		times.push_back(t2 - t1);
	}
	tOMP_d = average(times);
	printf("OMP:Work took %f seconds\n", tOMP_d);

	times.clear();

	// SAXPY GPU
	for (int i = 0; i < iterations; i++)
	{
		resGPU_d = daxpy_gpu(threads_num, best_block_size, n, a, x_d, incx, y_d, incy, &times);
	}

	tGPU_d = average(times);
	printf("GPU:Work took %f seconds\n", tGPU_d);

	 OMP_correct = CompareArrays<double>(res_d, resOMP_d, n);
	 GPU_correct = CompareArrays<double>(res_d, resGPU_d, n);

	 if (OMP_correct == true && GPU_correct == true)
	{
		std::cout << "All double calculations correct!:)" << std::endl;
	}
	else
	{
		std::cout << "Something wrong with doubles:(" << std::endl;
	}


	delete[] x;
	delete[] y;
}


void getGPUINfo() {

	int deviceCount;
	cudaDeviceProp deviceProp;
	//Сколько устройств CUDA установлено на PC.
	cudaGetDeviceCount(&deviceCount);
	printf("-------------------------------------\n");

	for (int i = 0; i < deviceCount; i++)
	{
		//Получаем информацию об устройстве
		cudaGetDeviceProperties(&deviceProp, i);

		//Выводим иформацию об устройстве
		printf("| Device : %s|\n", deviceProp.name);
		printf("| Shared memory per block: %d    |\n", deviceProp.sharedMemPerBlock);
		printf("| Registers per block: %d        |\n", deviceProp.regsPerBlock);
		printf("| Max threads per block: %d       |\n", deviceProp.maxThreadsPerBlock);
		printf("| Total constant memory: %d      |\n", deviceProp.totalConstMem);
		printf("| Multiprocessor count: %d           |\n", deviceProp.multiProcessorCount);
		printf("Max threads dimensions: x = %d, y = %d, z = %d\n",
			deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);

		printf("Max grid size: x = %d, y = %d, z = %d\n",
			deviceProp.maxGridSize[0],
			deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
		
		printf("-------------------------------------\n");
		printf("                   |                 \n");
		printf("                   |                 \n");
		printf("-------------------------------------\n");
	}

}

int main()
{
	getGPUINfo();

	test();
	/*const int n = 13, incx = 1, incy = 1;
	const float a = 1;

	double* x = GenerateArrayRandom<double>(n);
	double* y = GenerateArrayRandom<double>(n);*/

	//float* resCPU= saxpy(n, a, x, incx, y, incy);
	//float* resGPU = saxpy_gpu(n, a, x, incx, y, incy);

	//double* xd = GenerateArrayRandom<double>(n);
	//double* yd = GenerateArrayRandom<double>(n);

	//double* resCPUd = daxpy(n, a, xd, incx, yd, incy);
	//double* resGPUd = daxpy_gpu(n, a, xd, incx, yd, incy);
	//
	//PrintArray<double>(resCPUd, n);
	//PrintArray<double>(resCPUd, n);

	/*double* res = daxpy(n, a, x, incx, y, incy);
	double* resOMP = daxpyOMP(n, a, x, incx, y, incy);
	PrintArray<double>(res, n);
	PrintArray<double>(resOMP, n);
	std::cout << CompareArrays<double>(res, resOMP, n);*/


	return 0;
}