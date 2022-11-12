#include <stdio.h>
#include <iostream>
#include <vector>
#include <omp.h>

#include "CPU.h"
#include "GPU.cuh"
#include "Utils.h"

double average(std::vector<double> arr)
{
	double res = 0;

	for (int i = 0; i < arr.size(); i++)
		res += arr[i];

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

int findBestBlockSize(const std::vector<int> block_sizes, const int threads_num, const int iterations, const int n, const int incx, const int incy, const double a)
{
	std::vector<double> times, avg_time;


	printf("--------------------------------------\n");
	printf("| Array Size (float): %-14i |\n", n);

	float* x = GenerateArray<float>(n);
	float* y = GenerateArray<float>(n);
	for (int i = 0; i < block_sizes.size(); i++)
	{


		for (int j = 0; j < iterations; j++)
		{
			saxpy_gpu(threads_num, block_sizes[i], n, a, x, incx, y, incy, &avg_time);
		}

		times.push_back(average(avg_time));
		printf("| Blocks: %-3i; Average time: %.5f |\n", block_sizes[i], average(avg_time));


	}
	int best_block_size = block_sizes[findMinElementIdx(times)];
	printf("| Best block size: %-18i|\n", best_block_size);
	printf("--------------------------------------\n");
	printf("                    |                 \n");
	printf("                    |                 \n");
	delete[] x;
	delete[] y;

	return best_block_size;
}

void PrintTable(int n, double tCPU_f, double tOMP_f, double tGPU_f, double tCPU_d, double tOMP_d, double tGPU_d)
{
	printf("----------------------------------------------------------------\n");
	printf("| Technology|   Time   | Acceleration | Array size | Data type |\n");
	printf("|-----------+----------+--------------+------------+-----------|\n");
	printf("|    CPU    | %f |   %f   |  %-8i  |  float    |\n", tCPU_f, tCPU_f / tCPU_f, n);
	printf("|    OMP    | %f |   %f   |  %-8i  |  float    |\n", tOMP_f, tCPU_f / tOMP_f, n);
	printf("|    GPU    | %f |   %f   |  %-8i  |  float    |\n", tGPU_f, tCPU_f / tGPU_f, n);
	printf("| ------------------------------------------------------------ |\n");
	printf("|    CPU    | %f |   %f   |  %-8i  |  double   |\n", tCPU_d, tCPU_d / tCPU_d, n);
	printf("|    OMP    | %f |   %f   |  %-8i  |  double   |\n", tOMP_d, tCPU_d / tOMP_d, n);
	printf("|    GPU    | %f |   %f   |  %-8i  |  double   |\n", tGPU_d, tCPU_d / tGPU_d, n);
	printf("----------------------------------------------------------------\n");
}


// 1 000 000 000 байт на GPU
void DoComputations()
{
	const int threads_num = 50000000; // 400 000
	const int iterations = 10;     // 500

	int n = threads_num;
	const int incx = 1, incy = 2;
	const double a = 0.75;

	const std::vector<int> block_sizes{ 8, 16, 32, 64, 128, 256 };
	int best_block_size = findBestBlockSize(block_sizes, threads_num, iterations, n, incx, incy, a);

	printf("--------------------------------------\n");
	printf("| Array Size (float): %-14i |\n", n);

	double t1, t2;
	double tCPU_f, tOMP_f, tGPU_f;
	std::vector<double> times;

	//--------float---------//
	float* x = GenerateArray<float>(n);
	float* y = GenerateArray<float>(n);

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
	printf("| CPU:Work took %f seconds     |\n", tCPU_f);


	times.clear();
	x = GenerateArray<float>(n);
	y = GenerateArray<float>(n);

	// SAXPY OMP
	for (int i = 0; i < iterations; i++)
	{
		t1 = omp_get_wtime();
		resOMP = saxpyOMP(n, a, x, incx, y, incy);
		t2 = omp_get_wtime();
		times.push_back(t2 - t1);
	}
	tOMP_f = average(times);
	printf("| OMP:Work took %f seconds     | \n", tOMP_f);

	x = GenerateArray<float>(n);
	y = GenerateArray<float>(n);
	times.clear();

	// SAXPY GPU
	for (int i = 0; i < iterations; i++)
	{
		resGPU = saxpy_gpu(threads_num, best_block_size, n, a, x, incx, y, incy, &times);
	}
	tGPU_f = average(times);
	printf("| GPU:Work took %f seconds     | \n", tGPU_f);
	times.clear();

	bool OMP_correct = CompareArrays<float>(res, resOMP, n);
	bool GPU_correct = CompareArrays<float>(res, resGPU, n);

	if (OMP_correct == true && GPU_correct == true)
		std::cout << "| All float calculations correct! :) |" << std::endl;
	else
		std::cout << "| Something wrong with floats :(     |" << std::endl;

	printf("| ---------------                    |\n");
	//------------------------//


	//--------double---------//
	n = threads_num;
	printf("| Array Size (double): %-13i |\n", n);
	double tCPU_d, tOMP_d, tGPU_d;

	double* x_d = GenerateArray<double>(n);
	double* y_d = GenerateArray<double>(n);

	double* res_d, * resOMP_d, * resGPU_d;


	// DAXPY CPU
	for (int i = 0; i < iterations; i++)
	{
		t1 = omp_get_wtime();
		res_d = daxpy(n, a, x_d, incx, y_d, incy);
		t2 = omp_get_wtime();
		times.push_back(t2 - t1);
	}
	tCPU_d = average(times);
	printf("| CPU:Work took %f seconds     |\n", tCPU_d);

	x_d = GenerateArray<double>(n);
	y_d = GenerateArray<double>(n);
	times.clear();

	// DAXPY OMP
	for (int i = 0; i < iterations; i++)
	{
		t1 = omp_get_wtime();
		resOMP_d = daxpyOMP(n, a, x_d, incx, y_d, incy);
		t2 = omp_get_wtime();
		times.push_back(t2 - t1);
	}
	tOMP_d = average(times);
	printf("| OMP:Work took %f seconds     |\n", tOMP_d);

	x_d = GenerateArray<double>(n);
	y_d = GenerateArray<double>(n);
	times.clear();

	// DAXPY GPU
	for (int i = 0; i < iterations; i++)
	{
		resGPU_d = daxpy_gpu(threads_num, best_block_size, n, a, x_d, incx, y_d, incy, &times);
	}

	tGPU_d = average(times);
	printf("| GPU:Work took %f seconds     |\n", tGPU_d);

	OMP_correct = CompareArrays<double>(res_d, resOMP_d, n);
	GPU_correct = CompareArrays<double>(res_d, resGPU_d, n);

	if (OMP_correct == true && GPU_correct == true)
		std::cout << "| All double calculations correct!:) |" << std::endl;
	else
		std::cout << "| Something wrong with doubles:(    |" << std::endl;

	printf("--------------------------------------\n");
	printf("                    |                 \n");
	printf("                    |                 \n");

	PrintTable(n, tCPU_f, tOMP_f, tGPU_f, tCPU_d, tOMP_d, tGPU_d);

	delete[] x, y, res, res, res_d, resOMP, resOMP_d, resGPU, resGPU_d;

}



void PrintGPUSpecs() {

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	printf("--------------------------------------\n");
	printf("| Device : %s |\n", deviceProp.name);
	printf("| Shared memory per block: %i     |\n", deviceProp.sharedMemPerBlock);
	printf("| Registers per block: %i         |\n", deviceProp.regsPerBlock);
	printf("| Max threads per block: %i        |\n", deviceProp.maxThreadsPerBlock);
	printf("| Max blocks in x dim: %i    |\n", deviceProp.maxGridSize[0]);
	printf("| Total constant memory: %i       |\n", deviceProp.totalConstMem);
	printf("| Total global memory: %i    |\n", deviceProp.totalGlobalMem);
	printf("--------------------------------------\n");
	printf("                    |                 \n");
	printf("                    |                 \n");
}

int main()
{
	PrintGPUSpecs();

	DoComputations();

	return 0;
}