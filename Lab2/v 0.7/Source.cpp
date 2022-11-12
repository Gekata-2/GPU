#include <stdio.h>
#include <iostream>
#include "CPU.h"
#include "GPU.cuh"
#include "Utils.h"
#include <vector>


void test()
{
	const std::vector<int> block_sizes{ 8, 16, 32, 64, 128, 256 };


	const int n = 13, incx = 1, incy = 1;
	const float a = 1;

	double* x = GenerateArrayRandom<double>(n);
	double* y = GenerateArrayRandom<double>(n);


	double* res = daxpy(n, a, x, incx, y, incy);
	double* resOMP = daxpyOMP(n, a, x, incx, y, incy);
	PrintArray<double>(res, n);
	PrintArray<double>(resOMP, n);
	std::cout << CompareArrays<double>(res, resOMP, n);
}

































int main()
{

	const int n = 13, incx = 1, incy = 1;
	const float a = 1;

	double* x = GenerateArrayRandom<double>(n);
	double* y = GenerateArrayRandom<double>(n);

	//float* resCPU= saxpy(n, a, x, incx, y, incy);
	//float* resGPU = saxpy_gpu(n, a, x, incx, y, incy);

	//double* xd = GenerateArrayRandom<double>(n);
	//double* yd = GenerateArrayRandom<double>(n);

	//double* resCPUd = daxpy(n, a, xd, incx, yd, incy);
	//double* resGPUd = daxpy_gpu(n, a, xd, incx, yd, incy);
	//
	//PrintArray<double>(resCPUd, n);
	//PrintArray<double>(resCPUd, n);

	double* res = daxpy(n, a, x, incx, y, incy);
	double* resOMP = daxpyOMP(n, a, x, incx, y, incy);
	PrintArray<double>(res, n);
	PrintArray<double>(resOMP, n);
	std::cout << CompareArrays<double>(res, resOMP, n);


	return 0;
}