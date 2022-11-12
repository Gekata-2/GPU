#include "CPU.h"
#include <algorithm>
#include <omp.h>




float* saxpy(int n, float a, float* x, int incx, float* y, int incy, bool print)
{
	int sizeX = GetBufferSize(n, incx), sizeY = GetBufferSize(n, incy);

	float* xBuff = new float[sizeX];
	float* yBuff = new float[sizeY];

	std::fill(xBuff, xBuff + sizeX, 0);
	std::fill(yBuff, yBuff + sizeY, 0);

	std::memcpy(xBuff, x, n * sizeof(float));
	std::memcpy(yBuff, y, n * sizeof(float));

	for (int i = 0; i < n; i++)
	{
		yBuff[i * incy] += a * xBuff[i * incx];
	}

	if (print)
	{
		printf("[");
		for (int i = 0; i < sizeX; i++)
		{
			if (i == sizeX - 1)
				printf("%.2f", xBuff[i]);
			else
				printf("%.2f,", xBuff[i]);
		}
		printf("]\n");

		printf("[");
		for (int i = 0; i < sizeY; i++)
		{
			if (i == sizeY - 1)
				printf("%.2f", yBuff[i]);
			else
				printf("%.2f,", yBuff[i]);
		}
		printf("]\n");
	}
	return yBuff;
}


double* daxpy(int n, double a, double* x, int incx, double* y, int incy, bool print)
{
	int sizeX = GetBufferSize(n, incx), sizeY = GetBufferSize(n, incy);

	double* xBuff = new double[sizeX];
	double* yBuff = new double[sizeY];

	std::fill(xBuff, xBuff + sizeX, 0);
	std::fill(yBuff, yBuff + sizeY, 0);

	std::memcpy(xBuff, x, n * sizeof(double));
	std::memcpy(yBuff, y, n * sizeof(double));

	for (int i = 0; i < n; i++)
	{
		yBuff[i * incy] += a * xBuff[i * incx];
	}

	if (print)
	{
		printf("[");
		for (int i = 0; i < sizeX; i++)
		{
			if (i == sizeX - 1)
				printf("%.2f", xBuff[i]);
			else
				printf("%.2f,", xBuff[i]);
		}
		printf("]\n");

		printf("[");
		for (int i = 0; i < sizeY; i++)
		{
			if (i == sizeY - 1)
				printf("%.2f", yBuff[i]);
			else
				printf("%.2f,", yBuff[i]);
		}
		printf("]\n");
	}
	return yBuff;
}



float* saxpyOMP(int n, float a, float* x, int incx, float* y, int incy)
{
	int sizeX = GetBufferSize(n, incx), sizeY = GetBufferSize(n, incy);

	float* xBuff = new float[sizeX];
	float* yBuff = new float[sizeY];

	std::fill(xBuff, xBuff + sizeX, 0);
	std::fill(yBuff, yBuff + sizeY, 0);

	std::memcpy(xBuff, x, n * sizeof(float));
	std::memcpy(yBuff, y, n * sizeof(float));



#pragma omp parallel for shared(xBuff,yBuff,incy,incx,a)schedule(static)
	for (int i = 0; i < n; i++)
	{
		yBuff[i * incy] += a * xBuff[i * incx];
	}

	return yBuff;
}


double* daxpyOMP(int n, double a, double* x, int incx, double* y, int incy)
{
	int sizeX = GetBufferSize(n, incx), sizeY = GetBufferSize(n, incy);

	double* xBuff = new double[sizeX];
	double* yBuff = new double[sizeY];

	std::fill(xBuff, xBuff + sizeX, 0);
	std::fill(yBuff, yBuff + sizeY, 0);

	std::memcpy(xBuff, x, n * sizeof(double));
	std::memcpy(yBuff, y, n * sizeof(double));



#pragma omp parallel for shared(xBuff,yBuff,incy,incx,a)schedule(static)
	for (int i = 0; i < n; i++)
	{
		yBuff[i * incy] += a * xBuff[i * incx];
	}

	return yBuff;
}
