#include <algorithm>
#include <omp.h>

#include "CPU.h"

float* saxpy(int n, float a, float* x, int incx, float* y, int incy, bool print)
{
	for (int i = 0; i < n; i++)
	{
		if (i * incx >= n || i * incy >= n)
			break;

		y[i * incy] += a * x[i * incx];
	}

	if (print)
	{
		printf("[");
		for (int i = 0; i < n; i++)
		{
			if (i == n - 1)
				printf("%.2f", x[i]);
			else
				printf("%.2f,", x[i]);
		}
		printf("]\n");

		printf("[");
		for (int i = 0; i < n; i++)
		{
			if (i == n - 1)
				printf("%.2f", y[i]);
			else
				printf("%.2f,", y[i]);
		}
		printf("]\n");
	}
	return y;
}


double* daxpy(int n, double a, double* x, int incx, double* y, int incy)
{
	for (int i = 0; i < n; i++)
	{
		if (i * incx >= n || i * incy >= n)
			break;

		y[i * incy] += a * x[i * incx];
	}

	return y;
}


float* saxpyOMP(int n, float a, float* x, int incx, float* y, int incy)
{
#pragma omp parallel for schedule(static) num_threads(8)
	for (int i = 0; i < n; i++)
	{
		if (i * incx >= n || i * incy >= n)
			break;
		y[i * incy] += a * x[i * incx];
	}
	return y;
}

double* daxpyOMP(int n, double a, double* x, int incx, double* y, int incy)
{
#pragma omp parallel for schedule(static) num_threads(8)
	for (int i = 0; i < n; i++)
	{
		if (i * incx >= n || i * incy >= n)
			break;

		y[i * incy] += a * x[i * incx];
	}
	return y;
}
