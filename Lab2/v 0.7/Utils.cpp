#include "Utils.h"




float* GenerateArrayZeros(int size)
{
	float* arr = new float[size];
	std::fill(arr, arr + size, 0);
	return arr;

}

double* GenerateArrayZerosD(int size)
{
	double* arr = new double[size];
	std::fill(arr, arr + size, 0);
	return arr;

}


int GetBufferSize(int n, int inc)
{
	return 1 + (n - 1) * std::abs(inc);
}

