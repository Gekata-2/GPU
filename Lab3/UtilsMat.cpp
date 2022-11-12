#include "UtilsMat.h"

float* GenerateMatrixZeros(int M, int N)
{
	int size = M * N;
	float* A = new float[size];
	for (int i = 0; i < size; i++)
		A[i] = 0;

	return A;
}

float* GenerateMatrix(int M, int N)
{
	int size = M * N;
	float* A = new float[size];

	for (int i = 0; i < size; i++)
		A[i] = (i + 10) % 11;

	return A;
}

void PrintMatrix1D(float* A, int M, int N)
{
	printf("[ ");
	for (int i = 0; i < M * N; i++)
	{
		if (i == M * N - 1)
			printf("%.4f ", A[i]);
		else
			printf("%.4f, ", A[i]);
	}

	printf("]\n");

}

void PrintMatrix2D(float* A, int M, int N)
{
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%-6.2f ", A[i * N + j]);
		}
		printf("\n");
	}

	printf("\n");
}

bool CompareMatricies(float* A, float* B, int M, int K)
{
	for (int i = 0; i < M * K; i++)
	{
		if (std::fabs(A[i] - B[i]) > 0.000001)
			return false;
	}

	return true;
}

void T(float* A, int M, int N)
{

	float tmp = 0;

	float* C = new float[N * M];

	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			C[j * M + i] = A[i * N + j];
		}
	}
	for (int i = 0; i < N * M; i++)
	{
		A[i] = C[i];

	}
	delete[] C;
}

int summ(float* A, int size) {
	float s = 0;
	for (int i = 0; i < size; i++)
	{
		s += A[i];
	}
	printf("summ = %f \n", s);
	return s;
}