#include <omp.h>

#include "CPU.h"

float* SeqMulti(float* A, float* B, int M, int N, int K) {

	float* C = GenerateMatrixZeros(M, K);

	for (int i = 0; i < M; i++) {
		for (int j = 0; j < K; j++) {
			for (int l = 0; l < N; l++) {
				C[i * K + j] += A[i * N + l] * B[l * K + j];
			}
		}
	}
	return C;
}

float* OMPMulti(float* A, float* B, int M, int N, int K) {

	float* C = GenerateMatrixZeros(M, K);

#pragma omp parallel for schedule(static) num_threads(N_THREADS)
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < K; j++) {
			for (int l = 0; l < N; l++) {
				C[i * K + j] += A[i * N + l] * B[l * K + j];
			}
		}
	}

	return C;
}