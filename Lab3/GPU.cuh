#include <stdio.h>
#include <vector>
#include <omp.h>
#include "cuda_runtime.h"

#include "device_launch_parameters.h"

#include"UtilsMat.h"
float* GPUMultiSuperNaive(float* A, float* B, int M, int N, int K,  std::vector<double>* times = nullptr);
float* GPUMultiNaiveOptimized(float* A, float* B, int M, int N, int K, std::vector<double>* times = nullptr);
float* GPUMultiOptimized(float* A, float* B, int M, int N, int K, std::vector<double>* times = nullptr);