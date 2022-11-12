#include "GPU.cuh"
/*
	 -------------       ------
	 |           |       |    |         ------
  M  |     A     |   x   |    |         |    |
	 |           |       |  B |   =  M  | C  |
	 -------------     N |    |         |    |
		   N             |    |         ------
						 |    |            K
						 ------
							K
*/

__global__ void kernel__super_naive(float* A, float* B, float* C, int M, int N, int K) {


	int global_x = blockIdx.x * blockDim.x + threadIdx.x;
	int global_y = blockIdx.y * blockDim.y + threadIdx.y;

	for (int l = 0; l < N; l++)
	{
		//C[i * K + j] += A[i * N + l] * B[l * K + j];
		// i = global_x;
		// j = global_y;
		// l = l;
		C[global_x * K + global_y] += A[global_x * N + l] * B[l * K + global_y];
	}
}

float* GPUMultiSuperNaive(float* A, float* B, int M, int N, int K, std::vector<double>* times)
{
	cudaError_t error;

	float* C = GenerateMatrixZeros(M, K);

	float* A_gpu;
	float* B_gpu;
	float* C_gpu;

	//----Malloc------//
	error = cudaMalloc((void**)&A_gpu, M * N * sizeof(float));
	if (error)
		printf("cudaMalloc A error : %i \n", error);

	error = cudaMalloc((void**)&B_gpu, N * K * sizeof(float));
	if (error)
		printf("cudaMalloc B error : %i \n", error);

	error = cudaMalloc((void**)&C_gpu, M * K * sizeof(float));
	if (error)
		printf("cudaMalloc C error : %i \n", error);


	//------Memcpy----//
	error = cudaMemcpy(A_gpu, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
	if (error)
		printf("cudaMemcpy A error: %i \n", error);

	error = cudaMemcpy(B_gpu, B, N * K * sizeof(float), cudaMemcpyHostToDevice);
	if (error)
		printf("cudaMemcpy B error: %i \n", error);

	error = cudaMemcpy(C_gpu, C, M * K * sizeof(float), cudaMemcpyHostToDevice);
	if (error)
		printf("cudaMemcpy C error: %i \n", error);

	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 num_blocks(M / BLOCK_SIZE, K / BLOCK_SIZE);
	

	double t1, t2;
	t1 = omp_get_wtime();
	kernel__super_naive << <num_blocks, block_size >> > (A_gpu, B_gpu, C_gpu, M, N, K);
	error = cudaDeviceSynchronize();
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", error);
	}
	t2 = omp_get_wtime();
	if (times != nullptr)
		(*times).push_back(t2 - t1);

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
	}


	error = cudaMemcpy(C, C_gpu, M * K * sizeof(float), cudaMemcpyDeviceToHost);
	if (error)
		printf("cudaMemcpy device to host error: %i \n", error);

	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);

	return C;
}



__global__ void kernel_naive_optimized(float* A, float* B, float* C, int M, int N, int K) {


	int global_x = blockIdx.x * blockDim.x + threadIdx.x;
	int global_y = blockIdx.y * blockDim.y + threadIdx.y;

	for (int l = 0; l < N; l++)
	{
		C[global_x * K + global_y] += A[l * M + global_x] * B[l * K + global_y];
	}
}

float* GPUMultiNaiveOptimized(float* A, float* B, int M, int N, int K, std::vector<double>* times)
{
	cudaError_t error;

	float* C = GenerateMatrixZeros(M, K);
	T(A, M, N);
	//PrintMatrix2D(A, N, M);

	float* A_gpu;
	float* B_gpu;
	float* C_gpu;

	//----Malloc------//
	error = cudaMalloc((void**)&A_gpu, M * N * sizeof(float));
	if (error)
		printf("cudaMalloc A error : %i \n", error);

	error = cudaMalloc((void**)&B_gpu, N * K * sizeof(float));
	if (error)
		printf("cudaMalloc B error : %i \n", error);

	error = cudaMalloc((void**)&C_gpu, M * K * sizeof(float));
	if (error)
		printf("cudaMalloc C error : %i \n", error);


	//------Memcpy----//
	error = cudaMemcpy(A_gpu, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
	if (error)
		printf("cudaMemcpy A error: %i \n", error);

	error = cudaMemcpy(B_gpu, B, N * K * sizeof(float), cudaMemcpyHostToDevice);
	if (error)
		printf("cudaMemcpy B error: %i \n", error);

	error = cudaMemcpy(C_gpu, C, M * K * sizeof(float), cudaMemcpyHostToDevice);
	if (error)
		printf("cudaMemcpy C error: %i \n", error);

	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 num_blocks(M / BLOCK_SIZE, K / BLOCK_SIZE);
	
	double t1, t2;
	t1 = omp_get_wtime();

	kernel_naive_optimized << <num_blocks, block_size >> > (A_gpu, B_gpu, C_gpu, M, N, K);
	error = cudaDeviceSynchronize();
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", error);
	}

	t2 = omp_get_wtime();
	if (times != nullptr)
		(*times).push_back(t2 - t1);


	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
	}

	error = cudaMemcpy(C, C_gpu, M * K * sizeof(float), cudaMemcpyDeviceToHost);
	if (error)
		printf("cudaMemcpy device to host error: %i \n", error);

	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
	T(A, M, N);
	return C;
}



__global__ void kernel_optimized(float* A, float* B, float* C, int M, int N, int K) {

	int b_x = blockIdx.x;
	int b_y = blockIdx.y;

	int t_x = threadIdx.x;
	int t_y = threadIdx.y;



	int A_start = b_x * BLOCK_SIZE;
	int B_start = b_y * BLOCK_SIZE;

	int A_step = BLOCK_SIZE * M;
	int B_step = BLOCK_SIZE * K;

	int end = N;

	float sum = 0.0f;
	__shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];
	for (int i = A_start, j = B_start, k = 0; k < end; i += A_step, j += B_step, k += BLOCK_SIZE)
	{
		A_shared[t_x][t_y] = A[i + t_x * M + t_y];
		B_shared[t_x][t_y] = B[j + t_x * K + t_y];

		__syncthreads();
		for (int l = 0; l < BLOCK_SIZE; l++)
		{
			sum += A_shared[l][t_x] * B_shared[l][t_y];
		}
		__syncthreads();
	}

	int block_C_start = b_x * K * BLOCK_SIZE + b_y * BLOCK_SIZE;
	int idx = block_C_start + t_x * K + t_y;


	C[idx] = sum;
}

float* GPUMultiOptimized(float* A, float* B, int M, int N, int K, std::vector<double>* times)
{
	cudaError_t error;

	float* C = GenerateMatrixZeros(M, K);
	T(A, M, N);

	float* A_gpu;
	float* B_gpu;
	float* C_gpu;

	//----Malloc------//
	error = cudaMalloc((void**)&A_gpu, M * N * sizeof(float));
	if (error)
		printf("cudaMalloc A error : %i \n", error);

	error = cudaMalloc((void**)&B_gpu, N * K * sizeof(float));
	if (error)
		printf("cudaMalloc B error : %i \n", error);

	error = cudaMalloc((void**)&C_gpu, M * K * sizeof(float));
	if (error)
		printf("cudaMalloc C error : %i \n", error);


	//------Memcpy----//
	error = cudaMemcpy(A_gpu, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
	if (error)
		printf("cudaMemcpy A error: %i \n", error);

	error = cudaMemcpy(B_gpu, B, N * K * sizeof(float), cudaMemcpyHostToDevice);
	if (error)
		printf("cudaMemcpy B error: %i \n", error);

	error = cudaMemcpy(C_gpu, C, M * K * sizeof(float), cudaMemcpyHostToDevice);
	if (error)
		printf("cudaMemcpy C error: %i \n", error);

	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 num_blocks(M / BLOCK_SIZE, K / BLOCK_SIZE, 1);
	
	double t1, t2;
	t1 = omp_get_wtime();

	kernel_optimized << <num_blocks, block_size >> > (A_gpu, B_gpu, C_gpu, M, N, K);
	error = cudaDeviceSynchronize();
	if (error != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", error);
	}

	t2 = omp_get_wtime();
	if (times != nullptr)
		(*times).push_back(t2 - t1);


	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(error));
	}

	error = cudaMemcpy(C, C_gpu, M * K * sizeof(float), cudaMemcpyDeviceToHost);
	if (error)
		printf("cudaMemcpy device to host error: %i \n", error);

	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
	T(A, M, N);
	
	return C;
}

