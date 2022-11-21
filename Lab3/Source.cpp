#include <stdio.h>
#include <omp.h>

#include "UtilsMat.h"
#include "GPU.cuh"
#include "CPU.h"


void PrintTable(double t_SEQ, double t_OMP, double t_SupN, double t_NOpt, double t_Opt)
{
	printf("-----------------------------------------\n");
	printf("|    Method   |   Time   | Acceleration |\n");
	printf("|-------------+----------+--------------|\n");
	printf("|     SEQ     | %-8.5f |   %-8.3f   |  \n", t_SEQ, t_SEQ / t_SEQ);
	printf("|     OMP     | %-8.5f |   %-8.3f   | \n", t_OMP, t_SEQ / t_OMP);
	printf("| SUPER_NAIVE | %-8.5f |   %-8.3f   | \n", t_SupN, t_SEQ / t_SupN);
	printf("|  NAIVE_OPT  | %-8.5f |   %-8.3f   | \n", t_NOpt, t_SEQ / t_NOpt);
	printf("|  OPT_SHARED | %-8.5f |   %-8.3f   | \n", t_Opt, t_SEQ / t_Opt);
	printf("-----------------------------------------\n");
}



int main()
{

	int size =75 * BLOCK_SIZE;
	printf("Size is [%i,%i]\n", size, size);
	printf("\n");
	int M = size, N = size, K = size;
	float* mat = GenerateMatrix(M, N);
	float* mat2 = GenerateMatrix(N, K);
	std::vector<double> times;

	double t1, t2;
	double t_SEQ, t_OMP, t_SupN, t_NOpt, t_Opt;


	t1 = omp_get_wtime();
	float* resSEQ = SeqMulti(mat, mat2, M, N, K);
	t2 = omp_get_wtime();
	t_SEQ = t2 - t1;
	std::cout << "Time seq = " << t_SEQ << std::endl;



	t1 = omp_get_wtime();
	float* resOMP = OMPMulti(mat, mat2, M, N, K);
	t2 = omp_get_wtime();
	t_OMP = t2 - t1;
	std::cout << "Time omp = " << t_OMP << std::endl;

	
	float* resGPUSuperNaive = GPUMultiSuperNaive(mat, mat2, M, N, K, &times);
	t_SupN = times[0];
	std::cout << "Time GPUMultiSuperNaive = " << t_SupN << std::endl;
	times.clear();



	float* resGPUNaiveOprimized = GPUMultiNaiveOptimized(mat, mat2, M, N, K, &times);
	t_NOpt = times[0];
	std::cout << "Time GPUMultiNaiveOptimized = " << t_NOpt << std::endl;
	times.clear();


	float* resGPUOptimized = GPUMultiOptimized(mat, mat2, M, N, K, &times);
	t_Opt = times[0];
	std::cout << "Time GPUMultiOptimized = " << t_Opt << std::endl;
	times.clear();

	bool OMP_Correct = CompareMatricies(resSEQ, resOMP, M, K);
	bool GPUSuperNaive_Correct = CompareMatricies(resSEQ, resGPUSuperNaive, M, K);
	bool GPUNaiveOprimized_Correct = CompareMatricies(resSEQ, resGPUNaiveOprimized, M, K);
	bool GPUOptimized_Correct = CompareMatricies(resSEQ, resGPUOptimized, M, K);

	if (OMP_Correct && GPUSuperNaive_Correct && GPUNaiveOprimized_Correct && GPUOptimized_Correct)
		printf("ALL MATRICIES ARE EQUAL  :)\n");
	else
		std::cout << "Something worng " << OMP_Correct << " " << GPUSuperNaive_Correct << " "
		<< GPUNaiveOprimized_Correct << " " << GPUOptimized_Correct << std::endl;

	printf("\n");
	PrintTable(t_SEQ, t_OMP, t_SupN, t_NOpt, t_Opt);

	return 0;
}