#include "cuda_runtime.h"
#include "device_launch_parameters.h"





float* saxpy_gpu(int n, float a, float* x, int incx, float* y, int incy, bool print = false);
double* daxpy_gpu(int n, double a, double* x, int incx, double* y, int incy, bool print = false);

