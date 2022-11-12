#pragma once
#include <stdio.h>

#include "Utils.h"

float* saxpy(int n, float a, float* x, int incx, float* y, int incy, bool print = false);
double* daxpy(int n, double a, double* x, int incx, double* y, int incy);


float* saxpyOMP(int n, float a, float* x, int incx, float* y, int incy);
double* daxpyOMP(int n, double a, double* x, int incx, double* y, int incy);
