#pragma once
#include "UtilsMat.h"

float* SeqMulti(float* A, float* B, int M, int N, int K);

float* OMPMulti(float* A, float* B, int M, int N, int K);
