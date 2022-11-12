#pragma once

#include <stdio.h>
#include <iostream>
#include <random>

#define N_THREADS 4
#define BLOCK_SIZE 16

float* GenerateMatrixZeros(int M, int N);

float* GenerateMatrix(int M, int N);

void PrintMatrix1D(float* A, int M, int N);

void PrintMatrix2D(float* A, int M, int N);

bool CompareMatricies(float* A, float* B, int M, int K);
void T(float* A, int M, int N);

int summ(float* A, int size);
