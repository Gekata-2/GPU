#pragma once
#include <stdio.h>
#include <iostream>
#include <random>

template<typename T>
void PrintArray(T* arr, int size)
{
	printf("\n");
	for (int i = 0; i < size; i++)
		std::cout << arr[i] << "; ";

	printf("\n");
}

template<typename T>
T* GenerateArray(int size, bool print = false)
{
	T* arr = new T[size];

	for (int i = 0; i < size; i++)
		arr[i] = (i % 11) - 5;

	if (print)
		PrintArray<T>(arr, size);

	return arr;
}

template<typename T>
float* GenerateArrayZeros(int size)
{
	T* arr = new T[size];
	std::fill(arr, arr + size, 0);

	return arr;
}

template<typename T>
bool CompareArrays(T* a, T* b, int size)
{
	for (int i = 0; i < size; i++)
	{
		if (std::fabs(a[i] - b[i]) > 0.000001)
			return false;
	}

	return true;
}

template<typename T>
T GenerateValue(T min, T max)
{
	std::random_device dev;
	std::mt19937 gen(dev());
	std::uniform_real_distribution<T> uni(min, max);

	return uni(gen);
}

template<typename T>
T* GenerateArrayRandom(int size, bool print = false)
{
	T* arr = new T[size];

	for (int i = 0; i < size; i++)
		arr[i] = GenerateValue<T>(-100, 101);

	if (print)
		PrintArray<T>(arr, size);

	return arr;
}
