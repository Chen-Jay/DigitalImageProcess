#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#ifndef PI
#define	PI 3.14159265 
#endif

//每个Block的最大线程数为1024
constexpr auto BlockXMaxThreadNum = 16;
constexpr auto BlockYMaxThreadNum = 16;
constexpr auto BlockZMaxThreadNum = 4;

__device__ struct ComplexNumber
{
	double realpart;
	double imagecypart;

	inline __device__ ComplexNumber(double realpart_, double imagecypart_) {
		realpart = realpart_;
		imagecypart = imagecypart_;
	}
	inline __device__ ComplexNumber() { realpart = 0.0; imagecypart = 0.0; };

	inline __device__ ComplexNumber operator +(const ComplexNumber &x)
	{
		return ComplexNumber(realpart + x.realpart, imagecypart + x.imagecypart);
	}

	inline __device__ ComplexNumber operator -(const ComplexNumber &x)
	{
		return ComplexNumber(realpart - x.realpart, imagecypart - x.imagecypart);
	}

	inline __device__ ComplexNumber operator *(const double x)
	{
		return ComplexNumber(realpart*x, imagecypart * x);
	}

	inline __device__ double getNorm()
	{
		return std::sqrt(realpart*realpart + imagecypart * imagecypart);
	}
};
