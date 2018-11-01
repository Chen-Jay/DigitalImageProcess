#pragma once

#include <cmath>

#define ToRadian(x) ((x) * PI / 180.0f)
#define ToDegree(x) ((x) * 180.0f / PI)

#ifndef PI
#define	PI 3.14159265 
#endif

struct ComplexNumber
{
	double realpart;
	double imagecypart;

	ComplexNumber(double realpart, double imagecypart);
	ComplexNumber() { realpart = 0.0; imagecypart = 0.0; };
	void setValue(double realpart, double imagecypart);


	inline ComplexNumber operator +(const ComplexNumber &x)
	{
		return ComplexNumber(realpart + x.realpart, imagecypart + x.imagecypart);
	}

	inline ComplexNumber operator -(const ComplexNumber &x)
	{
		return ComplexNumber(realpart - x.realpart, imagecypart - x.imagecypart);
	}

	inline ComplexNumber operator *(const ComplexNumber &x)
	{
		return ComplexNumber(realpart *x.realpart - imagecypart * x.imagecypart, imagecypart*x.realpart + realpart * x.imagecypart);
	}

	inline ComplexNumber operator *(const int x)
	{
		return ComplexNumber(realpart*x, imagecypart * x);
	}

	inline ComplexNumber operator *(const double x)
	{
		return ComplexNumber(realpart*x, imagecypart * x);
	}

	inline ComplexNumber operator /(const ComplexNumber &x)
	{
		if (x.realpart == 0 && x.imagecypart == 0)
		{
			return ComplexNumber(realpart, imagecypart);
		}
		else
		{
			return ComplexNumber((realpart*x.realpart + imagecypart * x.imagecypart) / (x.realpart*x.realpart + x.imagecypart*x.imagecypart), (imagecypart*x.realpart - realpart * x.imagecypart) / (x.realpart*x.realpart + x.imagecypart*x.imagecypart));
		}
		return ComplexNumber(realpart *x.realpart - imagecypart * x.imagecypart, imagecypart*x.realpart + realpart * x.imagecypart);
	}

	inline double getNorm()
	{
		return std::sqrt(realpart*realpart + imagecypart * imagecypart);
	}
};
