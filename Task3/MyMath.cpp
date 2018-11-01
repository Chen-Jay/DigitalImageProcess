#include "MyMath.h"

ComplexNumber::ComplexNumber(double realpart, double imagecypart)
{
	this->realpart = realpart;
	this->imagecypart = imagecypart;
}

void ComplexNumber::setValue(double realpart, double imagecypart)
{
	this->realpart = realpart;
	this->imagecypart = imagecypart;
}
