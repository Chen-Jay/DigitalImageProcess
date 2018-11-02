#pragma once
#include "stdafx.h"
class MyImage
{
private:
	CImage* myImg;
	byte* data;
	int width;
	int height;
	int pixelSize;
	int pitch;
	
public:
	MyImage(CImage * cImage);
	MyImage();

	void setImage(CImage * cImage);

	inline int getWidth()
	{
		return width;
	}

	inline int getHeight()
	{
		return height;
	}

	inline bool isColorful()
	{
		return pixelSize == 1 ? false : true;
	}

	inline int getBPP()
	{
		return myImg->GetBPP();
	}

	inline byte readImage(int x, int y)
	{
		return *(data + y * pitch + x * pixelSize);
	}

	inline byte readImage_R(int x, int y) 
	{
		return *(data + y * pitch + x * pixelSize);
	}

	inline byte readImage_G(int x, int y) 
	{
		return *(data + y * pitch + x * pixelSize + 1);
	}

	inline byte readImage_B(int x, int y)
	{
		return *(data + y * pitch + x * pixelSize + 2);
	}

	inline void writeImage(int x, int y, byte value)
	{
		*(data + y * pitch + x * pixelSize) = value;
	}

	inline void writeImage_R(int x, int y, byte value) 
	{
		*(data + y * pitch + x * pixelSize) = value;
	}

	inline void writeImage_G(int x, int y, byte value)
	{
		*(data + y * pitch + x * pixelSize + 1) = value;
	}

	inline void writeImage_B(int x, int y, byte value)
	{
		*(data + y * pitch + x * pixelSize + 2) = value;
	}

};