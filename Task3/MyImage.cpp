#include "stdafx.h"
#include "MyImage.h"

MyImage::MyImage(CImage * cImage)
{
	myImg = cImage;
	data = (byte*)myImg->GetBits();
	width = cImage->GetWidth();
	height = cImage->GetHeight();
	pixelSize = cImage->GetBPP() / 8;
	pitch = cImage->GetPitch();
}

MyImage::MyImage()
{
}

void MyImage::setImage(CImage * cImage)
{
	myImg = cImage;
	data = (byte*)myImg->GetBits();
	width = cImage->GetWidth();
	height = cImage->GetHeight();
	pixelSize = cImage->GetBPP() / 8;
	pitch = cImage->GetPitch();
}


