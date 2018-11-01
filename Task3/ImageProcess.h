#pragma once
#include "stdafx.h"
#include "MyMath.h"

enum class Operation
{
	Zoom, 
	Rotate,
	DFT,
	GaussNoise,
	MeanFilter,
	GaussFilter,
	WienerFilter
};

//线程的工作区
struct ThreadWorkSpace
{
	CImage * img;
	CImage * handled;
	long long  startIndex;
	long long endIndex;

	CWnd* window;
	void *ctx;
};


template<typename T>
struct point
{
	T x;
	T y;
	inline void setPoint(T x_, T y_)
	{
		x = x_;
		y = y_;
	}
	inline point() = default;
	inline point(T x, T y):
		x(x), y(y) { }
};

point<double> rotate_point(point<double> p, point<double> q, double angle);




//图像处理的各种算法实现
namespace ImageProcess
{
	 UINT zoom(LPVOID param);
	 UINT rotate(LPVOID param);
	 UINT DFT(LPVOID param);
	 UINT GaussNoise(LPVOID param);
	 UINT MeanFilter(LPVOID param);
	 UINT GaussFilter(LPVOID param);
	 UINT WienerFilter(LPVOID param);
}

//缩放的参数/方法
struct zoomParam
{
	double scale;

	void getNeighborPoints(point<int> n[4][4], double x_,double y_);	//得到双三次插值的16个点的坐标
	void getWeight(double n[4][4], double x, double y);	//根据坐标得到16个点的权重
	double bicubic(double x);	//双三次插值的系数计算
};

//旋转的参数/方法
struct rotateParam
{
	double angle;
	int originWidth;
	int originHeight;

	point<double> originCenter;
	point<double> getOriginProjection(point<int>now, point<double>center,rotateParam* param);
	void getNeighborPoints(point<int> neighbors[4][4], double x_, double y_);	//得到双三次插值的16个点的坐标
	void getWeight(double n[4][4], double x, double y);	//根据坐标得到16个点的权重
	double bicubic(double x);	//双三次插值的系数计算
};

//傅里叶变换的参数
struct DFTParam
{
	
};

//高斯噪声的参数/方法
struct GaussNoiseParam
{
	double means;
	double variance;

	inline double BoxMuller(double random_a,double random_b,double variance)
	{
		return variance*std::sqrt(-2 * std::log(random_a))*cos(2 * PI*random_b);
	}
};

//均值滤波的参数/方法
struct MeanFilterParam
{
	void getNeighborPoints(point<int> neighbors[3][3], int x, int y);	//得到图像对应像素周围（包括自己）9个点的坐标
};

//高斯滤波的参数/方法
struct GaussFilterParam
{
	double variance;
	void getNeighborPoints(point<int> neighbors[3][3], int x, int y);
	void getGaussFactor(double values[3][3],double variance);
};

//维纳滤波的参数/方法
struct WienerFilterParam
{
	double noise_variance;
	double noise_variance_R;
	double noise_variance_G;
	double noise_variance_B;
	void getNeighborPoints(point<int> neighbors[3][3], int x, int y);
};