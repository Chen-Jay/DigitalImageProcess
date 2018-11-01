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

//�̵߳Ĺ�����
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




//ͼ����ĸ����㷨ʵ��
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

//���ŵĲ���/����
struct zoomParam
{
	double scale;

	void getNeighborPoints(point<int> n[4][4], double x_,double y_);	//�õ�˫���β�ֵ��16���������
	void getWeight(double n[4][4], double x, double y);	//��������õ�16�����Ȩ��
	double bicubic(double x);	//˫���β�ֵ��ϵ������
};

//��ת�Ĳ���/����
struct rotateParam
{
	double angle;
	int originWidth;
	int originHeight;

	point<double> originCenter;
	point<double> getOriginProjection(point<int>now, point<double>center,rotateParam* param);
	void getNeighborPoints(point<int> neighbors[4][4], double x_, double y_);	//�õ�˫���β�ֵ��16���������
	void getWeight(double n[4][4], double x, double y);	//��������õ�16�����Ȩ��
	double bicubic(double x);	//˫���β�ֵ��ϵ������
};

//����Ҷ�任�Ĳ���
struct DFTParam
{
	
};

//��˹�����Ĳ���/����
struct GaussNoiseParam
{
	double means;
	double variance;

	inline double BoxMuller(double random_a,double random_b,double variance)
	{
		return variance*std::sqrt(-2 * std::log(random_a))*cos(2 * PI*random_b);
	}
};

//��ֵ�˲��Ĳ���/����
struct MeanFilterParam
{
	void getNeighborPoints(point<int> neighbors[3][3], int x, int y);	//�õ�ͼ���Ӧ������Χ�������Լ���9���������
};

//��˹�˲��Ĳ���/����
struct GaussFilterParam
{
	double variance;
	void getNeighborPoints(point<int> neighbors[3][3], int x, int y);
	void getGaussFactor(double values[3][3],double variance);
};

//ά���˲��Ĳ���/����
struct WienerFilterParam
{
	double noise_variance;
	double noise_variance_R;
	double noise_variance_G;
	double noise_variance_B;
	void getNeighborPoints(point<int> neighbors[3][3], int x, int y);
};