#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#define ToRadian(x) ((x) * PI / 180.0f)
#define ToDegree(x) ((x) * 180.0f / PI)

#ifndef PI
#define	PI 3.14159265 
#endif

constexpr auto BlockXMaxThreadNum = 32;
constexpr auto BlockYMaxThreadNum = 32;

__device__ struct point
{
	int x;
	int y;
	inline __device__ void setPoint(int x_, int y_)
	{
		x = x_;
		y = y_;
	}
	inline __device__ point() = default;
	inline __device__ point(int x, int y) :
		x(x), y(y) { }
};

__device__ struct point_double
{
	double x;
	double y;
	inline __device__ void setPoint(double x_, double y_)
	{
		x = x_;
		y = y_;
	}
	inline __device__ point_double() = default;
	inline __device__ point_double(double x, double y) :
		x(x), y(y) { }
};

//得到双三次插值的16个点的坐标
inline __device__ void getNeighborPoints(point neighbors[4][4], double x_, double y_)
{
	//int x = int(x_) ;
	int x = std::floor(x_);
	//int y = int(y_) ;
	int y = std::floor(y_);

	neighbors[0][0].setPoint(x - 1, y - 1);
	neighbors[0][1].setPoint(x, y - 1);
	neighbors[0][2].setPoint(x + 1, y - 1);
	neighbors[0][3].setPoint(x + 2, y - 1);

	neighbors[1][0].setPoint(x - 1, y);
	neighbors[1][1].setPoint(x, y);
	neighbors[1][2].setPoint(x + 1, y);
	neighbors[1][3].setPoint(x + 2, y);

	neighbors[2][0].setPoint(x - 1, y + 1);
	neighbors[2][1].setPoint(x, y + 1);
	neighbors[2][2].setPoint(x + 1, y + 1);
	neighbors[2][3].setPoint(x + 2, y + 1);

	neighbors[3][0].setPoint(x - 1, y + 2);
	neighbors[3][1].setPoint(x, y + 2);
	neighbors[3][2].setPoint(x + 1, y + 2);
	neighbors[3][3].setPoint(x + 2, y + 2);
}

//双三次插值的系数计算
inline __device__ double bicubic(double x)
{
	double a = -0.5;
	if (x < 0)
	{
		x = (-1.0)*x;
	}

	double x2 = x * x;
	double x3 = x * x2;

	if (x <= 1)
	{
		return (2 + a)*x3 - (3 + a)*x2 + 1;
	}
	else if (x < 2 && x>1)
	{
		return a * x3 - 5 * a*x2 + 8 * a*x - 4 * a;
	}
	else
	{
		return 0;
	}
}

//根据坐标得到16个点的权重
inline __device__ void getWeight(double weights[4][4], double x_, double y_)
{
	point neighbors[4][4];
	getNeighborPoints(neighbors, x_, y_);

	double k_x[4];
	double k_y[4];

	for (int i = 0; i < 4; i++)
	{
		k_x[i] = bicubic(x_ - (double)neighbors[0][i].x);
	}

	for (int j = 0; j < 4; j++)
	{
		k_y[j] = bicubic(y_ - (double)neighbors[j][0].y);
	}

	for (int j = 0; j < 4; j++)
	{
		for (int i = 0; i < 4; i++)
		{
			weights[j][i] = k_x[i] * k_y[j];
		}
	}
}


//旋转特有
inline __device__ point_double rotate_point(point_double p, point_double q, double angle)
{
	double x0 = p.x - q.x;
	double y0 = p.y - q.y;

	double x1 = x0 * cos(ToRadian(angle)) - y0 * sin(ToRadian(angle));
	double y1 = x0 * sin(ToRadian(angle)) + y0 * cos(ToRadian(angle));

	x1 += q.x;
	y1 += q.y;

	return point_double(x1, y1);
}

inline __device__ point_double getOriginProjection(point now, point_double center, point_double originCenter,double angle)
{
	point_double now_buffer(now.x, now.y);
	point_double abs_now = rotate_point(now_buffer, center, 360.0 - angle);

	point_double relative(abs_now.x - center.x, abs_now.y - center.y);

	point_double proj_orgin(originCenter.x + relative.x, originCenter.y + relative.y);
	return proj_orgin;
}