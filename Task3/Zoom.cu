#ifndef __ZOOM_CU_
#define __ZOOM_CU_
#endif // !__ZOOM_CU_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

//#define bicubic(x,y) {double a = -0.5;if (x < 0){x = (-1.0)*x;}double x2 = x * x;double x3 = x * x2;if (x <= 1){return (2 + a)*x3 - (3 + a)*x2 + 1;}\
//else if (x < 2 && x>1){return a * x3 - 5 * a*x2 + 8 * a*x - 4 * a;}else{return 0;}}

using byte = unsigned char;

constexpr auto BlockXMaxThreadNum = 32;
constexpr auto BlockYMaxThreadNum = 32;

inline void checkCudaErrors(cudaError err) //cuda error handle function
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA Runtime API error:%s.\n", cudaGetErrorString(err));
		return;
	}
}

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
	inline __device__ point(int x_, int y_) {
		x = x_;
		y = y_;
	}
};


//得到双三次插值的16个点的坐标
__device__ void getNeighborPoints(point neighbors[4][4], double x_, double y_)
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
__device__ double bicubic(double x)
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
__device__ void getWeight(double weights[4][4], double x_, double y_)
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



__global__ void Zoom_kernel(byte *GPU_source, byte *GPU_result, int HandleWidth, int HandleHeight, int SourceWidth, int SourceHeight, double param)
{
	//x，y坐标即为对应的线程在整个thread阵里面的x，y坐标
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= HandleWidth || y >= HandleHeight)
	{
		return;
	}

	//映射回原图中的坐标
	double x_origin = x / param;
	double y_origin = y / param;

	if (x_origin >= SourceWidth || y_origin >= SourceHeight)
	{
		return;
	}

	if (x_origin <= 1 || y_origin <= 1 || x_origin >= SourceWidth - 2 || y_origin >= SourceHeight - 2)
	{
		//图像边缘使用最近邻插值法
		GPU_result[y*HandleWidth + x] = GPU_source[(int)y_origin*SourceWidth + (int)x_origin];
	}
	else
	{
		//获得临近点
		point neighbors[4][4];
		getNeighborPoints(neighbors, x_origin, y_origin);
		//获得权值
		double weights[4][4];
		getWeight(weights, x_origin, y_origin);
		
		double value = 0;
		for (int y = 0; y < 4; y++)
		{
			for (int x = 0; x < 4; x++)
			{
				value += GPU_source[neighbors[x][y].y*SourceWidth + neighbors[x][y].x] * weights[x][y];
			}
		}

		value = value < 0 ? 0 : value;
		value = value > 255 ? 255 : value;
		GPU_result[y*HandleWidth + x] = (byte)value;
	}
}

extern "C" void Zoom_host(byte* source, byte* result_buf, int HandleWidth, int HandleHeight, int SourceWidth, int SourceHeight)
{
	//计算缩放比
	double param = (double)HandleHeight / SourceHeight;

	//指定GPU分配空间方式
	dim3 DimBlock(BlockXMaxThreadNum, BlockYMaxThreadNum);
	dim3 DimGrid(HandleWidth / BlockXMaxThreadNum + 1, HandleHeight / BlockYMaxThreadNum + 1);

	//用来在显存中进行操作的指针
	byte* GPU_source;
	byte* GPU_result;

	//在显存中为原图像和工作区分配空间
	checkCudaErrors(cudaMalloc((void **)&GPU_source, sizeof(byte)*SourceWidth*SourceHeight));
	checkCudaErrors(cudaMalloc((void **)&GPU_result, sizeof(byte)*HandleWidth*HandleHeight));

	checkCudaErrors(cudaMemcpy(GPU_source, source, sizeof(byte)*SourceHeight*SourceWidth, cudaMemcpyHostToDevice));

	Zoom_kernel << < DimGrid, DimBlock >> > (GPU_source, GPU_result, HandleWidth, HandleHeight, SourceWidth, SourceHeight, param);

	checkCudaErrors(cudaMemcpy(result_buf, GPU_result, sizeof(byte)*HandleWidth*HandleHeight, cudaMemcpyDeviceToHost));

	cudaFree(GPU_source);
	cudaFree(GPU_result);
}

