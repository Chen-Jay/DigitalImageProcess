#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "ZoomOrRotate.cuh"

using byte = unsigned char;

inline void checkCudaErrors(cudaError err) //cuda error handle function
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA Runtime API error:%s.\n", cudaGetErrorString(err));
		return;
	}
}

__global__ void Rotate_kernel(byte *GPU_source, byte *GPU_result, int HandleWidth, int HandleHeight, int SourceWidth, int SourceHeight, double angle)
{
	//x，y坐标即为对应的线程在整个thread阵里面的x，y坐标
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= HandleWidth || y >= HandleHeight)
	{
		return;
	}

	//映射回原图中的坐标
	point_double originCenter(SourceWidth / 2.0, SourceHeight / 2.0);
	point_double nowCenter(HandleWidth / 2.0, HandleHeight / 2.0);

	point now(x, y);
	point_double origin = getOriginProjection(now, nowCenter, originCenter, angle);

	if (origin.x >= SourceWidth || origin.y >= SourceHeight || origin.x < 0 || origin.y < 0)
	{
		return;
	}

	if (origin.x < 1 || origin.y < 1 || origin.x >= SourceWidth - 2 || origin.y >= SourceHeight - 2)
	{
		//图像边缘使用最近邻插值法
		GPU_result[y*HandleWidth + x] = GPU_source[(int)origin.y*SourceWidth + (int)origin.x];
	}
	else
	{
		//获得临近点
		point neighbors[4][4];
		getNeighborPoints(neighbors, origin.x, origin.y);

		//获得权值
		double weights[4][4];
		getWeight(weights, origin.x, origin.y);

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

extern "C" void Rotate_host(byte* source, byte* result_buf, int HandleWidth, int HandleHeight, int SourceWidth, int SourceHeight, double angle)
{
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

	Rotate_kernel << < DimGrid, DimBlock >> > (GPU_source, GPU_result, HandleWidth, HandleHeight, SourceWidth, SourceHeight, angle);

	checkCudaErrors(cudaMemcpy(result_buf, GPU_result, sizeof(byte)*HandleWidth*HandleHeight, cudaMemcpyDeviceToHost));

	cudaFree(GPU_source);
	cudaFree(GPU_result);
}

