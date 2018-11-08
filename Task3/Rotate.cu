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

__global__ void Rotate_kernel(byte *GPU_source, byte *GPU_result, int HandleWidth, int HandleHeight, int SourceWidth, int SourceHeight, double angle,
	int source_pitch, int source_pixelSize, int handle_pitch, int handle_pixelSize)
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
		GPU_result[(HandleHeight - 1 - y)*(-1)*handle_pitch + x * handle_pixelSize] = GPU_source[(SourceHeight - 1 - (int)origin.y)*(-1)*source_pitch + (int)origin.x * source_pixelSize];
		GPU_result[(HandleHeight - 1 - y)*(-1)*handle_pitch + x * handle_pixelSize + 1] = GPU_source[(SourceHeight - 1 - (int)origin.y)*(-1)*source_pitch + (int)origin.x * source_pixelSize + 1];
		GPU_result[(HandleHeight - 1 - y)*(-1)*handle_pitch + x * handle_pixelSize + 2] = GPU_source[(SourceHeight - 1 - (int)origin.y)*(-1)*source_pitch + (int)origin.x * source_pixelSize + 2];
	}
	else
	{
		//获得临近点
		point neighbors[4][4];
		getNeighborPoints(neighbors, origin.x, origin.y);

		//获得权值
		double weights[4][4];
		getWeight(weights, origin.x, origin.y);

		double value_R = 0;
		double value_G = 0;
		double value_B = 0;
		for (int y = 0; y < 4; y++)
		{
			for (int x = 0; x < 4; x++)
			{
				value_R += GPU_source[(SourceHeight - 1 - neighbors[x][y].y)*(-1)*source_pitch + neighbors[x][y].x * source_pixelSize] * weights[x][y];
				value_G += GPU_source[(SourceHeight - 1 - neighbors[x][y].y)*(-1)*source_pitch + neighbors[x][y].x * source_pixelSize + 1] * weights[x][y];
				value_B += GPU_source[(SourceHeight - 1 - neighbors[x][y].y)*(-1)*source_pitch + neighbors[x][y].x * source_pixelSize + 2] * weights[x][y];
			}
		}

		value_R = value_R < 0 ? 0 : value_R;
		value_R = value_R > 255 ? 255 : value_R;
		value_G = value_G < 0 ? 0 : value_G;
		value_G = value_G > 255 ? 255 : value_G;
		value_B = value_B < 0 ? 0 : value_B;
		value_B = value_B > 255 ? 255 : value_B;
		GPU_result[(HandleHeight - 1 - y)*(-1)*handle_pitch + x * handle_pixelSize] = (byte)value_R;
		GPU_result[(HandleHeight - 1 - y)*(-1)*handle_pitch + x * handle_pixelSize + 1] = (byte)value_G;
		GPU_result[(HandleHeight - 1 - y)*(-1)*handle_pitch + x * handle_pixelSize + 2] = (byte)value_B;
	}
}

extern "C" void Rotate_host(byte* source, byte* result_buf, int HandleWidth, int HandleHeight, int SourceWidth, int SourceHeight, double angle,
	int source_pitch, int source_pixelSize, int handle_pitch, int handle_pixelSize)
{
	//指定GPU分配空间方式
	dim3 DimBlock(BlockXMaxThreadNum, BlockYMaxThreadNum);
	dim3 DimGrid(HandleWidth / BlockXMaxThreadNum + 1, HandleHeight / BlockYMaxThreadNum + 1);

	//用来在显存中进行操作的指针
	byte* GPU_source;
	byte* GPU_result;

	//在显存中为原图像和工作区分配空间
	checkCudaErrors(cudaMalloc((void **)&GPU_source, sizeof(byte)*SourceHeight*((-1)*source_pitch)));
	checkCudaErrors(cudaMalloc((void **)&GPU_result, sizeof(byte)*HandleHeight*((-1)*handle_pitch)));

	checkCudaErrors(cudaMemcpy(GPU_source, source, sizeof(byte)*SourceHeight*((-1)*source_pitch), cudaMemcpyHostToDevice));

	Rotate_kernel << < DimGrid, DimBlock >> > (GPU_source, GPU_result, HandleWidth, HandleHeight, SourceWidth, SourceHeight, angle,
		source_pitch, source_pixelSize, handle_pitch, handle_pixelSize);

	checkCudaErrors(cudaMemcpy(result_buf, GPU_result, sizeof(byte)*HandleHeight*((-1)*handle_pitch), cudaMemcpyDeviceToHost));

	cudaFree(GPU_source);
	cudaFree(GPU_result);
}

