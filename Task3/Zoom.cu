#pragma once

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

