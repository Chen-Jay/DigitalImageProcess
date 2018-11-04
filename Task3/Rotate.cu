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
	//x��y���꼴Ϊ��Ӧ���߳�������thread�������x��y����
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= HandleWidth || y >= HandleHeight)
	{
		return;
	}

	//ӳ���ԭͼ�е�����
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
		//ͼ���Եʹ������ڲ�ֵ��
		GPU_result[y*HandleWidth + x] = GPU_source[(int)origin.y*SourceWidth + (int)origin.x];
	}
	else
	{
		//����ٽ���
		point neighbors[4][4];
		getNeighborPoints(neighbors, origin.x, origin.y);

		//���Ȩֵ
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
	//ָ��GPU����ռ䷽ʽ
	dim3 DimBlock(BlockXMaxThreadNum, BlockYMaxThreadNum);
	dim3 DimGrid(HandleWidth / BlockXMaxThreadNum + 1, HandleHeight / BlockYMaxThreadNum + 1);

	//�������Դ��н��в�����ָ��
	byte* GPU_source;
	byte* GPU_result;

	//���Դ���Ϊԭͼ��͹���������ռ�
	checkCudaErrors(cudaMalloc((void **)&GPU_source, sizeof(byte)*SourceWidth*SourceHeight));
	checkCudaErrors(cudaMalloc((void **)&GPU_result, sizeof(byte)*HandleWidth*HandleHeight));

	checkCudaErrors(cudaMemcpy(GPU_source, source, sizeof(byte)*SourceHeight*SourceWidth, cudaMemcpyHostToDevice));

	Rotate_kernel << < DimGrid, DimBlock >> > (GPU_source, GPU_result, HandleWidth, HandleHeight, SourceWidth, SourceHeight, angle);

	checkCudaErrors(cudaMemcpy(result_buf, GPU_result, sizeof(byte)*HandleWidth*HandleHeight, cudaMemcpyDeviceToHost));

	cudaFree(GPU_source);
	cudaFree(GPU_result);
}

