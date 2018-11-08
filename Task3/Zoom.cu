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

__global__ void Zoom_kernel(byte *GPU_source, byte *GPU_result, int HandleWidth, int HandleHeight, int SourceWidth, int SourceHeight, double param,
	int source_pitch, int source_pixelSize, int handle_pitch, int handle_pixelSize)
{
	//x��y���꼴Ϊ��Ӧ���߳�������thread�������x��y����
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= HandleWidth || y >= HandleHeight)
	{
		return;
	}

	//ӳ���ԭͼ�е�����
	double x_origin = x / param;
	double y_origin = y / param;

	if (x_origin >= SourceWidth || y_origin >= SourceHeight)
	{
		return;
	}

	if (x_origin <= 1 || y_origin <= 1 || x_origin >= SourceWidth - 2 || y_origin >= SourceHeight - 2)
	{
		//ͼ���Եʹ������ڲ�ֵ��
		GPU_result[(HandleHeight - 1 - y)*(-1)*handle_pitch + x * handle_pixelSize] = GPU_source[(SourceHeight - 1 - (int)y_origin)*(-1)*source_pitch + (int)x_origin * source_pixelSize];
		GPU_result[(HandleHeight - 1 - y)*(-1)*handle_pitch + x * handle_pixelSize + 1] = GPU_source[(SourceHeight - 1 - (int)y_origin)*(-1)*source_pitch + (int)x_origin * source_pixelSize + 1];
		GPU_result[(HandleHeight - 1 - y)*(-1)*handle_pitch + x * handle_pixelSize + 2] = GPU_source[(SourceHeight - 1 - (int)y_origin)*(-1)*source_pitch + (int)x_origin * source_pixelSize + 2];
	}
	else
	{
		//����ٽ���
		point neighbors[4][4];
		getNeighborPoints(neighbors, x_origin, y_origin);
		//���Ȩֵ
		double weights[4][4];
		getWeight(weights, x_origin, y_origin);

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

extern "C" void Zoom_host(byte* source, byte* result_buf, int HandleWidth, int HandleHeight, int SourceWidth, int SourceHeight,
	int source_pitch, int source_pixelSize, int handle_pitch, int handle_pixelSize)
{
	//�������ű�
	double param = (double)HandleHeight / SourceHeight;

	//ָ��GPU����ռ䷽ʽ
	dim3 DimBlock(BlockXMaxThreadNum, BlockYMaxThreadNum);
	dim3 DimGrid(HandleWidth / BlockXMaxThreadNum + 1, HandleHeight / BlockYMaxThreadNum + 1);

	//�������Դ��н��в�����ָ��
	byte* GPU_source;
	byte* GPU_result;

	//���Դ���Ϊԭͼ��͹���������ռ�
	checkCudaErrors(cudaMalloc((void **)&GPU_source, sizeof(byte)*SourceHeight*((-1)*source_pitch)));
	checkCudaErrors(cudaMalloc((void **)&GPU_result, sizeof(byte)*HandleHeight*((-1)*handle_pitch)));

	checkCudaErrors(cudaMemcpy(GPU_source, source, sizeof(byte)*SourceHeight*((-1)*source_pitch), cudaMemcpyHostToDevice));

	Zoom_kernel <<< DimGrid, DimBlock >>> (GPU_source, GPU_result, HandleWidth, HandleHeight, SourceWidth, SourceHeight, param,
		source_pitch,source_pixelSize,handle_pitch,handle_pixelSize);

	checkCudaErrors(cudaMemcpy(result_buf, GPU_result, sizeof(byte)*HandleHeight*((-1)*handle_pitch), cudaMemcpyDeviceToHost));

	cudaFree(GPU_source);
	cudaFree(GPU_result);
}

