#ifndef __ZOOM_CU_
#define __ZOOM_CU_
#endif // !__ZOOM_CU_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
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

__global__ void Zoom_kernel(byte *GPU_source, byte *GPU_result, int HandleWidth, int HandleHeight, int SourceWidth, int SourceHeight, double param)
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

	GPU_result[y*HandleWidth + x] = GPU_source[(int)y_origin*SourceWidth + (int)x_origin];
}

extern "C" void Zoom_host(byte* source, byte* result_buf, int HandleWidth, int HandleHeight, int SourceWidth, int SourceHeight)
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
	checkCudaErrors(cudaMalloc((void **)&GPU_source, sizeof(byte)*SourceWidth*SourceHeight));
	checkCudaErrors(cudaMalloc((void **)&GPU_result, sizeof(byte)*HandleWidth*HandleHeight));

	checkCudaErrors(cudaMemcpy(GPU_source, source, sizeof(byte)*SourceHeight*SourceWidth, cudaMemcpyHostToDevice));

	Zoom_kernel << <DimGrid, DimBlock >> > (GPU_source, GPU_result, HandleWidth, HandleHeight, SourceWidth, SourceHeight, param);

	checkCudaErrors(cudaMemcpy(result_buf, GPU_result, sizeof(byte)*HandleWidth*HandleHeight, cudaMemcpyDeviceToHost));

	cudaFree(GPU_source);
	cudaFree(GPU_result);
}

