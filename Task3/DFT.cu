#include "DFT.cuh"

using byte = unsigned char;

inline void checkCudaErrors(cudaError err) //cuda error handle function
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA Runtime API error:%s.\n", cudaGetErrorString(err));
		return;
	}
}

__global__ void DFT_kernel(byte *GPU_source, byte *GPU_result, int HandleWidth, int HandleHeight, int SourceWidth, int SourceHeight)
{
	//Ƶ������ϵ�µ�u��v���꼴Ϊ��Ӧ���߳�������thread�������x��y����
	int v = blockIdx.x*blockDim.x + threadIdx.x;
	int u = blockIdx.y*blockDim.y + threadIdx.y;

	if (v >= HandleWidth || u >= HandleHeight)
	{
		return;
	}

	ComplexNumber result;
	double greyValue;

	for (int x = 0; x < SourceHeight; x++)
	{
		for (int y = 0; y < SourceWidth; y++)
		{
			greyValue = GPU_source[x*SourceWidth + y];
			if ((x + y) & 1)
				greyValue = -1.0*greyValue;
			double factor = (double)u*x / (double)SourceHeight + (double)v * y / (double)SourceWidth;
			ComplexNumber buf(cos(-2 * PI*(factor)), sin(-2 * PI*(factor)));
			result = result + (buf)*greyValue;
		}
	}

	double result_norm = 15*log(result.getNorm()+1);

	result_norm= result_norm < 0.0 ? 0.0 : result_norm;
	result_norm= result_norm > 255.0 ? 255.0 : result_norm;

	GPU_result[u*SourceWidth + v] = (byte)result_norm;

	//GPU_result[u*SourceWidth + v] = GPU_source[u*SourceWidth +v];
}

extern "C" void DFT_host(byte* source, byte* result_buf, int HandleWidth, int HandleHeight, int SourceWidth, int SourceHeight)
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

	DFT_kernel << < DimGrid, DimBlock >> > (GPU_source, GPU_result, HandleWidth, HandleHeight, SourceWidth, SourceHeight);

	checkCudaErrors(cudaMemcpy(result_buf, GPU_result, sizeof(byte)*HandleWidth*HandleHeight, cudaMemcpyDeviceToHost));

	cudaFree(GPU_source);
	cudaFree(GPU_result);
}