#include "DFT.cuh"

using byte = unsigned char;

inline void checkCudaErrors(cudaError err, char* tag) //cuda error handle function
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA Runtime API error:%s. %s\n", cudaGetErrorString(err), tag);
		return;
	}
}

__global__ void DFT_kernel(byte *GPU_source, byte *GPU_result, int HandleWidth, int HandleHeight, int SourceWidth, int SourceHeight, int pitch, int pixelSize)
{
	//频率坐标系下的u，v坐标即为对应的线程在整个thread阵里面的x，y坐标
	int v = blockIdx.x*blockDim.x + threadIdx.x;
	int u = blockIdx.y*blockDim.y + threadIdx.y;

	if (v >= HandleWidth || u >= HandleHeight)
	{
		return;
	}

	ComplexNumber result;
	double realpart=0;
	double imaginepart =0;
	double greyValue;

	for (int x = 0; x < SourceHeight; x++)
	{
		for (int y = 0; y < SourceWidth; y++)
		{
			greyValue = (double)GPU_source[x*SourceWidth + y];
			if ((x + y) & 1)
				greyValue = -1.0*greyValue;

			double factor = (double)u*x / (double)SourceHeight + (double)v * y / (double)SourceWidth;

			double realpart_buf = cos(-2 * PI*(factor));
			double imaginepart_buf =sin(-2 * PI*(factor));

			realpart += realpart_buf * greyValue;
			imaginepart += imaginepart_buf * greyValue;
		}
	}
	double result_norm = 15 * log(std::sqrt(realpart*realpart+ imaginepart * imaginepart) + 1);

	result_norm = result_norm < 0.0 ? 0.0 : result_norm;
	result_norm = result_norm > 255.0 ? 255.0 : result_norm;

	GPU_result[(SourceHeight - 1 - u)*(-1)*pitch + v * pixelSize] = (byte)result_norm;
	GPU_result[(SourceHeight - 1 - u)*(-1)*pitch + v * pixelSize + 1] = (byte)result_norm;
	GPU_result[(SourceHeight - 1 - u)*(-1)*pitch + v * pixelSize + 2] = (byte)result_norm;

	//GPU_result[u*SourceWidth + v] = GPU_source[u*SourceWidth +v]; 
}

extern "C" void DFT_host(byte* source, byte* result_buf, int HandleWidth, int HandleHeight, int SourceWidth, int SourceHeight, int pitch, int pixelSize)
{

	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, (size_t)1024 * 1024 * 1024);

	//指定GPU分配空间方式
	dim3 DimBlock(BlockXMaxThreadNum, BlockYMaxThreadNum);
	dim3 DimGrid(HandleWidth / BlockXMaxThreadNum + 1, HandleHeight / BlockYMaxThreadNum + 1);

	byte* result;

	//用来在显存中进行操作的指针
	byte* GPU_source;

	//在显存中为原图像和工作区分配空间
	checkCudaErrors(cudaMalloc((void **)&GPU_source, sizeof(byte)*SourceWidth*SourceHeight), "a");
	checkCudaErrors(cudaMalloc((void **)&result, sizeof(byte)*HandleHeight*((-1)*pitch)), "b");

	checkCudaErrors(cudaMemcpy(GPU_source, source, sizeof(byte)*SourceHeight*SourceWidth, cudaMemcpyHostToDevice), "c");
	cudaThreadSynchronize();
	DFT_kernel <<< DimGrid, DimBlock >>> (GPU_source, result, HandleWidth, HandleHeight, SourceWidth, SourceHeight, pitch, pixelSize);
	cudaThreadSynchronize();
	checkCudaErrors(cudaMemcpy(result_buf, result, sizeof(byte)*HandleHeight*((-1) * pitch), cudaMemcpyDeviceToHost), "d");

	cudaFree(GPU_source);
	cudaFree(result);
}