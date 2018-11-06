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

__global__ void DFT_kernel(byte *GPU_source, double *GPU_result_real, double* GPU_result_image, int HandleWidth, int HandleHeight, int SourceWidth, int SourceHeight,int BlockZMaxThreadNum)
{
	//频率坐标系下的u，v坐标即为对应的线程在整个thread阵里面的x，y坐标,z坐标用于线程分配
	int v = blockIdx.x*blockDim.x + threadIdx.x;
	int u = blockIdx.y*blockDim.y + threadIdx.y;
	int z = threadIdx.z;

	if (v >= HandleWidth || u >= HandleHeight)
	{
		return;
	}


	int AreaSize = SourceWidth * SourceHeight / BlockZMaxThreadNum;
	int startIndex;
	int endIndex;


	if (z == BlockZMaxThreadNum - 1)
	{
		startIndex = AreaSize * z;
		endIndex = SourceHeight * SourceWidth - 1;
	}
	else
	{
		startIndex = AreaSize * z;
		endIndex = AreaSize * (z+1) - 1;
	}

	GPU_result_real[(u*SourceWidth + v)*BlockZMaxThreadNum + z] = 0;
	GPU_result_image[(u*SourceWidth + v)*BlockZMaxThreadNum + z] = 0;

	for (int i = startIndex; i <= endIndex; i++)
	{
		int x = i / SourceWidth;
		int y = i % SourceWidth;

		double greyValue = (double)GPU_source[x*SourceWidth + y];
		if ((x + y) & 1)
			greyValue = -1.0*greyValue;
		double factor = (double)u*x / (double)SourceHeight + (double)v * y / (double)SourceWidth;

		double realpart = cos(-2 * PI*(factor))*greyValue;
		double imagepart = sin(-2 * PI*(factor))*greyValue;
		GPU_result_real[(u*SourceWidth + v)*BlockZMaxThreadNum + z] += realpart;
		GPU_result_image[(u*SourceWidth + v)*BlockZMaxThreadNum + z] += imagepart;
	}

	/*for (int x = 0; x < SourceHeight; x++)
	{
		for (int y = 0; y < SourceWidth; y++)
		{
			greyValue = (double)GPU_source[x*SourceWidth + y];
			if ((x + y) & 1)
				greyValue = -1.0*greyValue;
			double factor = (double)u*x / (double)SourceHeight + (double)v * y / (double)SourceWidth;
			ComplexNumber buf(cos(-2 * PI*(factor)), sin(-2 * PI*(factor)));
			result = result + (buf)*greyValue;
		}
	}*/

}

extern "C" void DFT_host(byte* source, byte* result_buf, int HandleWidth, int HandleHeight, int SourceWidth, int SourceHeight)
{
	//指定GPU分配空间方式
	dim3 DimBlock(BlockXMaxThreadNum, BlockYMaxThreadNum, BlockZMaxThreadNum);
	dim3 DimGrid(HandleWidth / BlockXMaxThreadNum + 1, HandleHeight / BlockYMaxThreadNum + 1);

	//用来在显存中进行操作的指针
	byte* GPU_source;
	double* GPU_result_real;
	double* GPU_result_image;

	//内存中用于保存结果的数组
	double* CPU_result_real = new double[HandleWidth*HandleHeight*BlockZMaxThreadNum];
	double* CPU_result_image = new double[HandleWidth*HandleHeight*BlockZMaxThreadNum];

	//在显存中为原图像和工作区分配空间
	checkCudaErrors(cudaMalloc((void **)&GPU_source, sizeof(byte)*SourceWidth*SourceHeight));
	checkCudaErrors(cudaMalloc((void **)&GPU_result_real, sizeof(double)*HandleWidth*HandleHeight*BlockZMaxThreadNum));
	checkCudaErrors(cudaMalloc((void **)&GPU_result_image, sizeof(double)*HandleWidth*HandleHeight*BlockZMaxThreadNum));

	checkCudaErrors(cudaMemcpy(GPU_source, source, sizeof(byte)*SourceHeight*SourceWidth, cudaMemcpyHostToDevice));

	DFT_kernel << < DimGrid, DimBlock >> > (GPU_source, GPU_result_real, GPU_result_image, HandleWidth, HandleHeight, SourceWidth, SourceHeight,BlockZMaxThreadNum);
	cudaThreadSynchronize();

	checkCudaErrors(cudaMemcpy(CPU_result_real, GPU_result_real, sizeof(double)*HandleWidth*HandleHeight*BlockZMaxThreadNum, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(CPU_result_image, GPU_result_image, sizeof(double)*HandleWidth*HandleHeight*BlockZMaxThreadNum, cudaMemcpyDeviceToHost));

	for (int y = 0; y < SourceHeight; y++)
	{
		for (int x = 0; x < SourceWidth; x++)
		{
			for (int i = 1; i < BlockZMaxThreadNum; i++)
			{
				CPU_result_real[(y*SourceWidth + x)*BlockZMaxThreadNum] = CPU_result_real[(y*SourceWidth + x)*BlockZMaxThreadNum] + CPU_result_real[(y*SourceWidth + x)*BlockZMaxThreadNum + i];
				CPU_result_image[(y*SourceWidth + x)*BlockZMaxThreadNum] = CPU_result_image[(y*SourceWidth + x)*BlockZMaxThreadNum] + CPU_result_image[(y*SourceWidth + x)*BlockZMaxThreadNum + i];
			}
			double norm = std::sqrt(CPU_result_real[(y*SourceWidth + x)*BlockZMaxThreadNum] * CPU_result_real[(y*SourceWidth + x)*BlockZMaxThreadNum] + CPU_result_image[(y*SourceWidth + x)*BlockZMaxThreadNum] * CPU_result_image[(y*SourceWidth + x)*BlockZMaxThreadNum]);
			double result_norm = 15 * log(norm + 1);

			result_norm = result_norm < 0.0 ? 0.0 : result_norm;
			result_norm = result_norm > 255.0 ? 255.0 : result_norm;

			result_buf[y*SourceWidth + x] = (byte)result_norm;
		}
	}

	cudaFree(GPU_source);
	cudaFree(GPU_result_real);
	cudaFree(GPU_result_image);

	delete[]CPU_result_real;
	delete[]CPU_result_image;
}