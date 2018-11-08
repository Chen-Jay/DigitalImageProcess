#include "stdafx.h"
#include "ImageProcess.h"
#include "MyImage.h"
#include <algorithm>
#include "MyMath.h"
#include <complex>
#include <algorithm>

extern "C" void DFT_host(byte* source, byte* result_buf, int HandleWidth, int HandleHeight, int SourceWidth, int SourceHeight, int pitch, int pixelSize);

UINT ImageProcess::zoom(LPVOID workspaceNoType)
{

	ThreadWorkSpace* workspace = (ThreadWorkSpace*)workspaceNoType;

	zoomParam* param = (zoomParam*)workspace->ctx;

	MyImage originImage(workspace->img);
	MyImage handleImage(workspace->handled);

	long long startIndex = workspace->startIndex;
	long long endIndex = workspace->endIndex;

	for (long long index = startIndex; index <= endIndex; index++)
	{
		int x = index % handleImage.getWidth();
		int y = index / handleImage.getWidth();

		double x_origin = (double)x / param->scale;
		double y_origin = (double)y / param->scale;

		if (x_origin < 1 || y_origin < 1 || x_origin >= originImage.getWidth() - 2 || y_origin >= originImage.getHeight() - 2)
		{
			//处理边界(使用最近邻插值法处理)
			if (!originImage.isColorful())
			{
				//灰度图像
				handleImage.writeImage(x, y, originImage.readImage(int(x_origin), int(y_origin)));
			}
			else
			{
				//彩色图像
				handleImage.writeImage_R(x, y, originImage.readImage_R(int(x_origin), int(y_origin)));
				handleImage.writeImage_G(x, y, originImage.readImage_G(int(x_origin), int(y_origin)));
				handleImage.writeImage_B(x, y, originImage.readImage_B(int(x_origin), int(y_origin)));
			}
			continue;
		}

		point<int> neighbors[4][4];
		param->getNeighborPoints(neighbors, x_origin, y_origin);
		double weights[4][4];
		param->getWeight(weights, x_origin, y_origin);

		if (!originImage.isColorful())
		{
			//灰度图像
			double value = 0;
			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					value += weights[i][j] * originImage.readImage(neighbors[i][j].x, neighbors[i][j].y);
				}
			}
			handleImage.writeImage(x, y, (byte)value);
			//writheImage(x, y, handleIamge, readImage(int(x_origin + 0.5), int(y_origin + 0.5), workspace->img));

		}
		else
		{
			//彩色图像
			double rValue = 0;
			double gValue = 0;
			double bValue = 0;
			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					rValue += weights[i][j] * originImage.readImage_R(neighbors[i][j].x, neighbors[i][j].y);
					gValue += weights[i][j] * originImage.readImage_G(neighbors[i][j].x, neighbors[i][j].y);
					bValue += weights[i][j] * originImage.readImage_B(neighbors[i][j].x, neighbors[i][j].y);
				}
			}
			rValue = min(255.0, rValue); rValue = max(0.0, rValue);
			gValue = min(255.0, gValue); gValue = max(0.0, gValue);
			bValue = min(255.0, bValue); bValue = max(0.0, bValue);
			handleImage.writeImage_R(x, y, byte(int(rValue)));
			handleImage.writeImage_G(x, y, byte(int(gValue)));
			handleImage.writeImage_B(x, y, byte(int(bValue)));
		}
	}
	auto wnd = ((CWnd*)(workspace->window));
	PostMessageW(wnd->GetSafeHwnd(), WM_ZOOM, 1, NULL);
	delete param;
	return 0;
}

UINT ImageProcess::rotate(LPVOID workspaceNoType)
{
	ThreadWorkSpace* workspace = (ThreadWorkSpace*)workspaceNoType;

	rotateParam* param = (rotateParam*)workspace->ctx;

	MyImage originImage(workspace->img);
	MyImage handleImage(workspace->handled);

	long long startIndex = workspace->startIndex;
	long long endIndex = workspace->endIndex;

	point<double> center(handleImage.getWidth() / 2.0, handleImage.getHeight() / 2.0);
	for (long long index = startIndex; index <= endIndex; index++)
	{
		int x = index % handleImage.getWidth();
		int y = index / handleImage.getWidth();
		point<int> now(x, y);

		point<double> origin = param->getOriginProjection(now, center, param);

		if (origin.x < 0 || origin.y < 0 || origin.x >= originImage.getWidth() - 1 || origin.y >= originImage.getHeight() - 1)
		{
			//原图像中没有的点，统一写入白色
			if (!originImage.isColorful())
			{
				handleImage.writeImage(x, y, byte(0));
			}
			else
			{
				handleImage.writeImage_R(x, y, byte(0));
				handleImage.writeImage_G(x, y, byte(0));
				handleImage.writeImage_B(x, y, byte(0));
			}
		}
		else if (origin.x < 1 || origin.y < 1 || origin.x >= originImage.getWidth() - 2 || origin.y >= originImage.getHeight() - 2)
		{
			//边缘使用最近邻进行处理
			if (!originImage.isColorful())
			{
				handleImage.writeImage(x, y, originImage.readImage(int(origin.x), int(origin.y)));
			}
			else
			{
				handleImage.writeImage_R(x, y, originImage.readImage_R(int(origin.x), int(origin.y)));
				handleImage.writeImage_G(x, y, originImage.readImage_G(int(origin.x), int(origin.y)));
				handleImage.writeImage_B(x, y, originImage.readImage_B(int(origin.x), int(origin.y)));
			}
		}
		else
		{

			point<int> neighbors[4][4];
			param->getNeighborPoints(neighbors, origin.x, origin.y);
			double weights[4][4];
			param->getWeight(weights, origin.x, origin.y);

			if (!originImage.isColorful())
			{
				handleImage.writeImage(x, y, originImage.readImage(int(origin.x), int(origin.y)));
			}
			else
			{
				double rValue = 0;
				double gValue = 0;
				double bValue = 0;
				for (int i = 0; i < 4; i++)
				{
					for (int j = 0; j < 4; j++)
					{
						rValue += weights[i][j] * originImage.readImage_R(neighbors[i][j].x, neighbors[i][j].y);
						gValue += weights[i][j] * originImage.readImage_G(neighbors[i][j].x, neighbors[i][j].y);
						bValue += weights[i][j] * originImage.readImage_B(neighbors[i][j].x, neighbors[i][j].y);
					}
				}
				rValue = min(255.0, rValue); rValue = max(0.0, rValue);
				gValue = min(255.0, gValue); gValue = max(0.0, gValue);
				bValue = min(255.0, bValue); bValue = max(0.0, bValue);
				handleImage.writeImage_R(x, y, byte(int(rValue)));
				handleImage.writeImage_G(x, y, byte(int(gValue)));
				handleImage.writeImage_B(x, y, byte(int(bValue)));
			}
		}
	}
	auto wnd = ((CWnd*)(workspace->window));
	PostMessageW(wnd->GetSafeHwnd(), WM_ROTATE, 1, NULL);
	return 0;
}

UINT ImageProcess::DFT(LPVOID workspaceNoType)
{
	ThreadWorkSpace* workspace = (ThreadWorkSpace*)workspaceNoType;

	rotateParam* param = (rotateParam*)workspace->ctx;

	MyImage originImage(workspace->img);
	MyImage handleImage(workspace->handled);

	long long startIndex = workspace->startIndex;
	long long endIndex = workspace->endIndex;

	for (long long index = startIndex; index <= endIndex; index++)
	{
		int v = index % originImage.getWidth();
		int u = index / originImage.getWidth();

		ComplexNumber result;
		double greyValue;
		if (!originImage.isColorful())
		{
			for (int x = 0; x < originImage.getHeight(); x++)
			{
				for (int y = 0; y < originImage.getWidth(); y++)
				{
					greyValue = (double)originImage.readImage(y, x);
					if ((x + y) & 1)
						greyValue = -1.0*greyValue;
					double factor = (double)u*x / (double)originImage.getHeight() + (double)v * y / (double)originImage.getWidth();
					ComplexNumber buf(cos(-2 * PI*(factor)), sin(-2 * PI*(factor)));
					result = result + (buf)*greyValue;
				}
			}
			auto f = std::clamp(15 * log(result.getNorm() + 1), 0.0, 255.0);
			handleImage.writeImage(v, u, byte(f));
		}
		else
		{
			for (int x = 0; x < originImage.getHeight(); x++)
			{
				for (int y = 0; y < originImage.getWidth(); y++)
				{
					greyValue = 0.299*originImage.readImage_R(y, x) + 0.587*originImage.readImage_G(y, x) + 0.114*originImage.readImage_B(y, x);
					if ((x + y) & 1)
						greyValue = -1.0*greyValue;
					double factor = (double)u*(double)x / (double)originImage.getHeight() + (double)v * (double)y / (double)originImage.getWidth();
					ComplexNumber buf(cos(2 * PI*(factor)), sin(-2 * PI*(factor)));
					result = result + (buf)*greyValue;
				}
			}
			auto f = std::clamp(15 * log(result.getNorm() + 1), 0.0, 255.0);
			handleImage.writeImage_R(v, u, byte(f));
			handleImage.writeImage_G(v, u, byte(f));
			handleImage.writeImage_B(v, u, byte(f));
		}
	}
	auto wnd = ((CWnd*)(workspace->window));
	PostMessageW(wnd->GetSafeHwnd(), WM_DFT, 1, NULL);
	return 0;
}

UINT ImageProcess::GaussNoise(LPVOID workspaceNoType)
{
	srand(time(0));
	ThreadWorkSpace* workspace = (ThreadWorkSpace*)workspaceNoType;

	GaussNoiseParam* param = (GaussNoiseParam*)workspace->ctx;

	MyImage originImage(workspace->img);
	MyImage handleImage(workspace->handled);

	long long startIndex = workspace->startIndex;
	long long endIndex = workspace->endIndex;

	for (long long index = startIndex; index <= endIndex; index++)
	{
		int x = index % handleImage.getWidth();
		int y = index / handleImage.getWidth();

		double noise = param->means + param->BoxMuller((double)rand() / (double)RAND_MAX, (double)rand() / (double)RAND_MAX, param->variance);
		if (!originImage.isColorful())
		{
			handleImage.writeImage(x, y, (byte)std::clamp((double)originImage.readImage(x, y) + noise, 0.0, 255.0));
		}
		else
		{
			handleImage.writeImage_R(x, y, (byte)std::clamp((double)originImage.readImage_R(x, y) + noise, 0.0, 255.0));
			handleImage.writeImage_G(x, y, (byte)std::clamp((double)originImage.readImage_G(x, y) + noise, 0.0, 255.0));
			handleImage.writeImage_B(x, y, (byte)std::clamp((double)originImage.readImage_B(x, y) + noise, 0.0, 255.0));
		}
	}
	auto wnd = ((CWnd*)(workspace->window));
	PostMessageW(wnd->GetSafeHwnd(), WM_GAUSSNOISE, 1, NULL);
	return 0;
}

UINT ImageProcess::MeanFilter(LPVOID workspaceNoType)
{
	ThreadWorkSpace* workspace = (ThreadWorkSpace*)workspaceNoType;

	MeanFilterParam* param = (MeanFilterParam*)workspace->ctx;

	MyImage originImage(workspace->img);
	MyImage handleImage(workspace->handled);

	long long startIndex = workspace->startIndex;
	long long endIndex = workspace->endIndex;

	for (long long index = startIndex; index <= endIndex; index++)
	{
		int x = index % handleImage.getWidth();
		int y = index / handleImage.getWidth();

		point<int> neighbors[3][3];
		param->getNeighborPoints(neighbors, x, y);


		double value = 0;
		double value_R = 0;
		double value_G = 0;
		double value_B = 0;

		if (!originImage.isColorful())
		{
			if (x<1 || y<1 || x>originImage.getWidth() - 2 || y>originImage.getHeight() - 2)
			{
				handleImage.writeImage(x, y, (byte)originImage.readImage(x, y));
			}
			else
			{
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; i < 3; j++)
					{
						value += originImage.readImage(neighbors[i][j].x, neighbors[i][j].y) / 9.0;
					}
				}
				handleImage.writeImage(x, y, (byte)std::clamp(value, 0.0, 255.0));
			}
		}
		else
		{
			if (x<1 || y<1 || x>originImage.getWidth() - 2 || y>originImage.getHeight() - 2)
			{
				handleImage.writeImage_R(x, y, (byte)originImage.readImage_R(x, y));
				handleImage.writeImage_G(x, y, (byte)originImage.readImage_R(x, y));
				handleImage.writeImage_B(x, y, (byte)originImage.readImage_R(x, y));
			}
			else
			{
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						value_R += originImage.readImage_R(neighbors[i][j].x, neighbors[i][j].y) / 9.0;
						value_G += originImage.readImage_G(neighbors[i][j].x, neighbors[i][j].y) / 9.0;
						value_B += originImage.readImage_B(neighbors[i][j].x, neighbors[i][j].y) / 9.0;
					}
				}
				handleImage.writeImage_R(x, y, (byte)std::clamp(value_R, 0.0, 255.0));
				handleImage.writeImage_G(x, y, (byte)std::clamp(value_G, 0.0, 255.0));
				handleImage.writeImage_B(x, y, (byte)std::clamp(value_B, 0.0, 255.0));
			}

		}
	}
	auto wnd = ((CWnd*)(workspace->window));
	PostMessageW(wnd->GetSafeHwnd(), WM_MEANFILTER, 1, NULL);
	return 0;
}

UINT ImageProcess::GaussFilter(LPVOID workspaceNoType)
{
	ThreadWorkSpace* workspace = (ThreadWorkSpace*)workspaceNoType;

	GaussFilterParam* param = (GaussFilterParam*)workspace->ctx;

	MyImage originImage(workspace->img);
	MyImage handleImage(workspace->handled);

	long long startIndex = workspace->startIndex;
	long long endIndex = workspace->endIndex;

	double gaussFactor[3][3];
	param->getGaussFactor(gaussFactor, param->variance);

	for (long long index = startIndex; index <= endIndex; index++)
	{
		int x = index % handleImage.getWidth();
		int y = index / handleImage.getWidth();


		point<int> neighbors[3][3];
		param->getNeighborPoints(neighbors, x, y);

		double value = 0;
		double value_R = 0;
		double value_G = 0;
		double value_B = 0;
		if (!originImage.isColorful())
		{
			if (x<1 || y<1 || x>originImage.getWidth() - 2 || y>originImage.getHeight() - 2)
			{
				handleImage.writeImage(x, y, (byte)originImage.readImage(x, y));
			}
			else
			{
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; i < 3; j++)
					{
						value += originImage.readImage(neighbors[i][j].x, neighbors[i][j].y)*gaussFactor[i][j];
					}
				}
				handleImage.writeImage(x, y, (byte)std::clamp(value, 0.0, 255.0));
			}
		}
		else
		{
			if (x<1 || y<1 || x>originImage.getWidth() - 2 || y>originImage.getHeight() - 2)
			{
				handleImage.writeImage_R(x, y, (byte)originImage.readImage_R(x, y));
				handleImage.writeImage_G(x, y, (byte)originImage.readImage_R(x, y));
				handleImage.writeImage_B(x, y, (byte)originImage.readImage_R(x, y));
			}
			else
			{
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						value_R += originImage.readImage_R(neighbors[i][j].x, neighbors[i][j].y) *gaussFactor[i][j];
						value_G += originImage.readImage_G(neighbors[i][j].x, neighbors[i][j].y) *gaussFactor[i][j];
						value_B += originImage.readImage_B(neighbors[i][j].x, neighbors[i][j].y) *gaussFactor[i][j];
					}
				}
				handleImage.writeImage_R(x, y, (byte)std::clamp(value_R, 0.0, 255.0));
				handleImage.writeImage_G(x, y, (byte)std::clamp(value_G, 0.0, 255.0));
				handleImage.writeImage_B(x, y, (byte)std::clamp(value_B, 0.0, 255.0));
			}

		}

	}
	auto wnd = ((CWnd*)(workspace->window));
	PostMessageW(wnd->GetSafeHwnd(), WM_GAUSSFILTER, 1, NULL);
	return 0;
}

UINT ImageProcess::WienerFilter(LPVOID workspaceNoType)
{
	ThreadWorkSpace* workspace = (ThreadWorkSpace*)workspaceNoType;

	WienerFilterParam* param = (WienerFilterParam*)workspace->ctx;

	MyImage originImage(workspace->img);
	MyImage handleImage(workspace->handled);

	long long startIndex = workspace->startIndex;
	long long endIndex = workspace->endIndex;

	for (long long index = startIndex; index <= endIndex; index++)
	{
		int x = index % handleImage.getWidth();
		int y = index / handleImage.getWidth();

		point<int> neighbors[3][3];
		param->getNeighborPoints(neighbors, x, y);

		if (!originImage.isColorful())
		{
			if (x<1 || y<1 || x>originImage.getWidth() - 2 || y>originImage.getHeight() - 2)
			{
				handleImage.writeImage(x, y, (byte)originImage.readImage(x, y));
			}
			else
			{
				double sum = 0;
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						sum += originImage.readImage(neighbors[i][j].x, neighbors[i][j].y);
					}
				}
				double local_means = sum / 9.0;

				double variance = 0;
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						variance += (originImage.readImage(neighbors[i][j].x, neighbors[i][j].y) - local_means * originImage.readImage(neighbors[i][j].x, neighbors[i][j].y) - local_means) / 9.0;
					}
				}

				double value;
				value = local_means + (max(0, variance - param->noise_variance) / max(variance, param->noise_variance))*(originImage.readImage(x, y) - local_means);
				handleImage.writeImage(x, y, (byte)std::clamp(value, 0.0, 255.0));
			}
		}
		else
		{
			if (x<1 || y<1 || x>originImage.getWidth() - 2 || y>originImage.getHeight() - 2)
			{
				handleImage.writeImage_R(x, y, (byte)originImage.readImage_R(x, y));
				handleImage.writeImage_G(x, y, (byte)originImage.readImage_G(x, y));
				handleImage.writeImage_B(x, y, (byte)originImage.readImage_B(x, y));
			}
			else
			{
				double sum_R = 0;
				double sum_G = 0;
				double sum_B = 0;
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						sum_R += originImage.readImage_R(neighbors[i][j].x, neighbors[i][j].y);
						sum_G += originImage.readImage_G(neighbors[i][j].x, neighbors[i][j].y);
						sum_B += originImage.readImage_B(neighbors[i][j].x, neighbors[i][j].y);
					}
				}
				double local_means_R = sum_R / 9.0;
				double local_means_G = sum_G / 9.0;
				double local_means_B = sum_B / 9.0;

				double variance_R = 0;
				double variance_G = 0;
				double variance_B = 0;
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						variance_R += (originImage.readImage_R(neighbors[i][j].x, neighbors[i][j].y) - local_means_R * originImage.readImage_R(neighbors[i][j].x, neighbors[i][j].y) - local_means_R) / 9.0;
						variance_G += (originImage.readImage_G(neighbors[i][j].x, neighbors[i][j].y) - local_means_G * originImage.readImage_G(neighbors[i][j].x, neighbors[i][j].y) - local_means_G) / 9.0;
						variance_B += (originImage.readImage_B(neighbors[i][j].x, neighbors[i][j].y) - local_means_B * originImage.readImage_B(neighbors[i][j].x, neighbors[i][j].y) - local_means_B) / 9.0;
					}
				}

				double value_R;
				double value_G;
				double value_B;

				value_R = local_means_R + (max(0, variance_R - param->noise_variance_R) / max(variance_R, param->noise_variance_R))*(originImage.readImage_R(x, y) - local_means_R);
				value_G = local_means_G + (max(0, variance_G - param->noise_variance_G) / max(variance_G, param->noise_variance_G))*(originImage.readImage_G(x, y) - local_means_G);
				value_B = local_means_B + (max(0, variance_B - param->noise_variance_B) / max(variance_B, param->noise_variance_B))*(originImage.readImage_B(x, y) - local_means_B);

				handleImage.writeImage_R(x, y, (byte)std::clamp(value_R, 0.0, 255.0));
				handleImage.writeImage_G(x, y, (byte)std::clamp(value_G, 0.0, 255.0));
				handleImage.writeImage_B(x, y, (byte)std::clamp(value_B, 0.0, 255.0));

			}

		}
	}
	auto wnd = ((CWnd*)(workspace->window));
	PostMessageW(wnd->GetSafeHwnd(), WM_WIENERFILTER, 1, NULL);
	return 0;
}

UINT ImageProcess::DFT_CUDA(LPVOID workspaceNoType)
{
	ThreadWorkSpace* p = (ThreadWorkSpace*)workspaceNoType;

	MyImage srcImage(p->img);
	MyImage handleImage(p->handled);

	auto handledImage_buf = p->handled;

	if (srcImage.isColorful())
	{
		//图片是彩色的
		auto pixel = new byte[srcImage.getHeight()*srcImage.getWidth()];

		//将原图像转化为灰度图像
		for (int y = 0; y < srcImage.getHeight(); y++)
		{
			for (int x = 0; x < srcImage.getWidth(); x++)
			{
				pixel[y*srcImage.getWidth() + x] = (byte)(0.299*srcImage.readImage_R(x, y) + 0.587*srcImage.readImage_G(x, y) + 0.114*srcImage.readImage_B(x, y));
			}
		}

		byte* handleImage_data_start = (byte*)handledImage_buf->GetBits() + handledImage_buf->GetPitch()*(handledImage_buf->GetHeight() - 1);

		DFT_host(pixel, handleImage_data_start, handleImage.getWidth(), handleImage.getHeight(), srcImage.getWidth(), srcImage.getHeight(), handledImage_buf->GetPitch(), handledImage_buf->GetBPP() / 8);

		delete[] pixel;		
	}

	auto wnd = ((CWnd*)(p->window));
	PostMessageW(wnd->GetSafeHwnd(), WM_DFTCUDA, 1, NULL);
	return 0;
}

//缩放相关
void zoomParam::getNeighborPoints(point<int> neighbors[4][4], double x_, double y_)
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

void zoomParam::getWeight(double weights[4][4], double x_, double y_)
{
	point<int> neighbors[4][4];
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

double zoomParam::bicubic(double x)
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


//旋转相关
point<double> rotate_point(point<double> p, point<double> q, double angle)
{
	double x0 = p.x - q.x;
	double y0 = p.y - q.y;

	double x1 = x0 * cos(ToRadian(angle)) - y0 * sin(ToRadian(angle));
	double y1 = x0 * sin(ToRadian(angle)) + y0 * cos(ToRadian(angle));

	x1 += q.x;
	y1 += q.y;

	return point<double>(x1, y1);
}
point<double> rotateParam::getOriginProjection(point<int> now, point<double> center, rotateParam* param)
{
	point<double> now_buffer(now.x, now.y);
	point<double> abs_now = rotate_point(now_buffer, center, 360.0 - param->angle);

	point<double> relative(abs_now.x - center.x, abs_now.y - center.y);

	point<double> proj_orgin(param->originCenter.x + relative.x, param->originCenter.y + relative.y);
	return proj_orgin;
}

void rotateParam::getNeighborPoints(point<int> neighbors[4][4], double x_, double y_)
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

void rotateParam::getWeight(double weights[4][4], double x_, double y_)
{
	point<int> neighbors[4][4];
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

double rotateParam::bicubic(double x)
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


//均值滤波相关
void MeanFilterParam::getNeighborPoints(point<int> neighbors[3][3], int x, int y)
{
	neighbors[0][0].setPoint(x - 1, y - 1);
	neighbors[0][1].setPoint(x, y - 1);
	neighbors[0][2].setPoint(x + 1, y - 1);

	neighbors[1][0].setPoint(x - 1, y);
	neighbors[1][1].setPoint(x, y);
	neighbors[1][2].setPoint(x + 1, y);

	neighbors[2][0].setPoint(x - 1, y + 1);
	neighbors[2][1].setPoint(x, y + 1);
	neighbors[2][2].setPoint(x + 1, y + 1);
}

//高斯滤波相关
void GaussFilterParam::getNeighborPoints(point<int> neighbors[3][3], int x, int y)
{
	neighbors[0][0].setPoint(x - 1, y - 1);
	neighbors[0][1].setPoint(x, y - 1);
	neighbors[0][2].setPoint(x + 1, y - 1);

	neighbors[1][0].setPoint(x - 1, y);
	neighbors[1][1].setPoint(x, y);
	neighbors[1][2].setPoint(x + 1, y);

	neighbors[2][0].setPoint(x - 1, y + 1);
	neighbors[2][1].setPoint(x, y + 1);
	neighbors[2][2].setPoint(x + 1, y + 1);
}

void GaussFilterParam::getGaussFactor(double values[3][3], double variance)
{
	double sum = 0;
	double value;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			double numerator = -1.0 * ((i - 1)*(i - 1) + (j - 1) * (j - 1));
			double denominator = 2 * variance*variance;
			value = std::exp(numerator / denominator) / (2 * PI*variance*variance);
			values[i][j] = value;
			sum += value;
		}
	}

	//归一化
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			values[i][j] /= sum;
		}
	}

}

//维纳滤波相关
void WienerFilterParam::getNeighborPoints(point<int> neighbors[3][3], int x, int y)
{
	neighbors[0][0].setPoint(x - 1, y - 1);
	neighbors[0][1].setPoint(x, y - 1);
	neighbors[0][2].setPoint(x + 1, y - 1);

	neighbors[1][0].setPoint(x - 1, y);
	neighbors[1][1].setPoint(x, y);
	neighbors[1][2].setPoint(x + 1, y);

	neighbors[2][0].setPoint(x - 1, y + 1);
	neighbors[2][1].setPoint(x, y + 1);
	neighbors[2][2].setPoint(x + 1, y + 1);
}
