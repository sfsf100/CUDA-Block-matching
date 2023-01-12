#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>



#define BLOCK_SIZE 32
#define kernel 7

using namespace std;
using namespace cv;


// SAD on GPU
__global__ void SAD(unsigned char *srcImage, unsigned char *srcImage2, unsigned char *dstImage, unsigned int width, unsigned int height)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;//核心座標
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int half = (kernel - 1) / 2;
	int error;//灰階差值
			  // only threads inside image will write results
	if ((x >= kernel / 2) && (x<(width - kernel / 2)) && (y >= kernel / 2) && (y<(height - kernel / 2)))
	{
		int min = 100000;//SAD
		int minx = 0;
		for (int r = 0; r < 60; r++)//search range
		{
			error = 0;
			for (int a = -half;a <= half;a++){
				for (int b = -half;b <= half;b++){
					error += abs(srcImage[((y + a)*width + (x + b))] - srcImage2[((y+a)*width+(x+r+b))]);
				}
			}
			if (error < min) {
				min = error;
				minx = r;
			}
		}
		dstImage[(y*width + x)] = minx;
	}
}

void SAD_GPU(const Mat& input, const Mat& input2, Mat& output)
{
	// Use cuda event to catch time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	double time0 = static_cast<double>(getTickCount()); //計時器開始
														// Calculate number of input & output bytes in each block
	const int inputSize = input.cols * input.rows;
	const int outputSize = output.cols * output.rows;
	unsigned char *d_input;
	unsigned char *d_output;
	unsigned char *d_input2;


	// Allocate device memory
	cudaMalloc<unsigned char>(&d_input, inputSize);
	cudaMalloc<unsigned char>(&d_input2, inputSize);//img2
	cudaMalloc<unsigned char>(&d_output, outputSize);


	// Copy data from OpenCV input image to device memory
	cudaMemcpy(d_input, input.ptr(), inputSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_input2, input2.ptr(), inputSize, cudaMemcpyHostToDevice);

	// Specify block size
	const dim3 block(BLOCK_SIZE, BLOCK_SIZE);//BLOCK_SIZE 16
											 // Calculate grid size to cover the whole image
	const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);
	SAD << <grid, block >> > (d_input, d_input2, d_output, output.cols, output.rows);
	// Start time
	cudaEventRecord(start);

	// Run  kernel on CUDA 

	//// Stop time
	cudaEventRecord(stop);
	//Copy data from device memory to output image
	cudaMemcpy(output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);

	//Free the device memory
	cudaFree(d_input);
	cudaFree(d_input2);
	cudaFree(d_output);


	cudaEventSynchronize(stop);
	float milliseconds = 0;

	// Calculate elapsed time in milisecond  
	cudaEventElapsedTime(&milliseconds, start, stop);
	/*cout << "\nProcessing time on GPU (ms): " << milliseconds << "\n";*/
	time0 = ((double)getTickCount() - time0) / getTickFrequency(); //計時器结束
	printf("GPU的運算時間：%f s\n", time0);
}

int main() {
	Mat img1,img2, depth_map;
	
	img1 = imread("D://im2.png",1);
	img2 = imread("D://im1.png",1);
    depth_map = Mat::zeros(img2.rows, img2.cols,CV_8U);
	cvtColor(img1, img1, COLOR_BGR2GRAY);
	cvtColor(img2, img2, COLOR_BGR2GRAY);
	SAD_GPU(img1, img2, depth_map);

	imwrite("depth_map.png", depth_map);

	waitKey(0);
	system("pause");
	return 0;
}