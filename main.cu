#include <iostream>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <string>
#include <direct.h>
#include "integral_img.cuh"

using namespace std;

#define BLOCK_SIZE 16

__global__ void square_img_1ch(const unsigned char* d_ptrImg, unsigned short* d_out, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int index = y * cols + x;
        int short_index = y * cols + x;
        auto tmp = static_cast<unsigned short>(d_ptrImg[index]);
        d_out[short_index] = tmp * tmp;
    }
}
__global__ void char2short_1ch(const unsigned char* d_ptrImg, unsigned short* d_out, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int index = y * cols + x;
        int short_index = y * cols + x;
        d_out[short_index] = static_cast<unsigned short>(d_ptrImg[index]);
    }
}

// Helper function to generate a random image
void generate_random_image(unsigned char* img, int rows, int cols, int channels) {
    for(int i = 0; i < rows * cols * channels; i++) {
        img[i] = rand() % 256;
    }
}

void mkdirs(const std::string& path) {
    _mkdir(path.c_str());
}

int main() {
    srand(time(0));  // Seed random generator

    // Prepare the 1920x1080 image
    int rows = 1080;
    int cols = 1920;
    int channels = 1;  // Single channel image

    long long total_time = 0;
    int runtimes = 30;

    for (int run = 0; run < runtimes; run++) {
        // Output must exist
        //std::string output_dir = "E:/MyProject/CUDA Codes/Integral Image/Output/Run" + std::to_string(run) + "/";
        //mkdirs(output_dir);  // Create directories if they don't exist

        unsigned char* h_img = new unsigned char[rows * cols * channels];  // host memory for image

        // Generate a random image on host
        generate_random_image(h_img, rows, cols, channels);

        // Transfer to device
        unsigned char* d_img;  // device memory for image
        cudaMalloc((void**)&d_img, rows * cols * channels * sizeof(unsigned char));
        cudaMemcpy(d_img, h_img, rows * cols * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // Convert to short and square using CUDA kernels
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

        unsigned short* d_out_short;  // device memory for output short image
        unsigned short* d_out_square;  // device memory for output square image

        cudaMalloc((void**)&d_out_short, rows * cols * channels * sizeof(unsigned short));
        cudaMalloc((void**)&d_out_square, rows * cols * channels * sizeof(unsigned short));

        char2short_1ch<<<dimGrid, dimBlock>>>(d_img, d_out_short, rows, cols);
        square_img_1ch<<<dimGrid, dimBlock>>>(d_img, d_out_square, rows, cols);

        cudaDeviceSynchronize();

        unsigned long long* d_integral_image;
        unsigned long long* d_square_integral_image;
        cudaMalloc((void**)&d_integral_image, rows * cols * channels * sizeof(unsigned long long));
        cudaMalloc((void**)&d_square_integral_image, rows * cols * channels * sizeof(unsigned long long));

        // Timing
        auto start = std::chrono::high_resolution_clock::now();

        integral_image_cuda(d_out_short, d_integral_image, cols, rows, 0);
        integral_image_cuda(d_out_square, d_square_integral_image, cols, rows, 0);

        cudaDeviceSynchronize();

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        total_time += duration.count();
        std::cout << "Execution Time for Run " << run << ": " << duration.count() << " us" << std::endl;

        unsigned long long* h_integral_image = new unsigned long long[rows * cols * channels];
        unsigned long long* h_square_integral_image = new unsigned long long[rows * cols * channels];

        cudaMemcpy(h_integral_image, d_integral_image, rows * cols * channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_square_integral_image, d_square_integral_image, rows * cols * channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        // Write the images to .txt file
//        std::ofstream outOriginal(output_dir + "original_image.txt");
//        std::ofstream outIntegral(output_dir + "integral_image.txt");
//        std::ofstream outSquareIntegral(output_dir + "square_integral_image.txt");
//        for(int i = 0; i < rows; i++) {
//            for(int j = 0; j < cols; j++) {
//                for(int c = 0; c < channels; c++) {
//                    outOriginal << static_cast<int>(h_img[(i * cols + j) * channels + c]) << " ";
//                    outIntegral << h_integral_image[(i * cols + j) * channels + c] << " ";
//                    outSquareIntegral << h_square_integral_image[(i * cols + j) * channels + c] << " ";
//                }
//            }
//            outOriginal << std::endl;
//            outIntegral << std::endl;
//            outSquareIntegral << std::endl;
//        }
//        outOriginal.close();
//        outIntegral.close();
//        outSquareIntegral.close();

        // Cleanup
        delete[] h_img;
        delete[] h_integral_image;
        delete[] h_square_integral_image;
        cudaFree(d_img);
        cudaFree(d_out_short);
        cudaFree(d_out_square);
        cudaFree(d_integral_image);
        cudaFree(d_square_integral_image);
    }

    std::cout << "Average Execution Time: " << total_time / runtimes << " microseconds" << std::endl;

    return 0;
}