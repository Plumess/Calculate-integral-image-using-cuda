# CUDA Integral Image

[中文](README-zh.md) | [English](README.md)

This repo provides an implementation of integral image calculation using CUDA. It operates on an Intel i9, RTX4090, and can achieve single-channel 1080P single-frame processing in 0.874ms without using stream. This project is a reimplementation of shfl_scan from NVIDIA's official library, cuda-samples, using shfl operations to realize scan, also known as prefix sum or cumulative sum calculations. shfl_scan contains an example of two-dimensional prefix sum calculation, often referred to as an integral image or summed area table in a 2D context. For an introduction to integral images, please refer to this link: [Integral Image - MATLAB & Simulink (mathworks.com)](https://www.mathworks.com/help/images/integral-image.html)

## Explanation

This project has extracted and adapted code from the original example to calculate the integral image. It has changed the data types, converting the input from an unsigned char type in the range [0,255] to an unsigned short in the range [0, 65535], and reconfigured the corresponding shfl operation unpacking logic (the bitwise operation unpacking of the 16 units of shfl) and output types. This allows for the input of histograms, enabling the code to support the calculation of the square integral image of an image. The existence of the square integral image can optimize the speed of variance calculation for repeatedly occurring subregions in the same image.

## Experimental Results

The following are the running results provided by the project:

```
Execution Time for Run 0: 919 us
Execution Time for Run 1: 798 us
Execution Time for Run 2: 813 us
Execution Time for Run 3: 871 us
Execution Time for Run 4: 874 us
Execution Time for Run 5: 837 us
Execution Time for Run 6: 783 us
Execution Time for Run 7: 801 us
Execution Time for Run 8: 766 us
Execution Time for Run 9: 1052 us
Execution Time for Run 10: 999 us
Execution Time for Run 11: 808 us
Execution Time for Run 12: 982 us
Execution Time for Run 13: 1000 us
Execution Time for Run 14: 1021 us
Execution Time for Run 15: 840 us
Execution Time for Run 16: 808 us
Execution Time for Run 17: 874 us
Execution Time for Run 18: 917 us
Execution Time for Run 19: 887 us
Execution Time for Run 20: 880 us
Execution Time for Run 21: 761 us
Execution Time for Run 22: 813 us
Execution Time for Run 23: 923 us
Execution Time for Run 24: 1040 us
Execution Time for Run 25: 811 us
Execution Time for Run 26: 877 us
Execution Time for Run 27: 808 us
Execution Time for Run 28: 866 us
Execution Time for Run 29: 806 us
Average Execution Time: 874 microseconds
```

## PS：

Due to the fact that NVIDIA used fixed parameters in this project, only single-channel 1920*1080 integral image and square integral image operations are currently guaranteed to be correct. If modifications are needed, please determine the scanline, block and other parameters yourself. These parameters depend on the number of rows/columns when the input image is row/column priority.
