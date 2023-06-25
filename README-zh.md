# CUDA Integral Image

[中文](README-zh.md) | [English](README.md)

本Repo是使用CUDA实现的积分图计算，在Intel i9, RTX4090上运行，未使用stream流的情况下，可以做到单通道1080P单帧处理0.874ms，改写自NVIDIA官方库cuda-samples中的shfl_scan，其使用shfl操作实现了scan，或者说前缀和/累积和的计算。shfl_scan中有二维前缀和计算的例子，在二维层面通常称为积分图integral image或总和面积表summed area table。积分图的介绍可以参考这个链接：[Integral Image - MATLAB & Simulink (mathworks.com)](https://www.mathworks.com/help/images/integral-image.html)

## 说明

本项目将原示例的代码提取适用于计算积分图的代码，并且改写了数据类型，将输入的[0,255]的unsigned char类型改为了[0, 65535]的unsigned short，并且改写了对应的shfl操作拆解包逻辑（shfl的16个单位的位运算拆解）和输出类型，这样就可以输入平方图，使代码支持计算图像的平方积分图。平方积分图的存在可以优化针对同一张图的子区域反复出现的方差计算速度。

## 实验结果

项目中给出的运行结果如下

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

由于NVIDIA在该项目就使用的是固定参数的原因，目前仅确保单通道1920*1080的积分图和平方积分图的运算是正确的，如果需要修改，请自行确定scanline，block等参数，这些参数取决于输入图像在行/列优先时的行/列数。


