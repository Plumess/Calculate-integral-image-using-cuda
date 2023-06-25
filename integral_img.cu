#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper/helper_cuda.h"
#include "helper/helper_functions.h"

#include "integral_img.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ ushort4 ulonglong_to_ushort4(const unsigned long long in) {
    return make_ushort4((in & 0x000000000000ffff) >> 0,
                        (in & 0x00000000ffff0000) >> 16,
                        (in & 0x0000ffff00000000) >> 32,
                        (in & 0xffff000000000000) >> 48);
}

struct packed_result {
    ulonglong4 x, y, z, w;
};

__device__ packed_result get_prefix_sum(const ulonglong4 &data, const cg::thread_block &cta) {
    const auto tile = cg::tiled_partition<32>(cta);

    __shared__ unsigned long long sums[128];
    const unsigned int lane_id = tile.thread_rank();
    const unsigned int warp_id = tile.meta_group_rank();

    unsigned long long result[16] = {};
    {
        const ushort4 a = ulonglong_to_ushort4(data.x);
        const ushort4 b = ulonglong_to_ushort4(data.y);
        const ushort4 c = ulonglong_to_ushort4(data.z);
        const ushort4 d = ulonglong_to_ushort4(data.w);

        result[0] = a.x;
        result[1] = a.x + a.y;
        result[2] = a.x + a.y + a.z;
        result[3] = a.x + a.y + a.z + a.w;

        result[4] = b.x;
        result[5] = b.x + b.y;
        result[6] = b.x + b.y + b.z;
        result[7] = b.x + b.y + b.z + b.w;

        result[8] = c.x;
        result[9] = c.x + c.y;
        result[10] = c.x + c.y + c.z;
        result[11] = c.x + c.y + c.z + c.w;

        result[12] = d.x;
        result[13] = d.x + d.y;
        result[14] = d.x + d.y + d.z;
        result[15] = d.x + d.y + d.z + d.w;
    }

#pragma unroll
    for (unsigned int i = 4; i <= 7; i++) result[i] += result[3];

#pragma unroll
    for (unsigned int i = 8; i <= 11; i++) result[i] += result[7];

#pragma unroll
    for (unsigned int i = 12; i <= 15; i++) result[i] += result[11];

    unsigned long long sum = result[15];

#pragma unroll
    for (unsigned int i = 1; i < 32; i *= 2) {
        const unsigned long long n = tile.shfl_up(sum, i);

        if (lane_id >= i) {
#pragma unroll
            for (unsigned int j = 0; j < 16; j++) {
                result[j] += n;
            }

            sum += n;
        }
    }

    if (tile.thread_rank() == (tile.size() - 1)) {
        sums[warp_id] = result[15];
    }

    __syncthreads();

    if (warp_id == 0) {
        unsigned long long warp_sum = sums[lane_id];

#pragma unroll
        for (unsigned int i = 1; i <= 16; i *= 2) {
            const unsigned long long n = tile.shfl_up(warp_sum, i);

            if (lane_id >= i) warp_sum += n;
        }

        sums[lane_id] = warp_sum;
    }

    __syncthreads();

    // fold in unused warp
    if (warp_id > 0) {
        const unsigned long long blockSum = sums[warp_id - 1];

#pragma unroll
        for (unsigned int i = 0; i < 16; i++) {
            result[i] += blockSum;
        }
    }

    packed_result out;
    memcpy(&out, result, sizeof(out));
    return out;
}

__global__ void shfl_intimage_rows(const ulonglong4 *img, ulonglong4 *integral_image) {
    const auto cta = cg::this_thread_block();
    const auto tile = cg::tiled_partition<32>(cta);
    const unsigned int id = threadIdx.x;
    // pointer to head of current scanline
    const ulonglong4 *scanline = &img[blockIdx.x * 120];
    packed_result result = get_prefix_sum(scanline[id], cta);

    // This access helper allows packed_result to stay optimized as registers rather than spill to stack
    auto idxToElem = [&result](unsigned int idx) -> const ulonglong4 {
        switch (idx) {
            case 0:
                return result.x;
            case 1:
                return result.y;
            case 2:
                return result.z;
            case 3:
                return result.w;
        }
        return {};
    };

    const unsigned int idMask = id & 3;
    const unsigned int idSwizzle = (id + 2) & 3;
    const unsigned int idShift = (id >> 2) << 4;
    const unsigned int blockOffset = blockIdx.x * 480;

    // Use CG tile to warp shuffle vector types
    result.y = tile.shfl_xor(result.y, 1);
    result.z = tile.shfl_xor(result.z, 2);
    result.w = tile.shfl_xor(result.w, 3);

    // First batch
    integral_image[blockOffset + idMask + idShift] = idxToElem(idMask);
    // Second batch offset by 2
    integral_image[blockOffset + idSwizzle + idShift + 8] = idxToElem(idSwizzle);

    // continuing from the above example,
    // this use of __shfl_xor() places the y0..y3 and w0..w3 data in order.
    result.x = tile.shfl_xor(result.x, 1);
    result.y = tile.shfl_xor(result.y, 1);
    result.z = tile.shfl_xor(result.z, 1);
    result.w = tile.shfl_xor(result.w, 1);

    // First batch
    integral_image[blockOffset + idMask + idShift + 4] = idxToElem(idMask);
    // Second batch offset by 2
    integral_image[blockOffset + idSwizzle + idShift + 12] = idxToElem(idSwizzle);
}

__global__ void shfl_vertical_shfl(unsigned long long *img, int width, int height) {
    __shared__ unsigned long long sums[32][9];
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane_id = tidx % 8;
    unsigned long long stepSum = 0;
    unsigned long long mask = 0xffffffffffffffff;

    sums[threadIdx.x][threadIdx.y] = 0;
    __syncthreads();

    for (int step = 0; step < 135; step++) {
        unsigned long long sum = 0;
        unsigned long long *p = img + (threadIdx.y + step * 8) * width + tidx;

        sum = *p;
        sums[threadIdx.x][threadIdx.y] = sum;
        __syncthreads();

        // place into SMEM
        // shfl scan reduce the SMEM, reformating so the column sums are computed in a warp
        // then read out properly
        unsigned long long partial_sum = 0;
        int j = threadIdx.x % 8;
        int k = threadIdx.x / 8 + threadIdx.y * 4;

        partial_sum = sums[k][j];

        for (int i = 1; i <= 8; i *= 2) {
            unsigned long long n = __shfl_up_sync(mask, partial_sum, i, 32);

            if (lane_id >= i) partial_sum += n;
        }

        sums[k][j] = partial_sum;
        __syncthreads();

        if (threadIdx.y > 0) {
            sum += sums[threadIdx.x][threadIdx.y - 1];
        }

        sum += stepSum;
        stepSum += sums[threadIdx.x][blockDim.y - 1];
        __syncthreads();
        *p = sum;
    }
}

static unsigned int iDivUp(unsigned int dividend, unsigned int divisor) {
    return ((dividend % divisor) == 0) ? (dividend / divisor)
                                       : (dividend / divisor + 1);
}

template<typename T, typename R>
void calculate_integral_image_c(T* img, R* integral, int width, int height) {
    integral[0] = static_cast<R>(img[0]);
    for (int i = 1; i < width; i++) {
        integral[i] = integral[i - 1] + static_cast<R>(img[i]);
    }
    for (int i = 1; i < height; i++) {
        integral[i * width] = integral[(i - 1) * width] + static_cast<R>(img[i * width]);
    }
    for (int i = 1; i < height; i++) {
        for (int j = 1; j < width; j++) {
            int index = i * width + j;
            integral[index] = integral[index - 1] + integral[index - width] - integral[index - width - 1] + static_cast<R>(img[index]);
        }
    }
}

void integral_image_cuda(unsigned short *d_data, unsigned long long *d_integral_image,
                         int width, int height, cudaStream_t stream){
    int blockSize = iDivUp(width, 16);
    // launch 1 block / row
    int gridSize = height;

    shfl_intimage_rows<<<gridSize, blockSize, 0, stream>>>(
            reinterpret_cast<ulonglong4 *>(d_data),
            reinterpret_cast<ulonglong4 *>(d_integral_image));

    dim3 blockSz(32, 8);
    dim3 testGrid(width / blockSz.x, 1);

    shfl_vertical_shfl<<<testGrid, blockSz, 0, stream>>>((unsigned long long *)d_integral_image, width, height);
}