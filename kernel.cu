#include "kernels.h"
#include "utils.h"
#include <stdio.h>

__device__ double f_device(double x)
{
    return sin(x);
}

template <unsigned int block_size>
__device__ void warpReduce(volatile double* sdata, int tid)
{
    if (block_size >= 64) sdata[tid] += sdata[tid + 32];
    if (block_size >= 32) sdata[tid] += sdata[tid + 16];
    if (block_size >= 16) sdata[tid] += sdata[tid + 8];
    if (block_size >= 8) sdata[tid] += sdata[tid + 4];
    if (block_size >= 4) sdata[tid] += sdata[tid + 2];
    if (block_size >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int block_size>
__global__ void reduce(double a, double h, double *output, unsigned int n)
{
    const unsigned int thread_id = threadIdx.x;
    const unsigned int block_id = blockIdx.x;

    const unsigned int index = block_id * block_size + thread_id;
    const unsigned int offset = gridDim.x * block_size;

    __shared__ double shared[block_size];

    unsigned int i = 1;
    shared[thread_id] = f_device(a + index * h);
    while(i < n) {
      shared[thread_id] += f_device(a + (index + i*offset) * h);
      i++;
    }

    __syncthreads();

    if (block_size >= 1024) { if (thread_id < 512) { shared[thread_id] += shared[thread_id + 512]; } __syncthreads(); }
    if (block_size >= 512) { if (thread_id < 256) { shared[thread_id] += shared[thread_id + 256]; } __syncthreads(); }
    if (block_size >= 256) { if (thread_id < 128) { shared[thread_id] += shared[thread_id + 128]; } __syncthreads(); }
    if (block_size >= 128) { if (thread_id < 64) { shared[thread_id] += shared[thread_id + 64]; } __syncthreads(); }

    if (thread_id < 32) warpReduce<block_size>(shared, thread_id);

    if (!thread_id)
    {
        output[block_id] += shared[0];
    }
}

template __global__ void reduce<1024>(double a, double h, double *output, unsigned int n);
template __global__ void reduce<512>(double a, double h, double *output, unsigned int n);
template __global__ void reduce<256>(double a, double h, double *output, unsigned int n);
template __global__ void reduce<128>(double a, double h, double *output, unsigned int n);
