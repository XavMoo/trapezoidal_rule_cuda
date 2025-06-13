#include <stdio.h>
#include <stdlib.h>
#include "reduce.h"
#include "kernels.h"
#include "utils.h"
#include <cmath>

double func(double x) {
  return sin(x);
}

double trapezoid_kernel_CPU(double a, double b, uint64_t n) {

  const double h = (b - a) / n;
  double sum = 0.0;

  sum += func(a);
  for (uint64_t i = 1; i < n; ++i) {
      sum += 2.0 * func(a + i * h);
  }
  sum += func(b);

  return (h / 2.0) * sum;
}

double trapezoid_kernel_GPU(const double a, const double b, const uint64_t n, const unsigned int block_size, const unsigned int num_streams, const unsigned int N) {

  const float h = (b-a) / n;

  cudaError_t status;

  double final_sum = 0;
  double *partial_results[num_streams];
  double *dev_outputs[num_streams];

  const unsigned int n_half = n / N;
  double a_local = a;
  const unsigned int threads_needed = n_half / num_streams;
  const unsigned int num_blocks = threads_needed / block_size;

  for (int i = 0; i < num_streams; i++)
  {
      partial_results[i] = new double[num_blocks];
      status = cudaMalloc(&(dev_outputs[i]), num_blocks * sizeof(double));
      check_error(status, "Error allocating device buffer.");
      status = cudaMemset(dev_outputs[i], 0, num_blocks * sizeof(double));
      check_error(status, "Error initialising device buffer.");
  }

  cudaStream_t streams[num_streams];
  for (int i = 0; i < num_streams; i++)
  {
      status = cudaStreamCreate(&(streams[i]));
      check_error(status, "Error creating CUDA stream.");
  }

  if (num_blocks != 0)
  {
    for (int i = 0; i < num_streams; i++)
    {
        switch (block_size)
        {
          case 1024:
            reduce<1024><<<num_blocks, block_size, 0, streams[i]>>>(a_local, h, dev_outputs[i],N);
            break;
          case 512:
            reduce<512><<<num_blocks, block_size, 0, streams[i]>>>(a_local, h, dev_outputs[i],N);
            break;
          case 256:
            reduce<256><<<num_blocks, block_size, 0, streams[i]>>>(a_local, h, dev_outputs[i],N);
            break;
          case 128:
            reduce<128><<<num_blocks, block_size, 0, streams[i]>>>(a_local, h, dev_outputs[i],N);
            break;
        }
        a_local += N * num_blocks * block_size * h;
    }
  }

  check_error( cudaGetLastError(), "Error in kernel." );

  for (int i = 0; i < num_streams; i++)
  {
      status = cudaMemcpyAsync(partial_results[i], dev_outputs[i], num_blocks * sizeof(double), cudaMemcpyDeviceToHost, streams[i]);
      check_error(status, "Error on GPU->CPU cudaMemcpy for partial_result.");
  }

  for (int i = 0; i < num_streams; i++)
  {
    for (int j = 0; j < num_blocks; j++)
    {
      final_sum += partial_results[i][j];
    }
  }
  final_sum += 0.5*(func(b) - func(a));
  final_sum *= h;

  for (int i = 0; i < num_streams; i++)
  {
      delete[] partial_results[i];
      status = cudaStreamDestroy(streams[i]);
      check_error(status, "Error destroying stream.");
      status = cudaFree(dev_outputs[i]);
      check_error(status, "Error calling cudaFree on device buffer.");
  }

  return final_sum;
}
