#ifndef _REDUCE_H
#define _REDUCE_H

#include <cstdint>

double trapezoid_kernel_CPU(double a, double b, uint64_t n);
double trapezoid_kernel_GPU(const double a, const double b, const uint64_t n, const unsigned int block_size, const unsigned int num_streams, const unsigned int N);

#endif
