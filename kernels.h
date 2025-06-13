/* This header file is used to provide the prototypes for all device kernel functions that
  * can be called from the host. To call kernels, #include it in your main .cu source file.
  */

#ifndef _KERNELS_H
#define _KERNELS_H

/* Prototypes */

template <unsigned int block_size>
__global__ void reduce(double a, double h, double *output, unsigned int n);

#endif
