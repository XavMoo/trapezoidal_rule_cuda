#include "utils.h"
#include <stdio.h>

void check_error(cudaError_t status, const char *msg)
{
    if (status != cudaSuccess)
    {
        const char *errorStr = cudaGetErrorString(status);
        printf("%s:\n%s\nError Code: %d\n\n", msg, errorStr, status);
        exit(status); 
    }
}
