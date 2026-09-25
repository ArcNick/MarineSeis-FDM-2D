#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define IGN_SIZE_T(call) \
    do { size_t __t = (call); (void)__t; } while(0)

#define CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(error), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#endif // CUDA_CHECK_H