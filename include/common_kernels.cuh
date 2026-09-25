#ifndef KERNELS_CUH
#define KERNELS_CUH

// C = cA * A + cB * B
__global__ void matrix_add(
    const float * __restrict__ A, 
    const float * __restrict__ B, 
    float * __restrict__ C, 
    float cA, float cB, int row, int col
);

// A += cB * B
__global__ void matrix_self_add(
    float * __restrict__ A,
    const float * __restrict__ B,
    float cB, int row, int col
);


#endif