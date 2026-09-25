#include "common_kernels.cuh"

// C = cA * A + cB * B
__global__ void matrix_add(
    const float * __restrict__ A, 
    const float * __restrict__ B, 
    float * __restrict__ C, 
    float cA, float cB, int row, int col
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (idx >= col || idy >= row) return;
    
    int pos = idy * col + idx;
    C[pos] = cA * A[pos] + cB * B[pos];
}

// A += cB * B
__global__ void matrix_self_add(
    float * __restrict__ A,
    const float * __restrict__ B,
    float cB, int row, int col
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (idx >= col || idy >= row) return;
    
    int pos = idy * col + idx;
    A[pos] += cB * B[pos];
}