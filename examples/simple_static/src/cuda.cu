#include "cuda.h"
#include "stdio.h"

extern "C" __global__ void sin_kernel(float *out, const float *inp, const int32_t numel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = sin(inp[i]);
    }
}

extern "C" int launch_sin2(float *out, const float *inp, int32_t n,  cudaStream_t stream) {
    sin_kernel<<<n, 1, 0>>>(out, inp, n); 
    return 0;
}
