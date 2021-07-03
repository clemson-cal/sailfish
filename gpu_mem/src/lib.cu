#include <math.h>

extern "C" void *gpu_malloc(unsigned long size)
{
    void *ptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

extern "C" void gpu_free(void *ptr)
{
    cudaFree(&ptr);
}

extern "C" void gpu_memcpy_htod(void *dst, const void *src, unsigned long size)
{
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

extern "C" void gpu_memcpy_dtoh(void *dst, const void *src, unsigned long size)
{
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

extern "C" void gpu_memcpy_dtod(void *dst, const void *src, unsigned long size)
{
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
}


// Adapted from:
// https://sodocumentation.net/cuda/topic/6566/parallel-reduction--e-g--how-to-sum-an-array

#define REDUCE_BLOCK_SIZE 1024
#define REDUCE_GRID_SIZE 24

static __global__ void vec_max_f64_kernel(const double *in, unsigned long N, double *out)
{
    __shared__ double lds[REDUCE_BLOCK_SIZE];

    unsigned long start = threadIdx.x + blockIdx.x * REDUCE_BLOCK_SIZE;
    unsigned long gsize = gridDim.x * REDUCE_BLOCK_SIZE;
    double max = in[0];

    for (unsigned long i = start; i < N; i += gsize)
    {
        max = fmax(max, in[i]);
    }
    lds[threadIdx.x] = max;

    __syncthreads();

    for (unsigned long size = REDUCE_BLOCK_SIZE / 2; size > 0; size /= 2)
    {
        if (threadIdx.x < size)
        {
            lds[threadIdx.x] = fmax(lds[threadIdx.x], lds[threadIdx.x + size]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        out[blockIdx.x] = lds[0];
    }
}

extern "C" void gpu_vec_max_f64(const double *vec, unsigned long size, double *result)
{
    double* block_max;
    cudaMalloc(&block_max, sizeof(double) * REDUCE_GRID_SIZE);

    vec_max_f64_kernel<<<REDUCE_GRID_SIZE, REDUCE_BLOCK_SIZE>>>(vec, size, block_max);
    vec_max_f64_kernel<<<1, REDUCE_BLOCK_SIZE>>>(block_max, REDUCE_GRID_SIZE, block_max);

    cudaMemcpy(result, block_max, sizeof(double), cudaMemcpyDeviceToDevice);
    cudaFree(block_max);
}
