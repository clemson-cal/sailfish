#include <math.h>

#ifdef __NVCC__
#define gpuFree cudaFree
#define gpuMalloc cudaMalloc
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice

#else
#include <hip/hip_runtime.h>
#define gpuFree hipFree
#define gpuMalloc hipMalloc
#define gpuMemcpy hipMemcpy
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#endif

typedef unsigned long ulong;

extern "C" void *gpu_malloc(ulong size)
{
    void *ptr;
    gpuMalloc(&ptr, size);
    return ptr;
}

extern "C" void gpu_free(void *ptr)
{
    gpuFree(ptr);
}

extern "C" void gpu_memcpy_htod(void *dst, const void *src, ulong size)
{
    gpuMemcpy(dst, src, size, gpuMemcpyHostToDevice);
}

extern "C" void gpu_memcpy_dtoh(void *dst, const void *src, ulong size)
{
    gpuMemcpy(dst, src, size, gpuMemcpyDeviceToHost);
}

extern "C" void gpu_memcpy_dtod(void *dst, const void *src, ulong size)
{
    gpuMemcpy(dst, src, size, gpuMemcpyDeviceToDevice);
}

extern "C" void gpu_device_synchronize()
{
#ifdef __NVCC__
    cudaDeviceSynchronize();
#else
    hipDeviceSynchronize();
#endif
}

extern "C" int gpu_get_device_count()
{
    int count;
#ifdef __NVCC__
    cudaGetDeviceCount(&count);
#else
    hipGetDeviceCount(&count);
#endif
    return count;
}

extern "C" int gpu_get_device()
{
    int device;
#ifdef __NVCC__
    cudaGetDevice(&device);
#else
    hipGetDevice(&device);
#endif
    return device;
}

extern "C" void gpu_set_device(int device)
{
#ifdef __NVCC__
    cudaSetDevice(device);
#else
    hipSetDevice(device);
#endif
}

// Adapted from:
// https://sodocumentation.net/cuda/topic/6566/parallel-reduction--e-g--how-to-sum-an-array

#define REDUCE_BLOCK_SIZE 1024
#define REDUCE_GRID_SIZE 24

static __global__ void vec_max_f64_kernel(const double *in, ulong N, double *out)
{
    __shared__ double lds[REDUCE_BLOCK_SIZE];

    ulong start = threadIdx.x + blockIdx.x * REDUCE_BLOCK_SIZE;
    ulong gsize = gridDim.x * REDUCE_BLOCK_SIZE;
    double max = in[0];

    for (ulong i = start; i < N; i += gsize)
    {
        max = fmax(max, in[i]);
    }
    lds[threadIdx.x] = max;

    __syncthreads();

    for (ulong size = REDUCE_BLOCK_SIZE / 2; size > 0; size /= 2)
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

extern "C" void gpu_vec_max_f64(const double *vec, ulong size, double *result)
{
    if (size == 0) {
        return;
    }
    double* block_max;
    gpuMalloc(&block_max, sizeof(double) * REDUCE_GRID_SIZE);

    vec_max_f64_kernel<<<REDUCE_GRID_SIZE, REDUCE_BLOCK_SIZE>>>(vec, size, block_max);
    vec_max_f64_kernel<<<1, REDUCE_BLOCK_SIZE>>>(block_max, REDUCE_GRID_SIZE, block_max);

    gpuMemcpy(result, block_max, sizeof(double), gpuMemcpyDeviceToDevice);
    gpuFree(block_max);
}
