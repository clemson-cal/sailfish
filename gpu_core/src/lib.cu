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

extern "C" void gpu_memcpy_peer(void *dst, int dst_device, const void *src, int src_device, ulong size)
{
#ifdef __NVCC__
    cudaMemcpyPeer(dst, dst_device, src, src_device, size);
#else
    hipMemcpyPeer(dst, dst_device, src, src_device, size);
#endif    
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

extern "C" const char *gpu_get_last_error()
{
#ifdef __NVCC__
    cudaError_t error = cudaGetLastError();
    if (error)
        return cudaGetErrorString(error);
    return NULL;
#else
    hipError_t error = hipGetLastError();
    if (error) 
        return hipGetErrorString(error);
    return NULL;
#endif
}

static __global__ void gpu_memcpy_3d_kernel(
    char *dst,
    const char *src,
    ulong dst_start_i,
    ulong dst_start_j,
    ulong dst_start_k,
    ulong src_start_i,
    ulong src_start_j,
    ulong src_start_k,
    ulong dst_si,
    ulong dst_sj,
    ulong dst_sk,
    ulong src_si,
    ulong src_sj,
    ulong src_sk,
    ulong count_i,
    ulong count_j,
    ulong count_k,
    ulong bytes_per_elem)
{
    ulong i = threadIdx.z + blockIdx.z * blockDim.z;
    ulong j = threadIdx.y + blockIdx.y * blockDim.y;
    ulong k = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= count_i || j >= count_j || k >= count_k)
    {
        return;
    }

    ulong n_dst = (i - dst_start_i) * dst_si + (j - dst_start_j) * dst_sj + (k - dst_start_k) * dst_sk;
    ulong n_src = (i - src_start_i) * src_si + (j - src_start_j) * src_sj + (k - src_start_k) * src_sk;

    for (ulong q = 0; q < bytes_per_elem; ++q)
    {
        dst[n_dst + q] = src[n_src + q];
    }
}

extern "C" void gpu_memcpy_3d(
    char *dst,
    const char *src,
    ulong dst_start_i,
    ulong dst_start_j,
    ulong dst_start_k,
    ulong dst_shape_i,
    ulong dst_shape_j,
    ulong dst_shape_k,
    ulong src_start_i,
    ulong src_start_j,
    ulong src_start_k,
    ulong src_shape_i,
    ulong src_shape_j,
    ulong src_shape_k,
    ulong count_i,
    ulong count_j,
    ulong count_k,
    ulong bytes_per_elem)
{
    (void) src_shape_i; // unused
    (void) dst_shape_i; // unused

    dim3 bs = dim3(8, 8, 8);

    if (count_k == 1) {
        bs.y *= bs.x;
        bs.x = 1;
    }
    if (count_j == 1) {
        bs.z *= bs.y;
        bs.y = 1;
    }

    // strides in dst
    ulong dst_si = bytes_per_elem * dst_shape_k * dst_shape_j;
    ulong dst_sj = bytes_per_elem * dst_shape_k;
    ulong dst_sk = bytes_per_elem;

    // strides in src
    ulong src_si = bytes_per_elem * src_shape_k * src_shape_j;
    ulong src_sj = bytes_per_elem * src_shape_k;
    ulong src_sk = bytes_per_elem;

    dim3 bd = dim3((count_k + bs.z - 1) / bs.z, (count_j + bs.y - 1) / bs.x, (count_i + bs.x - 1) / bs.x);
    gpu_memcpy_3d_kernel<<<bd, bs>>>(
        dst,
        src,
        dst_start_i,
        dst_start_j,
        dst_start_k,
        src_start_i,
        src_start_j,
        src_start_k,
        dst_si,
        dst_sj,
        dst_sk,
        src_si,
        src_sj,
        src_sk,
        count_i,
        count_j,
        count_k,
        bytes_per_elem);
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
