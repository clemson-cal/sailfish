extern "C" void *gpu_malloc(unsigned long size) {
    void *ptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

extern "C" void gpu_free(void *ptr) {
    cudaFree(&ptr);
}

extern "C" void gpu_memcpy_htod(void *dst, const void *src, unsigned long size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

extern "C" void gpu_memcpy_dtoh(void *dst, const void *src, unsigned long size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

extern "C" void gpu_memcpy_dtod(void *dst, const void *src, unsigned long size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
}

static __global__ void max_f64(const double *vec, unsigned long size, double *result) {
    double max = vec[0];

    for (int i = 1; i < size; ++i)
    {
        if (max < vec[i]) {
            max = vec[i];
        }
    }
    *result = max;
}

extern "C" void gpu_vec_max_f64(const double *vec, unsigned long size, double *result) {
    max_f64<<<1, 1>>>(vec, size, result);
}
