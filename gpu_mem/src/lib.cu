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
