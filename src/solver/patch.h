#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef double real;
#define BUFFER_MODE_VIEW 0
#define BUFFER_MODE_HOST 1
#define BUFFER_MODE_DEVICE 2
#define FOR_EACH(p) \
    for (int i = p.start[0]; i < p.start[0] + p.count[0]; ++i) \
    for (int j = p.start[1]; j < p.start[1] + p.count[1]; ++j)
#define FOR_EACH_OMP(p) \
_Pragma("omp parallel for") \
    for (int i = p.start[0]; i < p.start[0] + p.count[0]; ++i) \
    for (int j = p.start[1]; j < p.start[1] + p.count[1]; ++j)
#define CONTAINS(p, q) \
        (p.start[0] <= q.start[0] && p.start[0] + p.count[0] >= q.start[0] + q.count[0]) && \
        (p.start[1] <= q.start[1] && p.start[1] + p.count[1] >= q.start[1] + q.count[1])
#define GET(p, i, j) (p.data + p.jumps[0] * ((i) - p.start[0]) + p.jumps[1] * ((j) - p.start[1]))
#define ELEMENTS(p) (p.count[0] * p.count[1] * p.num_fields)
#define BYTES(p) (ELEMENTS(p) * sizeof(real))

struct Patch
{
    int start[2];
    int count[2];
    int jumps[2];
    int num_fields;
    int buffer_mode;
    real *data;
};

#ifdef PATCH_LINKAGE

PATCH_LINKAGE struct Patch patch_new(
    int start_i,
    int start_j,
    int count_i,
    int count_j,
    int num_fields,
    int buffer_mode,
    real *data)
{
    struct Patch self;
    self.start[0] = start_i;
    self.start[1] = start_j;
    self.count[0] = count_i;
    self.count[1] = count_j;
    self.jumps[0] = num_fields * count_j;
    self.jumps[1] = num_fields;
    self.num_fields = num_fields;
    self.buffer_mode = buffer_mode;

    switch (buffer_mode)
    {
        case BUFFER_MODE_VIEW:
            self.data = data;
            break;
        case BUFFER_MODE_HOST:
            self.data = (real*)malloc(BYTES(self));
            break;
        case BUFFER_MODE_DEVICE:
        #ifdef __NVCC__
            cudaMalloc(&self.data, BYTES(self));
        #else
            self.data = NULL;
        #endif
            break;
    }
    return self;
}

PATCH_LINKAGE void patch_del(struct Patch self)
{
    switch (self.buffer_mode)
    {
        case BUFFER_MODE_VIEW:
            break;
        case BUFFER_MODE_HOST:
            free(self.data);
            break;
        case BUFFER_MODE_DEVICE:
        #ifdef __NVCC__
            cudaFree(self.data);
        #endif
            break;
    }
}

PATCH_LINKAGE void patch_set(struct Patch self, int i, int j, int q, real y)
{
    GET(self, i, j)[q] = y;
}

PATCH_LINKAGE real patch_get(struct Patch self, int i, int j, int q)
{
    return GET(self, i, j)[q];
}

PATCH_LINKAGE int patch_contains(struct Patch self, struct Patch other)
{
    return CONTAINS(self, other);
}

PATCH_LINKAGE struct Patch patch_clone(struct Patch self)
{
    struct Patch patch = self;

    switch (patch.buffer_mode)
    {
        case BUFFER_MODE_VIEW:
            break;
        case BUFFER_MODE_HOST:
            patch.data = (real*)malloc(BYTES(patch));
            memcpy(patch.data, self.data, BYTES(patch));
            break;
        case BUFFER_MODE_DEVICE:
        #ifdef __NVCC__
            cudaMalloc(&patch.data, BYTES(patch));
            cudaMemcpy(patch.data, self.data, BYTES(patch), cudaMemcpyDeviceToDevice);
        #endif
            break;
    }
    return patch;
}

#ifdef __NVCC__

PATCH_LINKAGE struct Patch patch_clone_to_device(struct Patch self)
{
    struct Patch device_patch = self;
    device_patch.buffer_mode = BUFFER_MODE_DEVICE;
    cudaMalloc(&device_patch.data, BYTES(device_patch));
    cudaMemcpy(device_patch.data, self.data, BYTES(self), cudaMemcpyHostToDevice);
    return device_patch;
}

PATCH_LINKAGE struct Patch patch_clone_to_host(struct Patch self)
{
    struct Patch host_patch = self;
    host_patch.data = (real*)malloc(BYTES(host_patch));
    host_patch.buffer_mode = BUFFER_MODE_HOST;
    cudaMemcpy(host_patch.data, self.data, BYTES(self), cudaMemcpyDeviceToHost);
    return host_patch;
}

#endif // __NVCC__
#endif // PATCH_LINKAGE
