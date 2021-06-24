#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>


// ============================ COMPAT ========================================
// ============================================================================
#define CONCAT(a, b) a ## _ ## b
#ifdef _OPENMP
#define PUBLIC(f) CONCAT(f, omp)
#define EXTERN_C
#elif defined __NVCC__
#define PUBLIC(f) CONCAT(f, gpu)
#define EXTERN_C extern "C"
#elif defined GPU_STUBS
#define PUBLIC(f) CONCAT(f, gpu)
#define EXTERN_C
#else
#define PUBLIC(f) CONCAT(f, cpu)
#define EXTERN_C
#endif
#ifndef __NVCC__
#define __device__
#define __host__
#endif


// ============================ MEMORY ========================================
// ============================================================================
#define BUFFER_MODE_VIEW 0
#define BUFFER_MODE_HOST 1
#define BUFFER_MODE_DEVICE 2

#ifndef __NVCC__
static void *compute_malloc(size_t count) { return malloc(count); }
static void compute_free(void *ptr) { free(ptr); }
static void compute_memcpy_host_to_device(void *dst, const void *src, size_t count) { memcpy(dst, src, count); }
static void compute_memcpy_device_to_host(void *dst, const void *src, size_t count) { memcpy(dst, src, count); }
static void compute_memcpy_device_to_device(void *dst, const void *src, size_t count) { memcpy(dst, src, count); }
#else
static void *compute_malloc(size_t count) { void *ptr; cudaMalloc(&ptr, count); return ptr; }
static void compute_free(void *ptr) { cudaFree(ptr); }
static void compute_memcpy_host_to_device(void *dst, const void *src, size_t count) { cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice); }
static void compute_memcpy_device_to_host(void *dst, const void *src, size_t count) { cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost); }
static void compute_memcpy_device_to_device(void *dst, const void *src, size_t count) { cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice); }
#endif


// ============================ PHYSICS =======================================
// ============================================================================
#define NCONS 3
#define PLM_THETA 1.5


// ============================ MATH ==========================================
// ============================================================================
#define real double
#define min2(a, b) ((a) < (b) ? (a) : (b))
#define max2(a, b) ((a) > (b) ? (a) : (b))
#define min3(a, b, c) min2(a, min2(b, c))
#define max3(a, b, c) max2(a, max2(b, c))
#define sign(x) copysign(1.0, x)
#define minabs(a, b, c) min3(fabs(a), fabs(b), fabs(c))

static __device__ real plm_gradient_scalar(real yl, real y0, real yr)
{
    real a = (y0 - yl) * PLM_THETA;
    real b = (yr - yl) * 0.5;
    real c = (yr - y0) * PLM_THETA;
    return 0.25 * fabs(sign(a) + sign(b)) * (sign(a) + sign(c)) * minabs(a, b, c);
}

static __device__ void plm_gradient(real *yl, real *y0, real *yr, real *g)
{
    for (int q = 0; q < NCONS; ++q)
    {
        g[q] = plm_gradient_scalar(yl[q], y0[q], yr[q]);
    }
}


// ============================ PATCH API =====================================
// ============================================================================
#define GET(p, i, j) (p.data + p.jumps[0] * ((i) - p.start[0]) + p.jumps[1] * ((j) - p.start[1]))
#define FOR_EACH(p) \
for (int i = p.start[0]; i < p.start[0] + p.count[0]; ++i) \
for (int j = p.start[1]; j < p.start[1] + p.count[1]; ++j) \

#define FOR_EACH_OMP(p) \
_Pragma("omp parallel for") \
for (int i = p.start[0]; i < p.start[0] + p.count[0]; ++i) \
for (int j = p.start[1]; j < p.start[1] + p.count[1]; ++j) \


#define ELEMENTS(p) (p.count[0] * p.count[1] * NCONS)
#define BYTES(p) (ELEMENTS(p) * sizeof(real))

struct Patch
{
    int start[2];
    int count[2];
    int jumps[2];
    int buffer_mode;
    real *data;
};

static struct Patch patch_new(int start_i, int start_j, int count_i, int count_j, real *data, int buffer_mode)
{
    struct Patch self;
    self.start[0] = start_i;
    self.start[1] = start_j;
    self.count[0] = count_i;
    self.count[1] = count_j;
    self.jumps[0] = NCONS * count_j;
    self.jumps[1] = NCONS;
    self.buffer_mode = buffer_mode;

    switch (buffer_mode)
    {
        case BUFFER_MODE_VIEW:
            self.data = data;
            break;
        case BUFFER_MODE_HOST:
            self.data = (real*)malloc(NCONS * count_i * count_j * sizeof(real));
            break;
        case BUFFER_MODE_DEVICE:
            self.data = (real*)compute_malloc(NCONS * count_i * count_j * sizeof(real));
            break;
    }

    return self;
}

static void patch_release(struct Patch self)
{
    switch (self.buffer_mode)
    {
        case BUFFER_MODE_VIEW:
            break;
        case BUFFER_MODE_HOST:
            free(self.data);
            break;
        case BUFFER_MODE_DEVICE:
            compute_free(self.data);
            break;
    }
}


// ============================ MESH ==========================================
// ============================================================================
struct Mesh
{
    int ni, nj;
    real x0, y0;
    real dx, dy;
};
#define X(m, i) (m.x0 + (i) * m.dx)
#define Y(m, i) (m.y0 + (j) * m.dy)


// ============================ HYDRO =========================================
// ============================================================================
static __device__ void conserved_to_primitive(const real *cons, real *prim)
{
    const real rho = cons[0];
    const real px = cons[1];
    const real py = cons[2];
    const real vx = px / rho;
    const real vy = py / rho;

    prim[0] = rho;
    prim[1] = vx;
    prim[2] = vy;
}

static __device__ __host__ void primitive_to_conserved(const real *prim, real *cons)
{
    const real rho = prim[0];
    const real vx = prim[1];
    const real vy = prim[2];
    const real px = vx * rho;
    const real py = vy * rho;

    cons[0] = rho;
    cons[1] = px;
    cons[2] = py;
}

static __device__ real primitive_to_velocity(const real *prim, int direction)
{
    switch (direction)
    {
        case 0: return prim[1];
        case 1: return prim[2];
        default: return 0.0;
    }
}

static __device__ void primitive_to_flux(
    const real *prim,
    const real *cons,
    real *flux,
    real cs2,
    int direction)
{
    const real vn = primitive_to_velocity(prim, direction);
    const real rho = prim[0];
    const real pressure = rho * cs2;

    flux[0] = vn * cons[0];
    flux[1] = vn * cons[1] + pressure * (direction == 0);
    flux[2] = vn * cons[2] + pressure * (direction == 1);
}

static __device__ void primitive_to_outer_wavespeeds(
    const real *prim,
    real *wavespeeds,
    real cs2,
    int direction)
{
    const real cs = sqrt(cs2);
    const real vn = primitive_to_velocity(prim, direction);
    wavespeeds[0] = vn - cs;
    wavespeeds[1] = vn + cs;
}

static __device__ void riemann_hlle(const real *pl, const real *pr, real *flux, real cs2, int direction)
{
    real ul[NCONS];
    real ur[NCONS];
    real fl[NCONS];
    real fr[NCONS];
    real al[2];
    real ar[2];

    primitive_to_conserved(pl, ul);
    primitive_to_conserved(pr, ur);
    primitive_to_flux(pl, ul, fl, cs2, direction);
    primitive_to_flux(pr, ur, fr, cs2, direction);
    primitive_to_outer_wavespeeds(pl, al, cs2, direction);
    primitive_to_outer_wavespeeds(pr, ar, cs2, direction);

    const real am = min3(0.0, al[0], ar[0]);
    const real ap = max3(0.0, al[1], ar[1]);

    for (int q = 0; q < NCONS; ++q)
    {
        flux[q] = (fl[q] * ap - fr[q] * am - (ul[q] - ur[q]) * ap * am) / (ap - am);
    }
}


// ============================ SCHEME ========================================
// ============================================================================
static __device__ void gradient_i(const struct Patch p, struct Patch g, int i, int j)
{
    real *pl = GET(p, i - 1, j);
    real *pc = GET(p, i + 0, j);
    real *pr = GET(p, i + 1, j);
    real *gc = GET(g, i, j);
    plm_gradient(pl, pc, pr, gc);
}

static __device__ void gradient_j(const struct Patch p, struct Patch g, int i, int j)
{
    real *pl = GET(p, i, j - 1);
    real *pc = GET(p, i, j + 0);
    real *pr = GET(p, i, j + 1);
    real *gc = GET(g, i, j);
    plm_gradient(pl, pc, pr, gc);
}

static __device__ void godunov_i(const struct Patch p, struct Patch g, struct Patch f, int i, int j)
{
    real *pl = GET(p, i - 1, j);
    real *pr = GET(p, i + 0, j);
    real *gl = GET(g, i - 1, j);
    real *gr = GET(g, i + 0, j);

    real pm[NCONS];
    real pp[NCONS];

    for (int q = 0; q < NCONS; ++q)
    {
        pm[q] = pl[q] + 0.5 * gl[q];
        pp[q] = pr[q] - 0.5 * gr[q];
    }
    riemann_hlle(pm, pp, GET(f, i, j), 1.0, 0);
}

static __device__ void godunov_j(const struct Patch p, struct Patch g, struct Patch f, int i, int j)
{
    real *pl = GET(p, i, j - 1);
    real *pr = GET(p, i, j + 0);
    real *gl = GET(g, i, j - 1);
    real *gr = GET(g, i, j + 0);

    real pm[NCONS];
    real pp[NCONS];

    for (int q = 0; q < NCONS; ++q)
    {
        pm[q] = pl[q] + 0.5 * gl[q];
        pp[q] = pr[q] - 0.5 * gr[q];
    }
    riemann_hlle(pm, pp, GET(f, i, j), 1.0, 1);
}

static __device__ void update_conserved_and_primitive(
    struct Patch p,
    struct Patch u,
    struct Patch u0,
    struct Patch flux_i,
    struct Patch flux_j,
    struct Mesh mesh,
    real a,
    real dt,
    int i,
    int j)
{
    real dx = mesh.dx;
    real dy = mesh.dy;
    real *fli = GET(flux_i, i + 0, j);
    real *fri = GET(flux_i, i + 1, j);
    real *flj = GET(flux_j, i, j + 0);
    real *frj = GET(flux_j, i, j + 1);
    real *pc = GET(p, i, j);
    real *uc = GET(u, i, j);
    real *un = GET(u0, i, j);

    for (int q = 0; q < NCONS; ++q)
    {
        uc[q] -= ((fri[q] - fli[q]) / dx + (frj[q] - flj[q]) / dy) * dt;
        uc[q] = (1.0 - a) * uc[q] + a * un[q];
    }
    conserved_to_primitive(uc, pc);
}


// ============================ SOLVER ========================================
// ============================================================================
struct Solver
{
    struct Mesh mesh;
    struct Patch primitive;
    struct Patch primitive_out;
    struct Patch conserved;
    struct Patch conserved_rk;
    struct Patch grad_i;
    struct Patch grad_j;
    struct Patch flux_i;
    struct Patch flux_j;
};

EXTERN_C struct Solver *PUBLIC(solver_new)(struct Mesh mesh)
{
    int i0 = 0;
    int j0 = 0;
    int ni = mesh.ni;
    int nj = mesh.nj;

    struct Solver *self = (struct Solver*)malloc(sizeof(struct Solver));
    self->mesh = mesh;
    self->primitive     = patch_new(i0 - 2, j0 - 2, ni + 4, nj + 4, NULL, BUFFER_MODE_DEVICE);
    self->primitive_out = patch_new(i0 - 2, j0 - 2, ni + 4, nj + 4, NULL, BUFFER_MODE_DEVICE);
    self->conserved = patch_new(i0, i0, ni, nj, NULL, BUFFER_MODE_DEVICE);
    self->conserved_rk = patch_new(i0, j0, ni, nj, NULL, BUFFER_MODE_DEVICE);
    self->grad_i = patch_new(i0 - 1, j0, ni + 2, nj, NULL, BUFFER_MODE_DEVICE);
    self->grad_j = patch_new(i0, j0 - 1, ni, nj + 2, NULL, BUFFER_MODE_DEVICE);
    self->flux_i = patch_new(i0, j0, ni + 1, nj, NULL, BUFFER_MODE_DEVICE);
    self->flux_j = patch_new(i0, j0, ni, nj + 1, NULL, BUFFER_MODE_DEVICE);

#ifdef __NVCC__
    cudaError_t error = cudaGetLastError();
    if (error)
        printf("%s\n", cudaGetErrorString(error));
#endif

    return self;
}

EXTERN_C void PUBLIC(solver_del)(struct Solver *self)
{
    patch_release(self->primitive);
    patch_release(self->primitive_out);
    patch_release(self->conserved);
    patch_release(self->conserved_rk);
    patch_release(self->grad_i);
    patch_release(self->grad_j);
    patch_release(self->flux_i);
    patch_release(self->flux_j);
    free(self);
}

EXTERN_C struct Mesh PUBLIC(solver_get_mesh)(struct Solver *self)
{
    return self->mesh;
}

EXTERN_C void PUBLIC(solver_get_primitive)(struct Solver *self, real *data)
{
    struct Patch user_primitive = patch_new( 0,  0, self->mesh.ni + 0, self->mesh.nj + 0, data, BUFFER_MODE_VIEW);
    struct Patch host_primitive = patch_new(-2, -2, self->mesh.ni + 4, self->mesh.nj + 4, NULL, BUFFER_MODE_HOST);
    compute_memcpy_device_to_host(host_primitive.data, self->primitive.data, BYTES(host_primitive));

    FOR_EACH(user_primitive) {
        real *ps = GET(host_primitive, i, j);
        real *pd = GET(user_primitive, i, j);

        for (int q = 0; q < NCONS; ++q)
        {
            pd[q] = ps[q];
        }        
    }
    patch_release(host_primitive);
}

EXTERN_C void PUBLIC(solver_set_primitive)(struct Solver *self, real *data)
{
    struct Patch user_primitive = patch_new( 0,  0, self->mesh.ni + 0, self->mesh.nj + 0, data, BUFFER_MODE_VIEW);
    struct Patch host_primitive = patch_new(-2, -2, self->mesh.ni + 4, self->mesh.nj + 4, NULL, BUFFER_MODE_HOST);
    struct Patch host_conserved = patch_new( 0,  0, self->mesh.ni + 0, self->mesh.nj + 0, NULL, BUFFER_MODE_HOST);

    FOR_EACH(host_primitive) {
        int ii = min2(max2(i, 0), self->mesh.ni - 1);
        int jj = min2(max2(j, 0), self->mesh.nj - 1);

        real *ps = GET(user_primitive, ii, jj);
        real *pd = GET(host_primitive, i, j);
        real *ud = GET(host_conserved, ii, jj);

        for (int q = 0; q < NCONS; ++q)
        {
            pd[q] = ps[q];
        }
        primitive_to_conserved(pd, ud);
    }

    compute_memcpy_host_to_device(self->primitive.data, host_primitive.data, BYTES(host_primitive));
    compute_memcpy_host_to_device(self->conserved.data, host_conserved.data, BYTES(host_conserved));
    compute_memcpy_device_to_device(self->primitive_out.data, self->primitive.data, BYTES(self->primitive));

    patch_release(host_primitive);
    patch_release(host_conserved);
}

#ifdef _OPENMP
void solver_advance_rk_omp(struct Solver *self, real a, real dt)
{
    struct Patch p = self->primitive;
    struct Patch u = self->conserved;
    struct Patch u0 = self->conserved_rk;
    struct Patch grad_i = self->grad_i;
    struct Patch grad_j = self->grad_j;
    struct Patch flux_i = self->flux_i;
    struct Patch flux_j = self->flux_j;
    int ni = self->mesh.ni;
    int nj = self->mesh.nj;

    #pragma omp parallel for
    for (int i = -1; i < ni + 1; ++i)
    {
        for (int j = -1; j < nj + 1; ++j)
        {
            if (0 <= j && j < nj)
            {
                gradient_i(p, grad_i, i, j);
            }
            if (0 <= i && i < ni)
            {
                gradient_j(p, grad_j, i, j);
            }
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < ni + 1; ++i)
    {
        for (int j = 0; j < nj + 1; ++j)
        {
            if (0 <= j && j < nj)
            {
                godunov_i(p, grad_i, flux_i, i, j);
            }
            if (0 <= i && i < ni)
            {
                godunov_j(p, grad_j, flux_j, i, j);
            }
        }
    }

    FOR_EACH_OMP(u) {
        update_conserved_and_primitive(p, u, u0, flux_i, flux_j, self->mesh, a, dt, i, j);
    }
}

void solver_new_timestep_omp(struct Solver *self)
{
    struct Patch u = self->conserved;
    struct Patch u0 = self->conserved_rk;

    FOR_EACH_OMP(u0) {
        memcpy(GET(u0, i, j), GET(u, i, j), NCONS * sizeof(real));
    }
}

#elif defined __NVCC__
static void __global__ kernel_advance_rk(struct Mesh mesh, struct Patch primitive_in, struct Patch primitive_out, struct Patch conserved_rk, real a, real dt)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    real dx = mesh.dx;
    real dy = mesh.dy;

    if (i >= mesh.ni || j >= mesh.nj) {
        return;
    }

    real *un = GET(conserved_rk, i, j);
    real *pc = GET(primitive_in, i, j);
    real *pli = GET(primitive_in, i - 1, j);
    real *pri = GET(primitive_in, i + 1, j);
    real *plj = GET(primitive_in, i, j - 1);
    real *prj = GET(primitive_in, i, j + 1);
    real *pki = GET(primitive_in, i - 2, j);
    real *pti = GET(primitive_in, i + 2, j);
    real *pkj = GET(primitive_in, i, j - 2);
    real *ptj = GET(primitive_in, i, j + 2);

    real *pll = GET(primitive_in, i - 1, j - 1);
    real *plr = GET(primitive_in, i - 1, j + 1);
    real *prl = GET(primitive_in, i + 1, j - 1);
    real *prr = GET(primitive_in, i + 1, j + 1);

    real plip[NCONS];
    real plim[NCONS];
    real prip[NCONS];
    real prim[NCONS];
    real pljp[NCONS];
    real pljm[NCONS];
    real prjp[NCONS];
    real prjm[NCONS];

    real gxli[NCONS];
    real gxri[NCONS];
    real gyli[NCONS];
    real gyri[NCONS];
    real gxlj[NCONS];
    real gxrj[NCONS];
    real gylj[NCONS];
    real gyrj[NCONS];
    real gxcc[NCONS];
    real gycc[NCONS];

    plm_gradient(pki, pli, pc, gxli);
    plm_gradient(pli, pc, pri, gxcc);
    plm_gradient(pc, pri, pti, gxri);
    plm_gradient(pkj, plj, pc, gylj);
    plm_gradient(plj, pc, prj, gycc);
    plm_gradient(pc, prj, ptj, gyrj);
    plm_gradient(pll, pli, plr, gyli);
    plm_gradient(prl, pri, prr, gyri);
    plm_gradient(pll, plj, prl, gxlj);
    plm_gradient(plr, prj, prr, gxrj);

    for (int q = 0; q < NCONS; ++q)
    {
        //      +-------+-------+-------+
        //      |       |       |       |
        //  k   |   l  -|+  c  -|+  r   |   t
        //      |       |       |       |
        //      +-------+-------+-------|

        plim[q] = pli[q] + 0.5 * gxli[q];
        plip[q] = pc [q] - 0.5 * gxcc[q];
        prim[q] = pc [q] + 0.5 * gxcc[q];
        prip[q] = pri[q] - 0.5 * gxri[q];

        pljm[q] = plj[q] + 0.5 * gylj[q];
        pljp[q] = pc [q] - 0.5 * gycc[q];
        prjm[q] = pc [q] + 0.5 * gycc[q];
        prjp[q] = prj[q] - 0.5 * gyrj[q];
    }

    real fli[NCONS];
    real fri[NCONS];
    real flj[NCONS];
    real frj[NCONS];
    real uc[NCONS];

    riemann_hlle(plim, plip, fli, 1.0, 0);
    riemann_hlle(prim, prip, fri, 1.0, 0);
    riemann_hlle(pljm, pljp, flj, 1.0, 1);
    riemann_hlle(prjm, prjp, frj, 1.0, 1);

    // totally ad-hoc viscous flux, just to force gradients to be used:
    fli[1] += gxli[2] * 1e-6;
    flj[2] += gxlj[2] * 1e-6;
    fri[1] += gyri[1] * 1e-6;
    frj[2] += gyrj[1] * 1e-6;

    primitive_to_conserved(pc, uc);

    for (int q = 0; q < NCONS; ++q)
    {
        uc[q] -= ((fri[q] - fli[q]) / dx + (frj[q] - flj[q]) / dy) * dt;
        uc[q] = (1.0 - a) * uc[q] + a * un[q];
    }
    real *pout = GET(primitive_out, i, j);
    conserved_to_primitive(uc, pout);
}

void __global__ kernel_prim_to_cons(struct Patch primitive, struct Patch conserved)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= conserved.count[0] || j >= conserved.count[1]) {
        return;
    }
    real *p = GET(primitive, i, j);
    real *u = GET(conserved, i, j);
    __syncthreads();
    primitive_to_conserved(p, u);
}

#define SWAP(x, y, T) do { T SWAP = x; x = y; y = SWAP; } while (0)
#define THREAD_DIM 8

EXTERN_C void solver_advance_rk_gpu(struct Solver *self, real a, real dt)
{
    dim3 bs = dim3(THREAD_DIM, THREAD_DIM);
    dim3 bd = dim3((self->mesh.ni + bs.x - 1) / bs.x, (self->mesh.nj + bs.y - 1) / bs.y);

    kernel_advance_rk<<<bd, bs>>>(self->mesh, self->primitive, self->primitive_out, self->conserved_rk, a, dt);
    SWAP(self->primitive, self->primitive_out, struct Patch);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error)
        printf("%s\n", cudaGetErrorString(error));
}

EXTERN_C void solver_new_timestep_gpu(struct Solver *self)
{
    dim3 bs = dim3(THREAD_DIM, THREAD_DIM);
    dim3 bd = dim3((self->mesh.ni + bs.x - 1) / bs.x, (self->mesh.nj + bs.y - 1) / bs.y);
    kernel_prim_to_cons<<<bd, bs>>>(self->primitive, self->conserved_rk);
}

#elif defined GPU_STUBS
void solver_advance_rk_gpu(struct Solver *self, real a, real dt)
{
    (void)self;
    (void)a;
    (void)dt;
}
void solver_new_timestep_gpu(struct Solver* self)
{
    (void)self;
}

#else
void solver_advance_rk_cpu(struct Solver *self, real a, real dt)
{
    struct Patch p = self->primitive;
    struct Patch u = self->conserved;
    struct Patch u0 = self->conserved_rk;
    struct Patch grad_i = self->grad_i;
    struct Patch grad_j = self->grad_j;
    struct Patch flux_i = self->flux_i;
    struct Patch flux_j = self->flux_j;

    FOR_EACH(grad_i) gradient_i(p, grad_i, i, j);
    FOR_EACH(grad_j) gradient_j(p, grad_j, i, j);
    FOR_EACH(flux_i) godunov_i(p, grad_i, flux_i, i, j);
    FOR_EACH(flux_j) godunov_j(p, grad_j, flux_j, i, j);
    FOR_EACH(u) update_conserved_and_primitive(p, u, u0, flux_i, flux_j, self->mesh, a, dt, i, j);
}

void solver_new_timestep_cpu(struct Solver *self)
{
    struct Patch u = self->conserved;
    struct Patch u0 = self->conserved_rk;

    FOR_EACH(u0) {
        memcpy(GET(u0, i, j), GET(u, i, j), NCONS * sizeof(real));
    }
}
#endif
