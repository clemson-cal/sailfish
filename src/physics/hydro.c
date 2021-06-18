#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>


// ============================ COMPAT ========================================
// ============================================================================
#ifndef __NVCC__
#define __host__
#define __device__
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
#else
// static void *compute_malloc(size_t count) { void *ptr; cudaMalloc(&ptr, count); return ptr; }
// static void compute_free(void *ptr) { cudaFree(ptr); }
// static void compute_memcpy_host_to_device(void *dst, const void *src, size_t count) { cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice); }
// static void compute_memcpy_device_to_host(void *dst, const void *src, size_t count) { cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost); }
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

#ifndef __NVCC__
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
#endif


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

static __device__ void primitive_to_conserved(const real *prim, real *cons)
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
        uc[q] = a * un[q] + (1.0 - a) * uc[q];
    }
    conserved_to_primitive(uc, pc);
}


// ============================ SOLVER ========================================
// ============================================================================
struct Solver
{
    struct Mesh mesh;
    struct Patch primitive;
    struct Patch conserved;
    struct Patch conserved_rk;
    struct Patch grad_i;
    struct Patch grad_j;
    struct Patch flux_i;
    struct Patch flux_j;
};

#ifndef __NVCC__
struct Solver *solver_new(struct Mesh mesh)
{
    int i0 = 0;
    int j0 = 0;
    int ni = mesh.ni;
    int nj = mesh.nj;

    struct Solver *self = (struct Solver*)malloc(sizeof(struct Solver));
    self->mesh = mesh;
    self->primitive = patch_new(i0 - 2, j0 - 2, ni + 4, nj + 4, NULL, BUFFER_MODE_DEVICE);
    self->conserved = patch_new(i0, i0, ni, nj, NULL, BUFFER_MODE_DEVICE);
    self->conserved_rk = patch_new(i0, j0, ni, nj, NULL, BUFFER_MODE_DEVICE);
    self->grad_i = patch_new(i0 - 1, j0, ni + 2, nj, NULL, BUFFER_MODE_DEVICE);
    self->grad_j = patch_new(i0, j0 - 1, ni, nj + 2, NULL, BUFFER_MODE_DEVICE);
    self->flux_i = patch_new(i0, j0, ni + 1, nj, NULL, BUFFER_MODE_DEVICE);
    self->flux_j = patch_new(i0, j0, ni, nj + 1, NULL, BUFFER_MODE_DEVICE);
    return self;
}

void solver_del(struct Solver *self)
{
    patch_release(self->primitive);
    patch_release(self->conserved);
    patch_release(self->conserved_rk);
    patch_release(self->grad_i);
    patch_release(self->grad_j);
    patch_release(self->flux_i);
    patch_release(self->flux_j);
    free(self);
}

struct Mesh solver_get_mesh(struct Solver *self)
{
    return self->mesh;
}

void solver_get_primitive(struct Solver *self, real *primitive_data)
{
    struct Patch primitive = patch_new(0, 0, self->mesh.ni, self->mesh.nj, primitive_data, BUFFER_MODE_VIEW);

    FOR_EACH(primitive) {
        compute_memcpy_device_to_host(GET(primitive, i, j), GET(self->primitive, i, j), NCONS * sizeof(real));
    }
}

void solver_set_primitive(struct Solver *self, real *primitive_data)
{
    struct Patch user_primitive = patch_new(0, 0, self->mesh.ni, self->mesh.nj, primitive_data, BUFFER_MODE_VIEW);
    struct Patch primitive = patch_new(-2, -2, self->mesh.ni + 4, self->mesh.nj + 4, NULL, BUFFER_MODE_HOST);
    struct Patch conserved = patch_new(0, 0, self->mesh.ni, self->mesh.nj, NULL, BUFFER_MODE_HOST);

    FOR_EACH(self->primitive) {
        int ii = min2(max2(i, 0), self->mesh.ni - 1);
        int jj = min2(max2(j, 0), self->mesh.nj - 1);

        real *pc = GET(primitive, i, j);
        real *uc = GET(conserved, ii, jj);

        for (int q = 0; q < NCONS; ++q)
        {
            pc[q] = GET(user_primitive, ii, jj)[q];
        }
        primitive_to_conserved(pc, uc);
    }

    compute_memcpy_host_to_device(self->primitive.data, primitive.data, BYTES(primitive));
    compute_memcpy_host_to_device(self->conserved.data, conserved.data, BYTES(conserved));

    patch_release(primitive);
    patch_release(conserved);
}
#endif

#ifndef __NVCC__
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
#endif

#ifndef __NVCC__
void solver_advance_rk_omp(struct Solver *self, real a, real dt)
{
#ifdef _OPENMP
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
#else // avoid unused variable warnings
    (void)self;
    (void)a;
    (void)dt;
#endif
}
#endif

#ifdef __NVCC__
extern "C" void solver_advance_rk_gpu(struct Solver *self)
{
    (void)self;
}
#elif GPU_STUBS
void solver_advance_rk_gpu(struct Solver* self)
{
    (void)self;
}
#endif

#ifndef __NVCC__
void solver_new_timestep_cpu(struct Solver *self)
{
    struct Patch u = self->conserved;
    struct Patch u0 = self->conserved_rk;

    FOR_EACH(u0) {
        memcpy(GET(u0, i, j), GET(u, i, j), NCONS * sizeof(real));
    }
}
#endif

#ifndef __NVCC__
void solver_new_timestep_omp(struct Solver *self)
{
#ifdef _OPENMP
    struct Patch u = self->conserved;
    struct Patch u0 = self->conserved_rk;

    FOR_EACH_OMP(u0) {
        memcpy(GET(u0, i, j), GET(u, i, j), NCONS * sizeof(real));
    }
#else // avoid unused variable warnings
    (void)self;
#endif
}
#endif

#ifdef __NVCC__
extern "C" void solver_new_timestep_gpu(struct Solver *self)
{
    cudaMemcpy(self->conserved_rk.data, self->conserved.data, ELEMENTS(self->conserved), cudaMemcpyDeviceToDevice);
}
#elif GPU_STUBS
void solver_new_timestep_gpu(struct Solver* self)
{
    (void)self;
}
#endif
