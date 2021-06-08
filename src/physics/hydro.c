#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CONCAT(a, b) a ## _ ## b
#define FUNC(a, b) CONCAT(a, b)
#define NCONS 3

#define max2(a, b) (a) > (b) ? (a) : (b)
#define min2(a, b) (a) < (b) ? (a) : (b)
#define max3(a, b, c) max2(a, max2(b, c))
#define min3(a, b, c) min2(a, min2(b, c))

#ifdef __NVCC__
// ============================ CUDA VERSION ==================================
#ifdef SINGLE
// ============================ SINGLE PRECISION ==============================
#define PREFIX iso2d_cuda_f32
#define real float
#define square_root sqrtf
#define power powf
#define abs_val fabsf
#else
// ============================ DOUBLE PRECISION ==============================
#define PREFIX iso2d_cuda_f64
#define real double
#define square_root sqrt
#define power pow
#define abs_val fabs
#endif
// ============================ MEMORY =========================================
static void *compute_malloc(size_t count) { void *ptr; cudaMalloc(&ptr, count); return ptr; }
static void compute_free(void *ptr) { cudaFree(ptr); }
static void compute_memcpy_host_to_device(void *dst, const void *src, size_t count) { cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice); }
static void compute_memcpy_device_to_host(void *dst, const void *src, size_t count) { cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost); }

#else
// ============================ CPU VERSION ==================================
#define __host__
#define __device__
#ifdef SINGLE
// ============================ SINGLE PRECISION ==============================
#define PREFIX iso2d_cpu_f32
#define real float
#define square_root sqrtf
#define power powf
#define abs_val fabsf
#else
// ============================ DOUBLE PRECISION ==============================
#define PREFIX iso2d_cpu_f64
#define real double
#define square_root sqrt
#define power pow
#define abs_val fabs
#endif
// ============================ MEMORY =========================================
static void *compute_malloc(size_t count) { return malloc(count); }
static void compute_free(void *ptr) { free(ptr); }
static void compute_memcpy_host_to_device(void *dst, const void *src, size_t count) { memcpy(dst, src, count); }
static void compute_memcpy_device_to_host(void *dst, const void *src, size_t count) { memcpy(dst, src, count); }
#endif




// ============================================================================
static __host__ __device__ void conserved_to_primitive(const real *cons, real *prim)
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

static __host__ __device__ void primitive_to_conserved(const real *prim, real *cons)
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

static __host__ __device__ real primitive_to_velocity_component(const real *prim, int direction)
{
    switch (direction)
    {
        case 0: return prim[1];
        case 1: return prim[2];
        default: return 0.0;
    }
}

static __host__ __device__ void primitive_to_flux_vector(const real *prim, const real *cons, real *flux, real cs2, int direction)
{
    const real vn = primitive_to_velocity_component(prim, direction);
    const real rho = prim[0];
    const real pressure = rho * cs2;

    flux[0] = vn * cons[0];
    flux[1] = vn * cons[1] + pressure * (direction == 0);
    flux[2] = vn * cons[2] + pressure * (direction == 1);
}

static __host__ __device__ void primitive_to_outer_wavespeeds(const real *prim, real *wavespeeds, real cs2, int direction)
{
    const real cs = square_root(cs2);
    const real vn = primitive_to_velocity_component(prim, direction);
    wavespeeds[0] = vn - cs;
    wavespeeds[1] = vn + cs;
}

static __host__ __device__ void riemann_hlle(const real *pl, const real *pr, real *flux, real cs2, int direction)
{
    real ul[NCONS];
    real ur[NCONS];
    real fl[NCONS];
    real fr[NCONS];
    real al[2];
    real ar[2];

    primitive_to_conserved(pl, ul);
    primitive_to_conserved(pr, ur);
    primitive_to_flux_vector(pl, ul, fl, cs2, direction);
    primitive_to_flux_vector(pr, ur, fr, cs2, direction);
    primitive_to_outer_wavespeeds(pl, al, cs2, direction);
    primitive_to_outer_wavespeeds(pr, ar, cs2, direction);

    const real am = min3(0.0, al[0], ar[0]);
    const real ap = max3(0.0, al[1], ar[1]);

    for (int q = 0; q < NCONS; ++q)
    {
        flux[q] = (fl[q] * ap - fr[q] * am - (ul[q] - ur[q]) * ap * am) / (ap - am);
    }
}

struct Mesh
{
    unsigned long ni, nj;
    real x0, x1, y0, y1;
};

struct PointMass
{
    real x;
    real y;
    real mass;
    real rate;
    real radius;
};

enum EquationOfStateType
{
    Isothermal,
    LocallyIsothermal,
    GammaLaw,
};

struct EquationOfState
{
    enum EquationOfStateType type;

    union
    {
        struct
        {
            real sound_speed;
        } isothermal;

        struct
        {
            real mach_number;
        } locally_isothermal;

        struct
        {
            real gamma_law_index;
        } gamma_law;
    };
};

struct Solver
{
    struct Mesh mesh;
    real *primitive;
    real *conserved;
    real *conserved_rk;
    real *flux_i;
    real *flux_j;
    real *flux_k;
    real *gradient_i;
    real *gradient_j;
    real *gradient_k;
};

struct Solver* FUNC(PREFIX, solver_new)(struct Mesh mesh)
{
    struct Solver* self = (struct Solver*) malloc(sizeof(struct Solver));
    self->mesh = mesh;
    self->primitive = NULL;
    self->conserved = NULL;
    self->conserved_rk = NULL;
    self->flux_i = NULL;
    self->flux_j = NULL;
    self->flux_k = NULL;
    self->gradient_i = NULL;
    self->gradient_j = NULL;
    self->gradient_k = NULL;
    return self;
}

void FUNC(PREFIX, solver_del)(struct Solver *self)
{
    compute_free(self->primitive);
    compute_free(self->conserved);
    compute_free(self->conserved_rk);
    compute_free(self->flux_i);
    compute_free(self->flux_j);
    compute_free(self->flux_k);
    compute_free(self->gradient_i);
    compute_free(self->gradient_j);
    compute_free(self->gradient_k);
    free(self);
}

int FUNC(PREFIX, solver_set_primitive)(struct Solver *self, real *primitive)
{
    unsigned long ni = self->mesh.ni;
    unsigned long nj = self->mesh.nj;
    size_t num_bytes = NCONS * ni * nj * sizeof(real);
    real *conserved = (real*)malloc(num_bytes);

    for (unsigned long n = 0; n < ni * nj; ++n)
    {
        real *prim = &primitive[NCONS * n];
        real *cons = &conserved[NCONS * n];
        primitive_to_conserved(prim, cons);
    }
    if (self->primitive == NULL)
    {
        self->primitive = (real*) compute_malloc(num_bytes);
        self->conserved = (real*) compute_malloc(num_bytes);
    }
    compute_memcpy_host_to_device(self->primitive, primitive, num_bytes);
    compute_memcpy_host_to_device(self->conserved, conserved, num_bytes);
    free(conserved);
    return 0;
}

int FUNC(PREFIX, solver_get_primitive)(struct Solver *self, real *primitive)
{
    if (self->primitive == NULL)
    {
        return 1;
    }
    size_t num_bytes = NCONS * self->mesh.ni * self->mesh.nj * sizeof(real);
    compute_memcpy_device_to_host(primitive, self->primitive, num_bytes);
    return 0;
}

int FUNC(PREFIX, solver_advance_cons)(
    struct Solver *self,
    struct EquationOfState eos,
    struct PointMass *particles,
    unsigned long num_masses,
    real dt)
{
    if (self->primitive == NULL)
    {
        return 1;
    }

    long ni = self->mesh.ni;
    long nj = self->mesh.nj;
    const real dx = (self->mesh.x1 - self->mesh.x0) / ni;
    const real dy = (self->mesh.y1 - self->mesh.y0) / nj;

    for (long i = 0; i < ni; ++i)
    {
        for (long j = 0; j < nj; ++j)
        {
            long il = i - 1;
            long ir = i + 1;
            long jl = j - 1;
            long jr = j + 1;

            if (il == -1)
                il = 0;
            if (ir == ni)
                ir = ni - 1;
            if (jl == -1)
                jl = 0;
            if (jr == nj)
                jr = nj - 1;

            const real *pli = &self->primitive[NCONS * (il * nj + j)];
            const real *pri = &self->primitive[NCONS * (ir * nj + j)];
            const real *plj = &self->primitive[NCONS * (i * nj + jl)];
            const real *prj = &self->primitive[NCONS * (i * nj + jr)];
            real *cons = &self->conserved[NCONS * (i * nj + j)];
            real *prim = &self->primitive[NCONS * (i * nj + j)];

            real phi = 0.0;

            for (unsigned long p = 0; p < num_masses; ++p)
            {
                real sigma = prim[0];
                real x0 = particles[p].x;
                real y0 = particles[p].y;
                real mp = particles[p].mass;
                real rs = particles[p].radius;

                real x1 = self->mesh.x0 + (i + 0.5) * dx;
                real y1 = self->mesh.y0 + (j + 0.5) * dy;

                real dx = x1 - x0;
                real dy = y1 - y0;
                real r2 = dx * dx + dy * dy;
                real r2_soft = r2 + rs * rs;
                real dr = square_root(r2);
                real mag = sigma * mp / r2_soft;
                real fx = -mag * dx / dr;
                real fy = -mag * dy / dr;
                real sink_rate = particles[p].rate * (dr < rs);

                cons[0] -= sigma * sink_rate * dt;
                cons[1] += fx * dt;
                cons[2] += fy * dt;

                phi -= sigma * mp / square_root(r2_soft);
            }
            real cs2;

            switch (eos.type) {
                case Isothermal:
                    cs2 = power(eos.isothermal.sound_speed, 2.0);
                    break;
                case LocallyIsothermal:
                    cs2 = -phi / power(eos.locally_isothermal.mach_number, 2.0);
                    break;
                case GammaLaw:
                    cs2 = 1.0;
                    break;
            }

            real fli[NCONS];
            real fri[NCONS];
            real flj[NCONS];
            real frj[NCONS];
            riemann_hlle(pli, prim, fli, cs2, 0);
            riemann_hlle(prim, pri, fri, cs2, 0);
            riemann_hlle(plj, prim, flj, cs2, 1);
            riemann_hlle(prim, prj, frj, cs2, 1);

            for (unsigned long q = 0; q < NCONS; ++q)
            {
                cons[q] -= ((fri[q] - fli[q]) / dx + (frj[q] - flj[q]) / dy) * dt;
            }

        }
    }
    for (long n = 0; n < ni * nj; ++n)
    {
        real *prim = &self->primitive[NCONS * n];
        real *cons = &self->conserved[NCONS * n];
        conserved_to_primitive(cons, prim);
    }
    return 0;
}
