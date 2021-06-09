#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
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
#define hyperbolic_tangent tanhf
#define power powf
#define exponential expf
#define abs_val fabsf
#else
// ============================ DOUBLE PRECISION ==============================
#define PREFIX iso2d_cuda_f64
#define real double
#define square_root sqrt
#define hyperbolic_tangent tanh
#define power pow
#define exponential exp
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
#define hyperbolic_tangent tanhf
#define power powf
#define exponential expf
#define abs_val fabsf
#else
// ============================ DOUBLE PRECISION ==============================
#define PREFIX iso2d_cpu_f64
#define real double
#define square_root sqrt
#define hyperbolic_tangent tanh
#define power pow
#define exponential exp
#define abs_val fabs
#endif
// ============================ MEMORY =========================================
static void *compute_malloc(size_t count) { return malloc(count); }
static void compute_free(void *ptr) { free(ptr); }
static void compute_memcpy_host_to_device(void *dst, const void *src, size_t count) { memcpy(dst, src, count); }
static void compute_memcpy_device_to_host(void *dst, const void *src, size_t count) { memcpy(dst, src, count); }
#endif




// ============================ STRUCTS =======================================
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

enum BufferZoneType
{
    None,
    Keplerian,
};

struct BufferZone
{
    enum BufferZoneType type;

    union
    {
        struct
        {

        } none;

        struct
        {
            real surface_density;
            real central_mass;
            real driving_rate;
            real outer_radius;
            real onset_width;
        } keplerian;
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




// ============================ GRAVITY =======================================
static __device__ real gravitational_potential(const struct PointMass *masses, unsigned long num_masses, real x1, real y1)
{
    real phi = 0.0;

    for (unsigned long p = 0; p < num_masses; ++p)
    {
        real x0 = masses[p].x;
        real y0 = masses[p].y;
        real mp = masses[p].mass;
        real rs = masses[p].radius;

        real dx = x1 - x0;
        real dy = y1 - y0;
        real r2 = dx * dx + dy * dy;
        real r2_soft = r2 + rs * rs;

        phi -= mp / square_root(r2_soft);
    }
    return phi;
}

static __device__ void point_mass_source_term(struct PointMass mass, real x1, real y1, real dt, real sigma, real *delta_cons)
{
    real x0 = mass.x;
    real y0 = mass.y;
    real mp = mass.mass;
    real rs = mass.radius;

    real dx = x1 - x0;
    real dy = y1 - y0;
    real r2 = dx * dx + dy * dy;
    real r2_soft = r2 + rs * rs;
    real dr = square_root(r2);
    real mag = sigma * mp / r2_soft;
    real fx = -mag * dx / dr;
    real fy = -mag * dy / dr;
    real sink_rate = 0.0;

    if (dr < 4.0 * rs)
    {
        sink_rate = mass.rate * exponential(-power(dr / rs, 6.0));
    }
    delta_cons[0] = dt * sigma * sink_rate * -1.0;
    delta_cons[1] = dt * fx;
    delta_cons[2] = dt * fy;
}

static __device__ void point_masses_source_term(
    struct PointMass* masses,
    unsigned long num_masses,
    real x1,
    real y1,
    real dt,
    real sigma,
    real *cons)
{
    for (unsigned long p = 0; p < num_masses; ++p)
    {
        real delta_cons[NCONS];
        point_mass_source_term(masses[p], x1, y1, dt, sigma, delta_cons);

        for (int q = 0; q < NCONS; ++q)
        {
            cons[q] += delta_cons[q];
        }
    }
}




// ============================ HYDRO =========================================
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

static __device__ real primitive_to_velocity_component(const real *prim, int direction)
{
    switch (direction)
    {
        case 0: return prim[1];
        case 1: return prim[2];
        default: return 0.0;
    }
}

static __device__ void primitive_to_flux_vector(const real *prim, const real *cons, real *flux, real cs2, int direction)
{
    const real vn = primitive_to_velocity_component(prim, direction);
    const real rho = prim[0];
    const real pressure = rho * cs2;

    flux[0] = vn * cons[0];
    flux[1] = vn * cons[1] + pressure * (direction == 0);
    flux[2] = vn * cons[2] + pressure * (direction == 1);
}

static __device__ void primitive_to_outer_wavespeeds(const real *prim, real *wavespeeds, real cs2, int direction)
{
    const real cs = square_root(cs2);
    const real vn = primitive_to_velocity_component(prim, direction);
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

static inline __device__ real sound_speed_squared(
    struct EquationOfState *eos,
    real x,
    real y,
    struct PointMass *masses,
    unsigned long num_masses)
{
    switch (eos->type)
    {
        case Isothermal:
            return power(eos->isothermal.sound_speed, 2.0);
        case LocallyIsothermal:
            return -gravitational_potential(masses, num_masses, x, y) / power(eos->locally_isothermal.mach_number, 2.0);
        case GammaLaw:
            return 1.0; // WARNING
    }
}

static inline __device__ void buffer_source_term(
    struct BufferZone *buffer,
    real xc,
    real yc,
    real dt,
    real *cons)
{
    switch (buffer->type)
    {
        case None:
        {
            break;
        }
        case Keplerian:
        {
            real rc = square_root(xc * xc + yc * yc);
            real surface_density = buffer->keplerian.surface_density;
            real central_mass = buffer->keplerian.central_mass;
            real driving_rate = buffer->keplerian.driving_rate;
            real outer_radius = buffer->keplerian.outer_radius;
            real onset_width = buffer->keplerian.onset_width;
            real onset_radius = outer_radius - onset_width;

            if (rc > onset_radius)
            {
                real pf = surface_density * square_root(central_mass / rc);
                real px = pf * (-yc / rc);
                real py = pf * ( xc / rc);
                real u0[NCONS] = {surface_density, px, py};

                real omega_outer = square_root(central_mass / power(onset_radius, 3.0));
                real buffer_rate = driving_rate * omega_outer * max2(rc, 1.0);

                for (int q = 0; q < NCONS; ++q)
                {
                    cons[q] -= (cons[q] - u0[q]) * buffer_rate * dt;
                }
            }
            break;
        }
    }
}

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

struct Mesh FUNC(PREFIX, solver_get_mesh)(struct Solver *self)
{
    return self->mesh;
}

int FUNC(PREFIX, solver_advance)(
    struct Solver *self,
    struct EquationOfState eos,
    struct BufferZone buffer,
    struct PointMass *masses,
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

    #pragma omp parallel for
    for (long i = 0; i < ni; ++i)
    {
        for (long j = 0; j < nj; ++j)
        {
            real xc = self->mesh.x0 + (i + 0.5) * dx;
            real yc = self->mesh.y0 + (j + 0.5) * dy;

            real *cons = &self->conserved[NCONS * (i * nj + j)];
            real *prim = &self->primitive[NCONS * (i * nj + j)];

            if (self->flux_i != NULL && self->flux_j != NULL)
            {
                real *fli = self->flux_i + NCONS * ((i + 0) * (nj + 0) + j);
                real *fri = self->flux_i + NCONS * ((i + 1) * (nj + 0) + j);
                real *flj = self->flux_j + NCONS * (i * (nj + 1) + (j + 0));
                real *frj = self->flux_j + NCONS * (i * (nj + 1) + (j + 1));

                for (unsigned long q = 0; q < NCONS; ++q)
                {
                    cons[q] -= ((fri[q] - fli[q]) / dx + (frj[q] - flj[q]) / dy) * dt;
                }
            }
            else
            {
                real fli[NCONS];
                real fri[NCONS];
                real flj[NCONS];
                real frj[NCONS];

                long il = i - (i > 0);
                long ir = i + (i < ni - 1);
                long jl = j - (j > 0);
                long jr = j + (j < nj - 1);
                
                real xl = self->mesh.x0 + (i + 0.0) * dx;
                real xr = self->mesh.x0 + (i + 1.0) * dx;
                real yl = self->mesh.y0 + (j + 0.0) * dy;
                real yr = self->mesh.y0 + (j + 1.0) * dy;

                real cs2_li = sound_speed_squared(&eos, xl, yc, masses, num_masses);
                real cs2_ri = sound_speed_squared(&eos, xr, yc, masses, num_masses);
                real cs2_lj = sound_speed_squared(&eos, xc, yl, masses, num_masses);
                real cs2_rj = sound_speed_squared(&eos, xc, yr, masses, num_masses);

                real *pli = &self->primitive[NCONS * (il * nj + j)];
                real *pri = &self->primitive[NCONS * (ir * nj + j)];
                real *plj = &self->primitive[NCONS * (i * nj + jl)];
                real *prj = &self->primitive[NCONS * (i * nj + jr)];

                riemann_hlle(pli, prim, fli, cs2_li, 0);
                riemann_hlle(prim, pri, fri, cs2_ri, 0);
                riemann_hlle(plj, prim, flj, cs2_lj, 1);
                riemann_hlle(prim, prj, frj, cs2_rj, 1);

                for (unsigned long q = 0; q < NCONS; ++q)
                {
                    cons[q] -= ((fri[q] - fli[q]) / dx + (frj[q] - flj[q]) / dy) * dt;
                }
            }
            point_masses_source_term(masses, num_masses, xc, yc, dt, prim[0], cons);
            buffer_source_term(&buffer, xc, yc, dt, cons);

            // if (cons[0] <= 0.0)
            // {
            //     printf("ERROR: negative density %f at (%f %f)\n", cons[0], xc, yc);
            //     exit(1);
            // }
        }
    }

    #pragma omp parallel for
    for (long n = 0; n < ni * nj; ++n)
    {
        real *prim = &self->primitive[NCONS * n];
        real *cons = &self->conserved[NCONS * n];
        conserved_to_primitive(cons, prim);
    }
    return 0;
}

int FUNC(PREFIX, solver_compute_fluxes)(
    struct Solver *self,
    struct EquationOfState eos,
    struct PointMass *masses,
    unsigned long num_masses)
{
    long ni = self->mesh.ni;
    long nj = self->mesh.nj;
    const real dx = (self->mesh.x1 - self->mesh.x0) / ni;
    const real dy = (self->mesh.y1 - self->mesh.y0) / nj;

    if (self->primitive == NULL)
    {
        return 1;
    }
    if (self->flux_i == NULL && self->flux_j == NULL)
    {
        self->flux_i = (real*) compute_malloc((ni + 1) * nj * NCONS * sizeof(real));
        self->flux_j = (real*) compute_malloc(ni * (nj + 1) * NCONS * sizeof(real));
    }

    #pragma omp parallel for
    for (long i = 0; i < ni + 1; ++i)
    {
        for (long j = 0; j < nj; ++j)
        {
            long il = i - (i > 0);
            long ir = i;
            real x = self->mesh.x0 + (i + 0.0) * dx;
            real y = self->mesh.y0 + (j + 0.5) * dy;
            real cs2 = sound_speed_squared(&eos, x, y, masses, num_masses);
            real *pl = &self->primitive[NCONS * (il * nj + j)];
            real *pr = &self->primitive[NCONS * (ir * nj + j)];
            riemann_hlle(pl, pr, self->flux_i + NCONS * (i * (nj + 0) + j), cs2, 0);
        }
    }

    #pragma omp parallel for
    for (long i = 0; i < ni; ++i)
    {
        for (long j = 0; j < nj + 1; ++j)
        {
            long jl = j - (j > 0);
            long jr = j;
            real x = self->mesh.x0 + (i + 0.5) * dx;
            real y = self->mesh.y0 + (j + 0.0) * dy;
            real cs2 = sound_speed_squared(&eos, x, y, masses, num_masses);
            real *pl = &self->primitive[NCONS * (i * nj + jl)];
            real *pr = &self->primitive[NCONS * (i * nj + jr)];
            riemann_hlle(pl, pr, self->flux_j + NCONS * (i * (nj + 1) + j), cs2, 1);
        }
    }
    return 0;
}
