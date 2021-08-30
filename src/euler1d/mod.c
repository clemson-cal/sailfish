#include <math.h>
#include <stddef.h>
#include "../sailfish.h"


// ============================ COMPAT ========================================
// ============================================================================
#ifdef __ROCM__
#include <hip/hip_runtime.h>
#endif

#if !defined(__NVCC__) && !defined(__ROCM__)
#define __device__
#define __host__
#define EXTERN_C
#else
#define EXTERN_C extern "C"
#endif


// ============================ PHYSICS =======================================
// ============================================================================
#define NCONS 3
#define PLM_THETA 2.0
#define ADIABATIC_GAMMA (5.0 / 3.0)


// ============================ MATH ==========================================
// ============================================================================
#define real double
#define min2(a, b) ((a) < (b) ? (a) : (b))
#define max2(a, b) ((a) > (b) ? (a) : (b))
#define min3(a, b, c) min2(a, min2(b, c))
#define max3(a, b, c) max2(a, max2(b, c))
#define sign(x) copysign(1.0, x)
#define minabs(a, b, c) min3(fabs(a), fabs(b), fabs(c))

static __host__ __device__ real plm_gradient_scalar(real yl, real y0, real yr)
{
    real a = (y0 - yl) * PLM_THETA;
    real b = (yr - yl) * 0.5;
    real c = (yr - y0) * PLM_THETA;
    return 0.25 * fabs(sign(a) + sign(b)) * (sign(a) + sign(c)) * minabs(a, b, c);
}

static __host__ __device__ void plm_gradient(real *yl, real *y0, real *yr, real *g)
{
    if (yl && y0 && yr)
    {
        for (int q = 0; q < NCONS; ++q)
        {
            g[q] = plm_gradient_scalar(yl[q], y0[q], yr[q]);
        }
    }
    else
    {
        for (int q = 0; q < NCONS; ++q)
        {
            g[q] = 0.0;
        }
    }
}


// ============================ HYDRO =========================================
// ============================================================================
static __host__ __device__ void conserved_to_primitive(const real *cons, real *prim)
{
    const real rho = cons[0];
    const real px = cons[1];
    const real energy = cons[2];

    const real vx = px / rho;
    const real kinetic_energy = 0.5 * rho * vx * vx;
    const real thermal_energy = energy - kinetic_energy;
    const real pressure = thermal_energy * (ADIABATIC_GAMMA - 1.0);

    prim[0] = rho;
    prim[1] = vx;
    prim[2] = pressure;
}

static __device__ __host__ void primitive_to_conserved(const real *prim, real *cons)
{
    const real rho = prim[0];
    const real vx = prim[1];
    const real pressure = prim[2];

    const real px = vx * rho;
    const real kinetic_energy = 0.5 * rho * vx * vx;
    const real thermal_energy = pressure / (ADIABATIC_GAMMA - 1.0);

    cons[0] = rho;
    cons[1] = px;
    cons[2] = kinetic_energy + thermal_energy;
}

static __host__ __device__ void primitive_to_flux(const real *prim, const real *cons, real *flux)
{
    const real vn = prim[1];
    const real pressure = prim[2];

    flux[0] = vn * cons[0];
    flux[1] = vn * cons[1] + pressure;
    flux[2] = vn * cons[2] + pressure * vn;
}

static __host__ __device__ real primitive_to_sound_speed_squared(const real *prim)
{
    const real rho = prim[0];
    const real pressure = prim[2];
    return ADIABATIC_GAMMA * pressure / rho;
}

static __host__ __device__ void primitive_to_outer_wavespeeds(const real *prim, real *wavespeeds)
{
    const real cs = sqrt(primitive_to_sound_speed_squared(prim));
    const real vn = prim[1];
    wavespeeds[0] = vn - cs;
    wavespeeds[1] = vn + cs;
}

static __host__ __device__ real primitive_to_max_wavespeed(const real *prim)
{
    real cs = sqrt(primitive_to_sound_speed_squared(prim));
    real vx = prim[1];
    real ax = max2(fabs(vx - cs), fabs(vx + cs));
    return ax;
}

static __host__ __device__ void riemann_hlle(const real *pl, const real *pr, real *flux)
{
    real ul[NCONS];
    real ur[NCONS];
    real fl[NCONS];
    real fr[NCONS];
    real al[2];
    real ar[2];

    primitive_to_conserved(pl, ul);
    primitive_to_conserved(pr, ur);
    primitive_to_flux(pl, ul, fl);
    primitive_to_flux(pr, ur, fr);
    primitive_to_outer_wavespeeds(pl, al);
    primitive_to_outer_wavespeeds(pr, ar);

    const real am = min3(0.0, al[0], ar[0]);
    const real ap = max3(0.0, al[1], ar[1]);

    for (int q = 0; q < NCONS; ++q)
    {
        flux[q] = (fl[q] * ap - fr[q] * am - (ul[q] - ur[q]) * ap * am) / (ap - am);
    }
}


// ============================ PATCH =========================================
// ============================================================================
static __host__ __device__ real face_area(enum Coordinates coords, real x)
{
    switch (coords) {
        case Cartesian: return 1.0;
        case SphericalPolar: return x * x;
    }
    return 0.0;
}

static __host__ __device__ real cell_volume(enum Coordinates coords, real x0, real x1) 
{
    switch (coords) {
        case Cartesian: return x1 - x0;
        case SphericalPolar: return (pow(x1, 3) - pow(x0, 3)) / 3.0;
    }
    return 0.0;
}

static __host__ __device__ void geometric_source_terms(enum Coordinates coords, real x0, real x1, const real *prim, real *source)
{
    switch (coords) {
        case SphericalPolar: {
            double p = prim[2];
            source[0] = 0.0;
            source[1] = p * (x1 * x1 - x0 * x0);
            source[2] = 0.0;
            break;
        }
        default: {
            source[0] = 0.0;
            source[1] = 0.0;
            source[2] = 0.0;
        }   
    }
}


// ============================ PATCH =========================================
// ============================================================================
#define FOR_EACH(p) \
    for (int i = p.start; i < p.start + p.count; ++i)
#define FOR_EACH_OMP(p) \
_Pragma("omp parallel for") \
    for (int i = p.start; i < p.start + p.count; ++i)
#define GET(p, i) (p.data + p.jumps * ((i) - p.start))

struct Patch
{
    int start;
    int count;
    int jumps;
    int num_fields;
    real *data;
};

static struct Patch patch(int num_elements, int num_fields, real *data)
{
    struct Patch patch;
    patch.start = 0;
    patch.count = num_elements;
    patch.jumps = num_fields;
    patch.num_fields = num_fields;
    patch.data = data;
    return patch;
}


// ============================ SCHEME ========================================
// ============================================================================
static __host__ __device__ void primitive_to_conserved_zone(
    struct Patch primitive,
    struct Patch conserved,
    int i)
{
    real *p = GET(primitive, i);
    real *u = GET(conserved, i);
    primitive_to_conserved(p, u);
}

static __host__ __device__ void advance_rk_zone(
    struct Patch face_positions,
    struct Patch conserved_rk,
    struct Patch primitive_rd,
    struct Patch primitive_wr,
    struct BoundaryCondition bc,
    enum Coordinates coords,
    real a,
    real dt,
    int i)
{
    if (bc.type == Inflow && i == 0) {
        return;
    }
    int ni = face_positions.count - 1;

    real xl = *GET(face_positions, i);
    real xr = *GET(face_positions, i + 1);

    real *un = GET(conserved_rk, i);
    real *pcc = GET(primitive_rd, i);
    real *pli = i >= 0 + 1 ? GET(primitive_rd, i - 1) : NULL;
    real *pri = i < ni - 1 ? GET(primitive_rd, i + 1) : NULL;
    real *pki = i >= 0 + 2 ? GET(primitive_rd, i - 2) : NULL;
    real *pti = i < ni - 2 ? GET(primitive_rd, i + 2) : NULL;

    real plip[NCONS];
    real plim[NCONS];
    real prip[NCONS];
    real prim[NCONS];
    real gxli[NCONS];
    real gxri[NCONS];
    real gxcc[NCONS];

    // NOTE: the gradient calculation here assumes smoothly varying face
    // separations. Also note plm_gradient initializes the gradients to zero
    // if any of the inputs are NULL.
    plm_gradient(pki, pli, pcc, gxli);
    plm_gradient(pli, pcc, pri, gxcc);
    plm_gradient(pcc, pri, pti, gxri);

    for (int q = 0; q < NCONS; ++q)
    {
        plim[q] = pli ? pli[q] + 0.5 * gxli[q] : pcc[q];
        plip[q] = pcc[q] - 0.5 * gxcc[q];
        prim[q] = pcc[q] + 0.5 * gxcc[q];
        prip[q] = pri ? pri[q] - 0.5 * gxri[q] : pcc[q];
    }

    real fli[NCONS];
    real fri[NCONS];
    real ucc[NCONS];
    real sources[NCONS];
    real dal = face_area(coords, xl);
    real dar = face_area(coords, xr);
    real dv = cell_volume(coords, xl, xr);

    riemann_hlle(plim, plip, fli);
    riemann_hlle(prim, prip, fri);
    primitive_to_conserved(pcc, ucc);
    geometric_source_terms(coords, xl, xr, pcc, sources);

    for (int q = 0; q < NCONS; ++q)
    {
        ucc[q] += (fli[q] * dal - fri[q] * dar + sources[q]) / dv * dt;
        ucc[q] = (1.0 - a) * ucc[q] + a * un[q];
    }
    real *pout = GET(primitive_wr, i);
    conserved_to_primitive(ucc, pout);
}

static __host__ __device__ void wavespeed_zone(
    struct Patch primitive,
    struct Patch wavespeed,
    int i)
{
    real *pc = GET(primitive, i);
    real a = primitive_to_max_wavespeed(pc);
    GET(wavespeed, i)[0] = a;
}


// ============================ KERNELS =======================================
// ============================================================================
#if defined(__NVCC__) || defined(__ROCM__)

static void __global__ primitive_to_conserved_kernel(
    struct Patch primitive,
    struct Patch conserved)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < conserved.count)
    {
        primitive_to_conserved_zone(primitive, conserved, i);
    }
}

static void __global__ advance_rk_kernel(
    struct Patch faces,
    struct Patch conserved_rk,
    struct Patch primitive_rd,
    struct Patch primitive_wr,
    struct BoundaryCondition bc,
    enum Coordinates coords,
    real a,
    real dt)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < primitive_wr.count)
    {
        advance_rk_zone(faces, conserved_rk, primitive_rd, primitive_wr, bc, coords, a, dt, i);
    }
}

static void __global__ wavespeed_kernel(
    struct Patch primitive,
    struct Patch wavespeed)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < wavespeed.count)
    {
        wavespeed_zone(primitive, wavespeed, i);
    }
}

#endif


// ============================ PUBLIC API ====================================
// ============================================================================


/**
 * Converts an array of primitive data to an array of conserved data. The
 * array index space must follow the descriptions below.
 * @param faces              The faces [ni = num_zones]
 * @param conserved_ptr[in]  [0] [ni] [3]
 * @param primitive_ptr[out] [0] [ni] [3]
 * @param mode               The execution mode
 */
EXTERN_C void euler1d_primitive_to_conserved(
    int num_zones,
    real *primitive_ptr,
    real *conserved_ptr,
    enum ExecutionMode mode)
{
    struct Patch primitive = patch(num_zones, NCONS, primitive_ptr);
    struct Patch conserved = patch(num_zones, NCONS, conserved_ptr);    

    switch (mode) {
        case CPU: {
            FOR_EACH(conserved) {
                primitive_to_conserved_zone(primitive, conserved, i);
            }
            break;
        }

        case OMP: {
            #ifdef _OPENMP
            FOR_EACH_OMP(conserved) {
                primitive_to_conserved_zone(primitive, conserved, i);
            }
            #endif
            break;
        }

        case GPU: {
            #if defined(__NVCC__) || defined(__ROCM__)
            dim3 bs = dim3(256);
            dim3 bd = dim3((num_zones + bs.x - 1) / bs.x);
            primitive_to_conserved_kernel<<<bd, bs>>>(primitive, conserved);
            #endif
            break;
        }
    }
}


/**
 * Updates an array of primitive data by advancing it a single Runge-Kutta
 * step.
 * @param face_positions_ptr[in] [num_zones + 1] [1]
 * @param conserved_rk_ptr[in]   [num_zones] [3]
 * @param primitive_rd_ptr[in]   [num_zones] [3]
 * @param primitive_wr_ptr[out]  [num_zones] [3]
 * @param a                      The RK averaging parameter
 * @param dt                     The time step
 * @param bc                     The boundary conditions type
 * @param coords                 The coordinate system
 * @param mode                   The execution mode
 */
EXTERN_C void euler1d_advance_rk(
    int num_zones,
    real *face_positions_ptr,
    real *conserved_rk_ptr,
    real *primitive_rd_ptr,
    real *primitive_wr_ptr,
    real a,
    real dt,
    struct BoundaryCondition bc,
    enum Coordinates coords,
    enum ExecutionMode mode)
{
    struct Patch face_positions = patch(num_zones + 1, 1, face_positions_ptr);
    struct Patch conserved_rk = patch(num_zones, NCONS, conserved_rk_ptr);
    struct Patch primitive_rd = patch(num_zones, NCONS, primitive_rd_ptr);
    struct Patch primitive_wr = patch(num_zones, NCONS, primitive_wr_ptr);

    switch (mode) {
        case CPU: {
            FOR_EACH(conserved_rk) {
                advance_rk_zone(face_positions, conserved_rk, primitive_rd, primitive_wr, bc, coords, a, dt, i);
            }
            break;
        }

        case OMP: {
            #ifdef _OPENMP
            FOR_EACH_OMP(conserved_rk) {
                advance_rk_zone(face_positions, conserved_rk, primitive_rd, primitive_wr, bc, coords, a, dt, i);
            }
            #endif
            break;
        }

        case GPU: {
            #if defined(__NVCC__) || defined(__ROCM__)
            dim3 bs = dim3(256);
            dim3 bd = dim3((num_zones + bs.x - 1) / bs.x);
            advance_rk_kernel<<<bd, bs>>>(face_positions, conserved_rk, primitive_rd, primitive_wr, bc, coords, a, dt);
            #endif
            break;
        }
    }
}


/**
 * Fill a buffer with the maximum wavespeed in each zone.
 * @param primitive_ptr[in]   [num_zones] [3]
 * @param wavespeed_ptr[out]  [num_zones] [1]
 * @param mode                The execution mode
 */
EXTERN_C void euler1d_wavespeed(
    int num_zones,
    real *primitive_ptr,
    real *wavespeed_ptr,
    enum ExecutionMode mode)
{
    struct Patch primitive = patch(num_zones, NCONS, primitive_ptr);
    struct Patch wavespeed = patch(num_zones, 1,     wavespeed_ptr);

    switch (mode) {
        case CPU: {
            FOR_EACH(wavespeed) {
                wavespeed_zone(primitive, wavespeed, i);
            }
            break;
        }

        case OMP: {
            #ifdef _OPENMP
            FOR_EACH_OMP(wavespeed) {
                wavespeed_zone(primitive, wavespeed, i);
            }
            #endif
            break;
        }

        case GPU: {
            #if defined(__NVCC__) || defined(__ROCM__)
            dim3 bs = dim3(256);
            dim3 bd = dim3((num_zones + bs.x - 1) / bs.x);
            wavespeed_kernel<<<bd, bs>>>(primitive, wavespeed);
            #endif
            break;
        }
    }
}


/**
 * Return the maximum wavespeed over all zones. Not implemented for GPU
 * execution.
 * @param primitive_ptr[in]   [num_zones] [3]
 * @param mode                The execution mode
 */
EXTERN_C real euler1d_max_wavespeed(
    int num_zones,
    real *primitive_ptr,
    enum ExecutionMode mode)
{
    struct Patch primitive = patch(num_zones, NCONS, primitive_ptr);
    real a_max = 0.0;

    switch (mode) {
        case CPU: {
            for (int i = 0; i < num_zones; ++i)
            {
                a_max = max2(a_max, primitive_to_max_wavespeed(GET(primitive, i)));
            }
            break;
        }

        case OMP: {
            #ifdef _OPENMP
            #pragma omp parallel for reduction(max:a_max)
            for (int i = 0; i < num_zones; ++i)
            {
                a_max = max2(a_max, primitive_to_max_wavespeed(GET(primitive, i)));
            }
            #endif
            break;
        }

        case GPU: break; // Not implemented, use euler1d_wavespeed
                         // followed by a GPU reduction.
    }
    return a_max;
}
