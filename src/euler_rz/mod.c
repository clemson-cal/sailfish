#include <stdio.h>
#include <math.h>
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
#define NCONS 5
#define PLM_THETA 1.2
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
    for (int q = 0; q < NCONS; ++q)
    {
        g[q] = plm_gradient_scalar(yl[q], y0[q], yr[q]);
    }
}


// ============================ HYDRO =========================================
// ============================================================================
static __host__ __device__ real sound_speed_squared(const real *prim)
{
    real rho = prim[0];
    real pre = prim[3];
    return ADIABATIC_GAMMA * pre / rho;
}

static __host__ __device__ void conserved_to_primitive(const real *cons, real *prim)
{
    real rho = cons[0];
    real px = cons[1];
    real py = cons[2];
    real energy = cons[3];
    real vx = px / rho;
    real vy = py / rho;
    real kinetic_energy = 0.5 * rho * (vx * vx + vy * vy);
    real thermal_energy = energy - kinetic_energy;
    real pressure = thermal_energy * (ADIABATIC_GAMMA - 1.0);
    real scalar_concentration = cons[4] / rho;

    prim[0] = rho;
    prim[1] = vx;
    prim[2] = vy;
    prim[3] = pressure;
    prim[4] = scalar_concentration;
}

static __host__ __device__ void primitive_to_conserved(const real *prim, real *cons)
{
    real rho = prim[0];
    real vx = prim[1];
    real vy = prim[2];
    real pressure = prim[3];
    real px = vx * rho;
    real py = vy * rho;
    real kinetic_energy = 0.5 * rho * (vx * vx + vy * vy);
    real thermal_energy = pressure / (ADIABATIC_GAMMA - 1.0);
    real scalar_concentration = prim[4];

    cons[0] = rho;
    cons[1] = px;
    cons[2] = py;
    cons[3] = kinetic_energy + thermal_energy;
    cons[4] = scalar_concentration * rho;
}

static __host__ __device__ real primitive_to_velocity(const real *prim, int direction)
{
    switch (direction)
    {
        case 0: return prim[1];
        case 1: return prim[2];
        default: return 0.0;
    }
}

static __host__ __device__ void primitive_to_flux(
    const real *prim,
    const real *cons,
    real *flux,
    int direction)
{
    real vn = primitive_to_velocity(prim, direction);
    real pressure = prim[3];

    flux[0] = vn * cons[0];
    flux[1] = vn * cons[1] + pressure * (direction == 0);
    flux[2] = vn * cons[2] + pressure * (direction == 1);
    flux[3] = vn * cons[3] + pressure * vn;
    flux[4] = vn * cons[4];
}

static __host__ __device__ real primitive_to_a_star(const real *pl, const real *pr, const real am, const real ap, int direction)
{
    switch (direction)
    {
        case 0: return (pr[3] - pl[3] + pl[0] * pl[1] * (am - pl[1]) - pr[0] * pr[1] * (ap - pr[1])) / (pl[0] * (am - pl[1]) - pr[0] * (ap - pr[1]));
        case 1: return (pr[3] - pl[3] + pl[0] * pl[2] * (am - pl[2]) - pr[0] * pr[2] * (ap - pr[2])) / (pl[0] * (am - pl[2]) - pr[0] * (ap - pr[2]));
        default: return 0.0;
    }
}

static __host__ __device__ void primitive_to_flux_star(const real *pl, const real *ul, const real *pr, const real *ur, const real *fl, const real *fr, real *fl_star, real *fr_star, const real am, const real ap, const real a_star, int direction)
{
    real d_star[4]  = {0.0, 0.0, 0.0, a_star};

    d_star[direction + 1] = 1.0;

    real vnl = primitive_to_velocity(pl, direction);
    real vnr = primitive_to_velocity(pr, direction);

    for (int q = 0; q < (NCONS-1); ++q)
    {
        fl_star[q] = (a_star * (am * ul[q] - fl[q]) + am * (pl[3] + pl[0] * (am - vnl) * (a_star - vnl)) * d_star[q]) / (am - a_star);
        fr_star[q] = (a_star * (ap * ur[q] - fr[q]) + ap * (pr[3] + pr[0] * (ap - vnr) * (a_star - vnr)) * d_star[q]) / (ap - a_star);
    }
    fl_star[4] = (a_star * (am * ul[0] - fl[0])) * pl[4] / (am - a_star);
    fr_star[4] = (a_star * (ap * ur[0] - fr[0])) * pr[4] / (ap - a_star);
}

static __host__ __device__ void primitive_to_outer_wavespeeds(
    const real *prim,
    real *wavespeeds,
    int direction)
{
    real cs = sqrt(sound_speed_squared(prim));
    real vn = primitive_to_velocity(prim, direction);
    wavespeeds[0] = vn - cs;
    wavespeeds[1] = vn + cs;
}

static __host__ __device__ real primitive_max_wavespeed(const real *prim)
{
    real cs = sqrt(sound_speed_squared(prim));
    real vx = prim[1];
    real vy = prim[2];
    real ax = max2(fabs(vx - cs), fabs(vx + cs));
    real ay = max2(fabs(vy - cs), fabs(vy + cs));
    return max2(ax, ay);
}

static __host__ __device__ void geometrical_source_terms(const real *prim, real *cons, real xc, real dt)
{
    cons[1] += dt * prim[3] / xc;
}

static __host__ __device__ void physical_source_terms(const real *prim, real *cons, real yc, real dt)
{
    real g = 1.0;
    //real g = (4.491E-7) * 2.468 * 2.0 * 3.14159 * 0.228 * 0.228 * 0.228 / (yc - 1.0) / (yc - 1.0);
    cons[2] -= dt * prim[0] * g;
    cons[3] -= dt * prim[0] * prim[2] * g;
}

static __host__ __device__ void riemann_hlle(const real *pl, const real *pr, real *flux, int direction)
{
    real ul[NCONS];
    real ur[NCONS];
    real fl[NCONS];
    real fr[NCONS];
    real al[2];
    real ar[2];

    primitive_to_conserved(pl, ul);
    primitive_to_conserved(pr, ur);
    primitive_to_flux(pl, ul, fl, direction);
    primitive_to_flux(pr, ur, fr, direction);
    primitive_to_outer_wavespeeds(pl, al, direction);
    primitive_to_outer_wavespeeds(pr, ar, direction);

    const real am = min3(0.0, al[0], ar[0]); //Toro refers to am as Sl, ap as Sr
    const real ap = max3(0.0, al[1], ar[1]);

    for (int q = 0; q < NCONS; ++q)
    {
        flux[q] = (fl[q] * ap - fr[q] * am - (ul[q] - ur[q]) * ap * am) / (ap - am);
    }
}

static __host__ __device__ void riemann_hllc(const real *pl, const real *pr, real *flux, int direction)
{
    real ul[NCONS];
    real ur[NCONS];
    real fl[NCONS];
    real fr[NCONS];
    real al[2];
    real ar[2];

    real fl_star[NCONS];
    real fr_star[NCONS];

    primitive_to_conserved(pl, ul);
    primitive_to_conserved(pr, ur);
    primitive_to_flux(pl, ul, fl, direction);
    primitive_to_flux(pr, ur, fr, direction);
    primitive_to_outer_wavespeeds(pl, al, direction);
    primitive_to_outer_wavespeeds(pr, ar, direction);

    const real am = min3(0.0, al[0], ar[0]); //Toro refers to am as Sl, ap as Sr
    const real ap = max3(0.0, al[1], ar[1]);

    real a_star = primitive_to_a_star(pl, pr, am, ap, direction);
    primitive_to_flux_star(pl, ul, pr, ur, fl, fr, fl_star, fr_star, am, ap, a_star, direction);

    for (int q = 0; q < NCONS; ++q)
    {
        if (0 <= am) {
          flux[q] = fl[q];
        } else if ((am <= 0) && (0 < a_star)) {
          flux[q] = fl_star[q];
        } else if ((a_star <= 0) && (0 < ap)) {
          flux[q] = fr_star[q];
        } else {
          flux[q] = fr[q];
        }
    }
}


// ============================ PATCH =========================================
// ============================================================================
#define FOR_EACH(p) \
    for (int i = p.start[0]; i < p.start[0] + p.count[0]; ++i) \
    for (int j = p.start[1]; j < p.start[1] + p.count[1]; ++j)
#define FOR_EACH_OMP(p) \
_Pragma("omp parallel for") \
    for (int i = p.start[0]; i < p.start[0] + p.count[0]; ++i) \
    for (int j = p.start[1]; j < p.start[1] + p.count[1]; ++j)
#define GET(p, i, j) (p.data + p.jumps[0] * ((i) - p.start[0]) + p.jumps[1] * ((j) - p.start[1]))

struct Patch
{
    int start[2];
    int count[2];
    int jumps[2];
    int num_fields;
    real *data;
};

static struct Patch patch(struct Mesh mesh, int num_fields, int num_guard, real *data)
{
    struct Patch patch;
    patch.start[0] = -num_guard;
    patch.start[1] = -num_guard;
    patch.count[0] = mesh.ni + 2 * num_guard;
    patch.count[1] = mesh.nj + 2 * num_guard;
    patch.jumps[0] = num_fields * patch.count[1];
    patch.jumps[1] = num_fields;
    patch.num_fields = num_fields;
    patch.data = data;
    return patch;
}


// ============================ SCHEME ========================================
// ============================================================================
static __host__ __device__ void primitive_to_conserved_zone(
    struct Patch primitive,
    struct Patch conserved,
    int i,
    int j)
{
    real *p = GET(primitive, i, j);
    real *u = GET(conserved, i, j);
    primitive_to_conserved(p, u);
}

static __host__ __device__ void advance_rk_zone(
    struct Mesh mesh,
    struct Patch conserved_rk,
    struct Patch primitive_rd,
    struct Patch primitive_wr,
    real a,
    real dt,
    int i,
    int j)
{
    real dx = mesh.dx;
    real dy = mesh.dy;
    real xl = mesh.x0 + (i + 0.0) * dx;
    real xc = mesh.x0 + (i + 0.5) * dx;
    real xr = mesh.x0 + (i + 1.0) * dx;
    real yc = mesh.y0 + (j + 0.5) * dy;

    real *un = GET(conserved_rk, i, j);
    real *pcc = GET(primitive_rd, i, j);
    real *pli = GET(primitive_rd, i - 1, j);
    real *pri = GET(primitive_rd, i + 1, j);
    real *plj = GET(primitive_rd, i, j - 1);
    real *prj = GET(primitive_rd, i, j + 1);
    real *pki = GET(primitive_rd, i - 2, j);
    real *pti = GET(primitive_rd, i + 2, j);
    real *pkj = GET(primitive_rd, i, j - 2);
    real *ptj = GET(primitive_rd, i, j + 2);

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
    real gylj[NCONS];
    real gyrj[NCONS];
    real gxcc[NCONS];
    real gycc[NCONS];

    plm_gradient(pki, pli, pcc, gxli);
    plm_gradient(pli, pcc, pri, gxcc);
    plm_gradient(pcc, pri, pti, gxri);
    plm_gradient(pkj, plj, pcc, gylj);
    plm_gradient(plj, pcc, prj, gycc);
    plm_gradient(pcc, prj, ptj, gyrj);

    for (int q = 0; q < NCONS; ++q)
    {
        plim[q] = pli[q] + 0.5 * gxli[q];
        plip[q] = pcc[q] - 0.5 * gxcc[q];
        prim[q] = pcc[q] + 0.5 * gxcc[q];
        prip[q] = pri[q] - 0.5 * gxri[q];

        pljm[q] = plj[q] + 0.5 * gylj[q];
        pljp[q] = pcc[q] - 0.5 * gycc[q];
        prjm[q] = pcc[q] + 0.5 * gycc[q];
        prjp[q] = prj[q] - 0.5 * gyrj[q];
    }

    real fli[NCONS];
    real fri[NCONS];
    real flj[NCONS];
    real frj[NCONS];
    real ucc[NCONS];

    riemann_hlle(plim, plip, fli, 0);
    riemann_hlle(prim, prip, fri, 0);
    riemann_hlle(pljm, pljp, flj, 1);
    riemann_hlle(prjm, prjp, frj, 1);

    primitive_to_conserved(pcc, ucc);
    geometrical_source_terms(pcc, ucc, xc, dt);
    physical_source_terms(pcc, ucc, yc, dt);

    for (int q = 0; q < NCONS; ++q)
    {
        ucc[q] -= ((xr * fri[q] - xl * fli[q]) / xc / dx + (frj[q] - flj[q]) / dy) * dt;
        ucc[q] = (1.0 - a) * ucc[q] + a * un[q];
    }
    real *pout = GET(primitive_wr, i, j);
    conserved_to_primitive(ucc, pout);
}

static __host__ __device__ void wavespeed_zone(
    struct Patch primitive,
    struct Patch wavespeed,
    int i,
    int j)
{
    real *pc = GET(primitive, i, j);
    GET(wavespeed, i, j)[0] = primitive_max_wavespeed(pc);
}


// ============================ KERNELS =======================================
// ============================================================================
#if defined(__NVCC__) || defined(__ROCM__)

static void __global__ primitive_to_conserved_kernel(
    struct Mesh mesh,
    struct Patch primitive,
    struct Patch conserved)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < mesh.ni && j < mesh.nj)
    {
        primitive_to_conserved_zone(primitive, conserved, i, j);
    }
}

static void __global__ advance_rk_kernel(
    struct Mesh mesh,
    struct Patch conserved_rk,
    struct Patch primitive_rd,
    struct Patch primitive_wr,
    real a,
    real dt)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < mesh.ni && j < mesh.nj)
    {
        advance_rk_zone(
            mesh,
            conserved_rk,
            primitive_rd,
            primitive_wr,
            a,
            dt,
            i, j
        );
    }
}

static void __global__ wavespeed_kernel(
    struct Patch primitive,
    struct Patch wavespeed)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    wavespeed_zone(primitive, wavespeed, i, j);
}

#endif


// ============================ PUBLIC API ====================================
// ============================================================================


/**
 * Converts an array of primitive data to an array of conserved data. The
 * array index space must follow the descriptions below.
 * @param mesh               The mesh [ni,     nj]
 * @param primitive_ptr[in]  [-2, -2] [ni + 4, nj + 4] [4]
 * @param conserved_ptr[out] [ 0,  0] [ni,     nj]     [4]
 * @param mode               The execution mode
 */
EXTERN_C void euler_rz_primitive_to_conserved(
    struct Mesh mesh,
    real *primitive_ptr,
    real *conserved_ptr,
    enum ExecutionMode mode)
{
    struct Patch primitive = patch(mesh, NCONS, 2, primitive_ptr);
    struct Patch conserved = patch(mesh, NCONS, 0, conserved_ptr);

    switch (mode) {
        case CPU: {
            FOR_EACH(conserved) {
                primitive_to_conserved_zone(primitive, conserved, i, j);
            }
            break;
        }

        case OMP: {
            #ifdef _OPENMP
            FOR_EACH_OMP(conserved) {
                primitive_to_conserved_zone(primitive, conserved, i, j);
            }
            #endif
            break;
        }

        case GPU: {
            #if defined(__NVCC__) || defined(__ROCM__)
            dim3 bs = dim3(16, 16);
            dim3 bd = dim3((mesh.nj + bs.x - 1) / bs.x, (mesh.ni + bs.y - 1) / bs.y);
            primitive_to_conserved_kernel<<<bd, bs>>>(mesh, primitive, conserved);
            #endif
            break;
        }
    }
}


/**
 * Updates an array of primitive data by advancing it a single Runge-Kutta
 * step.
 * @param mesh                  The mesh [ni,     nj]
 * @param conserved_rk_ptr[in]  [ 0,  0] [ni,     nj]     [4]
 * @param primitive_rd_ptr[in]  [-2, -2] [ni + 4, nj + 4] [4]
 * @param primitive_wr_ptr[out] [-2, -2] [ni + 4, nj + 4] [4]
 * @param a                     The RK averaging parameter
 * @param dt                    The time step
 * @param mode                  The execution mode
 */
EXTERN_C void euler_rz_advance_rk(
    struct Mesh mesh,
    real *conserved_rk_ptr,
    real *primitive_rd_ptr,
    real *primitive_wr_ptr,
    real a,
    real dt,
    enum ExecutionMode mode)
{
    struct Patch conserved_rk = patch(mesh, NCONS, 0, conserved_rk_ptr);
    struct Patch primitive_rd = patch(mesh, NCONS, 2, primitive_rd_ptr);
    struct Patch primitive_wr = patch(mesh, NCONS, 2, primitive_wr_ptr);

    switch (mode) {
        case CPU: {
            FOR_EACH(conserved_rk) {
                advance_rk_zone(
                    mesh,
                    conserved_rk,
                    primitive_rd,
                    primitive_wr,
                    a,
                    dt,
                    i, j
                );
            }
            break;
        }

        case OMP: {
            #ifdef _OPENMP
            FOR_EACH_OMP(conserved_rk) {
                advance_rk_zone(
                    mesh,
                    conserved_rk,
                    primitive_rd,
                    primitive_wr,
                    a,
                    dt,
                    i, j
                );
            }
            #endif
            break;
        }

        case GPU: {
            #if defined(__NVCC__) || defined(__ROCM__)
            dim3 bs = dim3(16, 16);
            dim3 bd = dim3((mesh.nj + bs.x - 1) / bs.x, (mesh.ni + bs.y - 1) / bs.y);
            advance_rk_kernel<<<bd, bs>>>(
                mesh,
                conserved_rk,
                primitive_rd,
                primitive_wr,
                a,
                dt
            );
            #endif
            break;
        }
    }
}


/**
 * Fill a buffer with the maximum wavespeed in each zone.
 * @param  mesh               The mesh [ni,     nj]
 * @param  primitive_ptr[in]  [-2, -2] [ni + 4, nj + 4] [4]
 * @param  wavespeed_ptr[out] [ 0,  0] [ni,     nj]     [1]
 * @param eos                 The EOS
 * @param mass_list           A list of point mass objects
 * @param mode                The execution mode
 */
EXTERN_C void euler_rz_wavespeed(
    struct Mesh mesh,
    real *primitive_ptr,
    real *wavespeed_ptr,
    enum ExecutionMode mode)
{
    struct Patch primitive = patch(mesh, NCONS, 2, primitive_ptr);
    struct Patch wavespeed = patch(mesh, 1,     0, wavespeed_ptr);

    switch (mode) {
        case CPU: {
            FOR_EACH(wavespeed) {
                wavespeed_zone(primitive, wavespeed, i, j);
            }
            break;
        }

        case OMP: {
            #ifdef _OPENMP
            FOR_EACH_OMP(wavespeed) {
                wavespeed_zone(primitive, wavespeed, i, j);
            }
            #endif
            break;
        }

        case GPU: {
            #if defined(__NVCC__) || defined(__ROCM__)
            dim3 bs = dim3(16, 16);
            dim3 bd = dim3((mesh.nj + bs.x - 1) / bs.x, (mesh.ni + bs.y - 1) / bs.y);
            wavespeed_kernel<<<bd, bs>>>(primitive, wavespeed);
            #endif
            break;
        }
    }
}


/**
 * Obtain the maximum value in an array of double's, using either a sequential
 * or an OpenMP reduction. Not implemented for GPU execution.
 *
 * @param data          The data [size]
 * @param size          The number of elements
 * @param mode          The execution mode
 */
EXTERN_C real euler_rz_maximum(
    real *data,
    unsigned long size,
    enum ExecutionMode mode)
{
    real a_max = 0.0;

    switch (mode) {
        case CPU: {
            for (unsigned long i = 0; i < size; ++i)
            {
                a_max = max2(a_max, data[i]);
            }
            break;
        }

        case OMP: {
            #ifdef _OPENMP
            #pragma omp parallel for reduction(max:a_max)
            for (unsigned long i = 0; i < size; ++i)
            {
                a_max = max2(a_max, data[i]);
            }
            #endif
            break;
        }

        case GPU: break; // Not implemented, use euler_rz_wavespeed
                         // followed by a GPU reduction.
    }
    return a_max;
}
