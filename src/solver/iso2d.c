#include <math.h>
#include "patch.h"


// ============================ COMPAT ========================================
// ============================================================================
#ifndef __NVCC__
#define __device__
#define __host__
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


// ============================ PUBLIC API ====================================
// ============================================================================
#ifdef API_MODE_CPU

void primitive_to_conserved_cpu(struct Patch primitive, struct Patch conserved)
{
    FOR_EACH(conserved) {
        real *u = GET(conserved, i, j);
        real *p = GET(primitive, i, j);
        primitive_to_conserved(p, u);
    }
}

#elif API_MODE_OMP

void primitive_to_conserved_omp(struct Patch primitive, struct Patch conserved)
{
    FOR_EACH_OMP(conserved) {
        real *u = GET(conserved, i, j);
        real *p = GET(primitive, i, j);
        primitive_to_conserved(p, u);
    }
}

#elif API_MODE_GPU

static void __global__ kernel_primitive_to_conserved(struct Patch primitive, struct Patch conserved)
{
    int i = conserved.start[0] + threadIdx.y + blockIdx.y * blockDim.y;
    int j = conserved.start[1] + threadIdx.x + blockIdx.x * blockDim.x;

    if (conserved.start[0] <= i && i < conserved.start[0] + conserved.count[0] &&
        conserved.start[1] <= j && j < conserved.start[1] + conserved.count[1])
    {
        real *p = GET(primitive, i, j);
        real *u = GET(conserved, i, j);
        primitive_to_conserved(p, u);
    }
}

extern "C" void primitive_to_conserved_gpu(struct Patch primitive, struct Patch conserved)
{
    dim3 bs = dim3(8, 8);
    dim3 bd = dim3((conserved.count[0] + bs.x - 1) / bs.x, (conserved.count[1] + bs.y - 1) / bs.y);
    kernel_primitive_to_conserved<<<bd, bs>>>(primitive, conserved);
}

#endif




#ifdef API_MODE_CPU

void advance_rk_cpu(
    struct Mesh mesh,
    struct Patch conserved_rk,
    struct Patch primitive_rd,
    struct Patch primitive_wr,
    real a,
    real dt)
{
    real dx = mesh.dx;
    real dy = mesh.dy;

    // ------------------------------------------------------------------------
    //                 tj
    //
    //      +-------+-------+-------+
    //      |       |       |       |
    //      |  lr   |  rj   |   rr  |
    //      |       |       |       |
    //      +-------+-------+-------+
    //      |       |       |       |
    //  ki  |  li  -|+  c  -|+  ri  |  ti
    //      |       |       |       |
    //      +-------+-------+-------+
    //      |       |       |       |
    //      |  ll   |  lj   |   rl  |
    //      |       |       |       |
    //      +-------+-------+-------+
    //
    //                 kj
    // ------------------------------------------------------------------------
    FOR_EACH(conserved_rk)
    {
        real *un = GET(conserved_rk, i, j);
        real *pc = GET(primitive_rd, i, j);
        real *pli = GET(primitive_rd, i - 1, j);
        real *pri = GET(primitive_rd, i + 1, j);
        real *plj = GET(primitive_rd, i, j - 1);
        real *prj = GET(primitive_rd, i, j + 1);
        real *pki = GET(primitive_rd, i - 2, j);
        real *pti = GET(primitive_rd, i + 2, j);
        real *pkj = GET(primitive_rd, i, j - 2);
        real *ptj = GET(primitive_rd, i, j + 2);
        real *pll = GET(primitive_rd, i - 1, j - 1);
        real *plr = GET(primitive_rd, i - 1, j + 1);
        real *prl = GET(primitive_rd, i + 1, j - 1);
        real *prr = GET(primitive_rd, i + 1, j + 1);

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
        // fli[1] += gxli[2] * 1e-6;
        // flj[2] += gxlj[2] * 1e-6;
        // fri[1] += gyri[1] * 1e-6;
        // frj[2] += gyrj[1] * 1e-6;

        primitive_to_conserved(pc, uc);

        for (int q = 0; q < NCONS; ++q)
        {
            uc[q] -= ((fri[q] - fli[q]) / dx + (frj[q] - flj[q]) / dy) * dt;
            uc[q] = (1.0 - a) * uc[q] + a * un[q];
        }
        real *pout = GET(primitive_wr, i, j);
        conserved_to_primitive(uc, pout);
    }
}

#elif API_MODE_OMP

void advance_rk_omp(
    struct Mesh mesh,
    struct Patch conserved_rk,
    struct Patch primitive_rd,
    struct Patch primitive_wr,
    real a,
    real dt)
{
    real dx = mesh.dx;
    real dy = mesh.dy;

    // ------------------------------------------------------------------------
    //                 tj
    //
    //      +-------+-------+-------+
    //      |       |       |       |
    //      |  lr   |  rj   |   rr  |
    //      |       |       |       |
    //      +-------+-------+-------+
    //      |       |       |       |
    //  ki  |  li  -|+  c  -|+  ri  |  ti
    //      |       |       |       |
    //      +-------+-------+-------+
    //      |       |       |       |
    //      |  ll   |  lj   |   rl  |
    //      |       |       |       |
    //      +-------+-------+-------+
    //
    //                 kj
    // ------------------------------------------------------------------------
    FOR_EACH_OMP(conserved_rk)
    {
        real *un = GET(conserved_rk, i, j);
        real *pc = GET(primitive_rd, i, j);
        real *pli = GET(primitive_rd, i - 1, j);
        real *pri = GET(primitive_rd, i + 1, j);
        real *plj = GET(primitive_rd, i, j - 1);
        real *prj = GET(primitive_rd, i, j + 1);
        real *pki = GET(primitive_rd, i - 2, j);
        real *pti = GET(primitive_rd, i + 2, j);
        real *pkj = GET(primitive_rd, i, j - 2);
        real *ptj = GET(primitive_rd, i, j + 2);
        real *pll = GET(primitive_rd, i - 1, j - 1);
        real *plr = GET(primitive_rd, i - 1, j + 1);
        real *prl = GET(primitive_rd, i + 1, j - 1);
        real *prr = GET(primitive_rd, i + 1, j + 1);

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
        // fli[1] += gxli[2] * 1e-6;
        // flj[2] += gxlj[2] * 1e-6;
        // fri[1] += gyri[1] * 1e-6;
        // frj[2] += gyrj[1] * 1e-6;

        primitive_to_conserved(pc, uc);

        for (int q = 0; q < NCONS; ++q)
        {
            uc[q] -= ((fri[q] - fli[q]) / dx + (frj[q] - flj[q]) / dy) * dt;
            uc[q] = (1.0 - a) * uc[q] + a * un[q];
        }
        real *pout = GET(primitive_wr, i, j);
        conserved_to_primitive(uc, pout);
    }
}

#elif API_MODE_GPU

static void __global__ kernel_advance_rk(
    struct Mesh mesh,
    struct Patch conserved_rk,
    struct Patch primitive_rd,
    struct Patch primitive_wr,
    real a,
    real dt)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    real dx = mesh.dx;
    real dy = mesh.dy;

    if (i >= mesh.ni || j >= mesh.nj) {
        return;
    }

    // ------------------------------------------------------------------------
    //                 tj
    //
    //      +-------+-------+-------+
    //      |       |       |       |
    //      |  lr   |  rj   |   rr  |
    //      |       |       |       |
    //      +-------+-------+-------+
    //      |       |       |       |
    //  ki  |  li  -|+  c  -|+  ri  |  ti
    //      |       |       |       |
    //      +-------+-------+-------+
    //      |       |       |       |
    //      |  ll   |  lj   |   rl  |
    //      |       |       |       |
    //      +-------+-------+-------+
    //
    //                 kj
    // ------------------------------------------------------------------------
    real *un = GET(conserved_rk, i, j);
    real *pc = GET(primitive_rd, i, j);
    real *pli = GET(primitive_rd, i - 1, j);
    real *pri = GET(primitive_rd, i + 1, j);
    real *plj = GET(primitive_rd, i, j - 1);
    real *prj = GET(primitive_rd, i, j + 1);
    real *pki = GET(primitive_rd, i - 2, j);
    real *pti = GET(primitive_rd, i + 2, j);
    real *pkj = GET(primitive_rd, i, j - 2);
    real *ptj = GET(primitive_rd, i, j + 2);
    real *pll = GET(primitive_rd, i - 1, j - 1);
    real *plr = GET(primitive_rd, i - 1, j + 1);
    real *prl = GET(primitive_rd, i + 1, j - 1);
    real *prr = GET(primitive_rd, i + 1, j + 1);

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
    // fli[1] += gxli[2] * 1e-6;
    // flj[2] += gxlj[2] * 1e-6;
    // fri[1] += gyri[1] * 1e-6;
    // frj[2] += gyrj[1] * 1e-6;

    primitive_to_conserved(pc, uc);

    for (int q = 0; q < NCONS; ++q)
    {
        uc[q] -= ((fri[q] - fli[q]) / dx + (frj[q] - flj[q]) / dy) * dt;
        uc[q] = (1.0 - a) * uc[q] + a * un[q];
    }
    real *pout = GET(primitive_wr, i, j);
    conserved_to_primitive(uc, pout);
}

void advance_rk_gpu(
    struct Mesh mesh,
    struct Patch conserved_rk,
    struct Patch primitive_rd,
    struct Patch primitive_wr,
    real a,
    real dt)
{
    dim3 bs = dim3(8, 8);
    dim3 bd = dim3((mesh.ni + bs.x - 1) / bs.x, (mesh.nj + bs.y - 1) / bs.y);
    kernel_advance_rk<<<bd, bs>>>(mesh, primitive_rd, primitive_wr, conserved_rk, a, dt);
}

#endif
