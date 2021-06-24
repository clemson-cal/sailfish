#include <math.h>
#include "patch.h"


// ============================ MATH ==========================================
// ============================================================================
#define real double
#define min2(a, b) ((a) < (b) ? (a) : (b))
#define max2(a, b) ((a) > (b) ? (a) : (b))
#define min3(a, b, c) min2(a, min2(b, c))
#define max3(a, b, c) max2(a, max2(b, c))
#define sign(x) copysign(1.0, x)
#define minabs(a, b, c) min3(fabs(a), fabs(b), fabs(c))


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


// ============================ PUBLIC API ====================================
// ============================================================================
void conserved_to_primitive_cpu(struct Patch conserved, struct Patch primitive)
{
    int i0 = max2(conserved.start[0], primitive.start[0]);
    int j0 = max2(conserved.start[1], primitive.start[1]);
    int i1 = min2(conserved.start[0] + conserved.count[0], primitive.start[0] + conserved.count[0]);
    int j1 = min2(conserved.start[1] + conserved.count[1], primitive.start[1] + conserved.count[1]);

    for (int i = i0; i < i1; ++i)
    {
        for (int j = j0; j < j1; ++j)
        {
            real *u = GET(conserved, i, j);
            real *p = GET(primitive, i, j);
            conserved_to_primitive(u, p);
        }
    }
}

void primitive_to_conserved_cpu(struct Patch primitive, struct Patch conserved)
{
    int i0 = max2(conserved.start[0], primitive.start[0]);
    int j0 = max2(conserved.start[1], primitive.start[1]);
    int i1 = min2(conserved.start[0] + conserved.count[0], primitive.start[0] + conserved.count[0]);
    int j1 = min2(conserved.start[1] + conserved.count[1], primitive.start[1] + conserved.count[1]);

    for (int i = i0; i < i1; ++i)
    {
        for (int j = j0; j < j1; ++j)
        {
            real *u = GET(conserved, i, j);
            real *p = GET(primitive, i, j);
            primitive_to_conserved(p, u);
        }
    }
}
