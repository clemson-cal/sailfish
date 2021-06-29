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


// ============================ PHYSICS =======================================
// ============================================================================
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
            real sound_speed_squared;
        } isothermal;

        struct
        {
            real mach_number_squared;
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


// ============================ GRAVITY =======================================
// ============================================================================
static __device__ real gravitational_potential(
    struct PointMass *masses,
    int num_masses,
    real x1,
    real y1)
{
    real phi = 0.0;

    for (int p = 0; p < num_masses; ++p)
    {
        real x0 = masses[p].x;
        real y0 = masses[p].y;
        real mp = masses[p].mass;
        real rs = masses[p].radius;

        real dx = x1 - x0;
        real dy = y1 - y0;
        real r2 = dx * dx + dy * dy;
        real r2_soft = r2 + rs * rs;

        phi -= mp / sqrt(r2_soft);
    }
    return phi;
}

static __device__ void point_mass_source_term(
    struct PointMass *mass,
    real x1,
    real y1,
    real dt,
    real sigma,
    real *delta_cons)
{
    real x0 = mass->x;
    real y0 = mass->y;
    real mp = mass->mass;
    real rs = mass->radius;

    real dx = x1 - x0;
    real dy = y1 - y0;
    real r2 = dx * dx + dy * dy;
    real r2_soft = r2 + rs * rs;
    real dr = sqrt(r2);
    real mag = sigma * mp / r2_soft;
    real fx = -mag * dx / dr;
    real fy = -mag * dy / dr;
    real sink_rate = 0.0;

    if (dr < 4.0 * rs)
    {
        sink_rate = mass->rate * exp(-pow(dr / rs, 4.0));
    }

    // NOTE: This is a force-free sink.
    delta_cons[0] = dt * sigma * sink_rate * -1.0;
    delta_cons[1] = dt * fx;
    delta_cons[2] = dt * fy;
}

static __device__ void point_masses_source_term(
    struct PointMass* masses,
    int num_masses,
    real x1,
    real y1,
    real dt,
    real sigma,
    real *cons)
{
    for (int p = 0; p < num_masses; ++p)
    {
        real delta_cons[NCONS];
        point_mass_source_term(&masses[p], x1, y1, dt, sigma, delta_cons);

        for (int q = 0; q < NCONS; ++q)
        {
            cons[q] += delta_cons[q];
        }
    }
}


// ============================ EOS AND BUFFER ================================
// ============================================================================
static __device__ real sound_speed_squared(
    struct EquationOfState *eos,
    real x,
    real y,
    struct PointMass *masses,
    int num_masses)
{
    switch (eos->type)
    {
        case Isothermal:
            return eos->isothermal.sound_speed_squared;
        case LocallyIsothermal:
            return -gravitational_potential(masses, num_masses, x, y) / eos->locally_isothermal.mach_number_squared;
        case GammaLaw:
            return 1.0; // WARNING
    }
    return 0.0;
}

static __device__ void buffer_source_term(
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
            real rc = sqrt(xc * xc + yc * yc);
            real surface_density = buffer->keplerian.surface_density;
            real central_mass = buffer->keplerian.central_mass;
            real driving_rate = buffer->keplerian.driving_rate;
            real outer_radius = buffer->keplerian.outer_radius;
            real onset_width = buffer->keplerian.onset_width;
            real onset_radius = outer_radius - onset_width;

            if (rc > onset_radius)
            {
                real pf = surface_density * sqrt(central_mass / rc);
                real px = pf * (-yc / rc);
                real py = pf * ( xc / rc);
                real u0[NCONS] = {surface_density, px, py};

                real omega_outer = sqrt(central_mass / pow(onset_radius, 3.0));
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

static __device__ void shear_strain(const real *gx, const real *gy, real dx, real dy, real *s)
{
    real sxx = 4.0 / 3.0 * gx[1] / dx - 2.0 / 3.0 * gy[2] / dy;
    real sxy = 1.0 / 1.0 * gx[2] / dx + 1.0 / 1.0 * gy[1] / dy;
    real syx = 1.0 / 1.0 * gx[2] / dx + 1.0 / 1.0 * gy[1] / dy;
    real syy =-2.0 / 3.0 * gx[1] / dx + 4.0 / 3.0 * gy[2] / dy;
    s[0] = sxx;
    s[1] = sxy;
    s[2] = syx;
    s[3] = syy;
}

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

static void __global__ kernel(struct Patch primitive, struct Patch conserved)
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
    dim3 bs = dim3(16, 16);
    dim3 bd = dim3((conserved.count[0] + bs.x - 1) / bs.x, (conserved.count[1] + bs.y - 1) / bs.y);
    kernel<<<bd, bs>>>(primitive, conserved);
}

#endif

static __device__ void advance_rk_zone(
    struct Mesh mesh,
    struct Patch conserved_rk,
    struct Patch primitive_rd,
    struct Patch primitive_wr,
    struct EquationOfState eos,
    struct BufferZone buffer,
    struct PointMass *masses,
    int num_masses,
    real nu,
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
    real yl = mesh.y0 + (j + 0.0) * dy;
    real yc = mesh.y0 + (j + 0.5) * dy;
    real yr = mesh.y0 + (j + 1.0) * dy;

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
    real *pcc = GET(primitive_rd, i, j);
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

    plm_gradient(pki, pli, pcc, gxli);
    plm_gradient(pli, pcc, pri, gxcc);
    plm_gradient(pcc, pri, pti, gxri);
    plm_gradient(pkj, plj, pcc, gylj);
    plm_gradient(plj, pcc, prj, gycc);
    plm_gradient(pcc, prj, ptj, gyrj);
    plm_gradient(pll, pli, plr, gyli);
    plm_gradient(prl, pri, prr, gyri);
    plm_gradient(pll, plj, prl, gxlj);
    plm_gradient(plr, prj, prr, gxrj);

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

    real cs2li = sound_speed_squared(&eos, xl, yc, masses, num_masses);
    real cs2ri = sound_speed_squared(&eos, xr, yc, masses, num_masses);
    real cs2lj = sound_speed_squared(&eos, xc, yl, masses, num_masses);
    real cs2rj = sound_speed_squared(&eos, xc, yr, masses, num_masses);

    riemann_hlle(plim, plip, fli, cs2li, 0);
    riemann_hlle(prim, prip, fri, cs2ri, 0);
    riemann_hlle(pljm, pljp, flj, cs2lj, 1);
    riemann_hlle(prjm, prjp, frj, cs2rj, 1);

    if (nu > 0.0)
    {
        real sli[4];
        real sri[4];
        real slj[4];
        real srj[4];
        real scc[4];

        shear_strain(gxli, gyli, dx, dy, sli);
        shear_strain(gxri, gyri, dx, dy, sri);
        shear_strain(gxlj, gylj, dx, dy, slj);
        shear_strain(gxrj, gyrj, dx, dy, srj);
        shear_strain(gxcc, gycc, dx, dy, scc);

        fli[1] -= nu * (pli[0] * sli[0] + pcc[0] * scc[0]); // x-x
        fli[2] -= nu * (pli[0] * sli[1] + pcc[0] * scc[1]); // x-y
        fri[1] -= nu * (pcc[0] * scc[0] + pri[0] * sri[0]); // x-x
        fri[2] -= nu * (pcc[0] * scc[1] + pri[0] * sri[1]); // x-y
        flj[1] -= nu * (plj[0] * slj[2] + pcc[0] * scc[2]); // y-x
        flj[2] -= nu * (plj[0] * slj[3] + pcc[0] * scc[3]); // y-y
        frj[1] -= nu * (pcc[0] * scc[2] + prj[0] * srj[2]); // y-x
        frj[2] -= nu * (pcc[0] * scc[3] + prj[0] * srj[3]); // y-y
    }

    primitive_to_conserved(pcc, ucc);
    buffer_source_term(&buffer, xc, yc, dt, ucc);
    point_masses_source_term(masses, num_masses, xc, yc, dt, pcc[0], ucc);

    for (int q = 0; q < NCONS; ++q)
    {
        ucc[q] -= ((fri[q] - fli[q]) / dx + (frj[q] - flj[q]) / dy) * dt;
        ucc[q] = (1.0 - a) * ucc[q] + a * un[q];
    }
    real *pout = GET(primitive_wr, i, j);
    conserved_to_primitive(ucc, pout);
}

#ifdef API_MODE_CPU

void advance_rk_cpu(
    struct Mesh mesh,
    struct Patch conserved_rk,
    struct Patch primitive_rd,
    struct Patch primitive_wr,
    struct EquationOfState eos,
    struct BufferZone buffer,
    struct PointMass *masses,
    int num_masses,
    real nu,
    real a,
    real dt)
{
    FOR_EACH(conserved_rk)
    {
        advance_rk_zone(mesh, conserved_rk, primitive_rd, primitive_wr, eos, buffer, masses, num_masses, nu, a, dt, i, j);
    }
}

#elif API_MODE_OMP

void advance_rk_omp(
    struct Mesh mesh,
    struct Patch conserved_rk,
    struct Patch primitive_rd,
    struct Patch primitive_wr,
    struct EquationOfState eos,
    struct BufferZone buffer,
    struct PointMass *masses,
    int num_masses,
    real nu,
    real a,
    real dt)
{
    FOR_EACH_OMP(conserved_rk)
    {
        advance_rk_zone(mesh, conserved_rk, primitive_rd, primitive_wr, eos, buffer, masses, num_masses, nu, a, dt, i, j);
    }
}

#elif API_MODE_GPU

void __global__ kernel(
    struct Mesh mesh,
    struct Patch conserved_rk,
    struct Patch primitive_rd,
    struct Patch primitive_wr,
    struct EquationOfState eos,
    struct BufferZone buffer,
    struct PointMass *masses,
    int num_masses,
    real nu,
    real a,
    real dt)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    advance_rk_zone(mesh, conserved_rk, primitive_rd, primitive_wr, eos, buffer, masses, num_masses, nu, a, dt, i, j);
}

extern "C" void advance_rk_gpu(
    struct Mesh mesh,
    struct Patch conserved_rk,
    struct Patch primitive_rd,
    struct Patch primitive_wr,
    struct EquationOfState eos,
    struct BufferZone buffer,
    struct PointMass *masses,
    int num_masses,
    real nu,
    real a,
    real dt)
{
    dim3 bs = dim3(16, 16);
    dim3 bd = dim3((mesh.ni + bs.x - 1) / bs.x, (mesh.nj + bs.y - 1) / bs.y);
    struct PointMass *device_masses;
    cudaMalloc(&device_masses, num_masses * sizeof(struct PointMass));
    cudaMemcpy(device_masses, masses, num_masses * sizeof(struct PointMass), cudaMemcpyHostToDevice);
    kernel<<<bd, bs>>>(mesh, conserved_rk, primitive_rd, primitive_wr, eos, buffer, device_masses, num_masses, nu, a, dt);
    cudaFree(device_masses);
    cudaDeviceSynchronize();
}

#endif
