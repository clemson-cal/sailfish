// ============================ MODES =========================================
// ============================================================================
#define EXEC_CPU 0
#define EXEC_OMP 1
#define EXEC_GPU 2

#if (EXEC_MODE != EXEC_GPU)
#include <math.h>
#include <stddef.h>
#define PRIVATE static
#define PUBLIC
#else
#define PRIVATE static __device__
#define PUBLIC extern "C" __global__
#endif


#define BC_INFLOW 0
#define BC_ZEROFLUX 1
#define COORDS_CARTESIAN 0
#define COORDS_SPHERICAL 1


// ============================ PHYSICS =======================================
// ============================================================================
#define NCONS 4
#define PLM_THETA 2.0
#define ADIABATIC_GAMMA (4.0 / 3.0)


// ============================ MATH ==========================================
// ============================================================================
#define min2(a, b) ((a) < (b) ? (a) : (b))
#define max2(a, b) ((a) > (b) ? (a) : (b))
#define min3(a, b, c) min2(a, min2(b, c))
#define max3(a, b, c) max2(a, max2(b, c))
#define sign(x) copysign(1.0, x)
#define minabs(a, b, c) min3(fabs(a), fabs(b), fabs(c))

PRIVATE double plm_gradient_scalar(double yl, double y0, double yr)
{
    double a = (y0 - yl) * PLM_THETA;
    double b = (yr - yl) * 0.5;
    double c = (yr - y0) * PLM_THETA;
    return 0.25 * fabs(sign(a) + sign(b)) * (sign(a) + sign(c)) * minabs(a, b, c);
}

PRIVATE void plm_gradient(double *yl, double *y0, double *yr, double *g)
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
PRIVATE double primitive_to_gamma_beta_squared(const double *prim)
{
    const double u1 = prim[1];
    return u1 * u1;
}

PRIVATE double primitive_to_lorentz_factor(const double *prim)
{
    return sqrt(1.0 + primitive_to_gamma_beta_squared(prim));
}

PRIVATE double primitive_to_gamma_beta_component(const double *prim)
{
    return prim[1];
}

PRIVATE double primitive_to_beta_component(const double *prim)
{
    const double w = primitive_to_lorentz_factor(prim);
    return prim[1] / w;
}

PRIVATE double primitive_to_enthalpy_density(const double* prim)
{
    const double rho = prim[0];
    const double pre = prim[2];
    return rho + pre * (1.0 + 1.0 / (ADIABATIC_GAMMA - 1.0));
}

PRIVATE void conserved_to_primitive(const double *cons, double *prim, double dv)
{
    const double newton_iter_max = 50;
    const double error_tolerance = 1e-12 * cons[0] / dv;
    const double gm              = ADIABATIC_GAMMA;
    const double m               = cons[0] / dv;
    const double tau             = cons[2] / dv;
    const double ss              = cons[1] / dv * cons[1] / dv;
    int iteration                = 0;
    double p                     = prim[2];
    double w0;

    while (1) {
        const double et = tau + p + m;
        const double b2 = min2(ss / et / et, 1.0 - 1e-10);
        const double w2 = 1.0 / (1.0 - b2);
        const double w  = sqrt(w2);
        const double e  = (tau + m * (1.0 - w) + p * (1.0 - w2)) / (m * w);
        const double d  = m / w;
        const double h  = 1.0 + e + p / d;
        const double a2 = gm * p / (d * h);
        const double f  = d * e * (gm - 1.0) - p;
        const double g  = b2 * a2 - 1.0;

        p -= f / g;

        if (fabs(f) < error_tolerance || iteration == newton_iter_max) {
            w0 = w;
            break;
        }
        iteration += 1;
    }

    prim[0] = m / w0;
    prim[1] = w0 * cons[1] / dv / (tau + m + p);
    prim[2] = p;
    prim[3] = cons[3] / cons[0];

    double mach_ceiling = 1000.0;
    double u = prim[1];
    double e = prim[2] / prim[0] * 3.0;
    double emin = u * u / (1.0 + u * u) / pow(mach_ceiling, 2.0);

    if (e < emin) {
        prim[2] = prim[0] * emin * (ADIABATIC_GAMMA - 1.0);
    }

    // if (prim[2] < 0.0 || prim[2] != prim[2]) {
    //     printf("[FATAL] srhd_1d got negative pressure p=%e at r=%e\n", prim[2], 0.0);
    //     exit(1);
    // }
}

PRIVATE void primitive_to_conserved(const double *prim, double *cons, double dv)
{
    const double rho = prim[0];
    const double u1 = prim[1];
    const double pre = prim[2];

    const double w = primitive_to_lorentz_factor(prim);
    const double h = primitive_to_enthalpy_density(prim) / rho;
    const double m = rho * w;

    cons[0] = dv * m;
    cons[1] = dv * m * h * u1;
    cons[2] = dv * m * (h * w - 1.0) - dv * pre;
    cons[3] = dv * m * prim[3];
}

PRIVATE void primitive_to_flux(const double *prim, const double *cons, double *flux)
{
    const double vn = primitive_to_beta_component(prim);
    const double pre = prim[2];
    const double s = prim[3]; // scalar concentration

    flux[0] = vn * cons[0];
    flux[1] = vn * cons[1] + pre;
    flux[2] = vn * cons[2] + pre * vn;
    flux[3] = vn * cons[0] * s;
}

PRIVATE double primitive_to_sound_speed_squared(const double *prim)
{
    const double pre = prim[2];
    const double rho_h = primitive_to_enthalpy_density(prim);
    return ADIABATIC_GAMMA * pre / rho_h;
}

PRIVATE void primitive_to_outer_wavespeeds(const double *prim, double *wavespeeds)
{
    const double a2 = primitive_to_sound_speed_squared(prim);
    const double un = primitive_to_gamma_beta_component(prim);
    const double uu = primitive_to_gamma_beta_squared(prim);
    const double vv = uu / (1.0 + uu);
    const double v2 = un * un / (1.0 + uu);
    const double vn = sqrt(v2);
    const double k0 = sqrt(a2 * (1.0 - vv) * (1.0 - vv * a2 - v2 * (1.0 - a2)));

    wavespeeds[0] = (vn * (1.0 - a2) - k0) / (1.0 - vv * a2);
    wavespeeds[1] = (vn * (1.0 - a2) + k0) / (1.0 - vv * a2);
}

PRIVATE void riemann_hlle(const double *pl, const double *pr, double v_face, double *flux)
{
    double ul[NCONS];
    double ur[NCONS];
    double fl[NCONS];
    double fr[NCONS];
    double al[2];
    double ar[2];

    primitive_to_conserved(pl, ul, 1.0);
    primitive_to_conserved(pr, ur, 1.0);
    primitive_to_flux(pl, ul, fl);
    primitive_to_flux(pr, ur, fr);
    primitive_to_outer_wavespeeds(pl, al);
    primitive_to_outer_wavespeeds(pr, ar);

    const double am = min2(al[0], ar[0]);
    const double ap = max2(al[1], ar[1]);

    if (v_face < am)
    {
        for (int q = 0; q < NCONS; ++q)
        {
            flux[q] = fl[q] - v_face * ul[q];
        }
    }
    else if (v_face > ap)
    {
        for (int q = 0; q < NCONS; ++q)
        {
            flux[q] = fr[q] - v_face * ur[q];
        }
    }
    else
    {    
        for (int q = 0; q < NCONS; ++q)
        {
            double u_hll = (ur[q] * ap - ul[q] * am + (fl[q] - fr[q]))           / (ap - am);
            double f_hll = (fl[q] * ap - fr[q] * am - (ul[q] - ur[q]) * ap * am) / (ap - am);
            flux[q] = f_hll - v_face * u_hll;
        }
    }
}


// ============================ GEOMETRY ======================================
// ============================================================================
PRIVATE double face_area(int coords, double x)
{
    switch (coords) {
        case COORDS_CARTESIAN: return 1.0;
        case COORDS_SPHERICAL: return x * x;
    }
    return 0.0;
}

PRIVATE double cell_volume(int coords, double x0, double x1) 
{
    switch (coords) {
        case COORDS_CARTESIAN: return x1 - x0;
        case COORDS_SPHERICAL: return (pow(x1, 3.0) - pow(x0, 3.0)) / 3.0;
    }
    return 0.0;
}

PRIVATE void geometric_source_terms(int coords, double x0, double x1, const double *prim, double *source)
{
    switch (coords) {
        case COORDS_SPHERICAL: {
            double p = prim[2];
            source[0] = 0.0;
            source[1] = p * (x1 * x1 - x0 * x0);
            source[2] = 0.0;
            source[3] = 0.0;
            break;
        }
        default: {
            source[0] = 0.0;
            source[1] = 0.0;
            source[2] = 0.0;
            source[3] = 0.0;
        }   
    }
}


// ============================ KERNELS =======================================
// ============================================================================


/**
 * Converts an array of primitive data to an array of conserved data.
 */
PUBLIC void srhd_1d_primitive_to_conserved(
    int num_zones,
    double *face_positions,  // :: $.shape == (num_zones + 1,)
    double *primitive,       // :: $.shape == (num_zones, 4)
    double *conserved,       // :: $.shape == (num_zones, 4)
    double scale_factor,     // :: $ > 0.0
    int coords)              // :: $ in [0, 1]
{
    #if (EXEC_MODE == EXEC_CPU)
    for (int i = 0; i < num_zones; ++i)
    #elif (EXEC_MODE == EXEC_OMP)
    #pragma omp parallel for
    for (int i = 0; i < num_zones; ++i)
    #elif (EXEC_MODE == EXEC_GPU)
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_zones) return;
    #endif

    {
        double *p = &primitive[NCONS * i];
        double *u = &conserved[NCONS * i];
        double yl = face_positions[i];
        double yr = face_positions[i + 1];
        double xl = yl * scale_factor;
        double xr = yr * scale_factor;
        double dv = cell_volume(coords, xl, xr);
        primitive_to_conserved(p, u, dv);
    }
}


/**
 * Converts an array of conserved data to an array of primitive data.
 */
PUBLIC void srhd_1d_conserved_to_primitive(
    int num_zones,
    double *face_positions, // :: $.shape == (num_zones + 1,)
    double *conserved,      // :: $.shape == (num_zones, 4)
    double *primitive,      // :: $.shape == (num_zones, 4)
    double scale_factor,    // :: $ > 0.0
    int coords)             // :: $ in [0, 1]
{
    #if (EXEC_MODE == EXEC_CPU)
    for (int i = 0; i < num_zones; ++i)
    #elif (EXEC_MODE == EXEC_OMP)
    #pragma omp parallel for
    for (int i = 0; i < num_zones; ++i)
    #elif (EXEC_MODE == EXEC_GPU)
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_zones) return;
    #endif

    {
        double *p = &primitive[NCONS * i];
        double *u = &conserved[NCONS * i];
        double yl = face_positions[i];
        double yr = face_positions[i + 1];
        double xl = yl * scale_factor;
        double xr = yr * scale_factor;
        double dv = cell_volume(coords, xl, xr);
        conserved_to_primitive(u, p, dv);
    }
}


/**
 * Updates an array of primitive data by advancing it a single Runge-Kutta
 * step.
 */
PUBLIC void srhd_1d_advance_rk(
    int num_zones,          // number of zones in the grid
    double *face_positions, // :: $.shape == (num_zones + 1,)
    double *conserved_rk,   // :: $.shape == (num_zones, 4)
    double *primitive_rd,   // :: $.shape == (num_zones, 4)
    double *conserved_rd,   // :: $.shape == (num_zones, 4)
    double *conserved_wr,   // :: $.shape == (num_zones, 4)
    double a0,              // scale factor at t=0
    double adot,            // scale factor derivative
    double time,            // current time
    double rk_param,        // runge-kutta parameter
    double dt,              // timestep size
    int coords)             // :: $ in [0, 1]
{
    #if (EXEC_MODE == EXEC_CPU)
    for (int i = 0; i < num_zones; ++i)
    #elif (EXEC_MODE == EXEC_OMP)
    #pragma omp parallel for
    for (int i = 0; i < num_zones; ++i)
    #elif (EXEC_MODE == EXEC_GPU)
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_zones) return;
    #endif

    {
        int ni = num_zones;
        double yl = face_positions[i];
        double yr = face_positions[i + 1];
        double xl = yl * (a0 + adot * time);
        double xr = yr * (a0 + adot * time);

        double *urk = &conserved_rk[NCONS * i];
        double *prd = &primitive_rd[NCONS * i];
        double *urd = &conserved_rd[NCONS * i];
        double *uwr = &conserved_wr[NCONS * i];
        double *pli = i >= 0 + 1 ? &primitive_rd[NCONS * (i - 1)] : NULL;
        double *pri = i < ni - 1 ? &primitive_rd[NCONS * (i + 1)] : NULL;
        double *pki = i >= 0 + 2 ? &primitive_rd[NCONS * (i - 2)] : NULL;
        double *pti = i < ni - 2 ? &primitive_rd[NCONS * (i + 2)] : NULL;

        double plip[NCONS];
        double plim[NCONS];
        double prip[NCONS];
        double prim[NCONS];
        double gxli[NCONS];
        double gxri[NCONS];
        double gxcc[NCONS];

        // NOTE: the gradient calculation here assumes smoothly varying face
        // separations. Also note plm_gradient initializes the gradients to zero
        // if any of the inputs are NULL.
        plm_gradient(pki, pli, prd, gxli);
        plm_gradient(pli, prd, pri, gxcc);
        plm_gradient(prd, pri, pti, gxri);

        for (int q = 0; q < NCONS; ++q)
        {
            plim[q] = pli ? pli[q] + 0.5 * gxli[q] : prd[q];
            plip[q] = prd[q] - 0.5 * gxcc[q];
            prim[q] = prd[q] + 0.5 * gxcc[q];
            prip[q] = pri ? pri[q] - 0.5 * gxri[q] : prd[q];
        }

        double fli[NCONS];
        double fri[NCONS];
        double sources[NCONS];
        double dal = face_area(coords, xl);
        double dar = face_area(coords, xr);

        riemann_hlle(plim, plip, yl * adot, fli);
        riemann_hlle(prim, prip, yr * adot, fri);
        geometric_source_terms(coords, xl, xr, prd, sources);

        for (int q = 0; q < NCONS; ++q)
        {
            uwr[q] = urd[q] + (fli[q] * dal - fri[q] * dar + sources[q]) * dt;
            uwr[q] = (1.0 - rk_param) * uwr[q] + rk_param * urk[q];
        }
    }
}
