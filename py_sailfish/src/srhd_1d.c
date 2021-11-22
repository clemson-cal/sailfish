#include <math.h>
#include <stddef.h>


// ============================ MODES =========================================
// ============================================================================
enum ExecutionMode {
    CPU,
    OMP,
    GPU,
};

enum BoundaryCondition {
    Inflow,
    ZeroFlux,
};

enum Coordinates {
    Cartesian,
    Spherical,
};


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

static __host__ __device__ double plm_gradient_scalar(double yl, double y0, double yr)
{
    double a = (y0 - yl) * PLM_THETA;
    double b = (yr - yl) * 0.5;
    double c = (yr - y0) * PLM_THETA;
    return 0.25 * fabs(sign(a) + sign(b)) * (sign(a) + sign(c)) * minabs(a, b, c);
}

static __host__ __device__ void plm_gradient(double *yl, double *y0, double *yr, double *g)
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
static __host__ __device__ double primitive_to_gamma_beta_squared(const double *prim)
{
    const double u1 = prim[1];
    return u1 * u1;
}

static __host__ __device__ double primitive_to_lorentz_factor(const double *prim)
{
    return sqrt(1.0 + primitive_to_gamma_beta_squared(prim));
}

static __host__ __device__ double primitive_to_gamma_beta_component(const double *prim)
{
    return prim[1];
}

static __host__ __device__ double primitive_to_beta_component(const double *prim)
{
    const double w = primitive_to_lorentz_factor(prim);
    return prim[1] / w;
}

static __host__ __device__ double primitive_to_enthalpy_density(const double* prim)
{
    const double rho = prim[0];
    const double pre = prim[2];
    return rho + pre * (1.0 + 1.0 / (ADIABATIC_GAMMA - 1.0));
}

static __host__ __device__ void conserved_to_primitive(const double *cons, double *prim, double dv)
{
    const double newton_iter_max = 50;
    const double error_tolerance = 1e-12 * cons[0] / dv;
    const double gm              = ADIABATIC_GAMMA;
    const double m               = cons[0] / dv;
    const double tau             = cons[2] / dv;
    const double ss              = cons[1] / dv * cons[1] / dv;
    int iteration              = 0;
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

static __device__ __host__ void primitive_to_conserved(const double *prim, double *cons, double dv)
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

static __host__ __device__ void primitive_to_flux(const double *prim, const double *cons, double *flux)
{
    const double vn = primitive_to_beta_component(prim);
    const double pre = prim[2];
    const double s = prim[3]; // scalar concentration

    flux[0] = vn * cons[0];
    flux[1] = vn * cons[1] + pre;
    flux[2] = vn * cons[2] + pre * vn;
    flux[3] = vn * cons[0] * s;
}

static __host__ __device__ double primitive_to_sound_speed_squared(const double *prim)
{
    const double pre = prim[2];
    const double rho_h = primitive_to_enthalpy_density(prim);
    return ADIABATIC_GAMMA * pre / rho_h;
}

static __host__ __device__ void primitive_to_outer_wavespeeds(const double *prim, double *wavespeeds)
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

static __host__ __device__ void riemann_hlle(const double *pl, const double *pr, double v_face, double *flux)
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


// ============================ PATCH =========================================
// ============================================================================
static __host__ __device__ double face_area(enum Coordinates coords, double x)
{
    switch (coords) {
        case Cartesian: return 1.0;
        case Spherical: return x * x;
    }
    return 0.0;
}

static __host__ __device__ double cell_volume(enum Coordinates coords, double x0, double x1) 
{
    switch (coords) {
        case Cartesian: return x1 - x0;
        case Spherical: return (pow(x1, 3) - pow(x0, 3)) / 3.0;
    }
    return 0.0;
}

static __host__ __device__ void geometric_source_terms(enum Coordinates coords, double x0, double x1, const double *prim, double *source)
{
    switch (coords) {
        case Spherical: {
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
    double *data;
};

static struct Patch patch(int num_elements, int num_fields, double *data)
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
    struct Patch face_positions,
    struct Patch primitive,
    struct Patch conserved,
    double scale_factor,
    enum Coordinates coords,
    int i)
{
    double *p = GET(primitive, i);
    double *u = GET(conserved, i);
    double yl = *GET(face_positions, i);
    double yr = *GET(face_positions, i + 1);
    double xl = yl * scale_factor;
    double xr = yr * scale_factor;
    double dv = cell_volume(coords, xl, xr);
    primitive_to_conserved(p, u, dv);
}

static __host__ __device__ void conserved_to_primitive_zone(
    struct Patch face_positions,
    struct Patch conserved,
    struct Patch primitive,
    double scale_factor,
    enum Coordinates coords,
    int i)
{
    double *p = GET(primitive, i);
    double *u = GET(conserved, i);
    double yl = *GET(face_positions, i);
    double yr = *GET(face_positions, i + 1);
    double xl = yl * scale_factor;
    double xr = yr * scale_factor;
    double dv = cell_volume(coords, xl, xr);
    conserved_to_primitive(u, p, dv);
}

static __host__ __device__ void advance_rk_zone(
    struct Patch face_positions,
    struct Patch conserved_rk,
    struct Patch primitive_rd,
    struct Patch conserved_rd,
    struct Patch conserved_wr,
    enum BoundaryCondition bc,
    enum Coordinates coords,
    double a0,
    double adot,
    double t,
    double a,
    double dt,
    int i)
{
    if (bc == Inflow && i == 0) {
        return;
    }
    int ni = face_positions.count - 1;

    double yl = *GET(face_positions, i);
    double yr = *GET(face_positions, i + 1);
    double xl = yl * (a0 + adot * t);
    double xr = yr * (a0 + adot * t);

    double *urk = GET(conserved_rk, i);
    double *prd = GET(primitive_rd, i);
    double *urd = GET(conserved_rd, i);
    double *uwr = GET(conserved_wr, i);
    double *pli = i >= 0 + 1 ? GET(primitive_rd, i - 1) : NULL;
    double *pri = i < ni - 1 ? GET(primitive_rd, i + 1) : NULL;
    double *pki = i >= 0 + 2 ? GET(primitive_rd, i - 2) : NULL;
    double *pti = i < ni - 2 ? GET(primitive_rd, i + 2) : NULL;

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
        uwr[q] = (1.0 - a) * uwr[q] + a * urk[q];
    }
}


// ============================ KERNELS =======================================
// ============================================================================
#if defined(__NVCC__) || defined(__ROCM__)

static void __global__ primitive_to_conserved_kernel(
    struct Patch face_positions,
    struct Patch primitive,
    struct Patch conserved,
    double scale_factor,
    enum Coordinates coords)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < conserved.count)
    {
        primitive_to_conserved_zone(face_positions, primitive, conserved, scale_factor, coords, i);
    }
}

static void __global__ conserved_to_primitive_kernel(
    struct Patch face_positions,
    struct Patch conserved,
    struct Patch primitive,
    double scale_factor,
    enum Coordinates coords)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < conserved.count)
    {
        conserved_to_primitive_zone(face_positions, conserved, primitive, scale_factor, coords, i);
    }
}

static void __global__ advance_rk_kernel(
    struct Patch faces,
    struct Patch conserved_rk,
    struct Patch primitive_rd,
    struct Patch conserved_rd,
    struct Patch conserved_wr,
    enum BoundaryCondition bc,
    enum Coordinates coords,
    double a0,
    double adot,
    double t,
    double a,
    double dt)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < primitive_rd.count)
    {
        advance_rk_zone(faces, conserved_rk, primitive_rd, conserved_rd, conserved_wr, bc, coords, a0, adot, t, a, dt, i);
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
EXTERN_C int srhd_1d_primitive_to_conserved(
    int num_zones,
    double *face_positions_ptr,
    double *primitive_ptr,
    double *conserved_ptr,
    double scale_factor,
    enum Coordinates coords,
    enum ExecutionMode mode)
{
    struct Patch face_positions = patch(num_zones + 1, 1, face_positions_ptr);
    struct Patch primitive = patch(num_zones, NCONS, primitive_ptr);
    struct Patch conserved = patch(num_zones, NCONS, conserved_ptr);    

    switch (mode) {
        case CPU: {
            FOR_EACH(conserved) {
                primitive_to_conserved_zone(face_positions, primitive, conserved, scale_factor, coords, i);
            }
            return 0;
        }

        case OMP: {
            #ifdef _OPENMP
            FOR_EACH_OMP(conserved) {
                primitive_to_conserved_zone(face_positions, primitive, conserved, scale_factor, coords, i);
            }
            return 0;
            #else
            return 1;
            #endif
        }

        case GPU: {
            #if defined(__NVCC__) || defined(__ROCM__)
            dim3 bs = dim3(256);
            dim3 bd = dim3((num_zones + bs.x - 1) / bs.x);
            primitive_to_conserved_kernel<<<bd, bs>>>(face_positions, primitive, conserved, scale_factor, coords);
            return 0;
            #else
            return 1;
            #endif
        }
    }
    return 1;
}


/**
 * Converts an array of conserved data to an array of primitive data. The
 * array index space must follow the descriptions below.
 * @param faces              The faces [ni = num_zones]
 * @param primitive_ptr[in ] [0] [ni] [3]
 * @param conserved_ptr[out] [0] [ni] [3]
 * @param mode               The execution mode
 */
EXTERN_C int srhd_1d_conserved_to_primitive(
    int num_zones,
    double *face_positions_ptr,
    double *conserved_ptr,
    double *primitive_ptr,
    double scale_factor,
    enum Coordinates coords,
    enum ExecutionMode mode)
{
    struct Patch face_positions = patch(num_zones + 1, 1, face_positions_ptr);
    struct Patch primitive = patch(num_zones, NCONS, primitive_ptr);
    struct Patch conserved = patch(num_zones, NCONS, conserved_ptr);    

    switch (mode) {
        case CPU: {
            FOR_EACH(conserved) {
                conserved_to_primitive_zone(face_positions, conserved, primitive, scale_factor, coords, i);
            }
            return 0;
        }

        case OMP: {
            #ifdef _OPENMP
            FOR_EACH_OMP(conserved) {
                conserved_to_primitive_zone(face_positions, conserved, primitive, scale_factor, coords, i);
            }
            return 0;
            #else
            return 1;
            #endif
        }

        case GPU: {
            #if defined(__NVCC__) || defined(__ROCM__)
            dim3 bs = dim3(256);
            dim3 bd = dim3((num_zones + bs.x - 1) / bs.x);
            conserved_to_primitive_kernel<<<bd, bs>>>(face_positions, conserved, primitive, scale_factor, coords);
            return 0;
            #else
            return 1;
            #endif
        }
    }
    return 1;
}


/**
 * Updates an array of primitive data by advancing it a single Runge-Kutta
 * step.
 * @param face_positions_ptr[in] [num_zones + 1] [1]
 * @param conserved_rk_ptr[in]   [num_zones] [3]
 * @param primitive_rd_ptr[in]   [num_zones] [3]
 * @param conserved_rd_ptr[in]   [num_zones] [3]
 * @param conserved_wr_ptr[out]  [num_zones] [3]
 * @param a0                     The scale factor at t=0
 * @param adot                   The expansion rate
 * @param a                      The RK averaging parameter
 * @param dt                     The time step
 * @param bc                     The boundary conditions type
 * @param coords                 The coordinate system
 * @param mode                   The execution mode
 */
EXTERN_C int srhd_1d_advance_rk(
    int num_zones,
    double *face_positions_ptr,
    double *conserved_rk_ptr,
    double *primitive_rd_ptr,
    double *conserved_rd_ptr,
    double *conserved_wr_ptr,
    double a0,
    double adot,
    double t,
    double a,
    double dt,
    enum BoundaryCondition bc,
    enum Coordinates coords,
    enum ExecutionMode mode)
{
    struct Patch face_positions = patch(num_zones + 1, 1, face_positions_ptr);
    struct Patch conserved_rk = patch(num_zones, NCONS, conserved_rk_ptr);
    struct Patch primitive_rd = patch(num_zones, NCONS, primitive_rd_ptr);
    struct Patch conserved_rd = patch(num_zones, NCONS, conserved_rd_ptr);
    struct Patch conserved_wr = patch(num_zones, NCONS, conserved_wr_ptr);

    switch (mode) {
        case CPU: {
            FOR_EACH(conserved_rk) {
                advance_rk_zone(face_positions, conserved_rk, primitive_rd, conserved_rd, conserved_wr, bc, coords, a0, adot, t, a, dt, i);
            }
            return 0;
        }

        case OMP: {
            #ifdef _OPENMP
            FOR_EACH_OMP(conserved_rk) {
                advance_rk_zone(face_positions, conserved_rk, primitive_rd, conserved_rd, conserved_wr, bc, coords, a0, adot, t, a, dt, i);
            }
            return 0;
            #else
            return 1;
            #endif
        }

        case GPU: {
            #if defined(__NVCC__) || defined(__ROCM__)
            dim3 bs = dim3(256);
            dim3 bd = dim3((num_zones + bs.x - 1) / bs.x);
            advance_rk_kernel<<<bd, bs>>>(face_positions, conserved_rk, primitive_rd, conserved_rd, conserved_wr, bc, coords, a0, adot, t, a, dt);
            return 0;
            #else
            return 1;
            #endif
        }
    }
    return 1;
}
