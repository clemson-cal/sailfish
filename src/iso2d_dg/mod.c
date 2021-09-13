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

#define MAX_INTERIOR_NODES 25
#define MAX_FACE_NODES 5
#define MAX_POLYNOMIALS 15
#define NCONS 3

// ============================ MATH ==========================================
// ============================================================================
#define real double
#define min2(a, b) ((a) < (b) ? (a) : (b))
#define max2(a, b) ((a) > (b) ? (a) : (b))
#define min3(a, b, c) min2(a, min2(b, c))
#define max3(a, b, c) max2(a, max2(b, c))
#define sign(x) copysign(1.0, x)
#define minabs(a, b, c) min3(fabs(a), fabs(b), fabs(c))

struct NodeData {
    real xsi_x;
    real xsi_y;
    real phi[MAX_POLYNOMIALS];
    real dphi_dx[MAX_POLYNOMIALS];
    real dphi_dy[MAX_POLYNOMIALS];
    real weight; 
};


struct Cell {
    struct NodeData interior_nodes[MAX_INTERIOR_NODES];
    struct NodeData face_nodes_li[MAX_FACE_NODES];
    struct NodeData face_nodes_ri[MAX_FACE_NODES];
    struct NodeData face_nodes_lj[MAX_FACE_NODES];
    struct NodeData face_nodes_rj[MAX_FACE_NODES];
    int order;
};


static int num_polynomials(struct Cell cell)
{
    switch (cell.order)
    {
        case 1: return 1;
        case 2: return 3;
        case 3: return 6;
        case 4: return 10;
        case 5: return 15;
        default: return 0;
    }
}

static int num_quadrature_points(struct Cell cell)
{
    return cell.order * cell.order;
}

// ============================ EOS ===========================================
// ============================================================================
static __host__ __device__ real sound_speed_squared(
    struct EquationOfState *eos)
{
    switch (eos->type)
    {
        case Isothermal:
            return eos->isothermal.sound_speed_squared;
        case LocallyIsothermal:
            return 1.0; // WARNING
        default:
            return 1.0; // WARNING
    }
}

// ============================ HYDRO =========================================
// ============================================================================
static __host__ __device__ void primitive_to_conserved(const real *prim, real *cons)
{
    real rho = prim[0];
    real vx = prim[1];
    real vy = prim[2];
    real px = vx * rho;
    real py = vy * rho;

    cons[0] = rho;
    cons[1] = px;
    cons[2] = py;
}

static __host__ __device__ void conserved_to_primitive(const real *cons, real *prim)
{
    real rho = cons[0];
    real px = cons[1];
    real py = cons[2];
    real vx = px / rho;
    real vy = py / rho;

    prim[0] = rho;
    prim[1] = vx;
    prim[2] = vy;
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
    real cs2,
    int direction)
{
    real vn = primitive_to_velocity(prim, direction);
    real rho = prim[0];
    real pressure = rho * cs2;

    flux[0] = vn * cons[0];
    flux[1] = vn * cons[1] + pressure * (direction == 0);
    flux[2] = vn * cons[2] + pressure * (direction == 1);
}

static __host__ __device__ void primitive_to_outer_wavespeeds(
    const real *prim,
    real *wavespeeds,
    real cs2,
    int direction)
{
    real cs = sqrt(cs2);
    real vn = primitive_to_velocity(prim, direction);
    wavespeeds[0] = vn - cs;
    wavespeeds[1] = vn + cs;
}

/*
static __host__ __device__ real primitive_max_wavespeed(const real *prim, real cs2)
{
    real cs = sqrt(cs2);
    real vx = prim[1];
    real vy = prim[2];
    real ax = max2(fabs(vx - cs), fabs(vx + cs));
    real ay = max2(fabs(vy - cs), fabs(vy + cs));
    return max2(ax, ay);
}
*/

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
static __host__ __device__ void primitive_to_weights_zone(
    struct Cell cell,
    struct Patch primitive,
    struct Patch weights,
    int i,
    int j)
{
    int n_quad = num_quadrature_points(cell);
    int n_poly = num_polynomials(cell);

    real *p_cell = GET(primitive, i, j);
    real *w_cell = GET(weights, i, j);

    for (int q = 0; q < NCONS; ++q)
    {
        for (int l = 0; l < n_poly; ++l)
        {
            w_cell[q * n_poly + l] = 0.0;
        }
    }

    for (int qp = 0; qp < n_quad; ++qp)
    {
        real u[NCONS];
        real *p = &p_cell[qp * NCONS];
        primitive_to_conserved(p, u);
        struct NodeData node = cell.interior_nodes[qp];

        for (int q = 0; q < NCONS; ++q)
        {
            for (int l = 0; l < n_poly; ++l)
            {
                w_cell[q * n_poly + l] += 0.25 * u[q] * node.phi[l] * node.weight;
            }
        }
    }
}

static __host__ __device__ void advance_rk_zone_dg(
    struct Cell cell,
    struct Mesh mesh,
    struct Patch weights_rd,
    struct Patch weights_wr,
    struct EquationOfState eos,
    real dt,
    int i,
    int j)
{
    real dx = mesh.dx;
    //real dy = mesh.dy;

    int n_quad = num_quadrature_points(cell);
    int n_poly = num_polynomials(cell);
    int n_face = cell.order;

    real *wij = GET(weights_rd, i, j);
    real *wli = GET(weights_rd, i - 1, j);
    real *wri = GET(weights_rd, i + 1, j);
    real *wlj = GET(weights_rd, i, j - 1);
    real *wrj = GET(weights_rd, i, j + 1);

    real ulip[NCONS];
    real ulim[NCONS];
    real urip[NCONS];
    real urim[NCONS];
    real uljp[NCONS];
    real uljm[NCONS];
    real urjp[NCONS];
    real urjm[NCONS];

    real plip[NCONS];
    real plim[NCONS];
    real prip[NCONS];
    real prim[NCONS];
    real pljp[NCONS];
    real pljm[NCONS];
    real prjp[NCONS];
    real prjm[NCONS];

    real fli[NCONS];
    real fri[NCONS];
    real flj[NCONS];
    real frj[NCONS];

    real dwij[NCONS * n_poly];

    for (int q = 0; q < NCONS; ++q)
    {
        for (int l = 0; l < n_poly; ++l)
        {
            dwij[q * n_poly + l] = 0.0;  
        }
    }

    // surface term
    for (int qp = 0; qp < n_face; ++qp)
    {
        for (int q = 0; q < NCONS; ++q)
        {
            ulim[q] = 0.0;
            ulip[q] = 0.0;
            urim[q] = 0.0;
            urip[q] = 0.0;
            uljm[q] = 0.0;
            uljp[q] = 0.0;
            urjm[q] = 0.0;
            urjp[q] = 0.0;

            for (int l = 0; l < n_poly; ++l)
            {
                ulim[q] += wli[q * n_poly + l] * cell.face_nodes_ri[qp].phi[l]; // right face of zone i-1 
                ulip[q] += wij[q * n_poly + l] * cell.face_nodes_li[qp].phi[l]; // left face of zone i
                urim[q] += wij[q * n_poly + l] * cell.face_nodes_ri[qp].phi[l]; // right face of zone i  
                urip[q] += wri[q * n_poly + l] * cell.face_nodes_li[qp].phi[l]; // left face of zone i + 1
                uljm[q] += wlj[q * n_poly + l] * cell.face_nodes_rj[qp].phi[l]; // top face of zone j-1 
                uljp[q] += wij[q * n_poly + l] * cell.face_nodes_lj[qp].phi[l]; // bottom face of zone j
                urjm[q] += wij[q * n_poly + l] * cell.face_nodes_rj[qp].phi[l]; // top face of zone j  
                urjp[q] += wrj[q * n_poly + l] * cell.face_nodes_lj[qp].phi[l]; // bottom face of zone j + 1                       
            }
        }

        conserved_to_primitive(ulim, plim);
        conserved_to_primitive(ulip, plip);
        conserved_to_primitive(urim, prim);
        conserved_to_primitive(urip, prip);
        conserved_to_primitive(uljm, pljm);
        conserved_to_primitive(uljp, pljp);
        conserved_to_primitive(urjm, prjm);
        conserved_to_primitive(urjp, prjp);

        real cs2li = sound_speed_squared(&eos);
        real cs2ri = sound_speed_squared(&eos);
        real cs2lj = sound_speed_squared(&eos);
        real cs2rj = sound_speed_squared(&eos);

        riemann_hlle(plim, plip, fli, cs2li, 0);
        riemann_hlle(prim, prip, fri, cs2ri, 0);
        riemann_hlle(pljm, pljp, flj, cs2lj, 1);
        riemann_hlle(prjm, prjp, frj, cs2rj, 1);
        
        for (int q = 0; q < NCONS; ++q)
        {
            for (int l = 0; l < n_poly; ++l)
            {
                dwij[q * n_poly + l] -= fli[q] * cell.face_nodes_li[qp].phi[l] * cell.face_nodes_li[qp].weight;
                dwij[q * n_poly + l] -= fri[q] * cell.face_nodes_ri[qp].phi[l] * cell.face_nodes_ri[qp].weight;
                dwij[q * n_poly + l] -= flj[q] * cell.face_nodes_lj[qp].phi[l] * cell.face_nodes_lj[qp].weight;
                dwij[q * n_poly + l] -= frj[q] * cell.face_nodes_rj[qp].phi[l] * cell.face_nodes_rj[qp].weight;  
            }
        }
    }

    real cons[NCONS];
    real primitive[NCONS];

    real flux_x[NCONS];
    real flux_y[NCONS];

    // volume term
    for (int qp = 0; qp < n_quad; ++qp)
    {
        struct NodeData node = cell.interior_nodes[qp];

        for (int q = 0; q < NCONS; ++q)
        {
            cons[q] = 0.0;

            for (int l = 0; l < n_poly; ++l)
            {
                cons[q] += wij[q * n_poly + l] * node.phi[l];
            }
        }

        conserved_to_primitive(cons, primitive);

        real cs2 = sound_speed_squared(&eos);

        primitive_to_flux(primitive, cons, flux_x, cs2, 0);
        primitive_to_flux(primitive, cons, flux_y, cs2, 1);

        for (int q = 0; q < NCONS; ++q)
        {
            for (int l = 0; l < n_poly; ++l)
            {
                dwij[q * n_poly + l] += flux_x[q] * node.dphi_dx[l] * node.weight;
                dwij[q * n_poly + l] += flux_y[q] * node.dphi_dy[l] * node.weight; 
            }
        }
    }

    real *wout = GET(weights_wr, i, j);

    for (int q = 0; q < NCONS; ++q)
    {
        for (int l = 0; l < n_poly; ++l)
        {
            wout[q * n_poly + l] = wij[q * n_poly + l] + 0.5 * dwij[q * n_poly + l] * dt / dx; //assumes dy=dx
        }
    }
}

// ============================ KERNELS =======================================
// ============================================================================
#if defined(__NVCC__) || defined(__ROCM__)

static void __global__ primitive_to_weights_kernel(
    struct Mesh mesh,
    struct Cell cell,
    struct Patch primitive,
    struct Patch weights)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < mesh.ni && j < mesh.nj)
    {
        primitive_to_weights_zone(cell, primitive, weights, i, j);
    }
}

static void __global__ advance_rk_kernel_dg(
    struct Cell cell,
    struct Mesh mesh,
    struct Patch weights_rd,
    struct Patch weights_wr,
    struct EquationOfState eos,
    real dt
    )
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < mesh.ni && j < mesh.nj)
    {
        advance_rk_zone_dg(
            cell,
            mesh,
            weights_rd,
            weights_wr,
            eos,
            dt,
            i, j
        );
    }
}

#endif // defined(__NVCC__) || defined(__ROCM__)


// ============================ PUBLIC API ====================================
// ============================================================================

/**
 * Converts an array of primitive data to an array of conserved weights data.
 * The primitive data array index space must follow the descriptions below.
 *
 * @param cell               The cell [order]
 * @param mesh               The mesh [ni,     nj]
 * @param primitive_ptr[in]  [ 0,  0] [ni,     nj]     [3] [n_poly(order)]
 * @param weights[out]       [-1, -1] [ni + 2, nj + 2] [3] [n_poly(order)]
 * @param mode               The execution mode
 */
EXTERN_C void iso2d_dg_primitive_to_weights(
    struct Cell cell,
    struct Mesh mesh,
    real *primitive_ptr,
    real *weights_ptr,
    enum ExecutionMode mode)
{
    int n_quad = num_quadrature_points(cell);
    int n_poly = num_polynomials(cell);

    struct Patch primitive = patch(mesh, NCONS * n_quad, 0, primitive_ptr);
    struct Patch weights = patch(mesh, NCONS * n_poly, 0, weights_ptr);

    switch (mode) {
        case CPU: {
            FOR_EACH(weights)
            {
                primitive_to_weights_zone(cell, primitive, weights, i, j);
            }
            break;
        }

        case OMP: {
            #ifdef _OPENMP
            FOR_EACH_OMP(weights)
            {
                primitive_to_weights_zone(cell, primitive, weights, i, j);
            }
            #endif
            break;
        }

        case GPU: {
            #if defined(__NVCC__) || defined(__ROCM__)
            dim3 bs = dim3(16, 16);
            dim3 bd = dim3((mesh.nj + bs.x - 1) / bs.x, (mesh.ni + bs.y - 1) / bs.y);
            primitive_to_weights_kernel<<<bd, bs>>>(mesh, cell, primitive, weights);
            #endif
            break;
        }
    }
}

/**
 * Updates an array of DG weights data by advancing it a single Runge-Kutta
 * step.
 * @param mesh                  The mesh [ni,     nj]
 * @param weights_rd_ptr[in]  [-1, -1] [ni + 2, nj + 2] [3]
 * @param weights_wr_ptr[out] [-1, -1] [ni + 2, nj + 2] [3]
 * @param eos                   The EOS
 * @param dt                    The time step
 * @param mode                  The execution mode
 */
EXTERN_C void iso2d_advance_rk_dg(
    struct Cell cell,
    struct Mesh mesh,
    real *weights_rd_ptr,
    real *weights_wr_ptr,
    struct EquationOfState eos,
    real dt, 
    enum ExecutionMode mode)
{
    int n_poly = num_polynomials(cell);

    struct Patch weights_rd = patch(mesh, n_poly * NCONS, 1, weights_rd_ptr);
    struct Patch weights_wr = patch(mesh, n_poly * NCONS, 1, weights_wr_ptr);

    switch (mode) {
        case CPU: {
            FOR_EACH(weights_rd) {
                advance_rk_zone_dg(
                    cell,
                    mesh,
                    weights_rd,
                    weights_wr,
                    eos,
                    dt,
                    i, j
                );
            }
            break;
        }

        case OMP: {
            #ifdef _OPENMP
            FOR_EACH_OMP(weights_rd) {
                advance_rk_zone_dg(
                    cell
                    mesh,
                    weights_rd,
                    weights_wr,
                    eos,
                    dt,
                    i, j);
            }
            #endif
            break;
        }

        case GPU: {
            #if defined(__NVCC__) || defined(__ROCM__)
            dim3 bs = dim3(16, 16);
            dim3 bd = dim3((mesh.nj + bs.x - 1) / bs.x, (mesh.ni + bs.y - 1) / bs.y);
 
            advance_rk_kernel_dg<<<bd, bs>>>(
                cell,
                mesh,
                weights_rd,
                weights_wr,
                eos,
                dt,
            );           
            #endif
            break;
        }
    }
}

/**
 * Template for a public API function to be exposed to Rust code via FFI.
 * 
 * @param order          The DG order
 */
EXTERN_C int iso2d_dg_say_hello(int order)
{
    return order;
}


/**
 * Template for a public API function to be exposed to Rust code via FFI.
 * 
 * @param cell          The DG cell data
 */
EXTERN_C int iso2d_dg_get_order(struct Cell cell)
{
    return cell.order;
}

