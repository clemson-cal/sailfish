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

