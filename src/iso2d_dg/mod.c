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
    struct Patch weights,
    int i,
    int j,
    real x,
    real y,
    real dx,
    real dy)
{
    real prim[NCONS];
    real cons[NCONS];

    // assume that "weights" is now an array containing the NCONS * NUM_POLYNOMIALS 
    // weights of conserved variables per zone

    real *weights = GET(weights, i, j);

    // initialize to zero
    
    for (int l = 0; l < NUM_POLYNOMIALS; ++l)
    {
        for (int q = 0; q < NCONS; ++q)
        {
            weights[q * NUM_POLYNOMIALS + l] = 0.0;
        }
    }

    // number of interior quadrature points in cell

    int nq = cell.order * cell.order;

    // loop over cell's interior quadrature points

    for (qp = 0; qp < nq; ++qp)
    {
        // global position of quadrature point 

        real xq = x + cell.interior_nodes[qp].xsi_x * 0.5 * dx;
        real yq = y + cell.interior_nodes[qp].xsi_y * 0.5 * dy;

        // get initial condition for primitive variables at quadrature point 

        initial_primitive(xq, yq, prim);

        // convert to conserved variables at quadrature point

        primitive_to_conserved(prim, cons);

        for (int l = 0; l < NUM_POLYNOMIALS; ++l)
        {
            for (int q = 0; q < NCONS; ++q)
            {
                weights[q * NUM_POLYNOMIALS + l] += 
                0.25 * cons[q] * cell.interior_nodes[qp].phi[l] * cell.interior_nodes[qp].weight;
            }
        }
    }
}

// ============================ PUBLIC API ====================================
// ============================================================================

/**
 * Converts an array of primitive data to an array of conserved weights data. 
 * The primitive data 
 * array index space must follow the descriptions below.
 * @param mesh               The mesh [ni,     nj]
 * @param primitive_ptr[in]  [-1, -1] [ni + 2, nj + 2] [3]
 * @param conserved_ptr[out] [ 0,  0] [ni,     nj]     [3]
 * @param mode               The execution mode
 */
EXTERN_C int iso2d_dg_initial_primitive_to_weights(
    struct Cell cell
    struct Mesh mesh,
    real *primitive_ptr,
    real *weights_ptr,
    enum ExecutionMode mode
    )
{
    struct Patch weights   = patch(mesh, NCONS * NUM_POLYNOMIALS, 0, weights_ptr);
        
    FOR_EACH(weights) {
                real dx = mesh.dx;
                real dy = mesh.dy;
                real x = mesh.x0 + (i + 0.5) * dx;
                real y = mesh.y0 + (j + 0.5) * dy;
                primitive_to_weights_zone(cell, weights, i, j, x, y, dx, dy);
            }
    return;
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

