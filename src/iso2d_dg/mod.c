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
