#include <stdint.h>
#define real double

enum ExecutionMode {
    CPU,
    OMP,
    GPU,
};

enum SinkModel {
    Inactive,
    AccelerationFree,
    TorqueFree,
    ForceFree,
};

struct PointMass
{
    real x;
    real y;
    real vx;
    real vy;
    real mass;
    real rate;
    real radius;
    enum SinkModel model;
};

struct PointMassList
{
    struct PointMass masses[2];
    int count;
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

enum BoundaryConditionType
{
    Default,
    Inflow,
    KeplerianBuffer,
};

struct BoundaryCondition
{
    enum BoundaryConditionType type;

    union
    {
        // default;
        // inflow;
        struct
        {
            real surface_density;
            real surface_pressure;
            real central_mass;
            real driving_rate;
            real outer_radius;
            real onset_width;
        } keplerian_buffer;
    };
};

struct Mesh
{
    int64_t ni, nj;
    real x0, y0;
    real dx, dy;
};

enum Coordinates
{
    Cartesian,
    SphericalPolar,
};

#ifdef DG_SOLVER

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

#endif // DG_SOLVER
