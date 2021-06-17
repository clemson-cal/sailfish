#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>


// ============================ COMPAT ========================================
// ============================================================================
#define __host__
#define __device__
#define PREFIX iso2d_cpu
#define CONCAT(a, b) a ## _ ## b
#define PUBLIC(f) CONCAT(PREFIX, f)


// ============================ MEMORY ========================================
// ============================================================================
static void *compute_malloc(size_t count) { return malloc(count); }
static void compute_free(void *ptr) { free(ptr); }
// static void compute_memcpy_host_to_device(void *dst, const void *src, size_t count) { memcpy(dst, src, count); }
// static void compute_memcpy_device_to_host(void *dst, const void *src, size_t count) { memcpy(dst, src, count); }


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


// ============================ PATCH API =====================================
// ============================================================================
#define GET(p, i, j) (p.data + p.jumps[0] * ((i) - p.start[0]) + p.jumps[1] * ((j) - p.start[1]))
#define FOR_EACH(p) for (int i = p.start[0]; i < p.start[0] + p.count[0]; ++i) \
                    for (int j = p.start[1]; j < p.start[1] + p.count[1]; ++j)
#define ELEMENTS(p) (p.count[0] * p.count[1] * NCONS)
#define BYTES(p) (ELEMENTS(p) * sizeof(real))

struct Patch
{
    int start[2];
    int count[2];
    int jumps[2];
    int owned;
    real *data;
};

// static struct Patch patch_view(int start_i, int start_j, int count_i, int count_j, real *data)
// {
//     struct Patch self;
//     self.start[0] = start_i;
//     self.start[1] = start_j;
//     self.count[0] = count_i;
//     self.count[1] = count_j;
//     self.jumps[0] = NCONS * count_j;
//     self.jumps[1] = NCONS;
//     self.data = data;
//     self.owned = 0;
//     return self;
// }

static struct Patch patch_alloc(int start_i, int start_j, int count_i, int count_j)
{
    struct Patch self;
    self.start[0] = start_i;
    self.start[1] = start_j;
    self.count[0] = count_i;
    self.count[1] = count_j;
    self.jumps[0] = NCONS * count_j;
    self.jumps[1] = NCONS;
    self.data = compute_malloc(NCONS * count_i * count_j * sizeof(real));
    self.owned = 1;

    for (int n = 0; n < ELEMENTS(self); ++n)
    {
        self.data[n] = 0.0;
    }
    return self;
}

static void patch_release(struct Patch self)
{
    if (self.owned)
    {
        compute_free(self.data);
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


// ============================ SCHEME ========================================
// ============================================================================
static __device__ void gradient_i(const struct Patch p, struct Patch g, int i, int j)
{
    real *pl = GET(p, i - 1, j);
    real *pc = GET(p, i + 0, j);
    real *pr = GET(p, i + 1, j);
    real *gc = GET(g, i, j);
    plm_gradient(pl, pc, pr, gc);
}

static __device__ void gradient_j(const struct Patch p, struct Patch g, int i, int j)
{
    real *pl = GET(p, i, j - 1);
    real *pc = GET(p, i, j + 0);
    real *pr = GET(p, i, j + 1);
    real *gc = GET(g, i, j);
    plm_gradient(pl, pc, pr, gc);
}

static __device__ void godunov_i(const struct Patch p, struct Patch g, struct Patch f, int i, int j)
{
    real *pl = GET(p, i - 1, j);
    real *pr = GET(p, i + 0, j);
    real *gl = GET(g, i - 1, j);
    real *gr = GET(g, i + 0, j);

    real pm[NCONS];
    real pp[NCONS];

    for (int q = 0; q < NCONS; ++q)
    {
        pm[q] = pl[q] + 0.5 * gl[q];
        pp[q] = pr[q] - 0.5 * gr[q];
    }
    riemann_hlle(pm, pp, GET(f, i, j), 1.0, 0);
}

static __device__ void godunov_j(const struct Patch p, struct Patch g, struct Patch f, int i, int j)
{
    real *pl = GET(p, i, j - 1);
    real *pr = GET(p, i, j + 0);
    real *gl = GET(g, i, j - 1);
    real *gr = GET(g, i, j + 0);

    real pm[NCONS];
    real pp[NCONS];

    for (int q = 0; q < NCONS; ++q)
    {
        pm[q] = pl[q] + 0.5 * gl[q];
        pp[q] = pr[q] - 0.5 * gr[q];
    }
    riemann_hlle(pm, pp, GET(f, i, j), 1.0, 1);
}

static __device__ void advance_rk(
    struct Patch p,
    struct Patch u,
    struct Patch u0,
    struct Mesh mesh,
    struct Patch grad_i,
    struct Patch grad_j,
    struct Patch flux_i,
    struct Patch flux_j,
    real a,
    real dt)
{
    FOR_EACH(grad_i) {
        gradient_i(p, grad_i, i, j);
    }
    FOR_EACH(grad_j) {
        gradient_j(p, grad_j, i, j);
    }
    FOR_EACH(flux_i) {
        godunov_i(p, grad_i, flux_i, i, j);
    }
    FOR_EACH(flux_j) {
        godunov_j(p, grad_j, flux_j, i, j);
    }
    FOR_EACH(u) {
        real *fli = GET(flux_i, i + 0, j);
        real *fri = GET(flux_i, i + 1, j);
        real *flj = GET(flux_j, i, j + 0);
        real *frj = GET(flux_j, i, j + 1);
        real *pc = GET(p, i, j);
        real *uc = GET(u, i, j);
        real *un = GET(u0, i, j);

        for (int q = 0; q < NCONS; ++q)
        {
            uc[q] -= ((fri[q] - fli[q]) / mesh.dx + (frj[q] - flj[q]) / mesh.dy) * dt;
            uc[q] = a * un[q] + (1.0 - a) * uc[q];
        }
        conserved_to_primitive(uc, pc);
    }
}

static __device__ void advance(
    struct Patch p,
    struct Patch u,
    struct Patch u0,
    struct Mesh mesh,
    struct Patch grad_i,
    struct Patch grad_j,
    struct Patch flux_i,
    struct Patch flux_j,
    real dt)
{
    memcpy(u0.data, u.data, BYTES(u));
    advance_rk(p, u, u0, mesh, grad_i, grad_j, flux_i, flux_j, 0.0, dt);
    advance_rk(p, u, u0, mesh, grad_i, grad_j, flux_i, flux_j, 0.5, dt);

    // memcpy(u0.data, u.data, BYTES(u));

    // real a = 0.0;
    // FOR_EACH(grad_i) {
    //     gradient_i(p, grad_i, i, j);
    // }
    // FOR_EACH(grad_j) {
    //     gradient_j(p, grad_j, i, j);
    // }
    // FOR_EACH(flux_i) {
    //     godunov_i(p, grad_i, flux_i, i, j);
    // }
    // FOR_EACH(flux_j) {
    //     godunov_j(p, grad_j, flux_j, i, j);
    // }
    // FOR_EACH(u) {
    //     real *fli = GET(flux_i, i + 0, j);
    //     real *fri = GET(flux_i, i + 1, j);
    //     real *flj = GET(flux_j, i, j + 0);
    //     real *frj = GET(flux_j, i, j + 1);
    //     real *pc = GET(p, i, j);
    //     real *uc = GET(u, i, j);
    //     real *un = GET(u0, i, j);

    //     for (int q = 0; q < NCONS; ++q)
    //     {
    //         uc[q] -= ((fri[q] - fli[q]) / mesh.dx + (frj[q] - flj[q]) / mesh.dy) * dt;
    //         uc[q] = a * un[q] + (1.0 - a) * uc[q];
    //     }
    //     conserved_to_primitive(uc, pc);
    // }

    // real a = 0.5;

    // FOR_EACH(grad_i) {
    //     gradient_i(p, grad_i, i, j);
    // }
    // FOR_EACH(grad_j) {
    //     gradient_j(p, grad_j, i, j);
    // }
    // FOR_EACH(flux_i) {
    //     godunov_i(p, grad_i, flux_i, i, j);
    // }
    // FOR_EACH(flux_j) {
    //     godunov_j(p, grad_j, flux_j, i, j);
    // }
    // FOR_EACH(u) {
    //     real *fli = GET(flux_i, i + 0, j);
    //     real *fri = GET(flux_i, i + 1, j);
    //     real *flj = GET(flux_j, i, j + 0);
    //     real *frj = GET(flux_j, i, j + 1);
    //     real *pc = GET(p, i, j);
    //     real *uc = GET(u, i, j);
    //     real *un = GET(u0, i, j);

    //     for (int q = 0; q < NCONS; ++q)
    //     {
    //         uc[q] -= ((fri[q] - fli[q]) / mesh.dx + (frj[q] - flj[q]) / mesh.dy) * dt;
    //         uc[q] = a * un[q] + (1.0 - a) * uc[q];
    //     }
    //     conserved_to_primitive(uc, pc);
    // }

    // memcpy(u0.data, u.data, BYTES(u));
    // int rk_order = 1;

    // switch (rk_order) {
    //     case 1:
    //         advance_rk(p, u, u0, mesh, grad_i, grad_j, flux_i, flux_j, 0.00, dt);
    //         break;
    //     case 2:
    //         advance_rk(p, u, u0, mesh, grad_i, grad_j, flux_i, flux_j, 0./1, dt);
    //         advance_rk(p, u, u0, mesh, grad_i, grad_j, flux_i, flux_j, 1./2, dt);
    //         break;
    //     case 3:
    //         advance_rk(p, u, u0, mesh, grad_i, grad_j, flux_i, flux_j, 0./1, dt);
    //         advance_rk(p, u, u0, mesh, grad_i, grad_j, flux_i, flux_j, 3./4, dt);
    //         advance_rk(p, u, u0, mesh, grad_i, grad_j, flux_i, flux_j, 1./3, dt);
    //         break;
    // }
}


// ============================ MAIN ==========================================
// ============================================================================
int main() {
    int N = 1024;
    int i0 = 0;
    int j0 = 0;
    int ni = N;
    int nj = N;

    struct Mesh mesh = {
        .x0 = -1.0,
        .y0 = -1.0,
        .ni = N,
        .nj = N,
        .dx = 2.0 / N,
        .dy = 2.0 / N,
    };

    struct Patch primitive = patch_alloc(-2, -2, mesh.ni + 4, mesh.nj + 4);
    struct Patch conserved = patch_alloc(0, 0, mesh.ni, mesh.nj);
    struct Patch conserved_rk = patch_alloc(0, 0, mesh.ni, mesh.nj);
    struct Patch grad_i = patch_alloc(i0 - 1, j0, ni + 2, nj);
    struct Patch grad_j = patch_alloc(i0, j0 - 1, ni, nj + 2);
    struct Patch flux_i = patch_alloc(i0, j0, ni + 1, nj);
    struct Patch flux_j = patch_alloc(i0, j0, ni, nj + 1);

    FOR_EACH(primitive) {
        real *p = GET(primitive, i, j);
        real x = X(mesh, i);
        real y = Y(mesh, j);

        if (sqrt(x * x + y * y) < 0.25) {
            p[0] = 1.0;
            p[1] = 0.0;
            p[2] = 0.0;
        } else {
            p[0] = 0.1;
            p[1] = 0.0;
            p[2] = 0.0;
        }
    }
    FOR_EACH(conserved) {
        real *u = GET(conserved, i, j);
        real *p = GET(primitive, i, j);
        primitive_to_conserved(p, u);
    }

    int fold = 10;
    int iteration = 0;
    real time = 0.0;
    real dt = fmin(mesh.dx, mesh.dy) * 0.2;

    while (time < 0.1) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        for (int n = 0; n < fold; ++n) {
            advance(primitive, conserved, conserved_rk, mesh, grad_i, grad_j, flux_i, flux_j, dt);
            time += dt;
            iteration += 1;
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double seconds = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        double mzps = (N * N / 1e6) / seconds * fold;

        printf("[%d] t=%f Mzps=%f\n", iteration, time, mzps);
    }

    FILE *outfile = fopen("output.bin", "wb");
    fwrite(primitive.data, sizeof(real), ELEMENTS(primitive), outfile);
    fclose(outfile);

    patch_release(primitive);
    patch_release(conserved);
    patch_release(conserved_rk);
    patch_release(grad_i);
    patch_release(grad_j);
    patch_release(flux_i);
    patch_release(flux_j);
    return 0;
}
