from numpy.typing import NDArray
from kernels import device


static = R"""
#define min2(a, b) ((a) < (b) ? (a) : (b))
#define max2(a, b) ((a) > (b) ? (a) : (b))
#define min3(a, b, c) min2(a, min2(b, c))
#define max3(a, b, c) max2(a, max2(b, c))

#if NPRIM == 3
#define NCONS 3
#define NVECS 1
#define RHO 0
#define VXX 1
#define PRE 2
#define DEN 0
#define PXX 1
#define NRG 2
#elif NPRIM == 4
#define NCONS 4
#define NVECS 2
#define RHO 0
#define VXX 1
#define VYY 2
#define PRE 3
#define DEN 0
#define PXX 1
#define PYY 2
#define NRG 3
#elif NPRIM == 5
#define NCONS 5
#define NVECS 3
#define RHO 0
#define VXX 1
#define VYY 2
#define VZZ 3
#define PRE 4
#define DEN 0
#define PXX 1
#define PYY 2
#define PZZ 3
#define NRG 4
#else
#error("NPRIM must be 3, 4, or 5")
#endif

#ifndef GAMMA_LAW_INDEX
#define GAMMA_LAW_INDEX (5.0 / 3.0)
#endif
"""


@device(static=static)
def prim_to_cons(p: NDArray[float], u: NDArray[float]):
    R"""
    DEVICE void prim_to_cons(double *p, double *u)
    {
        #if NVECS == 1
        double rho = p[RHO];
        double vx  = p[VXX];
        double pre = p[PRE];
        double v_squared = vx * vx;
        u[DEN] = rho;
        u[PXX] = vx * rho;
        u[NRG] = 0.5 * rho * v_squared + pre / (GAMMA_LAW_INDEX - 1.0);

        #elif NVECS == 2
        double rho = p[RHO];
        double vx  = p[VXX];
        double vy  = p[VYY];
        double pre = p[PRE];
        double v_squared = vx * vx + vy * vy;
        u[DEN] = rho;
        u[PXX] = vx * rho;
        u[PYY] = vy * rho;
        u[NRG] = 0.5 * rho * v_squared + pre / (GAMMA_LAW_INDEX - 1.0);

        #elif NVECS == 3
        double rho = p[RHO];
        double vx  = p[VXX];
        double vy  = p[VYY];
        double vz  = p[VZZ];
        double pre = p[PRE];
        double v_squared = vx * vx + vy * vy + vz * vz;
        u[DEN] = rho;
        u[PXX] = vx * rho;
        u[PYY] = vy * rho;
        u[PZZ] = vz * rho;
        u[NRG] = 0.5 * rho * v_squared + pre / (GAMMA_LAW_INDEX - 1.0);
        #endif
    }
    """


@device(static=static)
def cons_to_prim(u: NDArray[float], p: NDArray[float]):
    R"""
    DEVICE void cons_to_prim(double *u, double *p)
    {
        #if NVECS == 1
        double rho = u[DEN];
        double px  = u[PXX];
        double nrg = u[NRG];
        double p_squared = px * px;
        p[RHO] = rho;
        p[VXX] = px / rho;
        p[PRE] = (nrg - 0.5 * p_squared / rho) * (GAMMA_LAW_INDEX - 1.0);

        #elif NVECS == 2
        double rho = u[DEN];
        double px  = u[PXX];
        double py  = u[PYY];
        double nrg = u[NRG];
        double p_squared = px * px + py * py;
        p[RHO] = rho;
        p[VXX] = px / rho;
        p[VYY] = py / rho;
        p[PRE] = (nrg - 0.5 * p_squared / rho) * (GAMMA_LAW_INDEX - 1.0);

        #elif NVECS == 3
        double rho = u[DEN];
        double px  = u[PXX];
        double py  = u[PYY];
        double pz  = u[PZZ];
        double nrg = u[NRG];
        double p_squared = px * px + py * py + pz * pz;
        p[RHO] = rho;
        p[VXX] = px / rho;
        p[VYY] = py / rho;
        p[VZZ] = pz / rho;
        p[PRE] = (nrg - 0.5 * p_squared / rho) * (GAMMA_LAW_INDEX - 1.0);
        #endif
    }
    """


@device(static=static, device_funcs=[cons_to_prim])
def cons_to_prim_check(u: NDArray[float], p: NDArray[float]) -> int:
    R"""
    DEVICE int cons_to_prim_check(double *u, double *p)
    {
        cons_to_prim(u, p);

        if (u[DEN] < 0.0) {
            return 1;
        }
        if (u[NRG] < 0.0) {
            return 2;
        }
        if (p[PRE] < 0.0) {
            return 3;
        }
        return 0;
    }
    """


@device(static=static)
def prim_and_cons_to_flux(
    p: NDArray[float],
    u: NDArray[float],
    f: NDArray[float],
    direction: int,
):
    R"""
    DEVICE void prim_and_cons_to_flux(double *p, double *u, double *f, int direction)
    {
        double pre = p[PRE];
        double nrg = u[NRG];
        double vn = p[direction];

        #if NVECS == 1
        f[DEN] = vn * u[DEN];
        f[PXX] = vn * u[PXX] + pre * (direction == 1);
        f[NRG] = vn * (nrg + pre);

        #elif NVECS == 2
        f[DEN] = vn * u[DEN];
        f[PXX] = vn * u[PXX] + pre * (direction == 1);
        f[PYY] = vn * u[PYY] + pre * (direction == 2);
        f[NRG] = vn * (nrg + pre);

        #elif NVECS == 3
        f[DEN] = vn * u[DEN];
        f[PXX] = vn * u[PXX] + pre * (direction == 1);
        f[PYY] = vn * u[PYY] + pre * (direction == 2);
        f[PZZ] = vn * u[PZZ] + pre * (direction == 3);
        f[NRG] = vn * (nrg + pre);
        #endif
    }
    """


@device(static=static, device_funcs=[prim_and_cons_to_flux])
def prim_to_flux(p: NDArray[float], f: NDArray[float], direction: int):
    R"""
    DEVICE void prim_and_cons_to_flux(double *p, double *f, int direction)
    {
        double u[NCONS];
        prim_to_cons(p, u);
        prim_and_cons_to_flux(p, u, f, direction);
    }
    """


@device(static=static)
def sound_speed_squared(p: NDArray[float]) -> float:
    R"""
    DEVICE double sound_speed_squared(double *p)
    {
        return p[PRE] / p[RHO] * GAMMA_LAW_INDEX;
    }
    """


@device(static=static, device_funcs=[sound_speed_squared])
def max_wavespeed(p: NDArray[float]) -> float:
    R"""
    DEVICE double max_wavespeed(double *p)
    {
        #if NVECS == 1
        double cs = sqrt(sound_speed_squared(p));
        double vx = p[VXX];
        double ax = max2(fabs(vx - cs), fabs(vx + cs));
        return ax;

        #elif NVECS == 2
        double cs = sqrt(sound_speed_squared(p));
        double vx = p[VXX];
        double vy = p[VYY];
        double ax = max2(fabs(vx - cs), fabs(vx + cs));
        double ay = max2(fabs(vy - cs), fabs(vy + cs));
        return max2(ax, ay);

        #elif NVECS == 3
        double cs = sqrt(sound_speed_squared(p));
        double vx = p[VXX];
        double vy = p[VYY];
        double vz = p[VZZ];
        double ax = max2(fabs(vx - cs), fabs(vx + cs));
        double ay = max2(fabs(vy - cs), fabs(vy + cs));
        double az = max2(fabs(vz - cs), fabs(vz + cs));
        return max3(ax, ay, az);
        #endif
    }
    """


@device(static=static, device_funcs=[sound_speed_squared])
def outer_wavespeeds(
    p: NDArray[float],
    wavespeeds: NDArray[float],
    direction: int,
):
    R"""
    DEVICE void outer_wavespeeds(
        double *p,
        double *wavespeeds,
        int direction)
    {
        double cs = sqrt(sound_speed_squared(p));
        double vn = p[direction];
        wavespeeds[0] = vn - cs;
        wavespeeds[1] = vn + cs;
    }
    """


@device(
    static=static,
    device_funcs=[
        prim_to_cons,
        prim_and_cons_to_flux,
        outer_wavespeeds,
    ],
)
def riemann_hlle(
    pl: NDArray[float],
    pr: NDArray[float],
    flux: NDArray[float],
    direction: int,
):
    R"""
    DEVICE void riemann_hlle(double *pl, double *pr, double *flux, int direction)
    {
        double ul[NCONS];
        double ur[NCONS];
        double fl[NCONS];
        double fr[NCONS];
        double al[2];
        double ar[2];

        prim_to_cons(pl, ul);
        prim_to_cons(pr, ur);
        prim_and_cons_to_flux(pl, ul, fl, direction);
        prim_and_cons_to_flux(pr, ur, fr, direction);
        outer_wavespeeds(pl, al, direction);
        outer_wavespeeds(pr, ar, direction);

        double am = min3(0.0, al[0], ar[0]);
        double ap = max3(0.0, al[1], ar[1]);

        for (int q = 0; q < NCONS; ++q)
        {
            flux[q] = (fl[q] * ap - fr[q] * am - (ul[q] - ur[q]) * ap * am) / (ap - am);
        }
    }
    """


@device(static=static)
def source_terms_spherical_polar():
    R"""
    DEVICE void source_terms_spherical_polar(
        double r0, double r1,
        double q0, double q1,
        double f0, double f1,
        double *prim,
        double *source)
    {
        // Forumulas below are A8 - A10 from Zhang & MacFadyen (2006), integrated
        // over the cell volume with finite radial and polar extent, and taking
        // the Newonian limit h -> 1 and W -> 1.
        //
        // https://iopscience.iop.org/article/10.1086/500792/pdf

        double dr2 = r1 * r1 - r0 * r0;
        double df = f1 - f0;

        #if NVECS == 1
        double dcosq = cos(q1) - cos(q0);
        double dg = prim[0];
        double uq = 0.0;
        double uf = 0.0;
        double pg = prim[2];
        double srdot = -0.5 * df * dr2 * dcosq * (dg * (uq * uq + uf * uf) + 2 * pg);
        source[0] = 0.0;
        source[1] = srdot;
        source[2] = 0.0;

        #elif NVECS == 2
        double dcosq = cos(q1) - cos(q0);
        double dsinq = sin(q1) - sin(q0);
        double dg = prim[0];
        double ur = prim[1];
        double uq = prim[2];
        double uf = 0.0;
        double pg = prim[3];
        double srdot = -0.5 * df * dr2 * dcosq * (dg * (uq * uq + uf * uf) + 2 * pg);
        double sqdot = +0.5 * df * dr2 * (dcosq * dg * ur * uq + dsinq * (pg + dg * uf * uf));
        source[0] = 0.0;
        source[1] = srdot;
        source[2] = sqdot;
        source[3] = 0.0;

        #elif NVECS == 3
        double dcosq = cos(q1) - cos(q0);
        double dsinq = sin(q1) - sin(q0);
        double dg = prim[0];
        double ur = prim[1];
        double uq = prim[2];
        double uf = prim[3];
        double pg = prim[4];
        double srdot = -0.5 * df * dr2 * dcosq * (dg * (uq * uq + uf * uf) + 2 * pg);
        double sqdot = +0.5 * df * dr2 * (dcosq * dg * ur * uq + dsinq * (pg + dg * uf * uf));
        double sfdot = -0.5 * df * dr2 * dg * uf * (uq * dsinq - ur * dcosq);
        source[0] = 0.0;
        source[1] = srdot;
        source[2] = sqdot;
        source[3] = sfdot;
        source[4] = 0.0;
        #endif
    }
    """


@device(static=static)
def source_terms_cylindrical_polar():
    R"""
    DEVICE void source_terms_cylindrical_polar(
        double r0, double r1,
        double f0, double f1,
        double z0, double z1,
        double *prim,
        double *source)
    {
        // Forumulas below are A2 - A4 from Zhang & MacFadyen (2006), integrated
        // over the cell volume with finite radial and polar extent, and taking
        // the Newonian limit h -> 1 and W -> 1.
        //
        // https://iopscience.iop.org/article/10.1086/500792/pdf

        #if NVECS == 1
        double dg = prim[0];
        double uf = 0.0;
        double pg = prim[2];
        double srdot = +(f1 - f0) * (r1 - r0) * (z1 - z0) * (pg + dg * uf * uf);
        source[0] = 0.0;
        source[1] = srdot;
        source[2] = 0.0;

        #elif NVECS == 2
        double dg = prim[0];
        double ur = prim[1];
        double uf = prim[2];
        double pg = prim[3];
        double srdot = +(f1 - f0) * (r1 - r0) * (z1 - z0) * (pg + dg * uf * uf);
        double sfdot = -(f1 - f0) * (r1 - r0) * (z1 - z0) * ur * uf;
        source[0] = 0.0;
        source[1] = srdot;
        source[2] = sqdot;
        source[3] = 0.0;

        #elif NVECS == 3
        double dg = prim[0];
        double ur = prim[1];
        double uf = prim[2];
        double pg = prim[4];
        double srdot = +(f1 - f0) * (r1 - r0) * (z1 - z0) * (pg + dg * uf * uf);
        double sfdot = -(f1 - f0) * (r1 - r0) * (z1 - z0) * ur * uf;
        double szdot = 0.0;
        source[0] = 0.0;
        source[1] = srdot;
        source[2] = sqdot;
        source[3] = szdot;
        source[4] = 0.0;
        #endif
    }
    """


if __name__ == "__main__":
    from numpy import array, zeros_like, allclose
    from kernels import kernel

    nprim = 4

    @kernel(device_funcs=[cons_to_prim], define_macros=dict(NPRIM=nprim))
    def kernel_cons_to_prim(u: NDArray[float], p: NDArray[float], ni: int = None):
        R"""
        KERNEL void kernel_cons_to_prim(double *u, double *p, int ni)
        {
            FOR_EACH_1D(ni)
            {
                cons_to_prim(&u[NCONS * i], &p[NCONS * i]);
            }
        }
        """
        return u.size // nprim, (u, p, u.size // nprim)

    @kernel(device_funcs=[prim_to_cons], define_macros=dict(NPRIM=nprim))
    def kernel_prim_to_cons(p: NDArray[float], u: NDArray[float], ni: int = None):
        R"""
        KERNEL void kernel_prim_to_cons(double *p, double *u, int ni)
        {
            FOR_EACH_1D(ni)
            {
                prim_to_cons(&p[NCONS * i], &u[NCONS * i]);
            }
        }
        """
        return p.size // nprim, (p, u, p.size // nprim)

    p = array([[1.0, 0.1, 0.2, 100.0]])
    u = zeros_like(p)
    q = zeros_like(p)
    kernel_prim_to_cons(p, u)
    kernel_cons_to_prim(u, q)
    assert allclose(p - q, 0.0)
