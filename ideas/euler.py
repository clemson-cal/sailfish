from contextlib import contextmanager
from time import perf_counter
from numpy.typing import NDArray
from new_kernels import kernel_class, kernel_method


@kernel_class
class Solver:
    R"""
    #define min2(a, b) ((a) < (b) ? (a) : (b))
    #define max2(a, b) ((a) > (b) ? (a) : (b))
    #define min3(a, b, c) min2(a, min2(b, c))
    #define max3(a, b, c) max2(a, max2(b, c))

    static const double gamma_law_index = 5.0 / 3.0;

    static void _prim_to_cons(double *p, double *u)
    {
        double rho = p[0];
        double vx  = p[1];
        double vy  = p[2];
        double vz  = p[3];
        double pre = p[4];
        double v_squared = vx * vx + vy * vy + vz * vz;

        u[0] = rho;
        u[1] = vx * rho;
        u[2] = vy * rho;
        u[3] = vz * rho;
        u[4] = 0.5 * rho * v_squared + pre / (gamma_law_index - 1.0);
    }

    static void _cons_to_prim(double *u, double *p)
    {
        double rho = u[0];
        double px  = u[1];
        double py  = u[2];
        double pz  = u[3];
        double nrg = u[4];
        double p_squared = px * px + py * py + pz * pz;

        p[0] = rho;
        p[1] = px / rho;
        p[2] = py / rho;
        p[3] = py / rho;
        p[4] = (nrg - 0.5 * p_squared / rho) * (gamma_law_index - 1.0);
    }

    static void _prim_to_flux(double *p, double *u, double *f, int direction)
    {
        double pre = p[4];
        double nrg = u[4];
        double vn = p[direction];

        f[0] = vn * u[0];
        f[1] = vn * u[1] + pre * (direction == 1);
        f[2] = vn * u[2] + pre * (direction == 2);
        f[3] = vn * u[3] + pre * (direction == 3);
        f[4] = vn * (nrg + pre);
    }

    static double _sound_speed_squared(double *p)
    {
        return p[4] / p[0] * gamma_law_index;
    }

    static void _outer_wavespeeds(
        double *p,
        double *wavespeeds,
        int direction)
    {
        double cs = sqrt(_sound_speed_squared(p));
        double vn = p[direction];
        wavespeeds[0] = vn - cs;
        wavespeeds[1] = vn + cs;
    }

    static double _max_wavespeed(double *p)
    {
        double cs = sqrt(_sound_speed_squared(p));
        double vx = p[1];
        double vy = p[2];
        double vz = p[3];
        double ax = max2(fabs(vx - cs), fabs(vx + cs));
        double ay = max2(fabs(vy - cs), fabs(vy + cs));
        double az = max2(fabs(vz - cs), fabs(vz + cs));
        return max3(ax, ay, az);
    }

    static void _hlle(double *pl, double *pr, double *flux, int direction)
    {
        double ul[5];
        double ur[5];
        double fl[5];
        double fr[5];
        double al[2];
        double ar[2];

        _prim_to_cons(pl, ul);
        _prim_to_cons(pr, ur);
        _prim_to_flux(pl, ul, fl, direction);
        _prim_to_flux(pr, ur, fr, direction);
        _outer_wavespeeds(pl, al, direction);
        _outer_wavespeeds(pr, ar, direction);

        double am = min3(0.0, al[0], ar[0]);
        double ap = max3(0.0, al[1], ar[1]);

        for (int q = 0; q < 5; ++q)
        {
            flux[q] = (fl[q] * ap - fr[q] * am - (ul[q] - ur[q]) * ap * am) / (ap - am);
        }
    }
    """

    @kernel_method(rank=1)
    def cons_to_prim(u: NDArray[float], p: NDArray[float]):
        R"""
        void cons_to_prim(int ni, double *u, double *p)
        {
            FOR_EACH_1D(ni)
            {
                _cons_to_prim(&u[5 * i], &p[5 * i]);
            }
        }
        """
        return (u.size // 5,)

    @kernel_method(rank=1)
    def prim_to_cons(p: NDArray[float], u: NDArray[float]):
        R"""
        void prim_to_cons(int ni, double *p, double *u)
        {
            FOR_EACH_1D(ni)
            {
                _prim_to_cons(&p[5 * i], &u[5 * i]);
            }
        }
        """
        return (p.size // 5,)

    @kernel_method(rank=1)
    def prim_to_flux(p: NDArray[float], f: NDArray[float], direction: int):
        R"""
        void prim_to_flux(int ni, double *p, double *f, int direction)
        {
            double u[5];

            FOR_EACH_1D(ni)
            {
                _prim_to_cons(&p[i * 5], u);
                _prim_to_flux(&p[i * 5], u, &f[i * 5], direction);
            }
        }
        """
        return (p.size // 5,)

    @kernel_method(rank=1)
    def max_wavespeed(p: NDArray[float], a: NDArray[float]):
        R"""
        void max_wavespeed(int ni, double *p, double *a)
        {
            FOR_EACH_1D(ni)
            {
                a[i] = _max_wavespeed(&p[i * 5]);
            }
        }
        """
        return (p.size // 5,)

    @kernel_method(rank=1)
    def godunov_flux(
        pl: NDArray[float],
        pr: NDArray[float],
        fhat: NDArray[float],
        direction: int,
    ):
        R"""
        void godunov_flux(int ni, double *pl, double *pr, double *fhat, int direction)
        {
            FOR_EACH_1D(ni)
            {
                _hlle(&pl[5 * i], &pr[5 * i], &fhat[5 * i], direction);
            }
        }
        """
        return (fhat.size // 5,)


@contextmanager
def measure_time() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


def main():
    from numpy import array, linspace, zeros, zeros_like, diff
    from matplotlib import pyplot as plt

    solver = Solver()

    num_zones = 10000
    dx = 1.0 / num_zones
    fold = 100
    dt = dx * 1e-1
    p = zeros((num_zones, 5))
    u = zeros_like(p)
    fhat = zeros((num_zones - 1, 5))

    p[: num_zones // 2, :] = [1.0, 0.0, 0.0, 0.0, 1.0]
    p[num_zones // 2 :, :] = [0.1, 0.0, 0.0, 0.0, 0.125]
    t = 0.0
    n = 0

    while t < 0.1:
        with measure_time() as fold_time:
            for _ in range(fold):
                pr = p[+1:]
                pl = p[:-1]
                solver.godunov_flux(pl, pr, fhat, 1)
                solver.prim_to_cons(p, u)
                u[1:-1] -= diff(fhat, axis=0) * (dt / dx)
                t += dt
                n += 1
                solver.cons_to_prim(u, p)

        kzps = num_zones / fold_time() * 1e-3 * fold
        print(f"[{n:04d}]: t={t:.4f} Mzps={kzps * 1e-3:.3f}")

    plt.plot(p[:, 0])
    plt.show()


main()
