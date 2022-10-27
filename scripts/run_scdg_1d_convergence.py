"""
Measure the L1 error with respect to the analytic solution by integrating L1 =
1/V Integral(abs(u_computed-u_analytic)) dV over the cell using Gaussian
quadrature with order l1_order. Schaal+15 recommend using 2 orders higher
quadrature than the simulation order: l1_order = simulation_order + 2 to make
sure the measured error is that due to the simulation itself, and not from the
analysis.
"""

import argparse
import pickle
import sys
from numpy.polynomial.legendre import leggauss, Legendre
from numpy import array, pi, sin, cos

sys.path.insert(1, ".")


def analytic(t, x):
    """
    Analytic solution from initial condition from Advection setup in
    sailfish/setups/simple1d.py
    """
    a = 0.1
    k = 2.0 * pi
    wavespeed = 1.0
    return 1.0 + a * sin(k * (x - wavespeed * t))


def burgers(t, x):
    """
    Analytic solution from initial condition from Burgers setup in
    sailfish/setups/simple1d.py
    u(x,t) > 0 for this setup
    Use root finder to find xsi such that xsi - x + f(xsi) * t = 0
    """
    from scipy import optimize

    a = 0.1
    k = 2.0 * pi
    average_wavespeed = 1.0

    def f(xsi):
        return xsi - x + t * (1.0 + a * sin(k * xsi))

    def fder(xsi):
        return 1.0 + t * k * a * cos(k * xsi)

    def fder2(xsi):
        return -t * k * k * a * sin(k * xsi)

    # xsi0 is initial guess for xsi
    xsi0 = x - average_wavespeed * t
    xsi = optimize.newton(f, xsi0, fprime=fder, fprime2=fder2)
    return 1.0 + a * sin(k * xsi)


def leg(x, n):
    """
    Legendre polynomials scaled by sqrt(2n + 1) used as basis functions
    """
    c = [(2 * n + 1) ** 0.5 if i is n else 0.0 for i in range(n + 1)]
    return Legendre(c)(x)


def dot(u, p):
    return sum(u[i] * p[i] for i in range(u.shape[0]))


def compute_error(state):
    scheme_order = state.solver.options["order"]

    # Schaal+15 recommends computing the L1 error using order p + 2 quadrature,
    # where p is the order at which the simulation was run.
    l1_order = scheme_order + 2

    # Gaussian quadrature points inside cell
    gauss_points, weights = leggauss(l1_order)

    # Value of basis functions at the quadrature points
    phi_value = array([[leg(x, n) for n in range(l1_order)] for x in gauss_points])

    time = state.solver.time
    mesh = state.mesh
    xc = mesh.zone_centers(time)
    dx = mesh.dx
    uw = state.solver.solution
    num_zones = mesh.shape[0]
    num_points = len(gauss_points)
    l1 = 0.0

    for i in range(num_zones):
        for j in range(num_points):
            xsi = gauss_points[j]
            xj = xc[i] + xsi * 0.5 * dx
            u_analytic = analytic(time, xj)
            u_computed = dot(uw[i, 0, :], phi_value[j])
            l1 += abs(u_computed - u_analytic) * weights[j]

    return l1 * dx


def main(args):
    from sailfish.driver import run
    from matplotlib import pyplot as plt
    import numpy as np

    errors = []
    resolutions = [20, 40, 80]
    solver_options = dict(order=3, integrator="rk3")

    print(f"solver_options = {solver_options}")

    for res in resolutions:
        state = run(
            "burgers",
            end_time=0.00001,
            cfl_number=0.2,
            resolution=res,
            solver_options=solver_options,
        )
        err = compute_error(state)
        errors.append(err)
        print(f"run with res = {res} error = {err:.3e}")

    time = state.solver.time
    mesh = state.mesh
    num_zones = mesh.shape[0]
    uan = np.zeros(num_zones)
    xc = mesh.zone_centers(time)

    for i in range(num_zones):
        uan[i] = analytic(time, xc[i])

    plt.plot(xc, state.solver.solution[:, 0, 0], "-o")
    plt.plot(xc, uan)
    plt.show()

    expected = (
        errors[0] * (array(resolutions) / resolutions[0]) ** -solver_options["order"]
    )
    plt.loglog(resolutions, errors, "-o", mfc="none", label=r"$L_1$")
    plt.loglog(resolutions, expected, label=r"$N^{-3}$")
    plt.xlabel(r"$N$")
    plt.ylabel(r"$L_1$")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    main(args.parse_args())
