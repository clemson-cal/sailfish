#!/usr/bin/env python3

# Take checkpoint files and return the L1 error with respect to the analytic solution
# by integrating L1 = 1/V Integral(abs(u_computed-u_analytic)) dV over the cell using Gaussian
# quadrature with order l1_order. Schaal+15 recommend using 2 orders higher quadrature than the 
# simulation order: l1_order = simulation_order + 2 to make sure the measured error is that due 
# to the simulation itself, and not from the analysis.

import argparse
import pickle
import sys
import numpy as np
from numpy.polynomial.legendre import leggauss, Legendre
from math import pi, sin

sys.path.insert(1, ".")

# Schaal+15 recommends computing the L1 error using order p + 2 quadrature, where p is the order
# at which the simulation was run.
l1_order = 5

# Analytic solution from initial condition from Advection setup in
# sailfish/setups/simple1d.py
def analytic(t, x):
    a = 0.1
    k = 2.0 * pi
    wavespeed = 1.0
    return 1.0 + a * sin(k * (x - wavespeed * t))

#Legendre polynomials scaled by sqrt(2n+1) used as basis functions
def leg(x, n):
            c = [(2 * n + 1) ** 0.5 if i is n else 0.0 for i in range(n + 1)]
            return Legendre(c)(x)

def dot(u, p):
    return sum(u[i] * p[i] for i in range(u.shape[0]))

#def sample(uw, j):
#    return dot(uw, phi_value[j])

# Gaussian quadrature points inside cell
gauss_points, weights = leggauss(l1_order)

# Value of basis functions at the quadrature points
phi_value = np.array([[leg(x, n) for n in range(l1_order)] for x in gauss_points])

def main(args):

    import sailfish

    for checkpoint in args.checkpoints:
        with open(checkpoint, "rb") as f:
            chkpt = pickle.load(f)

        print(checkpoint)
        time = chkpt["time"]
        print(checkpoint,"time = ", time)
        mesh = chkpt["mesh"]
        x  = mesh.zone_centers(chkpt["time"])
        dx = x[1] - x[0]
        num_zones = len(x)
        uw = chkpt["solution"]
        num_points = len(gauss_points)

        l1 = 0
        for i in range(num_zones):
            for j in range(num_points):
                xsi = gauss_points[j]
                xj = x[i] + xsi * 0.5 * dx
                u_analytic = analytic(time, xj)
                u_computed = dot(uw[i],phi_value[j])
                l1 += abs(u_computed - u_analytic) * weights[j] 

        print(l1)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("checkpoints", type=str, nargs="+")
    main(args.parse_args())
