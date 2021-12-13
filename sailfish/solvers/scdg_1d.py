"""
An n-th order discontinuous Galerkin solver for 1D scalar advection.
"""

from typing import NamedTuple
from numpy.polynomial.legendre import leggauss, Legendre
from sailfish.mesh import PlanarCartesianMesh
from sailfish.solver import SolverBase


class CellData:
    """
    Gauss weights, quadrature points, and tabulated Legendre polonomials.

    This class works for n-th order Gaussian quadrature in 1D.
    """

    def __init__(self, order=3):
        import numpy as np

        if order <= 0:
            raise ValueError("cell order must be at least 1")

        def leg(x, n, m=0):
            c = [(2 * n + 1) ** 0.5 if i is n else 0.0 for i in range(n + 1)]
            return Legendre(c).deriv(m)(x)

        f = [-1.0, 1.0]  # xsi-coordinate of faces
        g, w = leggauss(order)
        self.gauss_points = g
        self.weights = w
        self.phi_faces = np.array([[leg(x, n, m=0) for n in range(order)] for x in f])
        self.phi_value = np.array([[leg(x, n, m=0) for n in range(order)] for x in g])
        self.phi_deriv = np.array([[leg(x, n, m=1) for n in range(order)] for x in g])
        self.order = order

    def to_weights(self, ux):
        w = self.weights
        p = self.phi_value
        o = self.order
        return [sum(ux[j] * p[j][n] * w[j] for j in range(o)) * 0.5 for n in range(o)]

    def sample(self, uw, j):
        return dot(uw, self.phi_value[j])

    @property
    def num_points(self):
        return self.order


def dot(u, p):
    return sum(u[i] * p[i] for i in range(u.shape[0]))


def update(wavespeed, uw, cell, dx, dt):
    def flux(ux):
        return wavespeed * ux

    nz = uw.shape[0]
    pv = cell.phi_value
    pf = cell.phi_faces
    pd = cell.phi_deriv
    w = cell.weights
    h = [-1.0, 1.0]

    for i in range(nz):
        im1 = (i - 1 + nz) % nz
        ip1 = (i + 1 + nz) % nz

        # surface fluxes
        fimh_l = flux(dot(uw[im1], pf[1]))
        fimh_r = flux(dot(uw[i], pf[0]))
        fiph_l = flux(dot(uw[i], pf[1]))
        fiph_r = flux(dot(uw[ip1], pf[0]))

        if wavespeed > 0.0:
            fimh = fimh_l
            fiph = fiph_l
        else:
            fimh = fimh_r
            fiph = fiph_r

        fs = [fimh, fiph]
        ux = [cell.sample(uw[i], j) for j in range(cell.order)]
        fx = [flux(u) for u in ux]

        for n in range(cell.order):
            udot_s = -sum(fs[j] * pf[j][n] * h[j] for j in range(2)) / dx
            udot_v = +sum(fx[j] * pd[j][n] * w[j] for j in range(cell.num_points)) / dx
            uw[i, n] += (udot_s + udot_v) * dt

def rhs(wavespeed, uw, cell, dx, uwdot):
    import numpy as np

    def flux(ux):
        #return wavespeed * ux
        return 0.5 * ux * ux

    nz = uw.shape[0]
    pv = cell.phi_value
    pf = cell.phi_faces
    pd = cell.phi_deriv
    w = cell.weights
    h = [-1.0, 1.0]

    for i in range(nz):
        im1 = (i - 1 + nz) % nz
        ip1 = (i + 1 + nz) % nz

        # surface fluxes
        fimh_l = flux(dot(uw[im1], pf[1]))
        fimh_r = flux(dot(uw[i], pf[0]))
        fiph_l = flux(dot(uw[i], pf[1]))
        fiph_r = flux(dot(uw[ip1], pf[0]))

        if wavespeed > 0.0:
            fimh = fimh_l
            fiph = fiph_l
        else:
            fimh = fimh_r
            fiph = fiph_r

        fs = [fimh, fiph]
        ux = [cell.sample(uw[i], j) for j in range(cell.order)]
        fx = [flux(u) for u in ux]

        for n in range(cell.order):
            udot_s = -sum(fs[j] * pf[j][n] * h[j] for j in range(2)) / dx
            udot_v = +sum(fx[j] * pd[j][n] * w[j] for j in range(cell.num_points)) / dx
            uwdot[i, n] = udot_s + udot_v


class Options(NamedTuple):
    order: int = 3


class Physics(NamedTuple):
    wavespeed: float = 1.0


class Solver(SolverBase):
    """
    An n-th order, discontinuous Galerkin solver for 1D scalar advection.
    """

    def __init__(
        self,
        setup=None,
        mesh=None,
        time=0.0,
        solution=None,
        num_patches=1,
        mode="cpu",
        physics=dict(),
        options=dict(),
    ):
        import numpy as np

        if num_patches != 1:
            raise ValueError("only works on one patch")

        if type(mesh) != PlanarCartesianMesh:
            raise ValueError("only the planar cartesian mesh is supported")

        if mode != "cpu":
            raise ValueError("only cpu mode is supported")

        if setup.boundary_condition != "periodic":
            raise ValueError("only periodic boundaries are supported")

        options = Options(**options)
        physics = Physics(**physics)
        cell = CellData(order=options.order)

        if solution is None:
            num_zones = mesh.shape[0]
            xf = mesh.faces(0, num_zones)  # face coordinates
            px = np.zeros([num_zones, cell.num_points, 1])
            ux = np.zeros([num_zones, cell.num_points, 1])
            uw = np.zeros([num_zones, cell.order, 1])
            dx = mesh.dx

            for i in range(num_zones):
                for j in range(cell.num_points):
                    xsi = cell.gauss_points[j]
                    xj = xf[i] + (xsi + 1.0) * 0.5 * dx
                    setup.primitive(time, xj, px[i, j])

            ux[...] = px[...]  # the conserved variable is also the primitive

            for i in range(num_zones):
                uw[i] = cell.to_weights(ux[i])
            self.conserved_w = uw
        else:
            self.conserved_w = solution

        self.conserved_wdot = np.zeros_like(self.conserved_w)
        self.conserved_w_rk = np.zeros_like(self.conserved_w)
        self.t = time
        self.mesh = mesh
        self.cell = cell
        self._options = options
        self._physics = physics

    @property
    def solution(self):
        return self.conserved_w

    @property
    def primitive(self):
        return self.conserved_w[:, 0]

    @property
    def time(self):
        return self.t

    @property
    def maximum_cfl(self):
        return 1.0

    @property
    def options(self):
        return self._options._asdict()

    @property
    def physics(self):
        return self._physics._asdict()

    @property
    def maximum_cfl(self):
        return 1.0

    def maximum_wavespeed(self):
        return abs(self._physics.wavespeed)

    def advance(self, dt):

        # Euler 1st order
        #rhs(self._physics.wavespeed, self.conserved_w, self.cell, self.mesh.dx, self.conserved_wdot)
        #self.conserved_w = self.conserved_w + dt * self.conserved_wdot

        # Strong Stability Preserving (TVD) 2nd Order SSP-RK2 (Shu & Osher Eq. 2.15)
        #rhs(self._physics.wavespeed, self.conserved_w, self.cell, self.mesh.dx, self.conserved_wdot)
        #self.conserved_w_rk = self.conserved_w + dt * self.conserved_wdot
        #rhs(self._physics.wavespeed, self.conserved_w_rk, self.cell, self.mesh.dx, self.conserved_wdot)
        #self.conserved_w_rk += dt * self.conserved_wdot
        #self.conserved_w = 0.5 * (self.conserved_w + self.conserved_w_rk)       

        # Strong Stability Preserving (TVD) 3rd Order SSP-RK3 (Shu & Osher Eq. 2.18)
        rhs(self._physics.wavespeed, self.conserved_w, self.cell, self.mesh.dx, self.conserved_wdot)
        self.conserved_w_rk = self.conserved_w + dt * self.conserved_wdot
        rhs(self._physics.wavespeed, self.conserved_w_rk, self.cell, self.mesh.dx, self.conserved_wdot)
        self.conserved_w_rk = 0.75 * self.conserved_w + 0.25 * (self.conserved_w_rk + dt * self.conserved_wdot)
        rhs(self._physics.wavespeed, self.conserved_w_rk, self.cell, self.mesh.dx, self.conserved_wdot)
        self.conserved_w = (1.0 / 3.0) * self.conserved_w + (2.0 /3.0) * (self.conserved_w_rk + dt * self.conserved_wdot)

        # Strong Stability Preserving (TVD) Four-stage 3rd Order SSP-4RK3; Stable for CFL <= 2.0 (Spiteri & Ruuth (2002))
        #rhs(self._physics.wavespeed, self.conserved_w, self.cell, self.mesh.dx, self.conserved_wdot)
        #self.conserved_w_rk = self.conserved_w + 0.5 * dt * self.conserved_wdot
        #rhs(self._physics.wavespeed, self.conserved_w_rk, self.cell, self.mesh.dx, self.conserved_wdot)
        #self.conserved_w_rk = self.conserved_w_rk + 0.5 * dt * self.conserved_wdot
        #rhs(self._physics.wavespeed, self.conserved_w_rk, self.cell, self.mesh.dx, self.conserved_wdot)
        #self.conserved_w_rk = (2.0 / 3.0) * self.conserved_w + (1.0 / 3.0) * (self.conserved_w_rk + 0.5 * dt * self.conserved_wdot)
        #rhs(self._physics.wavespeed, self.conserved_w, self.cell, self.mesh.dx, self.conserved_wdot)
        #self.conserved_w = self.conserved_w_rk + 0.5 * dt * self.conserved_wdot


        self.t += dt
