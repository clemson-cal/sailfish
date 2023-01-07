"""
An n-th order discontinuous Galerkin solver for 1D scalar advection and inviscid Burgers eqn.
"""

from typing import NamedTuple
from sailfish.mesh import PlanarCartesianMesh
from sailfish.solver_base import SolverBase
from sailfish.kernel.library import Library
from numpy.polynomial.legendre import leggauss, Legendre
import numpy as np

NUM_CONS = 1


class CellData:
    """
    Gauss weights, quadrature points, and tabulated Legendre polonomials.

    This class works for n-th order Gaussian quadrature in 1D.
    """

    def __init__(self, order=1):
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
        uw = np.zeros([NUM_CONS, self.order])

        for q in range(NUM_CONS):
            for n in range(self.order):
                for j in range(self.num_points):
                    uw[q, n] += ux[q, j] * p[j][n] * w[j] * 0.5

        return uw

    def sample(self, uw, j):
        ux = np.zeros(NUM_CONS)

        for q in range(NUM_CONS):
            for n in range(self.order):
                ux[q] += uw[q, n] * self.phi_value[j, n]
        return ux

    def sample_face(self, uw, j):
        ux = np.zeros(NUM_CONS)

        for q in range(NUM_CONS):
            for n in range(self.order):
                ux[q] += uw[q, n] * self.phi_faces[j, n]
        return ux

    @property
    def num_points(self):
        return self.order


# def limit_troubled_cells(u):
#     def minmod(w1, w0l, w0, w0r):

#         BETA_TVB = 1.0
#         a = w1 * (3.0 ** 0.5)
#         b = (w0 - w0l) * BETA_TVB
#         c = (w0r - w0) * BETA_TVB

#         return (
#             (0.25 / (3.0 ** 0.5))
#             * abs(np.sign(a) + np.sign(b))
#             * (np.sign(a) + np.sign(c))
#             * min(abs(a), abs(b), abs(c))
#         )

#     nz = u.shape[0]

#     for i in range(nz):
#         im1 = (i - 1 + nz) % nz
#         ip1 = (i + 1 + nz) % nz

#         # integrating polynomial extended from left zone into this zone
#         a = (
#             1.0 * u[im1, 0]
#             + 2.0 * (3.0 ** 0.5) * u[im1, 1]
#             + 5.0 * (5.0 ** 0.5) / 3.0 * u[im1, 2]
#         )

#         # integrating polynomial extended from right zone into this zone
#         b = (
#             1.0 * u[ip1, 0]
#             - 2.0 * (3.0 ** 0.5) * u[ip1, 1]
#             + 5.0 * (5.0 ** 0.5) / 3.0 * u[ip1, 2]
#         )

#         tci = (abs(u[i, 0] - a) + abs(u[i, 0] - b)) / max(
#             abs(u[im1, 0]), abs(u[i, 0]), abs(u[ip1, 0])
#         )

#         if tci > 0.1:
#             w1t = minmod(u[i, 1], u[im1, 0], u[i, 0], u[ip1, 0])
#             if u[i, 1] != w1t:
#                 u[i, 1] = w1t
#                 u[i, 2] = 0.0


def rhs(physics, uw, cell, dx, uwdot):
    if physics.equation == "advection":
        wavespeed = physics.wavespeed

        def flux(ux):
            return wavespeed * ux

        def upwind(ul, ur):
            if wavespeed > 0.0:
                return flux(ul)
            else:
                return flux(ur)

    elif physics.equation == "burgers":

        def flux(ux):
            return np.array([0.5 * ux[0] * ux[0]])

        def upwind(ul, ur):
            al = ul[0]
            ar = ur[0]

            if al > 0.0 and ar > 0.0:
                return flux(ul)
            elif al < 0.0 and ar < 0.0:
                return flux(ur)
            else:
                return np.array([0.0])

    nz = uw.shape[0]
    pv = cell.phi_value
    pf = cell.phi_faces
    pd = cell.phi_deriv
    w = cell.weights
    nhat = np.array([-1.0, 1.0])

    for i in range(nz):
        im1 = (i - 1 + nz) % nz
        ip1 = (i + 1 + nz) % nz

        uimh_l = cell.sample_face(uw[im1], 1)
        uimh_r = cell.sample_face(uw[i], 0)
        uiph_l = cell.sample_face(uw[i], 1)
        uiph_r = cell.sample_face(uw[ip1], 0)
        fimh = upwind(uimh_l, uimh_r)
        fiph = upwind(uiph_l, uiph_r)

        fs = np.array([fimh, fiph]).T
        ux = np.array([cell.sample(uw[i], j) for j in range(cell.order)]).T
        fx = np.array([flux(u) for u in ux.T]).T

        for n in range(cell.order):
            udot_s = 0.0
            udot_v = 0.0

            for j in range(2):
                udot_s -= fs[0, j] * pf[j, n] * nhat[j] / dx

            for j in range(cell.num_points):
                udot_v += fx[0, j] * pd[j, n] * w[j] / dx

            uwdot[i, 0, n] = udot_s + udot_v


class Options(NamedTuple):
    order: int = 1
    integrator: str = "rk2"


class Physics(NamedTuple):
    wavespeed: float = 1.0
    equation: str = "advection"  # or burgers


class Solver(SolverBase):
    """
    An n-th order, discontinuous Galerkin solver for 1D scalar advection.

    Time-advance integrator options:

    - :code:`rk1`: Forward Euler
    - :code:`rk2`: SSP-RK2 of Shu & Osher (1988; Eq. 2.15)
    - :code:`rk3`: SSP-RK3 of Shu & Osher (1988; Eq. 2.18)
    - :code:`rk3-sr02`: four-stage 3rd Order SSP-4RK3 of Spiteri & Ruuth (2002)
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
        options = Options(**options)
        physics = Physics(**physics)
        cell = CellData(order=options.order)

        if num_patches != 1:
            raise ValueError("only works on one patch")

        if type(mesh) != PlanarCartesianMesh:
            raise ValueError("only the planar cartesian mesh is supported")

        if mode != "cpu":
            raise ValueError("only cpu mode is supported")

        if setup.boundary_condition != "periodic":
            raise ValueError("only periodic boundaries are supported")

        if physics.equation not in ["advection", "burgers"]:
            raise ValueError("physics.equation must be advection or burgers")

        if options.integrator not in [
            "rk1",
            "rk2",
            "rk3",
            "rk3-sr02",
            "SSPRK32",
            "SSPRK43",
            "SSPRK53",
            "SSPRK54",
        ]:
            raise ValueError(
                "options.integrator must be "
                "rk1|rk2|rk3|rk3-sr02|SSPRK32|SSPRK43|SSPRK53|SSPRK54"
            )

        if options.order <= 0:
            raise ValueError("option.order must be greater than 0")

        with open(__file__.replace(".py", ".c"), "r") as f:
            source = f.read()

        self.lib = Library(source, mode=mode, debug=True)

        if solution is None:
            num_zones = mesh.shape[0]
            xf = mesh.faces(0, num_zones)  # face coordinates
            px = np.zeros([num_zones, 1, cell.num_points])
            ux = np.zeros([num_zones, 1, cell.num_points])
            uw = np.zeros([num_zones, 1, cell.order])
            dx = mesh.dx

            for i in range(num_zones):
                for j in range(cell.num_points):
                    xsi = cell.gauss_points[j]
                    xj = xf[i] + (xsi + 1.0) * 0.5 * dx
                    setup.primitive(time, xj, px[i, :, j])

            ux[...] = px[...]  # the conserved variable is also the primitive

            for i in range(num_zones):
                uw[i] = cell.to_weights(ux[i])
            self.conserved_w = uw
        else:
            self.conserved_w = solution

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
        k = self.cell.order - 1

        if self._options.integrator == "rk1":
            return 1.0 / (2 * k + 1)
        if self._options.integrator == "rk2":
            return 1.0 / (2 * k + 1)
        if self._options.integrator == "rk3":
            return 1.0 / (2 * k + 1)
        if self._options.integrator == "rk3-sr02":
            return 2.0 / (2 * k + 1)
        if self._options.integrator == "SSPRK32":
            return 2.0 / (2 * k + 1)  # up to 2.2 / (2 * k + 1) seems to work
        if self._options.integrator == "SSPRK43":
            return 2.0 / (2 * k + 1)  # C = 1.683339717642499
        if self._options.integrator == "SSPRK53":
            return 2.387300839230550 / (2 * k + 1)  # C = 2.387300839230550
        if self._options.integrator == "SSPRK54":
            return 1.5 / (2 * k + 1)  # up to 1.7 / (2 * k + 1) seems to work

    def maximum_wavespeed(self):
        if self._physics.equation == "advection":
            return abs(self._physics.wavespeed)
        elif self._physics.equation == "burgers":
            return abs(self.conserved_w[:, 0]).max()

    def advance(self, dt):
        def udot(u):
            udot = np.zeros_like(u)
            # rhs(self._physics, u, self.cell, self.mesh.dx, udot)
            self.lib.scdg_1d_udot[u.shape[0]](u, udot, self.mesh.dx)
            return udot

        if self._options.integrator == "rk1":
            u = self.conserved_w
            u += dt * udot(u)

        if self._options.integrator == "rk2":
            b1 = 0.0
            b2 = 0.5
            u = u0 = self.conserved_w.copy()
            u = u0 * b1 + (1.0 - b1) * (u + dt * udot(u))
            u = u0 * b2 + (1.0 - b2) * (u + dt * udot(u))

        if self._options.integrator == "rk3":
            b1 = 0.0
            b2 = 3.0 / 4.0
            b3 = 1.0 / 3.0
            u = u0 = self.conserved_w.copy()
            u = u0 * b1 + (1.0 - b1) * (u + dt * udot(u))
            u = u0 * b2 + (1.0 - b2) * (u + dt * udot(u))
            u = u0 * b3 + (1.0 - b3) * (u + dt * udot(u))

        if self._options.integrator == "rk3-sr02":
            u = u0 = self.conserved_w.copy()
            u = u0 + 0.5 * dt * udot(u)
            u = u + 0.5 * dt * udot(u)
            u = 2.0 / 3.0 * u0 + 1.0 / 3.0 * (u + 0.5 * dt * udot(u))
            u = u + 0.5 * dt * udot(u)

        if self._options.integrator == "SSPRK32":
            """
            3-stage 2nd-order Strong Stability Preserving SSPRK(3,2) integrator in Shu-Osher form
            Not a low storage integrator: requires 3 copies of the conserved array, plus 2 copies
            of its time derivative (optionally) which could instead be recomputed e.g. on a GPU
            Reference: Kubatko+, J Sci Comput (2014) 60:313–344; Table 7
            """
            alpha = [
                [1.000000000000000, 0.000000000000000, 0.000000000000000],
                [0.087353119859156, 0.912646880140844, 0.000000000000000],
                [0.344956917166841, 0.000000000000000, 0.655043082833159],
            ]
            beta = [
                [0.528005024856522, 0.000000000000000, 0.000000000000000],
                [0.000000000000000, 0.481882138633993, 0.000000000000000],
                [0.022826837460491, 0.000000000000000, 0.345866039233415],
            ]

            u = u0 = self.conserved_w.copy()
            udot_0 = udot(u0)
            u1 = alpha[1 - 1][0] * u0 + beta[1 - 1][0] * dt * udot_0
            udot_1 = udot(u1)
            u2 = (
                alpha[2 - 1][0] * u0
                + beta[2 - 1][0] * dt * udot_0
                + alpha[2 - 1][1] * u1
                + beta[2 - 1][1] * dt * udot_1
            )
            # udot_2 = udot(u2)
            u = (
                alpha[3 - 1][0] * u0
                + beta[3 - 1][0] * dt * udot_0
                + alpha[3 - 1][1] * u1
                + beta[3 - 1][1] * dt * udot_1
                + alpha[3 - 1][2] * u2
                + beta[3 - 1][2] * dt * udot(u2)
            )

        if self._options.integrator == "SSPRK43":
            """
            4-stage 3rd-order Strong Stability Preserving SSPRK(4,3) integrator in Shu-Osher form
            Not a low storage integrator: requires 4 copies of the conserved array, plus 3 copies
            of its time derivative (optionally) which could instead be recomputed e.g. on a GPU
            C = 1.683339717642499
            Reference: Kubatko+, J Sci Comput (2014) 60:313–344; Table 13
            """
            alpha = [
                [
                    1.000000000000000,
                    0.000000000000000,
                    0.000000000000000,
                    0.000000000000000,
                ],
                [
                    0.522361915162541,
                    0.477638084837459,
                    0.000000000000000,
                    0.000000000000000,
                ],
                [
                    0.368530939472566,
                    0.000000000000000,
                    0.631469060527434,
                    0.000000000000000,
                ],
                [
                    0.334082932462285,
                    0.006966183666289,
                    0.000000000000000,
                    0.658950883871426,
                ],
            ]
            beta = [
                [
                    0.594057152884440,
                    0.000000000000000,
                    0.000000000000000,
                    0.000000000000000,
                ],
                [
                    0.000000000000000,
                    0.283744320787718,
                    0.000000000000000,
                    0.000000000000000,
                ],
                [
                    0.000000038023030,
                    0.000000000000000,
                    0.375128712231540,
                    0.000000000000000,
                ],
                [
                    0.116941419604231,
                    0.004138311235266,
                    0.000000000000000,
                    0.391454485963345,
                ],
            ]

            u = u0 = self.conserved_w.copy()
            udot_0 = udot(u0)
            u1 = alpha[1 - 1][0] * u0 + beta[1 - 1][0] * dt * udot_0
            udot_1 = udot(u1)
            u2 = (
                alpha[2 - 1][0] * u0
                + beta[2 - 1][0] * dt * udot_0
                + alpha[2 - 1][1] * u1
                + beta[2 - 1][1] * dt * udot_1
            )
            udot_2 = udot(u2)
            u3 = (
                alpha[3 - 1][0] * u0
                + beta[3 - 1][0] * dt * udot_0
                + alpha[3 - 1][1] * u1
                + beta[3 - 1][1] * dt * udot_1
                + alpha[3 - 1][2] * u2
                + beta[3 - 1][2] * dt * udot_2
            )
            # udot_3 = udot(u3)
            u = (
                alpha[4 - 1][0] * u0
                + beta[4 - 1][0] * dt * udot_0
                + alpha[4 - 1][1] * u1
                + beta[4 - 1][1] * dt * udot_1
                + alpha[4 - 1][2] * u2
                + beta[4 - 1][2] * dt * udot_2
                + alpha[4 - 1][3] * u3
                + beta[4 - 1][3] * dt * udot(u3)
            )

        if self._options.integrator == "SSPRK53":
            """
            5-stage 3rd-order Strong Stability Preserving SSPRK(5,3) integrator in Shu-Osher form
            Not a low storage integrator: requires 5 copies of the conserved array, plus 4 copies
            of its time derivative (optionally) which could instead be recomputed e.g. on a GPU
            C = 2.387300839230550
            Reference: Kubatko+, J Sci Comput (2014) 60:313–344; Table 18
            """
            alpha = [
                [
                    1.000000000000000,
                    0.000000000000000,
                    0.000000000000000,
                    0.000000000000000,
                    0.000000000000000,
                ],
                [
                    0.495124140877703,
                    0.504875859122297,
                    0.000000000000000,
                    0.000000000000000,
                    0.000000000000000,
                ],
                [
                    0.105701991897526,
                    0.000000000000000,
                    0.894298008102474,
                    0.000000000000000,
                    0.000000000000000,
                ],
                [
                    0.411551205755676,
                    0.011170516177380,
                    0.000000000000000,
                    0.577278278066944,
                    0.000000000000000,
                ],
                [
                    0.186911123548222,
                    0.013354480555382,
                    0.012758264566319,
                    0.000000000000000,
                    0.786976131330077,
                ],
            ]
            beta = [
                [
                    0.418883109982196,
                    0.000000000000000,
                    0.000000000000000,
                    0.000000000000000,
                    0.000000000000000,
                ],
                [
                    0.000000000000000,
                    0.211483970024081,
                    0.000000000000000,
                    0.000000000000000,
                    0.000000000000000,
                ],
                [
                    0.000000000612488,
                    0.000000000000000,
                    0.374606330884848,
                    0.000000000000000,
                    0.000000000000000,
                ],
                [
                    0.046744815663888,
                    0.004679140556487,
                    0.000000000000000,
                    0.241812120441849,
                    0.000000000000000,
                ],
                [
                    0.071938257223857,
                    0.005593966347235,
                    0.005344221539515,
                    0.000000000000000,
                    0.329651009373300,
                ],
            ]

            u = u0 = self.conserved_w.copy()
            udot_0 = udot(u0)
            u1 = alpha[1 - 1][0] * u0 + beta[1 - 1][0] * dt * udot_0
            udot_1 = udot(u1)
            u2 = (
                alpha[2 - 1][0] * u0
                + beta[2 - 1][0] * dt * udot_0
                + alpha[2 - 1][1] * u1
                + beta[2 - 1][1] * dt * udot_1
            )
            udot_2 = udot(u2)
            u3 = (
                alpha[3 - 1][0] * u0
                + beta[3 - 1][0] * dt * udot_0
                + alpha[3 - 1][1] * u1
                + beta[3 - 1][1] * dt * udot_1
                + alpha[3 - 1][2] * u2
                + beta[3 - 1][2] * dt * udot_2
            )
            udot_3 = udot(u3)
            u4 = (
                alpha[4 - 1][0] * u0
                + beta[4 - 1][0] * dt * udot_0
                + alpha[4 - 1][1] * u1
                + beta[4 - 1][1] * dt * udot_1
                + alpha[4 - 1][2] * u2
                + beta[4 - 1][2] * dt * udot_2
                + alpha[4 - 1][3] * u3
                + beta[4 - 1][3] * dt * udot_3
            )
            # udot_4 = udot(u4)
            u = (
                alpha[5 - 1][0] * u0
                + beta[5 - 1][0] * dt * udot_0
                + alpha[5 - 1][1] * u1
                + beta[5 - 1][1] * dt * udot_1
                + alpha[5 - 1][2] * u2
                + beta[5 - 1][2] * dt * udot_2
                + alpha[5 - 1][3] * u3
                + beta[5 - 1][3] * dt * udot_3
                + alpha[5 - 1][4] * u4
                + beta[5 - 1][4] * dt * udot(u4)
            )

        if self._options.integrator == "SSPRK54":
            """
            5-stage 4th-order Strong Stability Preserving SSPRK(5,4) integrator in Shu-Osher form
            Not a low storage integrator: requires 5 copies of the conserved array, plus 4 copies
            of its time derivative (optionally) which could instead be recomputed e.g. on a GPU
            Reference: Kubatko+, J Sci Comput (2014) 60:313–344; Table 18
            """
            alpha = [
                [
                    1.000000000000000,
                    0.000000000000000,
                    0.000000000000000,
                    0.000000000000000,
                    0.000000000000000,
                ],
                [
                    0.261216512493821,
                    0.738783487506179,
                    0.000000000000000,
                    0.000000000000000,
                    0.000000000000000,
                ],
                [
                    0.623613752757655,
                    0.000000000000000,
                    0.376386247242345,
                    0.000000000000000,
                    0.000000000000000,
                ],
                [
                    0.444745181201454,
                    0.120932584902288,
                    0.000000000000000,
                    0.434322233896258,
                    0.000000000000000,
                ],
                [
                    0.213357715199957,
                    0.209928473023448,
                    0.063353148180384,
                    0.000000000000000,
                    0.513360663596212,
                ],
            ]
            beta = [
                [
                    0.605491839566400,
                    0.000000000000000,
                    0.000000000000000,
                    0.000000000000000,
                    0.000000000000000,
                ],
                [
                    0.000000000000000,
                    0.447327372891397,
                    0.000000000000000,
                    0.000000000000000,
                    0.000000000000000,
                ],
                [
                    0.000000844149769,
                    0.000000000000000,
                    0.227898801230261,
                    0.000000000000000,
                    0.000000000000000,
                ],
                [
                    0.002856233144485,
                    0.073223693296006,
                    0.000000000000000,
                    0.262978568366434,
                    0.000000000000000,
                ],
                [
                    0.002362549760441,
                    0.127109977308333,
                    0.038359814234063,
                    0.000000000000000,
                    0.310835692561898,
                ],
            ]

            u = u0 = self.conserved_w.copy()
            udot_0 = udot(u0)
            u1 = alpha[1 - 1][0] * u0 + beta[1 - 1][0] * dt * udot_0
            udot_1 = udot(u1)
            u2 = (
                alpha[2 - 1][0] * u0
                + beta[2 - 1][0] * dt * udot_0
                + alpha[2 - 1][1] * u1
                + beta[2 - 1][1] * dt * udot_1
            )
            udot_2 = udot(u2)
            u3 = (
                alpha[3 - 1][0] * u0
                + beta[3 - 1][0] * dt * udot_0
                + alpha[3 - 1][1] * u1
                + beta[3 - 1][1] * dt * udot_1
                + alpha[3 - 1][2] * u2
                + beta[3 - 1][2] * dt * udot_2
            )
            udot_3 = udot(u3)
            u4 = (
                alpha[4 - 1][0] * u0
                + beta[4 - 1][0] * dt * udot_0
                + alpha[4 - 1][1] * u1
                + beta[4 - 1][1] * dt * udot_1
                + alpha[4 - 1][2] * u2
                + beta[4 - 1][2] * dt * udot_2
                + alpha[4 - 1][3] * u3
                + beta[4 - 1][3] * dt * udot_3
            )
            # udot_4 = udot(u4)
            u = (
                alpha[5 - 1][0] * u0
                + beta[5 - 1][0] * dt * udot_0
                + alpha[5 - 1][1] * u1
                + beta[5 - 1][1] * dt * udot_1
                + alpha[5 - 1][2] * u2
                + beta[5 - 1][2] * dt * udot_2
                + alpha[5 - 1][3] * u3
                + beta[5 - 1][3] * dt * udot_3
                + alpha[5 - 1][4] * u4
                + beta[5 - 1][4] * dt * udot(u4)
            )

        # limit_troubled_cells(u)

        self.conserved_w = u
        self.t += dt
