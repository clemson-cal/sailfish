"""
One-dimensional discontinuous Galerkin (DG) scalar advection and Burgers solver.
"""

from logging import getLogger
from sailfish.kernel.library import Library
from sailfish.kernel.system import get_array_module
from sailfish.subdivide import subdivide
from sailfish.mesh import PlanarCartesianMesh, LogSphericalMesh
from sailfish.solver import SolverBase

try:
    from contextlib import nullcontext

except ImportError:
    from contextlib import AbstractContextManager

    class nullcontext(AbstractContextManager):
        """
        Scraped from contextlib source in Python >= 3.7 for backwards compatibility.
        """

        def __init__(self, enter_result=None):
            self.enter_result = enter_result

        def __enter__(self):
            return self.enter_result

        def __exit__(self, *excinfo):
            pass

        async def __aenter__(self):
            return self.enter_result

        async def __aexit__(self, *excinfo):
            pass


logger = getLogger(__name__)

BC_PERIODIC = 0
BC_OUTFLOW = 1
BC_INFLOW = 2
BC_REFLECT = 3

BC_DICT = {
    "periodic": BC_PERIODIC,
    "outflow": BC_OUTFLOW,
    "inflow": BC_INFLOW,
    "reflect": BC_REFLECT,
}
COORDINATES_DICT = {
    PlanarCartesianMesh: 0,
    LogSphericalMesh: 1,
}


def initial_condition(setup, mesh, time):
    import numpy as np

    faces = np.array(mesh.faces(0, mesh.shape[0]))
    zones = 0.5 * (faces[:-1] + faces[1:])
    primitive = np.zeros([len(zones), 1])

    for x, p in zip(zones, primitive):
        setup.primitive(time, x, p)

    return primitive

class CellData:
    """
    Gauss weights, quadrature points, and Legendre polynomials scaled by sqrt(2 * n + 1).

    This class works for n-th order Gaussian quadrature in 1D.

    Unscaled Legendre polynomials at end points: P_k(-1) = (-1)^k  P_k(1) = 1
    where k = order - 1 
    """

    def __init__(self, order=1):
        import numpy as np

        if order <= 0:
            raise ValueError("cell order must be at least 1")

        def leg(x, n, m=0):
            c = [(2 * n + 1) ** 0.5 if i is n else 0.0 for i in range(n + 1)]
            return Legendre(c).deriv(m)(x)

        g, w = leggauss(order)
        self.gauss_points = g
        self.weights = w
        self.phi_value = np.array([[leg(x, n, m=0) for n in range(order)] for x in g])
        self.phi_deriv = np.array([[leg(x, n, m=1) for n in range(order)] for x in g])
        self.order = order

    def to_weights(self, ux):
        w = self.weights
        p = self.phi
        o = self.order
        return [sum(ux[j] * p[j][n] * w[j] for j in range(o)) * 0.5 for n in range(o)]

    def sample(self, uw, j):
        return dot(uw, self.phi_value[j])

    @property
    def num_points(self):
        return self.order


class Patch:
    """
    Holds the array buffer state for the solution on a subset of the
    solution domain.
    """

    def __init__(
        self,
        time,
        primitive,
        mesh,
        index_range,
        lib,
        xp,
    ):
        i0, i1 = index_range
        self.lib = lib
        self.xp = xp
        self.num_zones = index_range[1] - index_range[0]
        self.faces = self.xp.array(mesh.faces(*index_range))
        self.coordinates = COORDINATES_DICT[type(mesh)]
        try:
            adot = float(mesh.scale_factor_derivative)
            self.scale_factor_initial = 0.0
            self.scale_factor_derivative = adot
        except (TypeError, AttributeError):
            self.scale_factor_initial = 1.0
            self.scale_factor_derivative = 0.0

        self.time = self.time0 = time

        with self.execution_context():
            self.primitive1 = self.xp.array(primitive)
            self.conserved0 = self.primitive_to_conserved(self.primitive1)
            self.conserved1 = self.conserved0.copy()
            self.conserved2 = self.conserved0.copy()

    def execution_context(self):
        """TODO: return a CUDA context for execution on the assigned device"""
        return nullcontext()

    def primitive_to_conserved(self, primitive):
        with self.execution_context():
            conserved = self.xp.zeros_like(primitive)
            self.lib.srhd_1d_primitive_to_conserved[self.num_zones](
                self.faces,
                primitive,
                conserved,
                self.scale_factor,
                self.coordinates,
            )
            return conserved

    def recompute_primitive(self):
        with self.execution_context():
            self.lib.srhd_1d_conserved_to_primitive[self.num_zones](
                self.faces,
                self.conserved1,
                self.primitive1,
                self.scale_factor,
                self.coordinates,
            )

    def advance_rk(self, rk_param, dt):
        with self.execution_context():
            self.lib.scdg_1d_advance_rk[self.num_zones](
                self.conserved0,
                self.conserved1,
                self.conserved2,
                self.time,
                rk_param,
                dt,
                dx,
            )
        self.time = self.time0 * rk_param + (self.time0 + dt) * (1.0 - rk_param)
        self.conserved1, self.conserved2 = self.conserved2, self.conserved1

    @property
    def scale_factor(self):
        return self.scale_factor_initial + self.scale_factor_derivative * self.time

    def new_iteration(self):
        self.time0 = self.time
        self.conserved0[...] = self.conserved1[...]

    @property
    def primitive(self):
        self.recompute_primitive()
        return self.primitive1


class Solver(SolverBase):
    """
    Adapter class to drive the scdg_1d C extension module.
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
        try:
            bcl, bcr = setup.boundary_condition
        except ValueError:
            bcl = setup.boundary_condition
            bcr = setup.boundary_condition
        try:
            self.boundary_condition = BC_DICT[bcl], BC_DICT[bcr]
        except KeyError:
            raise ValueError(f"bad boundary condition {bcl}/{bcr}")

        xp = get_array_module(mode)
        ng = 1  # number of guard zones
        nq = 1  # number of conserved quantities
        with open(__file__.replace(".py", ".c")) as f:
            code = f.read()
        lib = Library(code, mode=mode, debug=False)

        logger.info(f"initiate with time={time:0.4f}")
        logger.info(f"subdivide grid over {num_patches} patches")
        logger.info(f"mesh is {mesh}")
        logger.info(f"boundary condition is {bcl}/{bcr}")

        self.mesh = mesh
        self.setup = setup
        self.num_guard = ng
        self.num_cons = nq
        self.xp = xp
        self.patches = []

        if solution is None:
            primitive = initial_condition(setup, mesh, time)
        else:
            primitive = solution

        for (a, b) in subdivide(mesh.shape[0], num_patches):
            prim = xp.zeros([b - a + 2 * ng, nq])
            prim[ng:-ng] = primitive[a:b]
            self.patches.append(Patch(time, prim, mesh, (a, b), lib, xp))

        self.set_bc("conserved1")

    @property
    def solution(self):
        return self.primitive

    @property
    def primitive(self):
        nz = self.mesh.shape[0]
        ng = self.num_guard
        nq = self.num_cons
        np = len(self.patches)
        primitive = self.xp.zeros([nz, nq])
        for (a, b), patch in zip(subdivide(nz, np), self.patches):
            primitive[a:b] = patch.primitive[ng:-ng]
        return primitive

    @property
    def time(self):
        return self.patches[0].time

    @property
    def options(self):
        return dict()

    @property
    def physics(self):
        return dict()

    @property
    def maximum_cfl(self):
        return 0.6

    def maximum_wavespeed(self):
        return 1.0

    def advance(self, dt):
        self.new_iteration()
        self.advance_rk(0.0, dt)
        self.advance_rk(0.75, dt)
        self.advance_rk(0.333333333333333, dt)

    def advance_rk(self, rk_param, dt, dx):
        self.set_bc("conserved1")
        for patch in self.patches:
            patch.advance_rk(rk_param, dt, dx)

    def set_bc(self, array):
        ng = self.num_guard
        num_patches = len(self.patches)
        for i0 in range(num_patches):
            il = (i0 + num_patches - 1) % num_patches
            ir = (i0 + num_patches + 1) % num_patches
            pl = getattr(self.patches[il], array)
            p0 = getattr(self.patches[i0], array)
            pr = getattr(self.patches[ir], array)
            self.set_bc_patch(pl, p0, pr, i0)

    def set_bc_patch(self, al, a0, ar, patch_index):
        t = self.time
        nz = self.mesh.shape[0]
        ng = self.num_guard
        bcl, bcr = self.boundary_condition

        a0[:+ng] = al[-2 * ng : -ng]
        a0[-ng:] = ar[+ng : +2 * ng]

        def negative_vel(p):
            return [p[0], -p[1], p[2], p[3]]

        if patch_index == 0:
            if bcl == BC_OUTFLOW:
                a0[:+ng] = a0[+ng : +2 * ng]
            elif bcl == BC_INFLOW:
                for i in range(-ng, 0):
                    x = self.mesh.zone_center(t, i)
                    self.setup.primitive(t, x, a0[i + ng])
            elif bcl == BC_REFLECT:
                a0[0] = negative_vel(a0[3])
                a0[1] = negative_vel(a0[2])

        if patch_index == len(self.patches) - 1:
            if bcr == BC_OUTFLOW:
                a0[-ng:] = a0[-2 * ng : -ng]
            elif bcr == BC_INFLOW:
                for i in range(nz, nz + ng):
                    x = self.mesh.zone_center(t, i)
                    self.setup.primitive(t, x, a0[i + ng])
            elif bcr == BC_REFLECT:
                a0[-2] = negative_vel(a0[-3])
                a0[-1] = negative_vel(a0[-4])

    def new_iteration(self):
        for patch in self.patches:
            patch.new_iteration()
