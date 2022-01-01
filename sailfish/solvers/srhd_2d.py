"""
One-dimensional relativistic hydro solver supporting homologous mesh motion.
"""

from logging import getLogger
from contextlib import nullcontext
from sailfish.kernel.library import Library
from sailfish.kernel.system import get_array_module
from sailfish.subdivide import subdivide
from sailfish.mesh import PlanarCartesianMesh, LogSphericalMesh
from sailfish.solver import SolverBase


logger = getLogger(__name__)


def initial_condition(setup, mesh, time):
    import numpy as np

    primitive = np.zeros(mesh.shape + (4,))

    for i in range(mesh.shape[0]):
        for j in range(mesh.shape[1]):
            r, q = mesh.cell_coordinates(time, i, j)
            setup.primitive(time, (r, q), primitive[i, j])

    return primitive


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
        self.shape = (i1 - i0, mesh.shape[1])  # not including guard zones
        self.faces = self.xp.array(mesh.faces(*index_range))
        self.polar_extent = mesh.polar_extent

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
            self.lib.srhd_2d_primitive_to_conserved[self.shape](
                self.faces,
                primitive,
                conserved,
                self.polar_extent,
                self.scale_factor,
            )
            return conserved

    def recompute_primitive(self):
        with self.execution_context():
            self.lib.srhd_2d_conserved_to_primitive[self.shape](
                self.faces,
                self.conserved1,
                self.primitive1,
                self.polar_extent,
                self.scale_factor,
            )

    def advance_rk(self, rk_param, dt):
        with self.execution_context():
            self.lib.srhd_2d_advance_rk[self.shape](
                self.faces,
                self.conserved0,
                self.primitive1,
                self.conserved1,
                self.conserved2,
                self.polar_extent,
                self.scale_factor_initial,
                self.scale_factor_derivative,
                self.time,
                rk_param,
                dt,
            )
        self.time = self.time0 * rk_param + (self.time + dt) * (1.0 - rk_param)
        self.conserved1, self.conserved2 = self.conserved2, self.conserved1
        self.recompute_primitive()

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
    Adapter class to drive the srhd_1d C extension module.
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
        xp = get_array_module(mode)
        ng = 2  # number of guard zones
        nq = 4  # number of conserved quantities
        with open(__file__.replace(".py", ".c")) as f:
            code = f.read()
        lib = Library(code, mode=mode, debug=True)

        logger.info(f"initiate with time={time:0.4f}")
        logger.info(f"subdivide grid over {num_patches} patches")
        logger.info(f"mesh is {mesh}")

        if setup.boundary_condition != "outflow":
            raise ValueError(f"srhd_2d solver only supports outflow radial boundaries")

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
            prim = xp.zeros([b - a + 2 * ng, mesh.shape[1], nq])
            prim[ng:-ng] = primitive[a:b]
            self.patches.append(Patch(time, prim, mesh, (a, b), lib, xp))

        self.set_bc("primitive1")

    @property
    def solution(self):
        return self.primitive

    @property
    def primitive(self):
        ni, nj = self.mesh.shape
        ng = self.num_guard
        nq = self.num_cons
        np = len(self.patches)
        primitive = self.xp.zeros([ni, nj, nq])
        for (a, b), patch in zip(subdivide(ni, np), self.patches):
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
        return 0.3

    def maximum_wavespeed(self):
        return 1.0

    def advance(self, dt):
        self.new_iteration()
        self.advance_rk(0.0, dt)
        self.advance_rk(0.5, dt)

    def advance_rk(self, rk_param, dt):
        self.set_bc("primitive1")
        for patch in self.patches:
            patch.advance_rk(rk_param, dt)

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

    def set_bc_patch(self, pl, pc, pr, patch_index):
        ni, nj = self.mesh.shape
        ng = self.num_guard

        # 1. write to the guard zones of pc, the internal BC
        pc[:+ng] = pl[-2 * ng : -ng]
        pc[-ng:] = pr[+ng : +2 * ng]

        # 2. Set outflow BC on the inner and outer patch edges
        if patch_index == 0:
            for i in range(ng):
                pc[i] = pc[ng]
        if patch_index == len(self.patches) - 1:
            for i in range(pc.shape[0] - ng, pc.shape[0]):
                pc[i] = pc[-ng - 1]

    def new_iteration(self):
        for patch in self.patches:
            patch.new_iteration()
