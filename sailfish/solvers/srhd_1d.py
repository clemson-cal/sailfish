from logging import getLogger
from contextlib import nullcontext
from sailfish.kernel.library import Library
from sailfish.kernel.system import get_array_module
from sailfish.subdivide import subdivide
from sailfish.mesh import PlanarCartesianMesh, LogSphericalMesh
from sailfish.solver import SolverBase

logger = getLogger(__name__)
BC_PERIODIC = 0
BC_OUTFLOW = 1
BC_INFLOW = 2
BC_DICT = {
    "periodic": BC_PERIODIC,
    "outflow": BC_OUTFLOW,
    "inflow": BC_INFLOW,
}
COORDINATES_DICT = {
    PlanarCartesianMesh: 0,
    LogSphericalMesh: 1,
}


def initial_condition(setup, mesh):
    import numpy as np

    faces = np.array(mesh.faces(0, mesh.shape[0]))
    zones = 0.5 * (faces[:-1] + faces[1:])
    primitive = np.zeros([len(zones), 4])

    for x, p in zip(zones, primitive):
        setup.primitive(0.0, x, p)

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
                self.scale_factor(),
                self.coordinates,
            )
            return conserved

    def recompute_primitive(self):
        with self.execution_context():
            self.lib.srhd_1d_conserved_to_primitive[self.num_zones](
                self.faces,
                self.conserved1,
                self.primitive1,
                self.scale_factor(),
                self.coordinates,
            )

    def advance_rk(self, rk_param, dt):
        with self.execution_context():
            self.lib.srhd_1d_advance_rk[self.num_zones](
                self.faces,
                self.conserved0,
                self.primitive1,
                self.conserved1,
                self.conserved2,
                self.scale_factor_initial,
                self.scale_factor_derivative,
                self.time,
                rk_param,
                dt,
                self.coordinates,
            )
        self.time = self.time0 * rk_param + (self.time0 + dt) * (1.0 - rk_param)
        self.conserved1, self.conserved2 = self.conserved2, self.conserved1
        self.recompute_primitive()

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
    ):
        try:
            try:
                bcl, bcr = setup.boundary_condition
            except ValueError:
                bcl = setup.boundary_condition
                bcr = setup.boundary_condition
            self.boundary_condition = BC_DICT[bcl], BC_DICT[bcr]
        except KeyError:
            raise ValueError(f"bad boundary condition {bcl}/{bcr}")

        xp = get_array_module(mode)
        ng = 2  # number of guard zones
        nq = 4  # number of conserved quantities
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
            primitive = initial_condition(setup, mesh)
        else:
            primitive = solution

        for (a, b) in subdivide(mesh.shape[0], num_patches):
            prim = xp.zeros([b - a + 2 * ng, nq])
            prim[ng:-ng] = primitive[a:b]
            self.patches.append(Patch(time, prim, mesh, (a, b), lib, xp))

        self.set_bc("primitive1")

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
    def maximum_cfl(self):
        return 0.05

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

    def set_bc_patch(self, al, a0, ar, patch_index):
        ng = self.num_guard
        nz = self.mesh.shape[0]
        bcl, bcr = self.boundary_condition
        t = self.patches[patch_index].time

        a0[:+ng] = al[-2 * ng : -ng]
        a0[-ng:] = ar[+ng : +2 * ng]

        if patch_index == 0:
            if bcl == BC_OUTFLOW:
                a0[:+ng] = a0[+ng : +2 * ng]
            elif bcl == BC_INFLOW:
                for i in range(-ng, 0):
                    x = self.mesh.zone_center(i)
                    self.setup.primitive(t, x, a0[i + ng])

        if patch_index == len(self.patches) - 1:
            if bcr == BC_OUTFLOW:
                a0[-ng:] = a0[-2 * ng : -ng]
            elif bcr == BC_INFLOW:
                for i in range(nz, nz + ng):
                    x = self.mesh.zone_center(i)
                    self.setup.primitive(t, x, a0[i + ng])

    def new_iteration(self):
        for patch in self.patches:
            patch.new_iteration()
