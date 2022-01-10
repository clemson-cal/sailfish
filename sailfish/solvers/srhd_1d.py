"""
One-dimensional relativistic hydro solver supporting homologous mesh motion.
"""

from logging import getLogger
from sailfish.kernel.library import Library
from sailfish.kernel.system import get_array_module, execution_context, num_devices
from sailfish.subdivide import subdivide
from sailfish.mesh import PlanarCartesianMesh, LogSphericalMesh
from sailfish.solver import SolverBase

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

    faces = np.array(mesh.faces())
    zones = 0.5 * (faces[:-1] + faces[1:])
    primitive = np.zeros([len(zones), 4])

    for x, p in zip(zones, primitive):
        setup.primitive(time, x, p)

    return primitive


class Patch:
    """
    Holds the array buffer state for the solution on a subset of the
    solution domain.
    """

    def __init__(
        self,
        time,
        conserved,
        mesh,
        index_range,
        lib,
        xp,
        execution_context,
    ):
        i0, i1 = index_range
        self.lib = lib
        self.xp = xp
        self.num_zones = num_zones = index_range[1] - index_range[0]
        self.faces = xp.array(mesh.faces(*index_range))
        self.coordinates = COORDINATES_DICT[type(mesh)]
        self.execution_context = execution_context

        try:
            adot = float(mesh.scale_factor_derivative)
            self.scale_factor_initial = 0.0
            self.scale_factor_derivative = adot
        except (TypeError, AttributeError):
            self.scale_factor_initial = 1.0
            self.scale_factor_derivative = 0.0

        self.time = self.time0 = time

        with self.execution_context:
            self.wavespeeds = xp.zeros(num_zones)
            self.primitive1 = xp.zeros_like(conserved)
            self.conserved0 = conserved.copy()
            self.conserved1 = conserved.copy()
            self.conserved2 = conserved.copy()

    def recompute_primitive(self):
        with self.execution_context:
            self.lib.srhd_1d_conserved_to_primitive[self.num_zones](
                self.faces,
                self.conserved1,
                self.primitive1,
                self.scale_factor,
                self.coordinates,
            )

    def advance_rk(self, rk_param, dt):
        with self.execution_context:
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
        self.time = self.time0 * rk_param + (self.time + dt) * (1.0 - rk_param)
        self.conserved1, self.conserved2 = self.conserved2, self.conserved1

    @property
    def maximum_wavespeed(self):
        self.recompute_primitive()
        with self.execution_context:
            self.lib.srhd_1d_max_wavespeeds[self.num_zones](
                self.faces,
                self.primitive1,
                self.wavespeeds,
                self.scale_factor_derivative,
            )
            return float(self.wavespeeds.max())

    @property
    def scale_factor(self):
        return self.scale_factor_initial + self.scale_factor_derivative * self.time

    def new_iteration(self):
        self.time0 = self.time
        self.conserved0[...] = self.conserved1[...]

    @property
    def conserved(self):
        return self.conserved1

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
            primitive = xp.array(initial_condition(setup, mesh, time))
            conserved = xp.zeros_like(primitive)
            coordinates = COORDINATES_DICT[type(mesh)]

            try:
                scale_factor = mesh.scale_factor(time)
            except AttributeError:
                scale_factor = 1.0

            lib.srhd_1d_primitive_to_conserved[mesh.shape](
                xp.array(mesh.faces()),
                primitive,
                conserved,
                scale_factor,
                coordinates,
            )
        else:
            conserved = solution

        for n, (a, b) in enumerate(subdivide(mesh.shape[0], num_patches)):
            cons = xp.zeros([b - a + 2 * ng, nq])
            cons[ng:-ng] = conserved[a:b]
            patch = Patch(
                time,
                cons,
                mesh,
                (a, b),
                lib,
                xp,
                execution_context(mode, device_id=n % num_devices(mode)),
            )
            self.patches.append(patch)

    @property
    def solution(self):
        return self.reconstruct("conserved")

    @property
    def primitive(self):
        return self.reconstruct("primitive")

    def reconstruct(self, array):
        import numpy

        nz = self.mesh.shape[0]
        ng = self.num_guard
        nq = self.num_cons
        np = len(self.patches)
        result = numpy.zeros([nz, nq])
        for (a, b), patch in zip(subdivide(nz, np), self.patches):
            result[a:b] = getattr(patch, array)[ng:-ng]
        return result

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
    def recommended_cfl(self):
        return 0.6

    @property
    def maximum_cfl(self):
        return 16.0

    def maximum_wavespeed(self):
        return 1.0
        # max(patch.maximum_wavespeed for patch in self.patches)

    def advance(self, dt):
        self.new_iteration()
        self.advance_rk(0.0, dt)
        self.advance_rk(0.5, dt)

    def advance_rk(self, rk_param, dt):
        for patch in self.patches:
            patch.recompute_primitive()

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
