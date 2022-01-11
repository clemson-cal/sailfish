"""
One-dimensional relativistic hydro solver supporting homologous mesh motion.
"""

from logging import getLogger
from sailfish.kernel.library import Library
from sailfish.kernel.system import get_array_module, execution_context, num_devices
from sailfish.subdivide import subdivide, concat_on_host
from sailfish.mesh import PlanarCartesianMesh, LogSphericalMesh
from sailfish.solver import SolverBase


logger = getLogger(__name__)

NUM_GUARD = 2
NUM_CONS = 4


def initial_condition(setup, mesh, i0, i1, j0, j1, time):
    import numpy as np

    primitive = np.zeros([i1 - i0, j1 - j0, NUM_CONS])

    for i in range(i0, i1):
        for j in range(j0, j1):
            r, q = mesh.cell_coordinates(time, i, j)
            setup.primitive(time, (r, q), primitive[i - i0, j - j0])

    return primitive


class Patch:
    """
    Holds the array buffer state for the solution on a subset of the
    solution domain.
    """

    def __init__(
        self,
        setup,
        time,
        conserved,
        mesh,
        index_range,
        lib,
        xp,
        execution_context,
    ):
        i0, i1 = index_range
        nj = mesh.shape[1]
        ng = NUM_GUARD
        nq = NUM_CONS
        self.lib = lib
        self.xp = xp
        self.shape = shape = (i1 - i0, mesh.shape[1])  # not including guard zones
        self.polar_extent = mesh.polar_extent
        self.execution_context = execution_context
        self.time = self.time0 = time

        try:
            adot = float(mesh.scale_factor_derivative)
            self.scale_factor_initial = 0.0
            self.scale_factor_derivative = adot
        except (TypeError, AttributeError):
            self.scale_factor_initial = 1.0
            self.scale_factor_derivative = 0.0

        with self.execution_context:
            faces = xp.array(mesh.faces(*index_range))

            if conserved is None:
                primitive = xp.array(
                    initial_condition(setup, mesh, i0, i1, 0, nj, time)
                )
                conserved = xp.zeros_like(primitive)

                lib.srhd_2d_primitive_to_conserved[shape](
                    faces,
                    primitive,
                    conserved,
                    mesh.polar_extent,
                    mesh.scale_factor(time),
                )
                conserved_with_guard = xp.zeros([shape[0] + 2 * ng, nj, nq])
                conserved_with_guard[ng:-ng] = conserved

            self.faces = faces
            self.wavespeeds = xp.zeros(shape)
            self.primitive1 = xp.zeros_like(conserved_with_guard)
            self.conserved0 = conserved_with_guard.copy()
            self.conserved1 = conserved_with_guard.copy()
            self.conserved2 = conserved_with_guard.copy()

    def recompute_primitive(self):
        with self.execution_context:
            self.lib.srhd_2d_conserved_to_primitive[self.shape](
                self.faces,
                self.conserved1,
                self.primitive1,
                self.polar_extent,
                self.scale_factor,
            )

    def advance_rk(self, rk_param, dt):
        with self.execution_context:
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

    @property
    def maximum_wavespeed(self):
        self.recompute_primitive()
        with self.execution_context:
            self.lib.srhd_2d_max_wavespeeds[self.shape](
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
        with open(__file__.replace(".py", ".c")) as f:
            code = f.read()

        xp = get_array_module(mode)
        lib = Library(code, mode=mode, debug=True)

        logger.info(f"initiate with time={time:0.4f}")
        logger.info(f"subdivide grid over {num_patches} patches")
        logger.info(f"mesh is {mesh}")

        if setup.boundary_condition != "outflow":
            raise ValueError(f"srhd_2d solver only supports outflow radial boundaries")

        patches = list()

        for n, (a, b) in enumerate(subdivide(mesh.shape[0], num_patches)):
            patch = Patch(
                setup,
                time,
                solution[a:b] if solution else None,
                mesh,
                (a, b),
                lib,
                xp,
                execution_context(mode, device_id=n % num_devices(mode)),
            )
            patches.append(patch)

        self.mesh = mesh
        self.setup = setup
        self.num_guard = NUM_GUARD
        self.num_cons = NUM_CONS
        self.xp = xp
        self.patches = patches

    @property
    def solution(self):
        return concat_on_host([p.conserved for p in self.patches], (self.num_guard, 0))

    @property
    def primitive(self):
        return concat_on_host([p.primitive for p in self.patches], (self.num_guard, 0))

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
        return 0.4

    @property
    def maximum_cfl(self):
        """
        When the mesh is expanding, it's possible to use a CFL number
        significantly larger than the theoretical max. I haven't seen
        how this makes sense, but it works in practice.
        """
        return 16.0

    def maximum_wavespeed(self):
        return 1.0
        # return max(patch.maximum_wavespeed for patch in self.patches)

    def advance(self, dt):
        bs_rk1 = [0 / 1]
        bs_rk2 = [0 / 1, 1 / 2]
        bs_rk3 = [0 / 1, 3 / 4, 1 / 3]

        self.new_iteration()

        for b in bs_rk2:
            self.advance_rk(b, dt)

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

    def set_bc_patch(self, pl, pc, pr, patch_index):
        ni, nj = self.mesh.shape
        ng = self.num_guard

        with self.patches[patch_index].execution_context:
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
