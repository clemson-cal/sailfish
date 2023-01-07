"""
One-dimensional relativistic hydro solver supporting homologous mesh motion.

The solver configuration is:

- Planar cartesian, or spherical coordinate system in 1d
- Possible homologous radial expansion in spherical coordinates mode
- Four conserved quantities: D, Sr, tau, s (scalar lab-frame mass density)
- Four primitive quantities: rho, ur, p, x (scalar concentration)
- Gamma-law index of 4/3

The Python code assumes RK2 time stepping, although coefficients are written
below for RK1 and low-storage RK3 as well. The C code hard-codes a PLM theta
value of 2.0.
"""

from logging import getLogger
from typing import NamedTuple
from sailfish.kernel.library import Library
from sailfish.kernel.system import get_array_module, execution_context, num_devices
from sailfish.subdivide import subdivide, concat_on_host, lazy_reduce
from sailfish.mesh import PlanarCartesianMesh, LogSphericalMesh
from sailfish.solver_base import SolverBase

logger = getLogger(__name__)

NUM_GUARD = 2
NUM_CONS = 4

BC_PERIODIC = 0
BC_OUTFLOW = 1
BC_INFLOW = 2
BC_REFLECT = 3
BC_FIXED = 4

BC_DICT = {
    "periodic": BC_PERIODIC,
    "outflow": BC_OUTFLOW,
    "inflow": BC_INFLOW,
    "reflect": BC_REFLECT,
    "fixed": BC_FIXED,
}
COORDINATES_DICT = {
    PlanarCartesianMesh: 0,
    LogSphericalMesh: 1,
}


def initial_condition(setup, mesh, i0, i1, time, xp):
    primitive = xp.zeros([i1 - i0, NUM_CONS])

    for i in range(i0, i1):
        r = mesh.zone_center(time, i)
        setup.primitive(time, r, primitive[i - i0])
    return primitive


class Options(NamedTuple):
    compute_wavespeed: bool = False
    rk_order: int = 2


class Physics(NamedTuple):
    pass


class Patch:
    """
    Buffers for the solution on a subset of the solution domain.

    This class also takes care of generating initial conditions if needed, and
    issuing calls to the solver kernel functions.
    """

    def __init__(
        self,
        setup,
        time,
        conserved,
        mesh,
        index_range,
        fix_i0,
        fix_i1,
        lib,
        xp,
        execution_context,
    ):
        import numpy as np

        ng = NUM_GUARD
        nq = NUM_CONS
        i0, i1 = index_range
        self.lib = lib
        self.xp = xp
        self.index_range = index_range
        self.fix_i0 = fix_i0
        self.fix_i1 = fix_i1
        self.num_zones = num_zones = index_range[1] - index_range[0]
        self.coordinates = coordinates = COORDINATES_DICT[type(mesh)]
        self.time = self.time0 = time
        self.execution_context = execution_context

        try:
            adot = float(mesh.scale_factor_derivative)
            self.scale_factor_initial = 0.0
            self.scale_factor_derivative = adot
        except (TypeError, AttributeError):
            self.scale_factor_initial = 1.0
            self.scale_factor_derivative = 0.0

        with execution_context:
            faces = xp.array(mesh.faces(*index_range))
            conserved_with_guard = xp.zeros([num_zones + 2 * ng, nq])

            if conserved is None:
                primitive = initial_condition(setup, mesh, i0, i1, time, xp)
                conserved = xp.zeros_like(primitive)

                lib.srhd_1d_primitive_to_conserved[num_zones](
                    faces,
                    primitive,
                    conserved,
                    self.scale_factor,
                    coordinates,
                )
                conserved_with_guard[ng:-ng] = conserved
            else:
                conserved_with_guard[ng:-ng] = xp.array(conserved)

            self.faces = faces
            self.wavespeeds = xp.zeros(num_zones)
            self.primitive1 = xp.zeros_like(conserved_with_guard)
            self.conserved0 = conserved_with_guard.copy()
            self.conserved1 = conserved_with_guard.copy()
            self.conserved2 = conserved_with_guard.copy()

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
                int(self.fix_i0),
                int(self.fix_i1),
                self.coordinates,
            )
        self.time = self.time0 * rk_param + (self.time + dt) * (1.0 - rk_param)
        self.conserved1, self.conserved2 = self.conserved2, self.conserved1

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
        with open(__file__.replace(".py", ".c")) as f:
            code = f.read()

        xp = get_array_module(mode)
        lib = Library(code, mode=mode, debug=False)

        self._physics = physics = Physics(**physics)
        self._options = options = Options(**options)

        try:
            bcl, bcr = setup.boundary_condition
        except ValueError:
            bcl = setup.boundary_condition
            bcr = setup.boundary_condition
        try:
            self.boundary_condition = BC_DICT[bcl], BC_DICT[bcr]
        except KeyError:
            raise ValueError(f"bad boundary condition {bcl}/{bcr}")

        if options.rk_order not in (1, 2, 3):
            raise ValueError("solver only supports rk_order in 1, 2, 3")

        logger.info(f"initiate with time={time:0.4f}")
        logger.info(f"subdivide grid over {num_patches} patches")
        logger.info(f"mesh is {mesh}")
        logger.info(f"boundary condition is {bcl}/{bcr}")
        patches = list()

        for n, (a, b) in enumerate(subdivide(mesh.shape[0], num_patches)):
            fix_i0 = self.boundary_condition[0] == BC_FIXED and n == 0
            fix_i1 = self.boundary_condition[1] == BC_FIXED and n == num_patches - 1
            patch = Patch(
                setup,
                time,
                solution[a:b] if solution is not None else None,
                mesh,
                (a, b),
                fix_i0,
                fix_i1,
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
        return concat_on_host([p.conserved for p in self.patches], self.num_guard)

    @property
    def primitive(self):
        return concat_on_host([p.primitive for p in self.patches], self.num_guard)

    @property
    def time(self):
        return self.patches[0].time

    @property
    def options(self):
        return self._options._asdict()

    @property
    def physics(self):
        return self._physics._asdict()

    @property
    def recommended_cfl(self):
        return 0.6

    @property
    def maximum_cfl(self):
        return 16.0

    def maximum_wavespeed(self):
        if self._options.compute_wavespeed:
            return lazy_reduce(
                max,
                float,
                (patch.maximum_wavespeed for patch in self.patches),
                (patch.execution_context for patch in self.patches),
            )
        else:
            return 1.0

    def advance(self, dt):
        bs_rk1 = [0 / 1]
        bs_rk2 = [0 / 1, 1 / 2]
        bs_rk3 = [0 / 1, 3 / 4, 1 / 3]
        bs = (bs_rk1, bs_rk2, bs_rk3)[self._options.rk_order - 1]

        self.new_iteration()

        for b in bs:
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
        for ic in range(num_patches):
            il = (ic + num_patches - 1) % num_patches
            ir = (ic + num_patches + 1) % num_patches
            pl = getattr(self.patches[il], array)
            pc = getattr(self.patches[ic], array)
            pr = getattr(self.patches[ir], array)
            self.set_bc_patch(pl, pc, pr, ic)

    def set_bc_patch(self, pl, pc, pr, patch_index):
        t = self.time
        ni = self.mesh.shape[0]
        ng = self.num_guard
        bcl, bcr = self.boundary_condition

        with self.patches[patch_index].execution_context:
            pc[:+ng] = pl[-2 * ng : -ng]
            pc[-ng:] = pr[+ng : +2 * ng]

            def negative_vel(p):
                return self.xp.asarray([p[0], -p[1], p[2], p[3]])

            if patch_index == 0:
                if bcl == BC_OUTFLOW:
                    pc[:+ng] = pc[+ng : +2 * ng]
                elif bcl == BC_INFLOW:
                    for i in range(-ng, 0):
                        x = self.mesh.zone_center(t, i)
                        self.setup.primitive(t, x, pc[i + ng])
                elif bcl == BC_REFLECT:
                    pc[0] = negative_vel(pc[3])
                    pc[1] = negative_vel(pc[2])

            if patch_index == len(self.patches) - 1:
                if bcr == BC_OUTFLOW:
                    pc[-ng:] = pc[-2 * ng : -ng]
                elif bcr == BC_INFLOW:
                    i0 = self.patches[patch_index].index_range[0]
                    for i in range(ni, ni + ng):
                        x = self.mesh.zone_center(t, i)
                        self.setup.primitive(t, x, pc[i - i0 + ng])
                elif bcr == BC_REFLECT:
                    pc[-2] = negative_vel(pc[-3])
                    pc[-1] = negative_vel(pc[-4])

    def new_iteration(self):
        for patch in self.patches:
            patch.new_iteration()
