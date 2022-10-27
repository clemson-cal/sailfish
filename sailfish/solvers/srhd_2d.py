"""
Two-dimensional relativistic hydro solver supporting homologous mesh motion.

The solver configuration is:

- Spherical polar coordinates in 2d
- Possible homologous radial expansion
- Four conserved quantities: D, Sr, Sq, tau
- Four primitive quantities: rho, ur, uq, p
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

BC_INTERNAL = 0  # internal BC (guard zones overlap a neighbor patch)
BC_PERIODIC = 1  # period BC (not handled in C)
BC_OUTFLOW = 2  # zero-gradient BC (not handled in C)
BC_INFLOW = 3  # user-supplied BC in guard zones (not handled in C)
BC_REFLECT = 4  # apply reflecting BC in guard zones (not handled in C)
BC_FIXED = 5  # don't evolve the first (or last) interior zones
BC_JET = 6  # use a jet inflow boundary condition, with supplied parameters


BC_DICT = {
    "periodic": BC_PERIODIC,
    "outflow": BC_OUTFLOW,
    "inflow": BC_INFLOW,
    "reflect": BC_REFLECT,
    "fixed": BC_FIXED,
    "jet": BC_JET,
}


def initial_condition(setup, mesh, i0, i1, j0, j1, time, xp):
    primitive = xp.zeros([i1 - i0, j1 - j0, NUM_CONS])

    r_list = [mesh.cell_coordinates(time, i, 0)[0] for i in range(i0, i1)]
    q_list = [mesh.cell_coordinates(time, 0, j)[1] for j in range(j0, j1)]

    for i, r in enumerate(r_list):
        for j, q in enumerate(q_list):
            setup.primitive(time, (r, q), primitive[i, j])

    return primitive


class Options(NamedTuple):
    compute_wavespeed: bool = False
    rk_order: int = 2
    plm_theta: float = 1.5
    mach_ceiling: float = 1e6


class Physics(NamedTuple):
    jet_mdot: float = 0.0
    jet_gamma_beta: float = 0.0
    jet_theta: float = 0.0
    jet_duration: float = 0.0


class Patch:
    """
    Buffers for the solution on a subset of the solution domain.

    This class also takes care of generating initial conditions if needed, and
    issuing calls to the solver kernel functions.
    """

    def __init__(
        self,
        setup,
        physics,
        time,
        conserved,
        mesh,
        index_range,
        num_first_order_zones,
        lib,
        xp,
        execution_context,
    ):
        ng = NUM_GUARD
        nq = NUM_CONS
        i0, i1 = index_range
        nj = mesh.shape[1]
        self.lib = lib
        self.xp = xp
        self.physics = physics
        self.index_range = index_range
        self.num_first_order_zones = num_first_order_zones
        self.shape = shape = (i1 - i0, mesh.shape[1])  # not including guard zones
        self.polar_extent = mesh.polar_extent
        self.time = self.time0 = time
        self.execution_context = execution_context

        try:
            adot = float(mesh.scale_factor_derivative)
            self.scale_factor_initial = 0.0
            self.scale_factor_derivative = adot
        except (TypeError, AttributeError):
            self.scale_factor_initial = 1.0
            self.scale_factor_derivative = 0.0

        with self.execution_context:
            faces = xp.array(mesh.faces(*index_range))
            conserved_with_guard = xp.zeros([shape[0] + 2 * ng, nj, nq])

            if conserved is None:
                primitive = initial_condition(setup, mesh, i0, i1, 0, nj, time, xp)
                conserved = xp.zeros_like(primitive)

                lib.srhd_2d_primitive_to_conserved[shape](
                    faces,
                    primitive,
                    conserved,
                    mesh.polar_extent,
                    mesh.scale_factor(time),
                )
                conserved_with_guard[ng:-ng] = conserved
            else:
                conserved_with_guard[ng:-ng] = xp.array(conserved)

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
                self.conserved2,
                self.primitive1,
                self.polar_extent,
                self.scale_factor,
            )
            self.conserved1, self.conserved2 = self.conserved2, self.conserved1

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
                self.physics.jet_mdot if self.index_range[0] == 0 else 0.0,
                self.physics.jet_gamma_beta,
                self.physics.jet_theta,
                self.physics.jet_duration,
                self.num_first_order_zones,
            )
        self.time = self.time0 * rk_param + (self.time + dt) * (1.0 - rk_param)
        self.conserved1, self.conserved2 = self.conserved2, self.conserved1

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

        self._physics = physics = Physics(**physics)
        self._options = options = Options(**options)

        xp = get_array_module(mode)
        lib = Library(
            code,
            mode=mode,
            debug=False,
            define_macros=dict(
                PLM_THETA=options.plm_theta,
                MACH_CEILING=options.mach_ceiling,
            ),
        )

        try:
            bcl, bcr = setup.boundary_condition
        except ValueError:
            bcl = setup.boundary_condition
            bcr = setup.boundary_condition
        try:
            self.boundary_condition = BC_DICT[bcl], BC_DICT[bcr]
        except KeyError:
            raise ValueError(f"bad boundary condition {bcl}/{bcr}")

        logger.info(f"initiate with time={time:0.4f}")
        logger.info(f"subdivide grid over {num_patches} patches")
        logger.info(f"mesh is {mesh}")

        if options.rk_order not in (1, 2, 3):
            raise ValueError("solver only supports rk_order in 1, 2, 3")

        patches = list()

        for n, (a, b) in enumerate(subdivide(mesh.shape[0], num_patches)):
            if n == 0 and bcl == "jet":
                # introduction of some extra diffusion near the jet inlet is
                # effective at preventing crashes
                num_first_order_zones = 4
            else:
                num_first_order_zones = 0
            patch = Patch(
                setup,
                physics,
                time,
                solution[a:b] if solution is not None else None,
                mesh,
                (a, b),
                num_first_order_zones,
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
        return self._options._asdict()

    @property
    def physics(self):
        return self._physics._asdict()

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
        nj = self.mesh.shape[1]
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
                        for j in range(nj):
                            x = self.mesh.cell_coordinates(t, i, j)
                            self.setup.primitive(t, x, pc[i + ng, j])
                elif bcl == BC_REFLECT:
                    for j in range(nj):
                        pc[0, j] = negative_vel(pc[3, j])
                        pc[1, j] = negative_vel(pc[2, j])

            if patch_index == len(self.patches) - 1:
                if bcr == BC_OUTFLOW:
                    pc[-ng:] = pc[-2 * ng : -ng]
                elif bcr == BC_INFLOW:
                    i0 = self.patches[patch_index].index_range[0]
                    for i in range(ni, ni + ng):
                        for j in range(nj):
                            x = self.mesh.zone_center(t, i, j)
                            self.setup.primitive(t, x, pc[i - i0 + ng, j])
                elif bcr == BC_REFLECT:
                    for j in range(nj):
                        pc[-2, j] = negative_vel(pc[-3, j])
                        pc[-1, j] = negative_vel(pc[-4, j])

    def new_iteration(self):
        for patch in self.patches:
            patch.new_iteration()
