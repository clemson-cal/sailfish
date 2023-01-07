"""
Isothermal solver for the binary accretion problem in 2D planar coordinates.
"""

from logging import getLogger
from typing import NamedTuple
from sailfish.kernel.library import Library
from sailfish.kernel.system import get_array_module, execution_context, num_devices
from sailfish.mesh import PlanarCartesian2DMesh
from sailfish.physics.circumbinary import Physics, EquationOfState, ViscosityModel
from sailfish.solver_base import SolverBase
from sailfish.subdivide import subdivide, concat_on_host, lazy_reduce


logger = getLogger(__name__)

NCONS = 3
NPOLY = 6
ORDER = 3
GUARD = 1


class Options(NamedTuple):
    """
    Contains parameters which are solver specific options.
    """

    velocity_ceiling: float = 1e12
    rk_order: int = 2
    limit_slopes: bool = True


def primitive_to_conserved(prim, cons):
    sigma, vx, vy = prim
    cons[0] = sigma
    cons[1] = sigma * vx
    cons[2] = sigma * vy


def initial_condition(setup, mesh, time):
    """
    Generate a 2D array of weights from a mesh and a setup.
    """
    import numpy as np

    g = (-0.774596669241483, +0.000000000000000, +0.774596669241483)
    w = (+0.555555555555556, +0.888888888888889, +0.555555555555556)
    p = (
        (+1.000000000000000, +1.000000000000000, +1.000000000000000),
        (-1.341640786499873, +0.000000000000000, +1.341640786499873),
        (+0.894427190999914, -1.118033988749900, +0.894427190999914),
    )

    ni, nj = mesh.shape
    dx, dy = mesh.dx, mesh.dy
    prim_node = np.zeros(NCONS)
    cons_node = np.zeros(NCONS)
    weights = np.zeros([ni, nj, NCONS, ORDER, ORDER])

    for i in range(ni):
        for j in range(nj):
            for i_quad in range(ORDER):
                for j_quad in range(ORDER):
                    xc, yc = mesh.cell_coordinates(i, j)
                    x = xc + 0.5 * dx * g[i_quad]
                    y = yc + 0.5 * dy * g[j_quad]
                    setup.primitive(time, (x, y), prim_node)
                    primitive_to_conserved(prim_node, cons_node)
                    for q in range(NCONS):
                        for m in range(ORDER):
                            for n in range(ORDER):
                                weights[i, j, q, m, n] += (
                                    0.25
                                    * cons_node[q]
                                    * p[m][i_quad]
                                    * p[n][j_quad]
                                    * w[i_quad]
                                    * w[j_quad]
                                )
    return weights


class Patch:
    """
    Holds the array buffer state for the solution on a subset of the
    solution domain.
    """

    def __init__(
        self,
        time,
        weights,
        mesh,
        index_range,
        physics,
        options,
        buffer_outer_radius,
        buffer_surface_density,
        lib,
        xp,
        execution_context,
    ):
        i0, i1 = index_range
        ni, nj = i1 - i0, mesh.shape[1]
        self.lib = lib
        self.mesh = mesh
        self.xp = xp
        self.execution_context = execution_context
        self.time = self.time0 = time
        self.shape = (i1 - i0, nj)  # not including guard zones
        self.physics = physics
        self.options = options
        self.xl, self.yl = mesh.vertex_coordinates(i0, 0)
        self.xr, self.yr = mesh.vertex_coordinates(i1, nj)
        self.buffer_outer_radius = buffer_outer_radius
        self.buffer_surface_density = buffer_surface_density

        with self.execution_context:
            self.wavespeeds = xp.zeros(weights.shape[:2])
            self.weights0 = xp.zeros(weights.shape)  # weights at the timestep start
            self.weights1 = xp.array(weights)  # weights to be read from
            self.weights2 = xp.array(weights)  # weights to be written to

    def point_mass_source_term(self, which_mass):
        """
        Return an array of the rates of conserved quantities, resulting from the application of
        gravitational and/or accretion source terms due to point masses.
        """
        if which_mass not in (1, 2):
            raise ValueError("the mass must be either 1 or 2")

        ng = GUARD  # number of guard cells
        ni, nj = self.shape
        m1, m2 = self.physics.point_masses(self.time)

        with self.execution_context:
            cons_rate = self.xp.zeros([ni + 2 * ng, nj + 2 * ng, NCONS])

            self.lib.cbdiso_2d_point_mass_source_term[self.shape](
                self.xl,
                self.xr,
                self.yl,
                self.yr,
                m1.position_x,
                m1.position_y,
                m1.velocity_x,
                m1.velocity_y,
                m1.mass,
                m1.softening_length,
                m1.sink_rate,
                m1.sink_radius,
                m1.sink_model.value,
                m2.position_x,
                m2.position_y,
                m2.velocity_x,
                m2.velocity_y,
                m2.mass,
                m2.softening_length,
                m2.sink_rate,
                m2.sink_radius,
                m2.sink_model.value,
                self.options.velocity_ceiling,
                which_mass,
                self.weights1,
                cons_rate,
            )
        return cons_rate[ng:-ng, ng:-ng]

    def maximum_wavespeed(self):
        """
        Returns the maximum wavespeed over a given patch.
        """
        m1, m2 = self.physics.point_masses(self.time)
        with self.execution_context:
            self.lib.cbdisodg_2d_wavespeed[self.shape](
                self.xl,
                self.xr,
                self.yl,
                self.yr,
                self.physics.sound_speed**2,
                self.physics.mach_number**2,
                self.physics.eos_type.value,
                m1.position_x,
                m1.position_y,
                m1.velocity_x,
                m1.velocity_y,
                m1.mass,
                m1.softening_length,
                m1.sink_rate,
                m1.sink_radius,
                m1.sink_model.value,
                m2.position_x,
                m2.position_y,
                m2.velocity_x,
                m2.velocity_y,
                m2.mass,
                m2.softening_length,
                m2.sink_rate,
                m2.sink_radius,
                m2.sink_model.value,
                self.options.velocity_ceiling,
                self.weights1,
                self.wavespeeds,
            )
            return self.wavespeeds.max()

    def copy_weights1_to_weights0(self):
        with self.execution_context:
            self.weights0[...] = self.weights1[...]

    def slope_limit(self):
        """
        Limit slopes using minmodTVB
        """
        m1, m2 = self.physics.point_masses(self.time)

        with self.execution_context:
            self.lib.cbdisodg_2d_slope_limit[self.shape](
                self.xl,
                self.xr,
                self.yl,
                self.yr,
                m1.position_x,
                m1.position_y,
                m2.position_x,
                m2.position_y,
                self.weights1,
                self.weights2,
            )
        self.weights1, self.weights2 = self.weights2, self.weights1

    def advance_rk(self, rk_param, dt):
        """
        Pass required parameters for time evolution of the setup.

        This function calls the C-module function responsible for performing time evolution using a
        RK algorithm to update the parameters of the setup.
        """
        m1, m2 = self.physics.point_masses(self.time)
        buffer_central_mass = m1.mass + m2.mass
        buffer_surface_density = self.buffer_surface_density

        with self.execution_context:
            self.lib.cbdisodg_2d_advance_rk[self.shape](
                self.xl,
                self.xr,
                self.yl,
                self.yr,
                self.weights0,
                self.weights1,
                self.weights2,
                buffer_surface_density,
                buffer_central_mass,
                self.physics.buffer_driving_rate,
                self.buffer_outer_radius,
                self.physics.buffer_onset_width,
                int(self.physics.buffer_is_enabled),
                m1.position_x,
                m1.position_y,
                m1.velocity_x,
                m1.velocity_y,
                m1.mass,
                m1.softening_length,
                m1.sink_rate,
                m1.sink_radius,
                m1.sink_model.value,
                m2.position_x,
                m2.position_y,
                m2.velocity_x,
                m2.velocity_y,
                m2.mass,
                m2.softening_length,
                m2.sink_rate,
                m2.sink_radius,
                m2.sink_model.value,
                self.physics.sound_speed**2,
                self.physics.mach_number**2,
                self.physics.eos_type.value,
                self.physics.viscosity_coefficient,
                rk_param,
                dt,
                self.options.velocity_ceiling,
            )
        self.time = self.time0 * rk_param + (self.time + dt) * (1.0 - rk_param)
        self.weights1, self.weights2 = self.weights2, self.weights1

    def new_iteration(self):
        self.time0 = self.time
        self.copy_weights1_to_weights0()

    @property
    def primitive(self):
        with self.execution_context:
            u0 = self.weights1[:, :, :, 0, 0]
            p0 = self.xp.zeros_like(u0)
            p0[..., 0] = u0[..., 0]
            p0[..., 1] = u0[..., 1] / u0[..., 0]
            p0[..., 2] = u0[..., 2] / u0[..., 0]
            return p0


class Solver(SolverBase):
    """
    Adapter class to drive the cbdisodg_2d C extension module.
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
        import numpy

        self._physics = physics = Physics(**physics)
        self._options = options = Options(**options)

        if type(mesh) is not PlanarCartesian2DMesh:
            raise ValueError("solver only supports 2D Cartesian mesh")

        if setup.boundary_condition != "outflow":
            raise ValueError("solver only supports outflow boundary condition")

        if physics.viscosity_model not in (
            ViscosityModel.NONE,
            ViscosityModel.CONSTANT_NU,
        ):
            raise ValueError("solver only supports constant-nu viscosity")

        if physics.eos_type not in (
            EquationOfState.GLOBALLY_ISOTHERMAL,
            EquationOfState.LOCALLY_ISOTHERMAL,
        ):
            raise ValueError("solver only supports isothermal equation of states")

        if physics.cooling_coefficient != 0.0:
            raise ValueError("solver does not support thermal cooling")

        if not physics.constant_softening:
            raise ValueError("solver only supports constant gravitational softening")

        xp = get_array_module(mode)
        ng = GUARD  # number of guard zones
        nq = NCONS  # number of conserved quantities
        np = NPOLY  # number of polynomials
        with open(__file__.replace(".py", ".c")) as f:
            code = f.read()
        lib = Library(code, mode=mode, debug=True)

        logger.info(f"initiate with time={time:0.4f}")
        logger.info(f"subdivide grid over {num_patches} patches")
        logger.info(f"mesh is {mesh}")
        logger.info(f"boundary condition is outflow")
        logger.info(f"viscosity is {physics.viscosity_coefficient}")

        self.mesh = mesh
        self.setup = setup
        self.num_guard = ng
        self.num_cons = nq
        self.xp = xp
        self.patches = []
        ni, nj = mesh.shape

        if solution is None:
            weights = initial_condition(setup, mesh, time)
        else:
            weights = solution

        if physics.buffer_is_enabled:
            # Here we sample the initial condition at the buffer onset radius
            # to determine the disk surface density at the radius where the
            # buffer begins to ramp up. This procedure makes sense as long as
            # the initial condition is axisymmetric.
            buffer_prim = [0.0] * 3
            buffer_outer_radius = mesh.x1  # this assumes the mesh is a centered squared
            buffer_onset_radius = buffer_outer_radius - physics.buffer_onset_width
            setup.primitive(time, [buffer_onset_radius, 0.0], buffer_prim)
            buffer_surface_density = buffer_prim[0]
        else:
            buffer_outer_radius = 0.0
            buffer_surface_density = 0.0

        for n, (a, b) in enumerate(subdivide(ni, num_patches)):
            weights_patch = numpy.zeros([b - a + 2 * ng, nj + 2 * ng, nq, ORDER, ORDER])
            weights_patch[ng:-ng, ng:-ng] = weights[a:b]
            patch = Patch(
                time,
                weights_patch,
                mesh,
                (a, b),
                physics,
                options,
                buffer_outer_radius,
                buffer_surface_density,
                lib,
                xp,
                execution_context(mode, device_id=n % num_devices(mode)),
            )
            self.patches.append(patch)
            self.set_bc("weights1")

    @property
    def solution(self):
        self.set_bc("weights1")
        return concat_on_host(
            [p.weights1 for p in self.patches],
            (self.num_guard, self.num_guard),
            rank=2,
        )

    @property
    def primitive(self):
        self.set_bc("weights1")
        return concat_on_host(
            [p.primitive for p in self.patches],
            (self.num_guard, self.num_guard),
            rank=2,
        )

    def reductions(self):
        """
        Generate runtime reductions on the solution data for time series.

        As of now, the reductions generated are the rates of mass accretion, and
        of x and y momentum (combined gravitational and accretion) resulting
        from each of the point masses. If there are 2 point masses, then the
        result of this function is a 7-element list: `[time, mdot1, fx1, fy1,
        mdot2, fx2, fy2]`.
        """

        def to_host(a):
            try:
                return a.get()
            except AttributeError:
                return a

        def patch_reduction(patch, which):
            return patch.point_mass_source_term(which).sum(axis=(0, 1)) * da

        da = self.mesh.dx * self.mesh.dy
        point_mass_reductions = [self.time]

        for n in range(self._physics.num_particles):
            point_mass_reductions.extend(
                lazy_reduce(
                    sum,
                    to_host,
                    (lambda: patch_reduction(patch, n + 1) for patch in self.patches),
                    (patch.execution_context for patch in self.patches),
                )
            )
        return point_mass_reductions

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
        return 0.08

    @property
    def maximum_cfl(self):
        return 0.4

    def maximum_wavespeed(self):
        """
        Return the global maximum wavespeed over the whole domain.
        """
        return lazy_reduce(
            max,
            float,
            (patch.maximum_wavespeed for patch in self.patches),
            (patch.execution_context for patch in self.patches),
        )

    def advance(self, dt):
        self.new_iteration()
        if self._options.rk_order == 1:
            self.advance_rk(0.0, dt)
        elif self._options.rk_order == 2:
            self.advance_rk(0.0, dt)
            self.advance_rk(0.5, dt)
        elif self._options.rk_order == 3:
            self.advance_rk(0.0, dt)
            self.advance_rk(0.75, dt)
            self.advance_rk(1.0 / 3.0, dt)

    def advance_rk(self, rk_param, dt):
        if self._options.limit_slopes:
            self.set_bc("weights1")
            for patch in self.patches:
                patch.slope_limit()
        self.set_bc("weights1")
        for patch in self.patches:
            patch.advance_rk(rk_param, dt)

    def set_bc(self, array):
        ng = self.num_guard
        num_patches = len(self.patches)
        for i0 in range(num_patches):
            il = (i0 + num_patches - 1) % num_patches
            ir = (i0 + num_patches + 1) % num_patches
            pl = getattr(self.patches[il], array)
            pc = getattr(self.patches[i0], array)
            pr = getattr(self.patches[ir], array)
            self.set_bc_patch(pl, pc, pr, i0)

    def set_bc_patch(self, pl, pc, pr, patch_index):
        ni, nj = self.mesh.shape
        ng = self.num_guard

        with self.patches[patch_index].execution_context:
            # 1. write to the guard zones of pc, the internal BC
            pc[:+ng] = pl[-2 * ng : -ng]
            pc[-ng:] = pr[+ng : +2 * ng]

            # 2. Set outflow BC on the left/right patch edges
            if patch_index == 0:
                for i in range(ng):
                    pc[i] = pc[ng]
            if patch_index == len(self.patches) - 1:
                for i in range(pc.shape[0] - ng, pc.shape[0]):
                    pc[i] = pc[-ng - 1]

            # 3. Set outflow BC on bottom and top edges
            for i in range(ng):
                pc[:, i] = pc[:, ng]

            for i in range(pc.shape[1] - ng, pc.shape[1]):
                pc[:, i] = pc[:, -ng - 1]

    def new_iteration(self):
        for patch in self.patches:
            patch.new_iteration()
