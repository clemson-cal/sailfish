"""
Isothermal solver for the binary accretion problem in 2D planar coordinates.
"""

from logging import getLogger
from typing import NamedTuple, List
from sailfish.kernel.library import Library
from sailfish.kernel.system import get_array_module, execution_context, num_devices
from sailfish.mesh import PlanarCartesian2DMesh
from sailfish.physics.circumbinary import (
    Physics,
    EquationOfState,
    ViscosityModel,
    Diagnostic,
)
from sailfish.solver import SolverBase
from sailfish.subdivide import subdivide, concat_on_host, lazy_reduce


logger = getLogger(__name__)


class Options(NamedTuple):
    """
    Contains parameters which are solver specific options.
    """

    velocity_ceiling: float = 1e12
    mach_ceiling: float = 1e12


def initial_condition(setup, mesh, time):
    """
    Generate a 2D array of primitive data from a mesh and a setup.
    """
    import numpy as np

    ni, nj = mesh.shape
    primitive = np.zeros([ni, nj, 3])

    for i in range(ni):
        for j in range(nj):
            setup.primitive(time, mesh.cell_coordinates(i, j), primitive[i, j])

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
            x0 = self.xl + 0.5 * mesh.dx
            x1 = self.xr - 0.5 * mesh.dx
            y0 = self.yl + 0.5 * mesh.dy
            y1 = self.yr - 0.5 * mesh.dy
            self.coordinate_array_x = xp.linspace(x0, x1, ni)[:, None]
            self.coordinate_array_y = xp.linspace(y0, y1, nj)[None, :]
            self.wavespeeds = xp.zeros(primitive.shape[:2])
            self.primitive1 = xp.array(primitive)
            self.primitive2 = xp.array(primitive)
            self.conserved0 = xp.zeros(primitive.shape)

    @property
    def cell_center_coordinate_arrays(self):
        """
        Return two 2d arrays, one with the cell-center X coordinates, and the
        other with the cell-center Y coordinates. The arrays are either numpy
        or cupy arrays, allocated for the device this patch is assigned to.
        """
        return self.coordinate_array_x, self.coordinate_array_y

    def point_mass_source_term(self, which_mass, gravity=False, accretion=False):
        """
        Return an array of the rates of conserved quantities, resulting from
        the application of gravitational and/or accretion source terms due to
        point masses.
        """

        ng = 2  # number of guard cells
        if which_mass not in (1, 2):
            raise ValueError("the mass must be either 1 or 2")

        m = self.physics.point_masses(self.time)[which_mass - 1]

        with self.execution_context:
            cons_rate = self.xp.zeros_like(self.conserved0)

            self.lib.cbdiso_2d_point_mass_source_term[self.shape](
                self.xl,
                self.xr,
                self.yl,
                self.yr,
                m.position_x,
                m.position_y,
                m.velocity_x,
                m.velocity_y,
                m.mass * float(gravity),
                m.softening_length,
                m.sink_rate * float(accretion),
                m.sink_radius,
                m.sink_model.value,
                self.primitive1,
                cons_rate,
            )
        return cons_rate[ng:-ng, ng:-ng]

    def maximum_wavespeed(self):
        """
        Return the maximum wavespeed over a given patch.
        """
        m1, m2 = self.physics.point_masses(self.time)
        with self.execution_context:
            self.lib.cbdiso_2d_wavespeed[self.shape](
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
                self.primitive1,
                self.wavespeeds,
            )
            return self.wavespeeds.max()

    def recompute_conserved(self):
        """
        Convert the most recent primitive array to conserved.
        """
        with self.execution_context:
            return self.lib.cbdiso_2d_primitive_to_conserved[self.shape](
                self.primitive1,
                self.conserved0,
            )

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
            self.lib.cbdiso_2d_advance_rk[self.shape](
                self.xl,
                self.xr,
                self.yl,
                self.yr,
                self.conserved0,
                self.primitive1,
                self.primitive2,
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
        self.primitive1, self.primitive2 = self.primitive2, self.primitive1

    def new_iteration(self):
        self.time0 = self.time
        self.recompute_conserved()

    @property
    def primitive(self):
        return self.primitive1


class Solver(SolverBase):
    """
    Adapter class to drive the iso_2d C extension module.
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

        physics["diagnostics"] = [
            Diagnostic(**v) for v in physics.get("diagnostics", [])
        ]

        self._physics = physics = Physics(**physics)
        self._options = options = Options(**options)

        if type(mesh) is not PlanarCartesian2DMesh:
            raise ValueError("solver only supports 2D cartesian mesh")

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
        ng = 2  # number of guard zones
        nq = 3  # number of conserved quantities
        with open(__file__.replace(".py", ".c")) as f:
            code = f.read()
        lib = Library(code, mode=mode, debug=False)

        logger.info(f"initiate with time={time:0.4f}")
        logger.info(f"subdivide grid over {num_patches} patches")
        logger.info(f"mesh is {mesh}")
        logger.info(f"boundary condition is outflow")

        self.mesh = mesh
        self.setup = setup
        self.num_guard = ng
        self.num_cons = nq
        self.xp = xp
        self.patches = []
        ni, nj = mesh.shape

        if solution is None:
            primitive = initial_condition(setup, mesh, time)
        else:
            primitive = solution

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
            prim = np.zeros([b - a + 2 * ng, nj + 2 * ng, nq])
            prim[ng:-ng, ng:-ng] = primitive[a:b]
            patch = Patch(
                time,
                prim,
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

    @property
    def solution(self):
        return self.primitive

    @property
    def primitive(self):
        return concat_on_host(
            [p.primitive for p in self.patches], (self.num_guard, self.num_guard)
        )

    def reductions(self):
        """
        Generate runtime reductions on the solution data for time series.

        As of now, the reductions generated are the rates of mass accretion,
        and of x and y momentum (combined gravitational and accretion)
        resulting from each of the point masses. If there are 2 point masses,
        then the result of this function is a 7-element list: :pyobj:`[time,
        mdot1, fx1, fy1, mdot2, fx2, fy2]`.
        """
        import numpy as np

        def to_host(a):
            try:
                return a.get()
            except AttributeError:
                return a

        def patch_reduction(patch, which):
            x, y = patch.cell_center_coordinate_arrays
            r = (x**2 + y**2) ** 0.5
            udot = patch.point_mass_source_term(which, gravity=True)
            mdot = udot[..., 0]
            fx = udot[..., 1]
            fy = udot[..., 2]
            torque = x * fy - y * fx
            return np.array(
                [
                    mdot.sum() * da,
                    fx.sum() * da,
                    fy.sum() * da,
                    torque.sum() * da,
                    torque[r > 1.0].sum() * da,
                ]
            )

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
        return 0.3

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
