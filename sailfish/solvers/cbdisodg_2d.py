"""
Isothermal solver for the binary accretion problem in 2D planar coordinates.
"""

from logging import getLogger
from typing import NamedTuple
from sailfish.kernel.library import Library
from sailfish.kernel.system import get_array_module, execution_context, num_devices
from sailfish.mesh import PlanarCartesian2DMesh
from sailfish.physics.circumbinary import Physics, EquationOfState, ViscosityModel
from sailfish.solver import SolverBase
from sailfish.subdivide import subdivide, concat_on_host, lazy_reduce


logger = getLogger(__name__)


class CellData:
    """
    Gauss weights, quadrature points, and tabulated Legendre polonomials.

    This class works for n-th order Gaussian quadrature in 2D.
    """

    def __init__(self, order=3):
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


class Options(NamedTuple):
    """
    Contains parameters which are solver specific options.
    """

    velocity_ceiling: float = 1e12
    mach_ceiling: float = 1e12


def primitive_to_conserved(prim, cons):
    sigma, vx, vy = prim
    cons[0] = sigma
    cons[1] = sigma * vx
    cons[2] = sigma * vx


def initial_condition(setup, mesh, time):
    """
    Generate a 2D array of weights from a mesh and a setup.
    """
    import numpy as np

    n_cons = 3
    n_poly = 3
    order = 3
    ni, nj = mesh.shape
    dx, dy = mesh.dx, mesh.dy
    prim_node = np.zeros(n_cons)
    cons_node = np.zeros(n_cons)
    weights = np.zeros([ni, nj, n_cons, n_poly])

    g = (-0.774596669241483, +0.000000000000000, +0.774596669241483)
    w = (+0.555555555555556, +0.888888888888889, +0.555555555555556)

    for i in range(ni):
        for j in range(nj):
            for ip in range(3):
                for jp in range(3):
                    xc, yc = mesh.cell_coordinates(i, j)
                    x = xc + 0.5 * dx * g[ip]
                    y = yc + 0.5 * dy * g[jp]
                    setup.primitive(time, (x, y), prim_node)
                    primitive_to_conserved(prim_node, cons_node)
                    for q in range(n_cons):
                        for p in range(n_poly):
                            weights[i, j, q, p] += cons_node[q] * w[ip] * w[jp]

    return weights


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
            self.wavespeeds = xp.zeros(primitive.shape[:2])
            self.primitive1 = xp.array(primitive)
            self.primitive2 = xp.array(primitive)
            self.conserved0 = xp.zeros(primitive.shape)

    def point_mass_source_term(self, which_mass):
        """
        Returns an array of conserved quantities over a patch.
        """
        ng = 2  # number of guard cells
        if which_mass not in (1, 2):
            raise ValueError("the mass must be either 1 or 2")

        m1, m2 = self.physics.point_masses(self.time)
        with self.execution_context:
            cons_rate = self.xp.zeros_like(self.conserved0)

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
                which_mass,
                self.primitive1,
                cons_rate,
            )
        return cons_rate[ng:-ng, ng:-ng]

    def maximum_wavespeed(self):
        """
        Returns the maximum wavespeed over a given patch.
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
        """Converts the most recent primitive array to conserved"""
        with self.execution_context:
            return self.lib.cbdiso_2d_primitive_to_conserved[self.shape](
                self.primitive1,
                self.conserved0,
            )

    def advance_rk(self, rk_param, dt):
        """
        Passes required parameters for time evolution of the setup.

        Calls the C-module function responsible for performing time evolution
        using a RK algorithm to update the parameters of the setup.
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
    Adapter class to drive the isodg_2d C extension module.
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

        cell = CellData(order=options.order)
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
        ng = 1  # number of guard zones
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
            xf = mesh.faces(0, num_zones)  # face coordinates
            px = np.zeros([num_zones, 1, cell.num_points])
            ux = np.zeros([num_zones, 1, cell.num_points])
            uw = np.zeros([num_zones, 1, cell.order])
            dx = mesh.dx

    conserved_w = np.zeros([nx, ny, 3 * 6])

    for i in range(ni):
        for j in range(nj):
            # global coordinates of zone centers
            xc, yc = mesh.cell_coordinates(i, j)
            # loop over quadrature points in each zone
            for ic in range(cell.num_points):
                for jc in range(cell.num_points):
                    xp = xc + 0.5 * mesh.dx * cell.gauss_points[ic]
                    yp = yc + 0.5 * mesh.dy * cell.gauss_points[jc]
                    # primitive values at quadrature points
                    setup.primitive(time, xp, yp, primitive[ic, jc])

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

        # if solution is None:
        #    primitive = initial_condition(setup, mesh, time)
        # else:
        #    primitive = solution

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
        return 0.3

    @property
    def maximum_cfl(self):
        return 0.4

    def maximum_wavespeed(self):
        "Returns the global maximum wavespeed over the whole domain."

        return lazy_reduce(
            max,
            float,
            (patch.maximum_wavespeed for patch in self.patches),
            (patch.execution_context for patch in self.patches),
        )

    #    def advance(self, dt):
    #        self.new_iteration()
    #        self.advance_rk(0.0, dt)
    #        self.advance_rk(0.5, dt)

    def advance(self, dt):
        self.new_iteration()
        self.advance_rk(0.0, dt)
        self.advance_rk(0.75, dt)
        self.advance_rk(1.0 / 3.0, dt)

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
