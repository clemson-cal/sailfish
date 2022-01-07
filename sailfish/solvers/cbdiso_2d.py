"""
One-dimensional relativistic hydro solver supporting homologous mesh motion.
"""

from logging import getLogger
from sailfish.kernel.library import Library
from typing import NamedTuple
from sailfish.kernel.system import get_array_module
from sailfish.subdivide import subdivide
from sailfish.mesh import PlanarCartesian2DMesh
from sailfish.solver import SolverBase
from sailfish.physics.kepler import OrbitalElements

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


SINK_MODEL_ACCELERATION_FREE = 1
SINK_MODEL_TORQUE_FREE = 2
SINK_MODEL_FORCE_FREE = 3

EOS_TYPE_LOCALLY_ISOTHERMAL = 1
EOS_TYPE_GLOBALLY_ISOTHEMRAL = 2


class Physics(NamedTuple):

    sound_speed_squared: float = 1.0
    """ Square of the sound speed, if EOS type is globally isothermal """

    mach_number: float = 10.0
    """ Square of the Mach number, if EOS type is locally isothermal """

    viscosity_coefficient: float = 0.01
    """ Kinematic viscosity value, in units of a^2 Omega """

    buffer_is_enabled: bool = False
    """ Whether the buffer zone is enabled """

    buffer_surface_density: float = 1.0
    """ Target surface density in the buffer zone, if it's enabled """

    buffer_central_mass: float = 1.0
    """ Used to determine the orbital velocity in the buffer region """

    buffer_driving_rate: float = 1000.0
    """ Rate of driving toward target solution in the buffer region """

    buffer_onset_width: float = 0.1
    """ Distance over which the buffer ramps up """

    eos_type: int = EOS_TYPE_GLOBALLY_ISOTHEMRAL
    """ EOS type: globally or locally isothermal """

    q: float = 1.0
    e: float = 0.0
    sink_rate: float = 10.0
    sink_radius: float = 0.05
    sink_model1: int = SINK_MODEL_ACCELERATION_FREE
    sink_model2: int = SINK_MODEL_ACCELERATION_FREE
    softening_length: float = 0.01


class Options(NamedTuple):
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
        lib,
        xp,
    ):
        i0, i1 = index_range
        ni, nj = i1 - i0, mesh.shape[0]
        self.lib = lib
        self.mesh = mesh
        self.xp = xp
        self.time = self.time0 = time
        self.shape = (i1 - i0, nj)  # not including guard zones
        self.physics = physics
        self.options = options
        self.xl, self.yl = self.mesh.vertex_coordinates(i0, 0)
        self.xr, self.yr = self.mesh.vertex_coordinates(i1, nj)
        self.buffer_outer_radius = self.mesh.x1 - self.physics.buffer_onset_width
        self.orbelement = OrbitalElements(1.0, 1.0, self.physics.q, self.physics.e)

        with self.execution_context():
            self.wavespeeds = self.xp.zeros(primitive.shape[:2])
            self.primitive1 = self.xp.array(primitive)
            self.primitive2 = self.xp.array(primitive)
            self.conserved0 = self.xp.zeros(primitive.shape)

    def execution_context(self):
        """TODO: return a CUDA context for execution on the assigned device"""
        return nullcontext()

    def maximum_wavespeed(self):
        m1, m2 = self.orbelement.orbital_state(self.time)
        with self.execution_context():
            self.lib.cbdiso_wavespeed[self.shape](
                self.xl,
                self.xr,
                self.yl,
                self.yr,
                self.physics.sound_speed_squared,
                self.physics.mach_number ** 2.0,
                self.physics.eos_type,
                m1.position_x,
                m1.position_y,
                m1.velocity_x,
                m1.velocity_y,
                m1.mass,
                self.physics.softening_length,
                self.physics.sink_rate,
                self.physics.sink_radius,
                self.physics.sink_model1,
                m2.position_x,
                m2.position_y,
                m2.velocity_x,
                m2.velocity_y,
                m2.mass,
                self.physics.softening_length,
                self.physics.sink_rate,
                self.physics.sink_radius,
                self.physics.sink_model2,
                self.primitive1,
                self.wavespeeds,
            )
            return self.wavespeeds.max()

    def recompute_conserved(self):
        with self.execution_context():
            return self.lib.iso2d_primitive_to_conserved[self.shape](
                self.primitive1,
                self.conserved0,
            )

    def advance_rk(self, rk_param, dt):
        m1, m2 = self.orbelement.orbital_state(self.time)
        with self.execution_context():
            self.lib.cbdiso_advance_rk[self.shape](
                self.xl,
                self.xr,
                self.yl,
                self.yr,
                self.conserved0,
                self.primitive1,
                self.primitive2,
                self.physics.buffer_surface_density,
                self.physics.buffer_central_mass,
                self.physics.buffer_driving_rate,
                self.buffer_outer_radius,
                self.physics.buffer_onset_width,
                int(self.physics.buffer_is_enabled),
                m1.position_x,
                m1.position_y,
                m1.velocity_x,
                m1.velocity_y,
                m1.mass,
                self.physics.softening_length,
                self.physics.sink_rate,
                self.physics.sink_radius,
                self.physics.sink_model1,
                m2.position_x,
                m2.position_y,
                m2.velocity_x,
                m2.velocity_y,
                m2.mass,
                self.physics.softening_length,
                self.physics.sink_rate,
                self.physics.sink_radius,
                self.physics.sink_model2,
                self.physics.sound_speed_squared,
                self.physics.mach_number ** 2.0,
                self.physics.eos_type,
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
        if type(mesh) is not PlanarCartesian2DMesh:
            raise ValueError("solver only supports 2D cartesian mesh")

        if setup.boundary_condition != "outflow":
            raise ValueError("solver only supports outflow boundary condition")

        self._physics = Physics(**physics)
        self._options = Options(**options)
        xp = get_array_module(mode)
        ng = 2  # number of guard zones
        nq = 3  # number of conserved quantities
        with open(__file__.replace(".py", ".c")) as f:
            code = f.read()
        lib = Library(code, mode=mode, debug=True)

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

        for (a, b) in subdivide(ni, num_patches):
            prim = xp.zeros([b - a + 2 * ng, nj + 2 * ng, nq])
            prim[ng:-ng, ng:-ng] = primitive[a:b]
            self.patches.append(
                Patch(time, prim, mesh, (a, b), self._physics, self._options, lib, xp)
            )

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
            primitive[a:b] = patch.primitive[ng:-ng, ng:-ng]
        return primitive

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
    def maximum_cfl(self):
        return 0.1

    def maximum_wavespeed(self):
        return max(patch.maximum_wavespeed() for patch in self.patches)

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
