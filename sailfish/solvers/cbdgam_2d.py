"""
Two-dimensional vertically integrated hydro with gamma-law EOS

This solver is still a work-in-progress.

Fill in more details about the physics here:
"""

from typing import NamedTuple
from logging import getLogger
from typing import NamedTuple
from sailfish.kernel.library import Library
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


class Physics(NamedTuple):
    alpha: float = 0.1
    sink_rate: float = 10.0
    sink_radius: float = 0.05
    mass_model1: int = 1 # [0,1,2,3]: [Inactive, AccelerationFree, TorqueFree, ForceFree]
    mass_model2: int = 1
    soft_length: float = 0.05
    q: float = 1.0
    e: float = 0.0
    gamma_law_index: float = 5.0 / 3.0
    cooling_coefficient: float = 0.0
    constant_softening: int = 1 # whether to use constant softening
    kb_mode: int = 0 # [0,1]: [no buffer, Keplerian buffer]

class Options(NamedTuple):
    pressure_floor: float = 1e-12
    density_floor: float = 1e-10
    velocity_ceiling: float = 1e16
    mach_ceiling: float = 1e5


def initial_condition(setup, mesh, time):
    """
    Generate a 2D array of primitive data from a mesh and a setup.
    """
    import numpy as np

    ni, nj = mesh.shape
    primitive = np.zeros([ni, nj, 4])

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
        kb_surface_density,
        kb_surface_pressure,
        lib,
        xp,
    ):
        i0, i1 = index_range
        self.lib = lib
        self.mesh = mesh
        self.xp = xp
        self.time = self.time0 = time
        ni, nj = mesh.shape()
        self.shape = (i1 - i0, nj)  # not including guard zones
        self.physics = physics
        self.options = options
        self.xl, self.yl = self.mesh.vertex_coordinates(i0, 0)
        self.xr, self.yr = self.xl + (i1 - i0) * self.mesh.dx, self.yl + nj * self.mesh.dy
        self.domain_radius = abs(self.mesh.x1)
        self.kb_outer_radius = self.domain_radius
        self.kb_onset_width = 0.1
        self.kb_surface_density = kb_surface_density
        self.kb_surface_pressure = kb_surface_pressure
        self.kb_driving_rate = 1000.0
        self.kb_central_mass = 1.0

        self.orbelement = OrbitalElements(1.0, 1.0, self.physics.q, self.physics.e)
        orbstate = self.orbelement.orbital_state(self.time)

        with self.execution_context():
            self.wavespeeds = self.xp.zeros(primitive.shape[:2])
            self.primitive1 = self.xp.array(primitive)
            self.primitive2 = self.xp.array(primitive)
            self.conserved0 = self.xp.zeros(primitive.shape)

    def execution_context(self):
        """TODO: return a CUDA context for execution on the assigned device"""
        return nullcontext()

    def maximum_wavespeed(self):
        with self.execution_context():
            self.lib.cbdgam_2d_wavespeed[self.shape](
                self.primitive1,
                self.wavespeeds,
                self.physics.gamma_law_index,
            )
            return self.wavespeeds.max()

    def recompute_conserved(self):
        with self.execution_context():
            return self.lib.cbdgam_2d_primitive_to_conserved[self.shape](
                self.primitive1,
                self.conserved0,
                self.physics.gamma_law_index,
            )

    def advance_rk(self, rk_param, dt):
        orbstate = OrbitalElements(1.0, 1.0, self.physics.q, self.physics.e).orbital_state(self.time)
        with self.execution_context():
            self.lib.cbdgam_2d_advance_rk[self.shape](
                self.xl,
                self.xr,
                self.yl,
                self.yr,
                self.conserved0,
                self.primitive1,
                self.primitive2,
                self.physics.gamma_law_index,
                self.kb_surface_density,
                self.kb_surface_pressure,
                self.kb_central_mass,
                self.kb_driving_rate,
                self.kb_outer_radius,
                self.kb_onset_width,
                self.physics.kb_mode,
                orbstate.primary.position_x,
                orbstate.primary.position_y,
                orbstate.primary.velocity_x,
                orbstate.primary.velocity_y,
                orbstate.primary.mass,
                self.physics.sink_rate,
                self.physics.sink_radius,
                self.physics.mass_model1,
                orbstate.secondary.position_x,
                orbstate.secondary.position_y,
                orbstate.secondary.velocity_x,
                orbstate.secondary.velocity_y,
                orbstate.secondary.mass,
                self.physics.sink_rate,
                self.physics.sink_radius,
                self.physics.mass_model2,
                self.physics.alpha,
                rk_param,
                dt,
                self.options.velocity_ceiling,
                self.physics.cooling_coefficient,
                self.options.mach_ceiling,
                self.options.density_floor,
                self.options.pressure_floor,
                self.physics.constant_softening,
                self.physics.soft_length,
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
    Adapter class to drive the cbdgam_2d C extension module.
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
        nq = 4  # number of conserved quantities
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
        self.domain_radius = self.mesh.x1
        self.kb_onset_width = 0.1

        kb_state = [0.0] * 4
        setup.primitive(time, [self.domain_radius - self.kb_onset_width, 0.0], kb_state)

        if solution is None:
            primitive = initial_condition(setup, mesh, time)
        else:
            primitive = solution

        for (a, b) in subdivide(ni, num_patches):
            prim = xp.zeros([b - a + 2 * ng, nj + 2 * ng, nq])
            prim[ng:-ng, ng:-ng] = primitive[a:b]
            self.patches.append(
                Patch(time, prim, mesh, (a, b), self._physics, self._options, kb_state[0], kb_state[3], lib, xp)
            )

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
