"""
Orbiting binary embedded in a wind

This code sets up a binary black hole in a wind of uniform density and Mach
number, to inspect the accretion rate of matter onto the binary under
different conditions. The accretion process in such scenarios is referred to
as Bondi-Hoyle-Lyttleton accretion. Some references below:

.. _Comerford & Izzard (2019): https://arxiv.org/abs/1910.13353
.. _Soker (2004): https://academic.oup.com/mnras/article/350/4/1366/986224
"""

from sailfish.mesh import PlanarCartesian2DMesh
from sailfish.physics.circumbinary import SinkModel, PointMass, EquationOfState
from sailfish.physics.kepler import OrbitalElements
from sailfish.setup_base import SetupBase, param


class BinaryBondi(SetupBase):
    """
    Simulates Bondi-Hoyle-Lyttleton type accretion in a binary system.

    The gas is isothermal with uniform sound speed.
    """

    aspect = param(1, "aspect ratio of the domain, width / height")
    height = param(1.0, "domain height in units of the binary separation")
    x_bin = param(0.0, "horizontal offset of binary COM from domain center")
    sound_speed = param(1.0, "speed of sound (uniform)")
    wind_vel = param(1.0, "horizontal speed of the background flow")
    bh_sep = param(0.25, "black hole semi-major axis")
    bh_mass = param(1.0, "mass of the primary")
    sink_radius = param(0.05, "black hole sink radius")
    sink_model = param("acceleration_free", "sink prescription")
    sink_rate = param(1e2, "sink rate")
    mass_ratio = param(1.0, "system mass ratio: [0.0-1.0]")
    eccentricity = param(0.0, "eccentricity")
    softening_length = param(0.05, "gravitational softening length", mutable=True)

    def primitive(self, t, coords, primitive):
        primitive[0] = 1.0
        primitive[1] = self.wind_vel
        primitive[2] = 0.0

    def mesh(self, resolution):
        return PlanarCartesian2DMesh.centered_rectangle(
            self.height, resolution, int(self.aspect)
        )

    @property
    def default_resolution(self):
        return 1000

    @property
    def physics(self):
        return dict(
            eos_type=EquationOfState.GLOBALLY_ISOTHERMAL,
            sound_speed=self.sound_speed,
            point_mass_function=self.point_masses,
        )

    @property
    def solver(self):
        return "cbdiso_2d"

    @property
    def boundary_condition(self):
        return "outflow"

    @property
    def default_end_time(self):
        return 10.0

    @property
    def orbital_elements(self):
        return OrbitalElements(
            semimajor_axis=self.bh_sep,
            total_mass=self.bh_mass * (1 + self.mass_ratio),
            mass_ratio=self.mass_ratio,
            eccentricity=self.eccentricity,
        )

    def point_masses(self, time):
        m1, m2 = self.orbital_elements.orbital_state(time)
        m1 = m1._replace(position_x=m1.position_x + self.x_bin)
        m2 = m2._replace(position_x=m2.position_x + self.x_bin)
        return (
            PointMass(
                softening_length=self.softening_length,
                sink_model=SinkModel[self.sink_model.upper()],
                sink_rate=self.sink_rate,
                sink_radius=self.sink_radius,
                **m1._asdict(),
            ),
            PointMass(
                softening_length=self.softening_length,
                sink_model=SinkModel[self.sink_model.upper()],
                sink_rate=self.sink_rate,
                sink_radius=self.sink_radius,
                **m2._asdict(),
            ),
        )
