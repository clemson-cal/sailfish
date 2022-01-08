"""
For simulating bondi-hoyle-lyttleton accretion in a binary in a wind setup.
"""
from typing import Callable
from sailfish.mesh import PlanarCartesian2DMesh
from sailfish.physics.circumbinary import SinkModel, PointMass
from sailfish.physics.kepler import OrbitalElements
from sailfish.setup import Setup, param


class BinaryBondi(Setup):

    aspect = param(1, "aspect ratio for domain")
    height = param(1.0, "length of the Y-axis of domain")
    x_bin = param(
        0.0, "horizontal position of binary COM (with respect to domain centroid)"
    )
    mach_number = param(10.0, "orbital Mach number (isothermal)", mutable=True)
    bh_sep = param(0.25, "black hole separation")
    bh_mass = param(1.0, "black hole's mass")
    sink_radius = param(5e-2, "black hole's (sink) radius")
    sink_model = param(SinkModel.ACCELERATION_FREE, "sink prescription")
    sink_rate = param(1e2, "sink rate")
    mass_ratio = param(1.0, "system mass ratio: [0.0-1.0]")
    eccentricity = param(0.0, "eccentricity")
    softening_length = param(0.05, "gravitational softening length", mutable=True)

    def primitive(self, t, coords, primitive):
        primitive[0] = 1.0
        primitive[1] = self.mach_number
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
            mach_number=self.mach_number,
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

    def point_masses(self, time):
        elements = OrbitalElements(
            semimajor_axis=self.bh_sep,
            total_mass=self.bh_mass * (1 + self.mass_ratio),
            mass_ratio=self.mass_ratio,
            eccentricity=self.eccentricity,
        )
        m1, m2 = elements.orbital_state(time)
        m1 = PointMass(**m1._asdict())
        m2 = PointMass(**m2._asdict())
        return (
            m1._replace(
                position_x=m1.position_x + self.x_bin,
                softening_length=self.softening_length,
                sink_model=self.sink_model,
                sink_rate=self.sink_rate,
                sink_radius=self.sink_radius,
            ),
            m2._replace(
                position_x=m2.position_x + self.x_bin,
                softening_length=self.softening_length,
                sink_model=self.sink_model,
                sink_rate=self.sink_rate,
                sink_radius=self.sink_radius,
            ),
        )
