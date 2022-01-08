"""
For simulating bondi-hoyle-lyttleton accretion in a binary in a wind setup.
"""
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
    bh_sep = param(1.0, "black hole separation")
    bh_mass = param(1.0, "black hole's mass")
    bh_rad = param(5e-2, "black hole's radius")  # sink_radius
    sink_model = param(SinkModel.ACCELERATION_FREE, "sink prescription")
    sink_rate = param(1e2, "sink rate")
    q = param(1.0, "system mass ratio: [0.0-1.0]")
    e = param(0.0, "eccentricity")

    def primitive(self, t, coords, primitive):
        primitive[0] = 1.0
        primitive[1] = self.physics.mach_number * self.physics.sound_speed

    def mesh(self, resolution):
        if isinstance(aspect, int):
            return PlanarCartesian2DMesh.centered_rectangle(
                self.height, resolution, int(self.aspect)
            )
        else:
            raise ValueError("Aspect has to be supplied an integer!")

    def masses(self):
        m1, m2 = OrbitalElements(
            1.0, self.bh_mass * (1 + self.q), self.q, self.e
        ).orbital_state

    @property
    def physics(self):
        return dict()
