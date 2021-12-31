from sailfish.mesh import LogSphericalMesh
from sailfish.setup import Setup


class UniformPolar(Setup):
    """
    Tests the srhd_2d solver goemtrical source terms.
    """

    def primitive(self, t, _, primitive):
        primitive[0] = 1.0
        primitive[1] = 0.0
        primitive[2] = 0.0
        primitive[3] = 1.0

    def mesh(self, num_zones_per_decade):
        return LogSphericalMesh(1.0, 50.0, num_zones_per_decade, polar_grid=True)

    @property
    def solver(self):
        return "srhd_2d"

    @property
    def boundary_condition(self):
        return "outflow"

    @property
    def default_end_time(self):
        return 1.0
