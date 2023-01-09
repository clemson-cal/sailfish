"""
Defines test problems and generic hydrodynamic setups

Note: There is a caveat in the way the application does hierarchical
configuration: dictionaries are recursively updated, and this means that
dictionaries that map to schemas that are discriminated unions will have their
keys joined if superseded, and this can introduce extra parameters that fail
validation. The workaround for this is to ensure that the default model in a
discriminated union does not define any parameters other than the
discriminating field.
"""


from typing import Literal, Union
from numpy import logical_not, zeros, sqrt, exp, sin, cos, pi
from .preset import preset
from .schema import schema
from .geometry import CoordinateBox


def two_state(region_a, state_a, state_b):
    region_b = logical_not(region_a)
    p = zeros(region_a.shape + (len(state_a),))
    p[region_a] = state_a
    p[region_b] = state_b
    return p


MODEL_DATA_CLASSES = list()


def modeldata(cls):
    MODEL_DATA_CLASSES.append(cls)
    cls.model = "".join(
        ["-" + c.lower() if c.isupper() else c for c in cls.__name__]
    ).lstrip("-")
    cls.__annotations__["model"] = Literal[cls.model]
    return schema(cls)


@modeldata
class Sod:
    """
    Classic Sod shocktube initial data
    """

    @property
    def primitive_fields(self):
        return "density", "x-velocity", "pressure"

    def primitive(self, box: CoordinateBox):
        x = box.cell_centers()
        return two_state(
            x < 0.5,
            [1.0, 0.0, 1.000],
            [0.1, 0.0, 0.125],
        )


@preset
def sod():
    return {
        "initial_data.model": "sod",
        "domain.num_zones": [200, 1, 1],
        "driver.tfinal": 0.1,
    }


@modeldata
class Uniform:
    """
    Uniform initial data

    This model data could be useful for things like testing the correctness of
    boundary conditions or geometrical source terms, see example presets below.
    """

    coordinates: str = "cartesian"

    @property
    def primitive_fields(self):
        return "density", "i-velocity", "j-velocity", "k-velocity", "pressure"

    def primitive(self, box: CoordinateBox):
        p = zeros(box.num_zones + (5,))
        p[...] = [1.0, 0.0, 0.0, 0.0, 1.0]
        return p


@preset
def uniform1d():
    """
    Uniform initial data in 1d spherical symmetry; tests source terms

    The solution should not evolve away from the initial value, either to
    machine precision or at least to a very good accuracy.
    """
    return {
        "initial_data.model": "uniform",
        "domain.num_zones": [200, 1, 1],
        "domain.extent_i": [1.0, 10.0],
        "coordinates": "spherical-polar",
        "driver.tfinal": 0.1,
    }


@preset
def uniform2d():
    """
    Uniform initial data in 2d spherical-polar coordinates; tests source terms

    The solution should not evolve away from the initial value, either to machine
    precision or at least to a very good accuracy.
    """
    return {
        "initial_data.model": "uniform",
        "domain.num_zones": [200, 200, 1],
        "domain.extent_i": [1.0, 10.0],
        "domain.extent_j": [0.0, pi],
        "coordinates": "spherical-polar",
        "driver.tfinal": 0.1,
    }


@modeldata
class IsothermalVortex:
    """
    An isothermal vortex
    """

    mach_number: float = 1.0

    @property
    def primitive_fields(self):
        return "density", "i-velocity", "j-velocity", "k-velocity", "pressure"

    def primitive(self, box: CoordinateBox):
        p = zeros(box.num_zones + (5,))

        if self.coordinates == "cartesian" and box.dimensionality == 2:
            x, y = box.cell_centers()
            r = sqrt(x**2 + y**2)
            rho, uf, pre = self.vortex(r)
            ux = -uf * y / r
            uy = +uf * x / r
            p[:, :, 0, 0] = rho
            p[:, :, 0, 1] = ux
            p[:, :, 0, 2] = uy
            p[:, :, 0, 3] = 0.0
            p[:, :, 0, 4] = pre
        elif self.coordinates == "cylindrical-polar" and box.dimensionality == 1:
            r = box.cell_centers()
            rho, uf, pre = self.vortex(r)
            p[:, 0, 0, 0] = rho
            p[:, 0, 0, 1] = 0.0
            p[:, 0, 0, 2] = 0.0
            p[:, 0, 0, 3] = uf
            p[:, 0, 0, 4] = pre
        else:
            raise ValueError("unsupported coordinates configuration")
        return p

    def vortex(self, r):
        omega0 = 1.0
        r0 = 1.0  # radius of vortex core
        cs = r0 * omega0 / self.mach_number  # nominal sound speed
        omega = omega0 * exp(-0.5 * r**2 / r0**2)
        rho = exp(-0.5 * self.mach_number**2 * exp(-((r / r0) ** 2)))
        return rho, omega * r, rho * cs**2


@preset
def isothermal_vortex1d():
    return {
        "initial_data.model": "isothermal-vortex",
        "initial_data.mach_number": 1.5,
        "domain.num_zones": [200, 1, 1],
        "domain.extent_i": [1.0, 10.0],
        "coordinates": "cylindrical-polar",
    }


@preset
def isothermal_vortex2d():
    return {
        "initial_data.model": "isothermal-vortex",
        "initial_data.mach_number": 1.5,
        "domain.num_zones": [200, 200, 1],
        "domain.extent_i": [-5.0, 5.0],
        "domain.extent_j": [-5.0, 5.0],
        "coordinates": "cartesian",
    }


@modeldata
class CylindricalExplosion:
    """
    Cylindrical explosion initial data

    Initializes a circular region of high pressure and density. A shock wave
    and contact discontinuity should expand outward with a circular profile.
    """

    @property
    def primitive_fields(self):
        return "density", "x-velocity", "y-velocity", "pressure"

    def primitive(self, box: CoordinateBox):
        x, y = box.cell_centers()
        return two_state(
            sqrt(x**2 + y**2) < 0.1,
            [1.0, 0.0, 0.0, 1.000],
            [0.1, 0.0, 0.0, 0.125],
        )


@preset
def cylindrical_explosion():
    return {
        "initial_data.model": "cylindrical-explosion",
        "domain.num_zones": [200, 200, 1],
        "domain.extent_i": [-0.50, 0.50],
        "domain.extent_j": [-0.50, 0.50],
        "driver.tfinal": 0.1,
    }


@modeldata
class CylinderInWind:
    """
    A round cylinder immersed in a dilute wind

    A circular region of high density is immersed in a low-density ambient
    medium moving from left to right. After some time, the high-density region
    is disrupted by Kelvin-Helmholtz instabilities.
    """

    @property
    def primitive_fields(self):
        return "density", "x-velocity", "y-velocity", "pressure"

    def primitive(self, box: CoordinateBox):
        x, y = box.cell_centers()
        return two_state(
            sqrt(x**2 + y**2) < 0.1,
            [1e2, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.1],
        )


@preset
def cylinder_in_wind():
    return {
        "initial_data.model": "cylinder-in-wind",
        "domain.num_zones": [200, 200, 1],
        "domain.extent_i": [-0.25, 0.75],
        "domain.extent_j": [-0.50, 0.50],
        "driver.tfinal": 1.0,
        "boundary_condition.upper_i": "outflow",
    }


@modeldata
class Ram41:
    """
    1d Riemann problem (RAM problem 1; Sec 4.1)

    Adapted from "RAM: A Relativistic Adaptive Mesh Refinement Hydrodynamics
    Code" The Astrophysical Journal Supplement Series, Volume 164, Issue 1,
    pp. 255-279.
    """

    @property
    def primitive_fields(self):
        return "proper-density", "x-gamma-beta", "pressure"

    def primitive(self, box: CoordinateBox):
        return two_state(box.cell_centers() < 0.5, [10.0, 0.0, 13.33], [1.0, 0.0, 1e-8])


@preset
def ram41():
    return {
        "initial_data.model": "ram41",
        "domain.num_zones": [400, 1, 1],
        "domain.extent_i": [0.0, 1.0],
        "driver.tfinal": 0.4,
        "physics.equation_of_state.gamma_law_index": 5.0 / 3.0,
        "physics.metric": "minkowski",
    }


@modeldata
class Ram42:
    """
    1d Riemann problem (RAM problem 2; Sec 4.2)

    Adapted from "RAM: A Relativistic Adaptive Mesh Refinement Hydrodynamics
    Code" The Astrophysical Journal Supplement Series, Volume 164, Issue 1,
    pp. 255-279.
    """

    @property
    def primitive_fields(self):
        return "proper-density", "x-gamma-beta", "pressure"

    def primitive(self, box: CoordinateBox):
        return two_state(box.cell_centers() < 0.5, [1.0, 0.0, 1000.0], [1.0, 0.0, 1e-2])


@preset
def ram42():
    return {
        "initial_data.model": "ram42",
        "domain.num_zones": [400, 1, 1],
        "domain.extent_i": [0.0, 1.0],
        "driver.tfinal": 0.4,
        "physics.equation_of_state.gamma_law_index": 5.0 / 3.0,
        "physics.metric": "minkowski",
    }


@modeldata
class Ram43:
    """
    1d Riemann problem (RAM problem 3; Sec 4.3)

    Adapted from "RAM: A Relativistic Adaptive Mesh Refinement Hydrodynamics
    Code" The Astrophysical Journal Supplement Series, Volume 164, Issue 1,
    pp. 255-279.
    """

    @property
    def primitive_fields(self):
        return "proper-density", "x-gamma-beta", "pressure"

    def primitive(self, box: CoordinateBox):
        v = 0.9
        u = v / sqrt(1.0 - v * v)
        x = box.cell_centers()
        return two_state(x < 0.5, [1.0, u, 1.0], [1.0, 0.0, 10.0])


@preset
def ram43():
    return {
        "initial_data.model": "ram43",
        "domain.num_zones": [400, 1, 1],
        "domain.extent_i": [0.0, 1.0],
        "driver.tfinal": 0.4,
        "physics.equation_of_state.gamma_law_index": 4.0 / 3.0,
        "physics.metric": "minkowski",
    }


@modeldata
class Ram44:
    """
    1d Riemann problem (RAM problem 4; Sec 4.4)

    Status: Failing

    Adapted from "RAM: A Relativistic Adaptive Mesh Refinement Hydrodynamics
    Code" The Astrophysical Journal Supplement Series, Volume 164, Issue 1,
    pp. 255-279.
    """

    @property
    def primitive_fields(self):
        return "proper-density", "x-gamma-beta", "y-gamma-beta", "pressure"

    def primitive(self, box: CoordinateBox):
        v = 0.99
        u = v / sqrt(1.0 - v * v)
        x = box.cell_centers()
        return two_state(x < 0.5, [1.0, 0.0, 0.0, 1e3], [1.0, 0.0, u, 1e-2])


@preset
def ram44():
    return {
        "initial_data.model": "ram44",
        "domain.num_zones": [400, 1, 1],
        "domain.extent_i": [0.0, 1.0],
        "driver.tfinal": 0.4,
        "physics.equation_of_state.gamma_law_index": 5.0 / 3.0,
        "physics.metric": "minkowski",
    }


@modeldata
class Ram45:
    """
    1d shock heating Riemann problem (RAM problem 5; Sec 4.5)

    Status: Failing

    Adapted from "RAM: A Relativistic Adaptive Mesh Refinement Hydrodynamics
    Code" The Astrophysical Journal Supplement Series, Volume 164, Issue 1,
    pp. 255-279.
    """

    @property
    def primitive_fields(self):
        return "proper-density", "x-gamma-beta", "pressure"

    def primitive(self, box: CoordinateBox):
        v = 1.0 - 1e-10
        u = v / sqrt(1.0 - v * v)
        pre_small = 1e-3
        p[...] = [1.0, u, pre_small]
        return p


@preset
def ram45():
    return {
        "initial_data.model": "ram45",
        "domain.num_zones": [100, 1, 1],
        "domain.extent_i": [0.0, 1.0],
        "boundary_condition.lower_i": "outflow",
        "boundary_condition.upper_i": "reflecting",
        "driver.tfinal": 2.0,
        "physics.equation_of_state.gamma_law_index": 4.0 / 3.0,
        "physics.metric": "minkowski",
    }


@modeldata
class Ram61:
    """
    1d Riemann problem with transverse velocity: (RAM Hard Test; sec 6.1)

    Status: Failing

    Adapted from "RAM: A Relativistic Adaptive Mesh Refinement Hydrodynamics
    Code" The Astrophysical Journal Supplement Series, Volume 164, Issue 1,
    pp. 255-279.
    """

    @property
    def primitive_fields(self):
        return "proper-density", "x-gamma-beta", "y-gamma-beta", "pressure"

    def primitive(self, box: CoordinateBox):
        v = 0.9
        u = v / sqrt(1.0 - v * v)
        x = box.cell_centers()
        return two_state(x < 0.5, [1.0, 0.0, u, 1e3], [1.0, 0.0, u, 1e-2])


@preset
def ram61():
    return {
        "initial_data.model": "ram61",
        "domain.num_zones": [400, 1, 1],
        "domain.extent_i": [0.0, 1.0],
        "driver.tfinal": 0.6,
        "physics.equation_of_state.gamma_law_index": 5.0 / 3.0,
        "physics.metric": "minkowski",
    }


@modeldata
class FuShu33:
    """
    Lax problem initial data

    Adapted from Example 3.3 from G. Fu and C.-W. Shu, "A new trouble-cell
    indicator for discontinuous Galerkin methods for hyperbolic conservation
    laws," Journal of Computational Physics, v347 (2017), pp.305-327.
    """

    @property
    def primitive_fields(self):
        return "density", "x-velocity", "pressure"

    def primitive(self, box: CoordinateBox):
        return two_state(
            box.cell_centers() < 0.0,
            [0.445, 0.698, 3.528],
            [0.500, 0.000, 0.571],
        )


@preset
def fu_shu33():
    return {
        "initial_data.model": "fu-shu33",
        "domain.num_zones": [200, 1, 1],
        "domain.extent_i": [-5.0, 5.0],
        "driver.tfinal": 1.3,
    }


@modeldata
class FuShu34:
    """
    Lax problem: double rarefaction wave

    Adapted from Example 3.4 from G. Fu and C.-W. Shu, "A new trouble-cell
    indicator for discontinuous Galerkin methods for hyperbolic conservation
    laws," Journal of Computational Physics, v347 (2017), pp.305-327.
    """

    @property
    def primitive_fields(self):
        return "density", "x-velocity", "pressure"

    def primitive(self, box: CoordinateBox):
        return two_state(
            box.cell_centers() < 0.0,
            [7.0, -1.0, 0.2],
            [7.0, +1.0, 0.2],
        )


@preset
def fu_shu34():
    return {
        "initial_data.model": "fu-shu34",
        "domain.num_zones": [200, 1, 1],
        "domain.extent_i": [-1.0, 1.0],
        "driver.tfinal": 0.6,
    }


@modeldata
class FuShu35:
    """
    LeBlanc problem

    Adapted from Example 3.5 from G. Fu and C.-W. Shu, "A new trouble-cell
    indicator for discontinuous Galerkin methods for hyperbolic conservation
    laws," Journal of Computational Physics, v347 (2017), pp.305-327.

    Note: use log-scaling to plot the mass density profile.
    """

    @property
    def primitive_fields(self):
        return "density", "x-velocity", "pressure"

    def primitive(self, box: CoordinateBox):
        return two_state(
            box.cell_centers() < 0.0,
            [1.00, 0.0, 2.0 / 3.0 * 1e-1],
            [1e-3, 0.0, 2.0 / 3.0 * 1e-10],
        )


@preset
def fu_shu35():
    return {
        "initial_data.model": "fu-shu35",
        "domain.num_zones": [600, 1, 1],
        "domain.extent_i": [-3.0, 6.0],
        "driver.tfinal": 6.0,
    }


@modeldata
class FuShu36:
    """
    Shu-Osher problem

    Adapted from Example 3.6 from G. Fu and C.-W. Shu, "A new trouble-cell
    indicator for discontinuous Galerkin methods for hyperbolic conservation
    laws," Journal of Computational Physics, v347 (2017), pp.305-327.
    """

    @property
    def primitive_fields(self):
        return "density", "x-velocity", "pressure"

    def primitive(self, box: CoordinateBox):
        x = box.cell_centers()
        p = zeros(x.shape + (3,))
        l = (x >= -5.0) * (x < -4.0)
        r = (x >= -4.0) * (x < +5.0)
        p[l] = [3.857143, 2.629369, 10.333333]
        p[r, 0] = 1.0 + 0.2 * sin(5.0 * x[r])
        p[r, 1] = 0.0
        p[r, 2] = 1.0
        return p


@preset
def fu_shu36():
    return {
        "initial_data.model": "fu-shu36",
        "domain.num_zones": [200, 1, 1],
        "domain.extent_i": [-5.0, 5.0],
        "driver.tfinal": 1.8,
    }


@modeldata
class FuShu37:
    """
    Blast wave interaction

    Adapted from Example 3.7 from G. Fu and C.-W. Shu, "A new trouble-cell
    indicator for discontinuous Galerkin methods for hyperbolic conservation
    laws," Journal of Computational Physics, v347 (2017), pp.305-327.

    Notes:
    - this problem should be run with a reflecting BC at each end
    - use log-scaling to plot the mass density profile
    """

    @property
    def primitive_fields(self):
        "density", "x-velocity", "pressure"

    def primitive(self, box: CoordinateBox):
        x = box.cell_centers()
        l = (x >= 0.0) * (x < 0.1)
        m = (x >= 0.1) * (x < 0.9)
        r = (x >= 0.9) * (x < 1.0)
        p = zeros(x.shape + (3,))
        p[l] = [1.0, 0.0, 1000.0]
        p[m] = [1.0, 0.0, 0.01]
        p[r] = [1.0, 0.0, 100.0]
        return p


@preset
def fu_shu37():
    return {
        "initial_data.model": "fu-shu37",
        "domain.num_zones": [400, 1, 1],
        "domain.extent_i": [0.0, 1.0],
        "driver.tfinal": 0.038,
    }


@modeldata
class DensityWave:
    """
    Sinusoidal density wave translating rigidly
    """

    amplitude: float = 0.2

    @property
    def primitive_fields(self):
        return "density", "x-velocity", "pressure"

    def primitive(self, box: CoordinateBox):
        x = box.cell_centers()
        p = zeros(x.shape + (3,))
        p[..., 0] = 1.0 + self.amplitude * sin(2 * pi * x)
        p[..., 1] = 1.0
        p[..., 2] = 1.0
        return p


@preset
def density_wave():
    return {
        "initial_data.model": "density-wave",
        "domain.num_zones": [400, 1, 1],
        "domain.extent_i": [0.0, 1.0],
        "driver.tfinal": 0.1,
        "boundary_condition": {
            "lower_i": "periodic",
            "upper_i": "periodic",
        },
    }


DefaultModelData = Sod
ModelData = Union[tuple(MODEL_DATA_CLASSES)]
