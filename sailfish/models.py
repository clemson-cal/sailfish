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


def modeldata(f):
    MODEL_DATA_CLASSES.append(f)
    return schema(f)


@modeldata
class Sod:
    """
    Classic Sod shocktube initial data
    """

    model: Literal["sod"] = "sod"

    @property
    def dimensionality(self):
        return 1

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

    model: Literal["uniform"] = "uniform"
    dimensionality: int = 1
    coordinates: str = "cartesian"

    @property
    def primitive_fields(self):
        if self.dimensionality == 1:
            return "density", "i-velocity", "pressure"
        if self.dimensionality == 2:
            return "density", "i-velocity", "j-velocity", "pressure"

    def primitive(self, box: CoordinateBox):
        if self.dimensionality == 1:
            p = zeros(box.num_zones + (3,))
            p[...] = [1.0, 0.0, 1.0]
        if self.dimensionality == 2:
            p = zeros(box.num_zones + (4,))
            p[...] = [1.0, 0.0, 0.0, 1.0]
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
        "initial_data.dimensionality": 1,
        "domain.num_zones": [200, 1, 1],
        "domain.extent_i": [1.0, 10.0],
        "coordinates": "cylindrical-polar",
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
        "initial_data.dimensionality": 2,
        "domain.num_zones": [200, 200, 1],
        "domain.extent_i": [1.0, 10.0],
        "domain.extent_j": [0.0, pi],
        # "domain.extent_j": [0.0, 9.0],
        "coordinates": "spherical-polar",
        "driver.tfinal": 0.1,
    }


@modeldata
class IsothermalVortex:
    """
    An isothermal vortex
    """

    model: Literal["isothermal-vortex"] = "isothermal-vortex"
    mach_number: float = 1.0

    @property
    def dimensionality(self):
        return 1

    @property
    def primitive_fields(self):
        return "density", "r-velocity", "z-velocity", "f-velocity", "pressure"

    def primitive(self, box: CoordinateBox):
        r = box.cell_centers()
        p = zeros(box.num_zones + (5,))
        r0 = 1.0  # radius of vortex core
        cs = 1.0  # nominal sound speed
        omega0 = self.mach_number * cs / r0
        omega = omega0 * exp(-0.5 * r**2 / r0**2)
        rho0 = 1.0
        rho = rho0 * exp(-0.5 * self.mach_number**2 * exp(-(r**2 / r0**2)))
        p[:, 0, 0, 0] = rho
        p[:, 0, 0, 1] = 0.0
        p[:, 0, 0, 2] = 0.0
        p[:, 0, 0, 3] = omega * r
        p[:, 0, 0, 4] = rho * cs**2
        return p


@preset
def isothermal_vortex():
    return {
        "initial_data.model": "isothermal-vortex",
        "domain.num_zones": [200, 1, 1],
        "domain.extent_i": [1.0, 10.0],
        "coordinates": "cylindrical-polar",
        "driver.tfinal": 0.1,
    }


@modeldata
class CylindricalExplosion:
    """
    Cylindrical explosion initial data

    Initializes a circular region of high pressure and density. A shock wave
    and contact discontinuity should expand outward with a circular profile.
    """

    model: Literal["cylindrical-explosion"] = "cylindrical-explosion"

    @property
    def dimensionality(self):
        return 2

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

    model: Literal["cylinder-in-wind"] = "cylinder-in-wind"

    @property
    def dimensionality(self):
        return 2

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

    model: Literal["ram-41"] = "ram-41"

    @property
    def dimensionality(self):
        return 1

    @property
    def primitive_fields(self):
        return "proper-density", "x-gamma-beta", "pressure"

    def primitive(self, box: CoordinateBox):
        return two_state(box.cell_centers() < 0.5, [10.0, 0.0, 13.33], [1.0, 0.0, 1e-8])


@preset
def ram_41():
    return {
        "initial_data.model": "ram-41",
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

    model: Literal["ram-42"] = "ram-42"

    @property
    def primitive_fields(self):
        return "proper-density", "x-gamma-beta", "pressure"

    @property
    def dimensionality(self):
        return 1

    def primitive(self, box: CoordinateBox):
        return two_state(box.cell_centers() < 0.5, [1.0, 0.0, 1000.0], [1.0, 0.0, 1e-2])


@preset
def ram_42():
    return {
        "initial_data.model": "ram-42",
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

    model: Literal["ram-43"] = "ram-43"

    @property
    def dimensionality(self):
        return 1

    @property
    def primitive_fields(self):
        return "proper-density", "x-gamma-beta", "pressure"

    def primitive(self, box: CoordinateBox):
        v = 0.9
        u = v / sqrt(1.0 - v * v)
        x = box.cell_centers()
        return two_state(x < 0.5, [1.0, u, 1.0], [1.0, 0.0, 10.0])


@preset
def ram_43():
    return {
        "initial_data.model": "ram-43",
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

    model: Literal["ram-44"] = "ram-44"

    @property
    def dimensionality(self):
        return 1

    @property
    def primitive_fields(self):
        return "proper-density", "x-gamma-beta", "y-gamma-beta", "pressure"

    def primitive(self, box: CoordinateBox):
        v = 0.99
        u = v / sqrt(1.0 - v * v)
        x = box.cell_centers()
        return two_state(x < 0.5, [1.0, 0.0, 0.0, 1e3], [1.0, 0.0, u, 1e-2])


@preset
def ram_44():
    return {
        "initial_data.model": "ram-44",
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

    model: Literal["ram-45"] = "ram-45"

    @property
    def dimensionality(self):
        return 1

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
def ram_45():
    return {
        "initial_data.model": "ram-45",
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

    model: Literal["ram-61"] = "ram-61"

    @property
    def dimensionality(self):
        return 1

    @property
    def primitive_fields(self):
        return "proper-density", "x-gamma-beta", "y-gamma-beta", "pressure"

    def primitive(self, box: CoordinateBox):
        v = 0.9
        u = v / sqrt(1.0 - v * v)
        x = box.cell_centers()
        return two_state(x < 0.5, [1.0, 0.0, u, 1e3], [1.0, 0.0, u, 1e-2])


@preset
def ram_61():
    return {
        "initial_data.model": "ram-61",
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

    model: Literal["fu-shu-33"] = "fu-shu-33"

    @property
    def dimensionality(self):
        return 1

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
def fu_shu_33():
    return {
        "initial_data.model": "fu-shu-33",
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

    model: Literal["fu-shu-34"] = "fu-shu-34"

    @property
    def dimensionality(self):
        return 1

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
def fu_shu_34():
    return {
        "initial_data.model": "fu-shu-34",
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

    model: Literal["fu-shu-35"] = "fu-shu-35"

    @property
    def dimensionality(self):
        return 1

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
def fu_shu_35():
    return {
        "initial_data.model": "fu-shu-35",
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

    model: Literal["fu-shu-36"] = "fu-shu-36"

    @property
    def dimensionality(self):
        return 1

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
def fu_shu_36():
    return {
        "initial_data.model": "fu-shu-36",
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

    model: Literal["fu-shu-37"] = "fu-shu-37"

    @property
    def dimensionality(self):
        return 1

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
def fu_shu_37():
    return {
        "initial_data.model": "fu-shu-37",
        "domain.num_zones": [400, 1, 1],
        "domain.extent_i": [0.0, 1.0],
        "driver.tfinal": 0.038,
    }


@modeldata
class DensityWave:
    """
    Sinusoidal density wave translating rigidly
    """

    model: Literal["density-wave"] = "density-wave"
    amplitude: float = 0.2

    @property
    def dimensionality(self):
        return 1

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
