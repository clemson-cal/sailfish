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
from numpy import logical_not, zeros, sqrt, sin, cos, pi
from preset import preset
from schema import schema
from geometry import CoordinateBox


@schema
class Shocktube:
    """
    Classic Sod shocktube initial data

    Note: this model should be specialized to 1d; a version generalized for
    1d/2d/3d should be added in a new models section.
    """

    model: Literal["shocktube"] = "shocktube"

    def primitive(self, box: CoordinateBox):
        if box.dimensionality != 1:
            raise NotImplementedError("model only works in 1d")
        x = box.cell_centers()
        l = x < 0.5
        r = logical_not(l)
        p = zeros(x.shape + (3,))
        p[l] = [1.0, 0.0, 1.000]
        p[r] = [0.1, 0.0, 0.125]
        return p


@schema
class CylindricalExplosion:
    """
    Cylindrical explosion initial data

    Initializes a circular region of high pressure and density. A shock wave
    and contact discontinuity should expand outward with a circular profile.
    """

    model: Literal["cylindrical-explosion"] = "cylindrical-explosion"

    def primitive(self, box: CoordinateBox):
        if box.dimensionality != 2:
            raise NotImplementedError("model only works in 2d")
        x, y = box.cell_centers()
        l = sqrt(x**2 + y**2) < 0.1
        r = logical_not(l)
        p = zeros(x.shape + (4,))
        p[l] = [1.0, 0.0, 0.0, 1.000]
        p[r] = [0.1, 0.0, 0.0, 0.125]
        return p


@preset
def cylindrical_explosion():
    return {
        "initial_data.model": "cylindrical-explosion",
        "domain.num_zones": [200, 200, 1],
        "domain.extent_i": [-0.5, 0.5],
        "domain.extent_j": [-0.5, 0.5],
        "driver.tfinal": 0.1,
    }


@schema
class CylinderInWind:
    """
    A round cylinder immersed in a dilute wind

    A circular region of high density is immersed in a low-density ambient
    medium moving from left to right. After some time, the high-density region
    is disrupted by Kelvin-Helmholtz instabilities.
    """

    model: Literal["cylinder-in-wind"] = "cylinder-in-wind"

    def primitive(self, box: CoordinateBox):
        if box.dimensionality != 2:
            raise NotImplementedError("model only works in 2d")
        x, y = box.cell_centers()
        l = sqrt(x**2 + y**2) < 0.1
        r = logical_not(l)
        p = zeros(x.shape + (4,))
        p[l] = [1e2, 0.0, 0.0, 1.0]
        p[r] = [1.0, 1.0, 0.0, 1.0]
        return p


@preset
def cylinder_in_wind():
    return {
        "initial_data.model": "cylinder-in-wind",
        "domain.num_zones": [200, 200, 1],
        "domain.extent_i": [-0.25, 0.75],
        "domain.extent_j": [-0.50, 0.50],
        "driver.tfinal": 1.0,
    }


@schema
class FuShu33:
    """
    Lax problem initial data

    Adapted from Example 3.3 from G. Fu and C.-W. Shu, "A new trouble-cell
    indicator for discontinuous Galerkin methods for hyperbolic conservation
    laws," Journal of Computational Physics, v347 (2017), pp.305-327.
    """

    model: Literal["fu-shu-33"] = "fu-shu-33"

    def primitive(self, box: CoordinateBox):
        if box.dimensionality != 1:
            raise NotImplementedError("model only works in 1d")
        x = box.cell_centers()
        l = x < 0.0
        r = logical_not(l)
        p = zeros(x.shape + (3,))
        p[l] = [0.445, 0.698, 3.528]
        p[r] = [0.500, 0.000, 0.571]
        return p


@preset
def fu_shu_33():
    return {
        "initial_data.model": "fu-shu-33",
        "domain.num_zones": [200, 1, 1],
        "domain.extent_i": [-5.0, 5.0],
        "driver.tfinal": 1.3,
    }


@schema
class FuShu34:
    """
    Lax problem: double rarefaction wave

    Adapted from Example 3.4 from G. Fu and C.-W. Shu, "A new trouble-cell
    indicator for discontinuous Galerkin methods for hyperbolic conservation
    laws," Journal of Computational Physics, v347 (2017), pp.305-327.
    """

    model: Literal["fu-shu-34"] = "fu-shu-34"

    def primitive(self, box: CoordinateBox):
        if box.dimensionality != 1:
            raise NotImplementedError("model only works in 1d")
        x = box.cell_centers()
        l = x < 0.0
        r = logical_not(l)
        p = zeros(x.shape + (3,))
        p[l] = [7.0, -1.0, 0.2]
        p[r] = [7.0, +1.0, 0.2]
        return p


@preset
def fu_shu_34():
    return {
        "initial_data.model": "fu-shu-34",
        "domain.num_zones": [200, 1, 1],
        "domain.extent_i": [-1.0, 1.0],
        "driver.tfinal": 0.6,
    }


@schema
class FuShu35:
    """
    LeBlanc problem

    Adapted from Example 3.5 from G. Fu and C.-W. Shu, "A new trouble-cell
    indicator for discontinuous Galerkin methods for hyperbolic conservation
    laws," Journal of Computational Physics, v347 (2017), pp.305-327.

    Note: use log-scaling to plot the mass density profile.
    """

    model: Literal["fu-shu-35"] = "fu-shu-35"

    def primitive(self, box: CoordinateBox):
        if box.dimensionality != 1:
            raise NotImplementedError("model only works in 1d")
        x = box.cell_centers()
        l = x < 0.0
        r = logical_not(l)
        p = zeros(x.shape + (3,))
        p[l] = [1.00, 0.0, 2.0 / 3.0 * 1e-1]
        p[r] = [1e-3, 0.0, 2.0 / 3.0 * 1e-10]
        return p


@preset
def fu_shu_35():
    return {
        "initial_data.model": "fu-shu-35",
        "domain.num_zones": [600, 1, 1],
        "domain.extent_i": [-3.0, 6.0],
        "driver.tfinal": 6.0,
    }


@schema
class FuShu36:
    """
    Shu-Osher problem

    Adapted from Example 3.6 from G. Fu and C.-W. Shu, "A new trouble-cell
    indicator for discontinuous Galerkin methods for hyperbolic conservation
    laws," Journal of Computational Physics, v347 (2017), pp.305-327.
    """

    model: Literal["fu-shu-36"] = "fu-shu-36"

    def primitive(self, box: CoordinateBox):
        if box.dimensionality != 1:
            raise NotImplementedError("model only works in 1d")
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


@schema
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

    def primitive(self, box: CoordinateBox):
        if box.dimensionality != 1:
            raise NotImplementedError("model only works in 1d")
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


@schema
class DensityWave:
    """
    Sinusoidal density wave translating rigidly
    """

    model: Literal["density-wave"] = "density-wave"

    def primitive(self, box: CoordinateBox):
        if box.dimensionality != 1:
            raise NotImplementedError("model only works in 1d")
        x = box.cell_centers()
        p = zeros(x.shape + (3,))
        p[..., 0] = 1.0 + 0.2 * sin(2 * pi * x)
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


InitialData = Union[
    Shocktube,
    CylindricalExplosion,
    CylinderInWind,
    FuShu33,
    FuShu34,
    FuShu35,
    FuShu36,
    FuShu37,
    DensityWave,
]
