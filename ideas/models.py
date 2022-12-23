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
from schema import schema
from geometry import CoordinateBox
from numpy import logical_not, zeros, sqrt, sin, cos, pi


@schema
class Shocktube:
    """
    A linear shocktube setup
    """

    model: Literal["shocktube"] = "shocktube"

    def primitive(self, box: CoordinateBox):
        if box.dimensionality == 1:
            x = box.cell_centers()
            l = x < 0.5
            r = logical_not(l)
            p = zeros(x.shape + (3,))
            p[l] = [1.0, 0.0, 1.000]
            p[r] = [0.1, 0.0, 0.125]

        if box.dimensionality == 2:
            x, y = box.cell_centers()
            angle = 0.0
            a = cos(0.5 * angle * pi)
            b = sin(0.5 * angle * pi)
            l = a * x + b * y < 0.5
            r = logical_not(l)
            p = zeros(x.shape + (4,))
            p[l] = [1.0, 0.0, 0.0, 1.000]
            p[r] = [0.1, 0.0, 0.0, 0.125]

        return p


@schema
class CylindricalExplosion:
    """
    A cylindrical explosion setup
    """

    model: Literal["cylindrical-explosion"] = "cylindrical-explosion"

    def primitive(self, box: CoordinateBox):
        if box.dimensionality != 2:
            raise NotImplementedError("setup only works in 2d")

        x, y = box.cell_centers()
        l = sqrt(x**2 + y**2) < 0.1
        r = logical_not(l)
        p = zeros(x.shape + (4,))
        p[l] = [1.0, 0.0, 0.0, 1.000]
        p[r] = [0.1, 0.0, 0.0, 0.125]

        return p


@schema
class CylinderInWind:
    """
    A round cylinder immersed in a dilute wind
    """

    model: Literal["cylinder-in-wind"] = "cylinder-in-wind"

    def primitive(self, box: CoordinateBox):
        if box.dimensionality != 2:
            raise NotImplementedError("setup only works in 2d")

        x, y = box.cell_centers()
        l = sqrt(x**2 + y**2) < 0.1
        r = logical_not(l)
        p = zeros(x.shape + (4,))

        p[l] = [1e2, 0.0, 0.0, 1.0]
        p[r] = [1.0, 1.0, 0.0, 1.0]

        return p


@schema
class FuShu33:
    """
    Lax problem. Run until t=0.13. Adapted from Example 3.3 from
    G. Fu and C.-W. Shu, "A new trouble-cell indicator for discontinuous Galerkin methods for 
    hyperbolic conservation laws," Journal of Computational Physics, v347 (2017), pp.305-327.
    """

    model: Literal["fu-shu-33"] = "fu-shu-33"

    def primitive(self, box: CoordinateBox):
        if box.dimensionality == 1:
            x = box.cell_centers()
            l = x < 0.5
            r = logical_not(l)
            p = zeros(x.shape + (3,))
            p[l] = [0.445, 0.698, 3.528]
            p[r] = [0.5, 0.0, 0.571]

        if box.dimensionality == 2:
            x, y = box.cell_centers()
            angle = 0.0
            a = cos(0.5 * angle * pi)
            b = sin(0.5 * angle * pi)
            l = a * x + b * y < 0.5
            r = logical_not(l)
            p = zeros(x.shape + (4,))
            p[l] = [0.445, 0.698, 3.528]
            p[r] = [0.5, 0.0, 0.571]

        return p


@schema
class FuShu34:
    """
    Lax problem

    Run until t=0.3. Adapted from Example 3.4 from G. Fu and C.-W. Shu, "A new
    trouble-cell indicator for discontinuous Galerkin methods for hyperbolic
    conservation laws," Journal of Computational Physics, v347 (2017),
    pp.305-327.
    """

    model: Literal["fu-shu-34"] = "fu-shu-34"

    def primitive(self, box: CoordinateBox):
        if box.dimensionality != 1:
            raise NotImplementedError("model only works in 1d")
        x = box.cell_centers()
        l = x < 0.5
        r = logical_not(l)
        p = zeros(x.shape + (3,))
        p[l] = [7.0, -1.0, 0.2]
        p[r] = [7.0, 1.0, 0.2]
        return p


@schema
class FuShu35:
    """
    LeBlanc problem

    Run until t=1.0. Adapted from Example 3.5 from G. Fu and C.-W. Shu, "A new
    trouble-cell indicator for discontinuous Galerkin methods for hyperbolic
    conservation laws," Journal of Computational Physics, v347 (2017),
    pp.305-327.
    """

    model: Literal["fu-shu-35"] = "fu-shu-35"

    def primitive(self, box: CoordinateBox):
        if box.dimensionality != 1:
            raise NotImplementedError("model only works in 1d")
        x = box.cell_centers()
        l = x < 1.0 / 3.0
        r = logical_not(l)
        p = zeros(x.shape + (3,))
        p[l] = [1.0, 0.0, 0.2 / 3.0]
        p[r] = [1e-3, 0.0, 2.0 / 3.0 * 1e-10]
        return p


@schema
class FuShu36:
    """
    Shu-Osher problem

    Run until t=0.18. Adapted from Example 3.6 from G. Fu and C.-W. Shu, "A
    new trouble-cell indicator for discontinuous Galerkin methods for
    hyperbolic conservation laws," Journal of Computational Physics, v347
    (2017), pp.305-327.
    """

    model: Literal["fu-shu-36"] = "fu-shu-36"

    def primitive(self, box: CoordinateBox):
        if box.dimensionality != 1:
            raise NotImplementedError("model only works in 1d")
        x = box.cell_centers()
        p = zeros(x.shape + (3,))
        l = (x >= 0.0) * (x < 0.1)
        r = (x >= 0.1) * (x < 1.0)
        p[l] = [3.857143, 2.629369, 10.333333]
        p[r, 0] = 1.0 + 0.2 * sin(50.0 * x[r])
        p[r, 1] = 0.0
        p[r, 2] = 1.0
        return p


@schema
class FuShu37:
    """
    Shu-Osher problem

    Run until t=0.38. Adapted from Example 3.7 from G. Fu and C.-W. Shu, "A
    new trouble-cell indicator for discontinuous Galerkin methods for
    hyperbolic conservation laws," Journal of Computational Physics, v347
    (2017), pp.305-327.
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


InitialData = Union[
    Shocktube,
    CylindricalExplosion,
    CylinderInWind,
    FuShu33,
    FuShu34,
    FuShu35,
    FuShu36,
    FuShu37,
]
