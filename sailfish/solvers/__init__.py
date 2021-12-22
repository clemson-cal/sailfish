"""
Physics solvers and solution schemes.
"""

from . import srhd_1d
from . import scdg_1d
from . import cbdgam_2d


def make_solver(name, physics, options, **kwargs):
    """
    Find a solver with the given name and construct it.
    """
    return globals()[name].Solver(
        physics=physics or dict(), options=options or dict(), **kwargs
    )
