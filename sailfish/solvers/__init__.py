"""
Physics solvers and solution schemes.
"""

from . import srhd_1d
from . import scdg_1d


def make_solver(name, *args, **kwargs):
    """
    Find a solver with the given name and construct it.
    """
    return globals()[name].Solver(*args, **kwargs)
