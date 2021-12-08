"""
Physics solvers and solution schemes.
"""

from . import srhd_1d
from . import scdg_1d


def make_solver(name, options, *args, **kwargs):
    """
    Find a solver with the given name and construct it.
    """
    kwargs.update(options)
    return globals()[name].Solver(*args, **kwargs)
