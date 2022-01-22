"""
Physics solvers and solution schemes.
"""


class SolverInitializationError(Exception):
    """An invalid runtime configuration"""


def make_solver(name, physics, options, **kwargs):
    from . import srhd_1d
    from . import srhd_2d
    from . import scdg_1d
    from . import cbdgam_2d
    from . import cbdiso_2d

    """
    Find a solver with the given name and construct it.
    """
    try:
        return globals()[name].Solver(
            physics=physics or dict(), options=options or dict(), **kwargs
        )
    except (TypeError, ValueError) as e:
        raise SolverInitializationError(e)
