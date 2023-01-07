"""
Physics solvers and solution schemes.
"""


class SolverInitializationError(Exception):
    """An invalid runtime configuration"""


__solver_extension_modules = list()


def register_solver_extension(solver_name):
    """ """
    __solver_extension_modules.append(solver_name)


def make_solver(name, physics, options, **kwargs):
    """
    Find a solver with the given name and construct it.
    """
    from importlib import import_module
    from . import srhd_1d
    from . import srhd_2d
    from . import scdg_1d
    from . import cbdgam_2d
    from . import cbdiso_2d
    from . import cbdisodg_2d

    solvers = dict(
        srhd_1d=srhd_1d,
        srhd_2d=srhd_2d,
        scdg_1d=scdg_1d,
        cbdgam_2d=cbdgam_2d,
        cbdiso_2d=cbdiso_2d,
        cbdisodg_2d=cbdisodg_2d,
    )
    for ext_name in __solver_extension_modules:
        solvers[ext_name] = import_module(ext_name)

    try:
        return solvers[name].Solver(
            physics=physics or dict(), options=options or dict(), **kwargs
        )
    except (TypeError, ValueError) as e:
        raise SolverInitializationError(e)
