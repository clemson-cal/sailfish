"""
The `Solver` class maintains physics state and implements a solution scheme.
"""

from abc import ABC, abstractmethod


class SolverBase(ABC):
    """
    Base class for solver implementations.
    """

    @property
    @abstractmethod
    def solution(self):
        """
        Return an object representing the solution state.

        The solution object is generally going to be either an array of
        primitive hydrodynamic data, or of conserved quantities, depending on
        which the solver considers to be more fundamental. If the solver
        maintains state in addition to the hydrodynamic data, such as a list
        of tracer or star particles, then objects like that should also be
        returned as part of the solution object.

        The solution object will be written to the checkpoint file, read back
        on restarts, and passed to the solver constructor. When the run is not
        a restart, the solver should instead construct its initial solution
        object from the setup and mesh instances.
        """
        pass

    @property
    @abstractmethod
    def primitive(self):
        """
        Return primitive hydrodynamic data for use in plotting.

        The primitive variable data returned from this function could be
        written to checkpoint files, but it might also be written to
        non-checkpoint output files for offline analysis, or passed along to
        runtime post-processing routines.
        """
        pass

    @property
    @abstractmethod
    def time(self):
        """
        Return the simulation time.
        """
        pass

    @property
    @abstractmethod
    def maximum_cfl(self):
        """
        Return the largest CFL number that should be used for this solver.
        """
        pass

    @abstractmethod
    def advance(self, dt):
        """
        Advance the solution state by one iteration.
        """
        pass
