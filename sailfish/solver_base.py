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
    def options(self) -> dict:
        """
        A dictionary that reflects the solver solution scheme.

        An example of an options object would be :py:obj:`dict(plm=1.5,
        rk=2)`. The items should describe the solver algorithm, not the
        parameters of the physical system being solved, such as an equation of
        state or cooling prescription.

        Solver options and physics parameters are kept logically separate
        because their configuration comes from different places: the physics
        parameters come from the :py:obj:`sailfish.setup.Setup` class instance
        (which can in turn be set from model parameters), whereas the solver
        options come directly from the driver instance (i.e. command line
        :code:`--solver key=val` if sailfish is invoked as an executable).
        """
        pass

    @property
    @abstractmethod
    def physics(self) -> dict:
        """
        A dictionary describing the physics parameters of the system.

        The physics parameters are supplied by the setup. They can include
        things like an equation of state, a cooling prescription, external
        gravity, a wavespeed in the case of passive scalar advection, fluid
        viscocity model, etc. These items can still be influenced at runtime
        through the setup's model parameters.
        """
        pass

    @property
    def recommended_cfl(self):
        """
        Return a recommended CFL number to the driver, default to the maximum.
        """
        return self.maximum_cfl

    @property
    @abstractmethod
    def maximum_cfl(self):
        """
        Return the largest CFL number that should be used for this solver.
        """
        pass

    @abstractmethod
    def maximum_wavespeed(self):
        """
        Return the largest wavespeed on the grid.

        This function is not implemented as a property, to emphasize that it
        might be relatively expensive to compute.
        """
        pass

    @abstractmethod
    def advance(self, dt):
        """
        Advance the solution state by one iteration.
        """
        pass

    def reductions(self):
        """
        Return a set of measurements derived from the solution state.

        Solvers do not need to implement this. If they do, the diagnostic
        outputs they will return when this function is called should be provided
        to the solver by the setup when the setup is first constructed.
        """
        pass
