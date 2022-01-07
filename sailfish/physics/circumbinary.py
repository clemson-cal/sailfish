from typing import NamedTuple, List, Callable


SINK_MODEL_INACTIVE = 0
SINK_MODEL_ACCELERATION_FREE = 1
SINK_MODEL_TORQUE_FREE = 2
SINK_MODEL_FORCE_FREE = 3

EOS_TYPE_GLOBALLY_ISOTHERMAL = 1
EOS_TYPE_LOCALLY_ISOTHERMAL = 2
EOS_TYPE_GAMMA_LAW = 3


class PointMass(NamedTuple):
    r"""
    Describes a gravitating point mass

    The point mass's mass, x-y position, and softening length fully determine
    the gravitational potential:

    .. math::
        \phi = \frac{G M}{\sqrt{\Delta r^2 + r_{\rm soft}^2}} \, .

    Here, :math:`\Delta r` is the distance from the particle to the field
    point, and :math:`r_{\rm soft}` is the :obj:`softening_length` parameter
    below. Note that the gravitational constant is assumed to be 1 by the
    physics solvers.

    Point masses can also act as sinks of mass and momentum. This behavior is
    controlled by the sink model, sink rate, and a sink radius. If the sink
    model has the value :obj:`SINK_MODEL_INACTIVE`, then that particle may
    still be a source of gravitational potential, but it does model accretion
    by subtracting mass or momentum. The acceleration-free and torque-free
    modes are described in literature here:

    - `Dittmann & Ryan (2021) <https://ui.adsabs.harvard.edu/abs/2021arXiv210205684D>`_
    - `Dempsey et. al (2020) <https://ui.adsabs.harvard.edu/abs/arXiv:2002.05164>`_

    The acceleration-free sink model is there for logical completeness, but it
    should not be used in practice.
    """

    mass: float = 0.0
    """ The mass (really G * M since solvers assume G = 1) """

    position_x: float = 0.0
    """ The x-position """

    position_y: float = 0.0
    """ The y-position """

    velocity_x: float = 0.0
    """ The x-velocity """

    velocity_y: float = 0.0
    """ The y-velocity """

    softening_length: float = 0.0
    """ Gravitational softening length """

    sink_model: int = SINK_MODEL_INACTIVE
    """ The equation used to control how momentum is subtracted """

    sink_rate: float = 0.0
    """ The sink rate: how fast mass and momentum are removed """

    sink_radius: float = 0.0
    """ The sink radius: how far from the particle the sink extends """


class Physics(NamedTuple):
    """
    Physics configuration for the binary accretion solvers

    Configuration categories are:

    1. Equation of state: globally isothermal, locally isothermal, or
       gamma-law

       In globally isothermal mode, a uniform sound speed is specified. In
       globally isothermal mode, an orbital mach number is specified, and used
       to assign the sound speed based on the local gravitational potential
       (this option only makes sense when gravitating point masses are
       provided). In gamma-law mode, the sound speed is determined
       self-consistently from the internal energy. The cbdiso_2d solver only
       supports the former two modes, and the cbdgam_2d solver only upports
       the last mode.

    2. Gravitating point masses

       Point masses can be optionally provided to model stars or black holes.
       The point mass properties include a point mass's mass, and its x-y
       position. These properties fully determine the gravitational potential
       sourced by the particle. Point masses can also act as sinks of mass and
       momentum (see the :obj:`PointMass` struct above for details). The point
       masses are supplied to the solver implicitly through a callback
       function, mapping the simulation time to a sequence of particles.
       Currently, solvers support either zero, one, or two particles.

    3. Viscosity model

       Two different viscosity models are nominally supported: constant-nu,
       and constant alpha. Currently the isothermal solver only supports
       constant-nu viscosity.

    4. Thermal cooling

       Todo.

    5. An outer buffer zone

       For binary accretion problems in a square domain, it can be useful to
       impose a wave-damping zone (or sponge layer) to avoid artifacts fromt
       the outer boundary on the orbiting gas. The buffer zone, if enabled,
       will drive the solution outside a "buffer onset radius" toward an
       appropriate hydrodynamic state. This should be gas with uniform gas
       pressure and density (which may be sampled from the initial condition),
       and that gas should be on circular Keplerian orbits. The
       :obj:`buffer_driving_rate` parameter controls how fast the solution is
       driven toward the smooth one. The buffer is applied in a region
       between the domain radius (the half-width of a square domain), extending
       inwards by an amount specified by the :obj:`buffer_onset_width`
       parameter.

    """

    eos_type: int = EOS_TYPE_GLOBALLY_ISOTHERMAL
    """ EOS type: globally or locally isothermal """

    sound_speed: float = 1.0
    """ Isothermal sound speed, if EOS type is globally isothermal """

    mach_number: float = 10.0
    """ Square of the Mach number, if EOS type is locally isothermal """

    gamma_law_index: float = 5.0 / 3.0
    """ Adiabatic index, if the EOS type is globally isothermal """

    viscosity_coefficient: float = 0.01
    """ Kinematic viscosity value, in units of a^2 Omega """

    buffer_is_enabled: bool = False
    """ Whether the buffer zone is enabled """

    buffer_driving_rate: float = 1000.0
    """ Rate of driving toward target solution in the buffer region """

    buffer_onset_width: float = 0.1
    """ Distance over which the buffer ramps up """

    point_mass_function: Callable[float, List[PointMass]] = None
    """ Callback function to supply point masses as a function of time """

    def point_masses(self, time):
        """
        Generate point masses from the simulation time and supplied callback.
        """
        if self.point_mass_function is None:
            return PointMass(), PointMass()

        masses = self.point_mass_function(time)

        if masses is None:
            return PointMass(), PointMass()

        if isinstance(masses, PointMass):
            return masses, PointMass()

        if isinstance(masses, tuple) or isinstance(masses, list):
            if len(masses) == 1:
                return masses[0], PointMass()
            if len(masses) == 2:
                return masses

        raise ValueError(
            "point_mass_function returned an unsupported description of point masses"
        )
