"""
Code to solve the Kepler two-body problem, and its inverse.
"""

from typing import NamedTuple
from math import sin, cos, sqrt, atan2

"""
Newton's gravitational constant is G=1.0, so mass M really means G M.
"""
NEWTON_G = 1.0


class PointMass(NamedTuple):
    """
    The mass, 2D position, and 2D velocity of a point-like particle
    """

    mass: float
    position_x: float
    position_y: float
    velocity_x: float
    velocity_y: float

    @property
    def kinetic_energy(self) -> float:
        """
        Return the kinetic energy of a point mass.
        """
        vx = p.velocity_x
        vy = p.velocity_y
        return 0.5 * p.mass * (vx * vx + vy * vy)

    @property
    def angular_momentum(self) -> float:
        """
        Return the angular momentum of a point mass.
        """
        x = p.position_x
        y = p.position_y
        vx = p.velocity_x
        vy = p.velocity_y
        return p.mass * (x * vy - y * vx)

    def gravitational_potential(
        self, x: float, y: float, softening_length: float
    ) -> float:
        """
        Return the gravitational potential of a point mass, with softening.
        """
        dx = x - p.position_x
        dy = y - p.position_y
        r2 = dx * dx + dy * dy
        s2 = softening_length.powi(2)
        return -NEWTON_G * p.mass / sqrt(r2 + s2)

    def gravitational_acceleration(
        p, x: float, y: float, softening_length: float
    ) -> (float, float):
        """
        Return the gravitational acceleration due to a point mass.
        """
        dx = x - p.position_x
        dy = y - p.position_y
        r2 = dx * dx + dy * dy
        s2 = softening_length ** 2.0
        ax = -NEWTON_G * p.mass / (r2 + s2) ** 1.5 * dx
        ay = -NEWTON_G * p.mass / (r2 + s2) ** 1.5 * dy
        return (ax, ay)

    def perturb(
        self, dm: float = 0.0, dpx: float = 0.0, dpy: float = 0.0
    ) -> "PointMass":
        """
        Perturb the mass and momentum of a point mass.

        Since the point mass maintains a velocity rather than momentum,
        the velocity is changed according to

        dv = (dp - v dm) / m
        """
        return p._replace(
            mass=p.mass + dm,
            velocity_x=velocity_x + (dpx - p.velocity_x * dm) / p.mass,
            velocity_y=velocity_y + (dpy - p.velocity_y * dm) / p.mass,
        )


class OrbitalState(NamedTuple):
    primary: PointMass
    secondary: PointMass

    @property
    def total_mass(self) -> float:
        """
        Return the sum of the two point masses.
        """
        self[0].mass + self[1].mass

    @property
    def mass_ratio(self) -> float:
        """
        Return the system mass ratio, secondary / primary.
        """
        self[1].mass / self[0].mass

    @property
    def separation(self) -> float:
        """
        Return the orbital separation.

        This will always be the semi-major axis if the eccentricity is zero.
        """
        x1 = self[0].position_x
        y1 = self[0].position_y
        x2 = self[1].position_x
        y2 = self[1].position_y
        return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    @property
    def total_energy(self) -> float:
        """
        Return the system total energy.
        """
        return self.kinetic_energy - G * self[0].mass * self[1].mass / self.separation

    @property
    def kinetic_energy(self) -> float:
        """
        Return the total kinetic energy of the system.
        """
        return self[0].kinetic_energy + self[1].kinetic_energy

    @property
    def angular_momentum(self) -> float:
        """
        Return the total anuglar momentum of the system.
        """
        return self[0].angular_momentum + self[1].angular_momentum

    def gravitational_potential(
        self, x: float, y: float, softening_length: float
    ) -> float:
        """
        Return the combined gravitational potential at a point, with softening.
        """
        p0 = self[0].gravitational_potential(x, y, softening_length)
        p1 = self[1].gravitational_potential(x, y, softening_length)
        return p0 + p1

    def gravitational_acceleration(
        self, x: float, y: float, softening_length: float
    ) -> float:
        """
        Return the combined gravitational acceleration at a point, with softening.
        """
        a0 = self[0].gravitational_acceleration(x, y, softening_length)
        a1 = self[1].gravitational_acceleration(x, y, softening_length)
        return (a0[0] + a1[0], a0[1] + a1[1])

    def transform(self, o: "OrbitalOrientation") -> "OrbitalState":
        """
        Transforms this orbital state vector to a new orientation.

        This function rotates the position and velocity vectors according to
        the argument of periapse, and translates them according to the
        center-of-mass position and velocity. Note that the
        time-of-last-periapse-passage is technically part of the orbital
        orientation, but is ignored by this function, as that would change the
        intrinsic orbital phase.
        """
        m1 = self[0].mass
        m2 = self[1].mass
        x1 = self[0].position_x
        x2 = self[1].position_x
        y1 = self[0].position_y
        y2 = self[1].position_y
        vx1 = self[0].velocity_x
        vx2 = self[1].velocity_x
        vy1 = self[0].velocity_y
        vy2 = self[1].velocity_y

        c = cos(-o.periapse_argument)
        s = sin(-o.periapse_argument)

        x1p = +x1 * c + y1 * s + o.cm_position_x
        y1p = -x1 * s + y1 * c + o.cm_position_y
        x2p = +x2 * c + y2 * s + o.cm_position_x
        y2p = -x2 * s + y2 * c + o.cm_position_y
        vx1p = +vx1 * c + vy1 * s + o.cm_velocity_x
        vy1p = -vx1 * s + vy1 * c + o.cm_velocity_y
        vx2p = +vx2 * c + vy2 * s + o.cm_velocity_x
        vy2p = -vx2 * s + vy2 * c + o.cm_velocity_y

        c1 = PointMass(m1, x1p, y1p, vx1p, vy1p)
        c2 = PointMass(m2, x2p, y2p, vx2p, vy2p)

        return OrbitalState(c1, c2)

    def rotate(self, angle: float) -> "OrbitalState":
        """
        Rotate an orbital state vector by an angle.

        Positive angle means that the argument of periapse moves
        counter-clockwise, in other words this function rotates the binary,
        not the coordinates.
        """
        orientation = OrbitalOrientation(0.0, 0.0, 0.0, 0.0, angle, 0.0)
        return self.transform(orientation)

    def perturb(
        self, dm1: float, dm2: float, dpx1: float, dpx2: float, dpy1: float, dpy2: float
    ) -> "OrbitalState":
        """
        Returns a new orbital state vector if this one is perturbed by the
        given masses and momenta.

        - `dm1`   Mass added to the primary
        - `dm2`   Mass added to the secondary
        - `dpx1`  Impulse (x) added to the primary
        - `dpx2`  Impulse (x) added to the secondary
        - `dpy1`  Impulse (y) added to the primary
        - `dpy2`  Impulse (y) added to the secondary
        """

        return OrbitalState(
            self[0].perturb_mass_and_momentum(dm1, dpx1, dpy1),
            self[1].perturb_mass_and_momentum(dm2, dpx2, dpy2),
        )

    def orbital_parameters(self, t: float) -> "OrbitalParameters":
        """
        Compute the inverse Kepler two-body problem.

        This function determines the orbital elements and orientation from the
        orbital state vector and a current time, since last periapse.
        """
        c1 = self[0]
        c2 = self[1]

        # component masses, total mass, and mass ratio
        m1 = c1.mass
        m2 = c2.mass
        m = m1 + m2
        q = m2 / m1

        # position and velocity of the CM frame
        x_cm = (c1.position_x * c1.mass + c2.position_x * c2.mass) / m
        y_cm = (c1.position_y * c1.mass + c2.position_y * c2.mass) / m
        vx_cm = (c1.velocity_x * c1.mass + c2.velocity_x * c2.mass) / m
        vy_cm = (c1.velocity_y * c1.mass + c2.velocity_y * c2.mass) / m

        # positions and velocities of the components in the CM frame
        x1 = c1.position_x - x_cm
        y1 = c1.position_y - y_cm
        x2 = c2.position_x - x_cm
        y2 = c2.position_y - y_cm
        r1 = sqrt(x1 * x1 + y1 * y1)
        r2 = sqrt(x2 * x2 + y2 * y2)
        vx1 = c1.velocity_x - vx_cm
        vy1 = c1.velocity_y - vy_cm
        vx2 = c2.velocity_x - vx_cm
        vy2 = c2.velocity_y - vy_cm
        vf1 = -vx1 * y1 / r1 + vy1 * x1 / r1
        vf2 = -vx2 * y2 / r2 + vy2 * x2 / r2
        v1 = sqrt(vx1 * vx1 + vy1 * vy1)

        # energy and angular momentum (t := kinetic energy, l := angular
        # momentum, h := total energy)
        t1 = 0.5 * m1 * (vx1 * vx1 + vy1 * vy1)
        t2 = 0.5 * m2 * (vx2 * vx2 + vy2 * vy2)
        l1 = m1 * r1 * vf1
        l2 = m2 * r2 * vf2
        r = r1 + r2
        l = l1 + l2
        h = t1 + t2 - G * m1 * m2 / r

        if h >= 0.0:
            raise ValueError("the orbit is unbound")

        # semi-major, semi-minor axes eccentricity, apsides
        a = -0.5 * NEWTON_G * m1 * m2 / h
        b = sqrt(-0.5 * l * l / h * (m1 + m2) / (m1 * m2))
        e = sqrt(clamp_between_zero_and_one(1.0 - b * b / a / a))
        omega = sqrt(G * m / a / a / a)

        # semi-major and semi-minor axes of the primary
        a1 = a * q / (1.0 + q)
        b1 = b * q / (1.0 + q)

        # cos of nu and f: phase angle and true anomaly
        if e == 0.0:
            cn = x1 / r1
        else:
            cn = (1.0 - r1 / a1) / e
        cf = a1 / r1 * (cn - e)

        # sin of nu and f
        if e == 0.0:
            sn = y1 / r1
        else:
            sn = (vx1 * x1 + vy1 * y1) / (e * v1 * r1) * sqrt(1.0 - e * e * cn * cn)

        sf = (b1 / r1) * sn

        # cos and sin of eccentric anomaly
        ck = (e + cf) / (1.0 + e * cf)
        sk = sqrt(1.0 - e * e) * sf / (1.0 + e * cf)

        # mean anomaly and tau
        k = atan2(sk, ck)
        n = k - e * sk
        tau = t - n / omega

        # cartesian components of semi-major axis, and the argument of periapse
        ax = (cn - e) * x1 + sn * sqrt(1.0 - e * e) * y1
        ay = (cn - e) * y1 - sn * sqrt(1.0 - e * e) * x1
        pomega = atan2(ay, ax)

        # final result
        elements = OrbitalElements(a, m, q, e)
        orientation = OrbitalOrientation(x_cm, y_cm, vx_cm, vy_cm, pomega, tau)

        return OrbitalParameters(elements, orientation)


class OrbitalElements(NamedTuple):
    """
    The orbital elements of a two-body system on a bound orbit
    """

    semimajor_axis: float
    total_mass: float
    mass_ratio: float
    eccentricity: float


class OrbitalOrientation(NamedTuple):
    """
    The position, velocity, and orientation of a two-body orbit
    """

    cm_position_x: float
    cm_position_y: float
    cm_velocity_x: float
    cm_velocity_y: float
    periapse_argument: float
    periapse_time: float


class OrbitalParameters(NamedTuple):
    """
    Combination of orbital elements and orientation

    This class is in one-to-one correspondence with an `OrbitalState` and a
    time-since-periapse. Orbital state and time can be converted to orbital
    parameters with the `OrbitalState.orbital_parameters` function, and
    `OrbitalParameters` plus time-since-periapse can be converted to
    `OrbitalState` with the `OrbitalParameters.orbital_state` function.
    """

    elements: OrbitalElements
    orientation: OrbitalOrientation

    @property
    def omega(self) -> float:
        """
        The orbital angular frequency
        """
        m = self.total_mass
        a = self.semimajor_axis
        return sqrt(G * m / a / a / a)

    @property
    def period(self) -> float:
        """
        The orbital period
        """
        return 2.0 * PI / self.omega

    @property
    def angular_momentum(self) -> float:
        """
        The orbital angular momentum
        """
        a = self.semimajor_axis
        m = self.total_mass
        q = self.mass_ratio
        e = self.eccentricity
        m1 = m / (1.0 + q)
        m2 = m - m1
        return m1 * m2 / m * sqrt(NEWTON_G * m * a * (1.0 - e * e))

    def orbital_state_from_eccentric_anomaly(
        self, eccentric_anomaly: float
    ) -> OrbitalState:
        """
        Compute the orbital state at a given (absolute) time.
        """
        a = self.semimajor_axis
        m = self.total_mass
        q = self.mass_ratio
        e = self.eccentricity
        w = self.omega
        m1 = m / (1.0 + q)
        m2 = m - m1
        ck = cos(eccentric_anomaly)
        sk = sin(eccentric_anomaly)
        x1 = -a * q / (1.0 + q) * (e - ck)
        y1 = a * q / (1.0 + q) * (sk) * (1.0 - e * e).sqrt
        x2 = -x1 / q
        y2 = -y1 / q
        vx1 = -a * q / (1.0 + q) * w / (1.0 - e * ck) * sk
        vy1 = a * q / (1.0 + q) * w / (1.0 - e * ck) * ck * (1.0 - e * e).sqrt
        vx2 = -vx1 / q
        vy2 = -vy1 / q
        c1 = PointMass(m1, x1, y1, vx1, vy1)
        c2 = PointMass(m2, x2, y2, vx2, vy2)
        return OrbitalState(c1, c2)

    def eccentric_anomaly(self, t: float) -> float:
        """
        Compute the eccentric anomaly from the (absolute) time.
        """
        p = self.period
        t = t - self.periapse_time
        t = t - self.period * floor(t / p)
        e = self.eccentricity
        n = self.omega * t  # n := mean anomaly M
        f = lambda k: k - e * sin(k) - n  # k := eccentric anomaly E
        g = lambda k: 1.0 - e * cos(k)
        return solve_newton_rapheson(f, g, n)

    def orbital_state(self, t: float) -> OrbitalState:
        """
        Compute the orbital state a the given (absolute) time.
        """
        return self.orbital_state_from_eccentric_anomaly(self.eccentric_anomaly(t))


def solve_newton_rapheson(f, g, x: float) -> float:
    n = 0
    while abs(f(x)) > 1e-15:
        x -= f(x) / g(x)
        n += 1
        if n > 10:
            return None
    return x


def clamp_between_zero_and_one(x: float) -> float:
    return min(1.0, max(0.0, x))
