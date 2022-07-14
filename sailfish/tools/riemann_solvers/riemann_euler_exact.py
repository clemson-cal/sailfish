#!/usr/bin/env python3

"""
Calculate exact solution to the Riemann problem for the Euler equations.
"""

from scipy import optimize
from matplotlib import pyplot as plt
import math
import numpy as np

g = 7.0 / 5.0
ge = -2.0 * g / (g - 1.0)

time = 0.4

# Left and right States primitive variables (rho, u, p)

priml = [1.0, 0.0, 1.0]
primr = [0.125, 0.0, 0.1]

# left state
u4 = priml[1]
p4 = priml[2]
a4 = math.sqrt(g * priml[2] / priml[0])

# right state
u1 = primr[1]
p1 = primr[2]
a1 = math.sqrt(g * primr[2] / primr[0])

# pressure ratio across initial discontinuity
p41 = priml[2] / primr[2]

# Riemann problem - high pressure on left, low pressure on right
# 4 states separated by Head (H) and tail (T) of rarefaction fan
# Contact discontinuity (CD) and shock (S)
# H and T move left, CD and S move right
# ---------------------
# 4 |||| 3  | 2 | 1
# ---------------------
#   H  T    CD  S
# see Laney, C.B. "Computational Gas Dynamics" Eq. 5.6

# Solve for the pressure ratio p21 across the shock connecting states 2 and 1


def f(p21):
    return (
        p21
        * (
            1.0
            + (g - 1.0)
            / (2.0 * a4)
            * (
                u4
                - u1
                - a1
                / g
                * (p21 - 1.0)
                / math.sqrt((g + 1.0) / (2.0 * g) * (p21 - 1.0) + 1.0)
            )
        )
        ** (-2.0 * g / (g - 1.0))
        - p41
    )


# use root finder (Secant method)
p21_0 = 0.5 * p41
# p21 = optimize.newton(f, p21_0)
result = optimize.newton(f, p21_0, full_output=True)
p21 = result[0]
print("Root finder # of iterations: ", result[1].iterations)

# state 2
p2 = p21 * p1
a2 = a4 * math.sqrt(
    p21 * (((g + 1.0) / (g - 1.0)) + p21) / (1.0 + (g + 1.0) / (g - 1.0) * p21)
)
u2 = u4 + 2.0 * a4 / (g - 1.0) * (1.0 - (p21 / p41) ** ((g - 1.0) / (2.0 * g)))
rho2 = g * p2 / a2 ** 2

# shock speed separating states 1 and 2
u_shock = u4 + a4 * math.sqrt((g + 1.0) / (2.0 * g) * (p21 - 1.0) + 1.0)

# contact discontinuity between states 2 and 3
u_cd = u2
u3 = u2
p3 = p2
a3 = 0.5 * (g - 1.0) * (u4 - u3 + 2.0 * a4 / (g - 1.0))
rho3 = g * p3 / a3 ** 2

# Solution inside rarefaction wave

# speed of head of rarefaction fan
u_head = u4 - a4

# speed of tail of rarefaction fan
u_tail = u3 - a3


def u(x, t):
    # print(x / t, u_head, u_tail, u_cd, u_shock)
    if x / t <= u_head:
        return u4
    elif (x / t > u_head) and (x / t <= u_tail):
        return 2.0 / (g + 1.0) * (x / t + 0.5 * (g - 1.0) * u4 + a4)
    elif (x / t > u_tail) and (x / t <= u_cd):
        return u3
    elif (x / t > u_cd) and (x / t <= u_shock):
        return u2
    else:
        return u1


def a(x, t):
    if x / t <= u_head:
        return a4
    elif (x / t > u_head) and (x / t <= u_tail):
        return u(x, t) - x / t
    elif (x / t > u_tail) and (x / t <= u_cd):
        return a3
    elif (x / t > u_cd) and (x / t <= u_shock):
        return a2
    else:
        return a1


def p(x, t):
    if x / t <= u_head:
        return p4
    elif (x / t > u_head) and (x / t <= u_tail):
        return p4 * (a(x, t) / a4) ** (2.0 * g / (g - 1.0))
    elif (x / t > u_tail) and (x / t <= u_cd):
        return p3
    elif (x / t > u_cd) and (x / t <= u_shock):
        return p2
    else:
        return p1


def rho(x, t):
    return g * p(x, t) / a(x, t) ** 2


xi = np.linspace(-1.0, 1.0, 1000)
ui = np.zeros_like(xi)
ai = np.zeros_like(xi)
pi = np.zeros_like(xi)
rhoi = np.zeros_like(xi)


for i in range(len(ui)):
    ui[i] = u(xi[i], time)
    ai[i] = a(xi[i], time)
    pi[i] = p(xi[i], time)
    rhoi[i] = rho(xi[i], time)
# print(xi, ui)

# f = plt.figure()
# ax = f.add_subplot(111)
# ax.yaxis.tick_right()

f = plt.figure(1)

ax = f.add_subplot(221)
plt.plot(xi, rhoi)
plt.ylabel(r"$\rho$")

ax = f.add_subplot(222)
# ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
plt.plot(xi, ui)
plt.ylabel(r"u")

ax = f.add_subplot(223)
plt.plot(xi, ui / ai)
plt.xlabel(r"$x$")
plt.ylabel(r"$\mathcal{M}$")

ax = f.add_subplot(224)
plt.plot(xi, pi)
plt.xlabel(r"$x$")
# ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
plt.ylabel(r"P")

plt.show()
