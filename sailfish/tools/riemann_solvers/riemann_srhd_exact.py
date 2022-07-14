#!/usr/bin/env python3

"""
Calculate exact solution to the Riemann problem for the SRHD equations.
Assume high pressure on left, low pressure on right
States separated by Head (H) and Tail (T) of rarefaction fan
Contact Discontinuity (CD) and Shock (S)
Three cases (middle case in either direction)
  ---<---------->----------
 -> 1 | 3  | 3'| 2  <-       Case 1: 2 Shocks
  -------------------------
      S   CD  
  --<--<---------->--------
<- 1 |4|| 3  | 3'| 2   <-    Case 2: 1 Shock and 1 Rarefaction
  -------------------------
     H  T    CD  
  --<---<--------->--->-----
<- 1 |4|| 3  | 3'||4'|| 2 ->  Case 3: 2 Rarefactions
  --------------------------
     H  T    CD  T    
Solve for the pressure ratio p21 across the shock connecting states 2 and 
Left and right States primitive variables (rho, v, p)
Ref: Rezzolla & Zanotti, JFM 449, 395 (2001)
"""

from scipy import optimize
from matplotlib import pyplot as plt
import numpy as np

g = 5.0 / 3.0
g1 = np.sqrt(g - 1.0)


def sound_speed(p, rho):
    return np.sqrt(g * (g - 1.0) * p / ((g - 1.0) * rho + g * p))


time = 0.4

# 1D Riemann problem 1 from RAM Zhang & MacFadyen (2006)
priml = [10.0, 0.0, 13.33]
primr = [1.0, 0.0, 1e-8]


# left state
rho1 = priml[0]
v1 = priml[1]
p1 = priml[2]

e1 = rho1 + p1 / (g - 1.0)
h1 = 1.0 + p1 / rho1 * (g / (g - 1.0))
cs1 = sound_speed(p1, rho1)

# right state
rho2 = primr[0]
v2 = primr[1]
p2 = primr[2]

e2 = rho2 + p2 / (g - 1.0)
h2 = 1.0 + p2 / rho2 * (g / (g - 1.0))
cs2 = sound_speed(p2, rho2)

# relative velocity between left and right states
v12 = (v1 - v2) / (1.0 - v1 * v2)

# limiting relative velocities for 3 Cases

# Case 1: Two Shocks

# calculate hhat
aa = 1.0 + (g - 1.0) * (p2 - p1) / (g * p1)
bb = -(g - 1.0) * (p2 - p1) / (g * p1)
cc = h2 * (p2 - p1) / rho2 - h2 * h2

hhat = (-bb + np.sqrt(bb * bb - 4 * aa * cc)) / (2 * aa)

ehat = hhat * (g * p1 / ((g - 1.0) * (hhat - 1.0))) - p1

v12_2s = np.sqrt((p1 - p2) * (ehat - e2) / ((ehat + p2) * (e2 + p1)))

# Case 2: Shock plus Rarefaction

# across rarefaction entropy is conserved so p1/p3 = (rho1/rho3)**gamma

aplus = (((g1 - cs2) / (g1 + cs2)) * ((g1 + cs1) / (g1 - cs1))) ** (2.0 / g1)

v12_sr = (1.0 - aplus) / (1.0 + aplus)

# Case 3: Two Rarefactions

s1 = ((g1 + cs1) / (g1 - cs1)) ** (2.0 / g1)
s2 = ((g1 + cs2) / (g1 - cs2)) ** (-2.0 / g1)

v12_2r = -1.0 * (s1 - s2) / (s1 + s2)

print(v12, v12_2s, v12_sr, v12_2r)


def h(ps, rhos, p):
    """
    Given a pre-shock state with pressure ps and density rhos,
    calculate the enthalpy behind the shock
    """
    hs = 1.0 + g / (g - 1.0) * ps / rhos
    # calculate hhat
    a = 1.0 + (g - 1.0) * (ps - p) / (g * p)
    b = -(g - 1.0) * (ps - p) / (g * p)
    c = hs * (ps - p) / rho2 - hs * hs

    return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)


def aplus(ps, rhos, p):
    """
    Given state ahead of left-moving rarefaction with pressure ps and density rhos,
    calculate A+ at tail of rarefaction
    """
    css = sound_speed(ps, rhos)
    rho = rhos * (p / ps) ** (1.0 / g)
    cs = sound_speed(p, rho)
    return (((g1 - cs) / (g1 + cs)) * ((g1 + css) / (g1 - css))) ** (2.0 / g1)


def aminus(ps, rhos, p):
    """
    Given state ahead of right-moving rarefaction with pressure ps and density rhos,
    calculate A+ at tail of rarefaction
    """
    css = sound_speed(ps, rhos)
    rho = rhos * (p / ps) ** (1.0 / g)
    cs = sound_speed(p, rho)

    return (((g1 - cs) / (g1 + cs)) * ((g1 + css) / (g1 - css))) ** (-2.0 / g1)


def v12_2s(pstar):
    h3 = h(p1, rho1, pstar)
    e3 = h3 * g * pstar / ((g - 1.0) * (h3 - 1.0)) - pstar
    h3p = h(p2, rho2, pstar)
    e3p = h3p * g * pstar / ((g - 1.0) * (h3p - 1.0)) - pstar

    v1 = np.sqrt((pstar - p1) * (e3 - e1) / ((e1 + pstar) * (e3 + p1)))
    v2 = -1.0 * np.sqrt((pstar - p2) * (e3p - e2) / ((e2 + pstar) * (e3p + p2)))

    return (v1 - v2) / (1.0 - v1 * v2) - v12


def v12_sr(pstar):
    ap = aplus(p1, rho1, pstar)
    v1 = (1.0 - ap) / (1.0 + ap)
    h3p = h(p2, rho2, pstar)
    e3p = h3p * g * pstar / ((g - 1.0) * (h3p - 1.0)) - pstar
    v2 = -1.0 * np.sqrt((pstar - p2) * (e3p - e2) / ((e2 + pstar) * (e3p + p2)))

    return (v1 - v2) / (1.0 - v1 * v2) - v12


def v12_2r(pstar):
    ap = aplus(p1, rho1, pstar)
    v1 = (1.0 - ap) / (1.0 + ap)
    am = aminus(p2, rho2, pstar)
    v2 = (1.0 - am) / (1.0 + am)

    return (v1 - v2) / (1.0 - v1 * v2) - v12


if v12 > v12_2s:
    pmin = max(p1, p2)
    pmax = 1e10
    result = optimize.brentq(v12_2s, pmin, pmax, full_output=True)
    p3 = result[0]
    print("v12_2s Root finder # of iterations: ", result[1].iterations)

elif v12 > v12_sr and v12 <= vt12_2s:
    pmin = min(p1, p2)
    pmax = max(p1, p2)
    result = optimize.brentq(v12_sr, pmin, pmax, full_output=True)
    p3 = result[0]
    print("v12_sr Root finder # of iterations: ", result[1].iterations)

elif v12 > v12_2r and v12 <= vt12_sr:
    pmin = 0.0
    pmax = min(p1, p2)
    result = optimize.brentq(v12_2s, pmin, pmax, full_output=True)
    p3 = result[0]
    print("v12_2s Root finder # of iterations: ", result[1].iterations)
