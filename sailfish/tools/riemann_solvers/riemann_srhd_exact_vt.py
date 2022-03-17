#!/usr/bin/env python3

"""
Calculate exact solution to the Riemann problem for the SRHD equations
including transverse velocities.
Assume high pressure on left, low pressure on right
States separated by Head (H) and Tail (T) of rarefaction fan
Contact Discontinuity (CD) and Shock (S)
Three cases (middle case in either direction)
  ---<---------->----------
 -> 1 | 3  | 3'| 2  <-       Case 1: 2 Shocks
  -------------------------
      S   CD   S
  --<--<---------->--------
<- 1 |4|| 3  | 3'| 2   <-    Case 2: 1 Shock and 1 Rarefaction
  -------------------------
     H  T    CD  S
  --<---<--------->--->-----
<- 1 ||4|| 3  | 3'||4'|| 2 ->  Case 3: 2 Rarefactions
  --------------------------
     H   T   CD   T   H
Solve for the pressure ratio p21 across the shock connecting states 2 and 
Left and right States primitive variables (rho, vx, vt, p)
Refs: Rezzolla, Zanotti & Pons, JFM 479, 199 (2003); Rezzolla & Zanotti, JFM 449, 395 (2001)
"""

from scipy import integrate
from matplotlib import pyplot as plt
import numpy as np

g = 5.0 / 3.0
g1 = np.sqrt(g - 1.0)


def sound_speed(p, rho):
    return np.sqrt(g * (g - 1.0) * p / ((g - 1.0) * rho + g * p))


time = 0.4

# 1D Riemann problem 1 from RAM Zhang & MacFadyen (2006)
priml = [10.0, 0.0, 0.0, 13.33]
primr = [ 1.0, 0.0, 0.0, 1e-8]


# left state
rho1 = priml[0]
vx1 = priml[1]
vt1 = priml[2]
p1 = priml[3]

e1 = rho1 + p1 / (g - 1.0)
h1 = 1.0 + p1 / rho1 * (g / (g - 1.0))
cs1 = sound_speed(p1, rho1)
w1 = 1.0 / np.sqrt(1.0 - vx1**2 - vt1**2)

# right state
rho2 = primr[0]
vx2 = primr[1]
vt2 = primr[2]
p2 = primr[3]

e2 = rho2 + p2 / (g - 1.0)
h2 = 1.0 + p2 / rho2 * (g / (g - 1.0))
cs2 = sound_speed(p2, rho2)
w2 = 1.0 / np.sqrt(1.0 - vx2**2 - vt2**2)

# relative normal velocity between left and right states
vx12 = (vx1 - vx2) / (1.0 - vx1 * vx2)

# limiting relative velocities for 3 Cases

# Case 1: Two Shocks

d = 1.0 - 4.0 * g * p1 * ((g - 1.0) * p2 + p1) / ((g - 1.0)**2 * (p1 - p2)**2) * (h2 * (p2 - p1) / rho2 - h2 * h2)
h3p = (np.sqrt(d) - 1.0) * (g - 1.0) * (p1 - p2) / ((g - 1.0) * p2 + p1) / 2.0
j2 = -(g / (g - 1.0)) * (p1 - p2) / (h3p * (h3p - 1.0) / p1 - h2 * (h2 - 1.0) / p2)
vbars = (rho2**2 * w2**2 * vx2 + abs(np.sqrt(j2)) * np.sqrt(j2 + rho2**2 * w2**2 * (1.0 - vx2**2))) / (rho2**2 * w2**2 + j2)
v12_2s = (p1 - p2) * (1.0 - vx2 * vbars) / ((vbars - v2x) * (h2 * rho2 * w2 * w2 * (1.0 - vx2 * vx2) + p1 - p2))

# Case 2: Shock plus Rarefaction

# across rarefaction wave the entropy is conserved so use isentropic relations 
# starting in, e.g. state s: p/ps = (rho/rhos)**gamma which yields
# rho(p) = rhos * (p/ps)**(1/gamma)

def rho(ps, rhos, p):
	return rhos * (p / ps)**(1.0/g)

def cs(ps, rhos, p):
	return np.sqrt(g * p / rho(ps, rhos, p))

def h(ps, rhos, p):
	return 1.0 + g / (g - 1.0) * p / rho(ps, rhos, p) 

def left_integrand(p):
	a1 = h1 * w1 * vt1
	np.sqrt(h(p1, rho1, p)**2 + a1**2 * (1.0 - cs(p1, rho1, p)**2)) / ((h(p1, rho1, p)**2 + a1**2) * rho(p1, rho1, p) * cs(p1, rho1, p))

def right_integrand(p):
	a2 = h2 * w2 * vt2
	np.sqrt(h(p2, rho2, p)**2 + a2**2 * (1.0 - cs(p2, rho2, p)**2)) / ((h(p2, rho2, p)**2 + a2**2) * rho(p2, rho2, p) * cs(p2, rho2, p))

if (p1 > p2):
	# rarefaction from left state (1)
	v12_sr = np.tanh(integrate.quadrature(left_integrand, p1, p2))
else:
	# rarefaction from right state (2)
	v12_sr = np.tanh(integrate.quadrature(right_integrand, p2, p1))

print("v12 = ", v12, " v12_2s = ", v12_2s, " v12_sr = ", v12_sr)

def vbx(pa, rhoa, vxa, vta, pb, dir):
	# dir =  1.0 is for shock moving to right
	# dir = -1.0 is for shock moving to left
	ha = 1.0 + g / (g - 1.0) * pa / rhoa
	wa = 1.0 / np.sqrt(1.0 - vxa * vxa - vta * vta)
	j2 = mass_flux_squared(pa, rhoa, pb)
	vs = (rhoa**2 * wa**2 * vxa + dir * abs(np.sqrt(j2)) * np.sqrt(j2 + rhoa**2 * wa**2 * (1.0 - vxa**2))) / (rhoa**2 * wa**2 + j2)
	ws = 1.0 / np.sqrt(1.0 - vs * vs)

	return (ha * wa * vxa + ws * (pb - pa) / j) / (ha * wa + (pb - pa) * (ws * vxa / j + 1.0 / (rhoa * wa)))

def mass_flux_squared(ps, rhos, p):
    hs = 1.0 + g / (g - 1.0) * ps / rhos
    h = enthalpy(ps, rhos, p)
	return -g / (g - 1.0) * (ps - p)/((hs * (hs - 1.0) / ps) - (h * (h - 1.0) / p))

def enthalpy(ps, rhos, p):
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
    calculate A- at tail of rarefaction
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

elif v12 >= v12_sr and v12 <= v12_2s:
    pmin = min(p1, p2)
    pmax = max(p1, p2)
    result = optimize.brentq(v12_sr, pmin, pmax, full_output=True)
    p3 = result[0]
    print("v12_sr Root finder # of iterations: ", result[1].iterations)

else:
    pmin = 0.0
    pmax = min(p1, p2)
    result = optimize.brentq(v12_2r, pmin, pmax, full_output=True)
    p3 = result[0]
    print("v12_2r Root finder # of iterations: ", result[1].iterations)
