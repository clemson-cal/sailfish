#!/usr/bin/env python3
# HLLC Approximate SRHD Riemann solver
# References: Toro 3rd ed., Mignone & Bodo (2005, 2006)
import math
import numpy as np

# adiabatic index
g = 4.0 / 3.0

# enthalpy (per unit volume)
def rhoh(prim):
    return prim[0] + prim[2] * g / (g - 1.0)


def cs(prim):
    return math.sqrt(g * prim[2] / rhoh(prim))


def lorentz_factor(v):
    return (1.0 - v**2) ** (-0.5)


def prim2cons(prim):
    lf = lorentz_factor(prim[1])
    fac = rhoh(prim) * lf * lf
    return prim[0] * lf, fac * prim[1], fac - prim[2]


def cons2prim(cons, p0=1.0):
    from scipy import optimize

    def f(p):
        v2 = (cons[1] / (cons[2] + p)) ** 2
        W = (1.0 - v2) ** (-0.5)
        eint = (cons[2] - cons[0] * W + p * (1.0 - W * W)) / (cons[0] * W)
        rho = cons[0] / W
        return rho * eint * (g - 1.0) - p

    def fder(p):
        v2 = (cons[1] / (cons[2] + p)) ** 2
        W = (1.0 - v2) ** (-0.5)
        rhoh = cons[0] / W + p * g / (g - 1.0)
        cs2 = g * p / rhoh
        return v2 * cs2 - 1.0

    p = optimize.newton(f, p0, fprime=fder)

    v2 = (cons[1] / (cons[2] + p)) ** 2
    W = (1.0 - v2) ** (-0.5)
    return cons[0] / W, math.sqrt(v2), p


def prim2flux(prim):
    cons = prim2cons(prim)
    return cons[0] * prim[1], cons[1] * prim[1] + prim[2], cons[1]


def wavespeeds(priml, primr):
    csl = cs(priml)
    csr = cs(primr)
    lfl = lorentz_factor(priml[1])
    lfr = lorentz_factor(primr[1])
    sigmal = csl**2 / (lfl * lfl * (1.0 - csl**2))
    facl = math.sqrt(sigmal * (1.0 - priml[1] ** 2 + sigmal))
    sigmar = csr**2 / (lfr * lfr * (1.0 - csr**2))
    facr = math.sqrt(sigmar * (1.0 - primr[1] ** 2 + sigmar))
    sl = min((priml[1] - facl) / (1.0 + sigmal), (primr[1] - facr) / (1.0 + sigmar))
    sr = max((priml[1] + facl) / (1.0 + sigmal), (primr[1] + facr) / (1.0 + sigmar))
    return sl, sr


def wavespeeds_simple(priml, primr):
    csl = cs(priml)
    csr = cs(primr)
    sl = min(
        (priml[1] - csl) / (1.0 - priml[1] * csl),
        (primr[1] - csr) / (1.0 - primr[1] * csr),
    )
    sr = max(
        (priml[1] + csl) / (1.0 + priml[1] * csl),
        (primr[1] + csr) / (1.0 + primr[1] * csr),
    )
    return sl, sr


def u_hlle(priml, primr, s=0.0):
    sl, sr = wavespeeds(priml, primr)
    uhlle = [0.0, 0.0, 0.0]
    if s <= sl:
        uhlle = prim2cons(priml)
    elif s >= sr:
        uhlle = prim2cons(primr)
    else:
        consl = prim2cons(priml)
        consr = prim2cons(primr)
        fl = prim2flux(priml)
        fr = prim2flux(primr)
        for i in range(3):
            uhlle[i] = (sr * consr[i] - sl * consl[i] + fl[i] - fr[i]) / (sr - sl)
    return uhlle


def f_hlle(priml, primr, s=0.0):
    fhlle = [0.0, 0.0, 0.0]
    sl, sr = wavespeeds(priml, primr)
    if s <= sl:
        fhlle = prim2flux(priml)
    elif s >= sr:
        fhlle = prim2flux(primr)
    else:
        consl = prim2cons(priml)
        consr = prim2cons(primr)
        fl = prim2flux(priml)
        fr = prim2flux(primr)
        for i in range(3):
            fhlle[i] = (sr * fl[i] - sl * fr[i] + sl * sr * (consr[i] - consl[i])) / (
                sr - sl
            )
    return fhlle


def sp_star(priml, primr):
    u = u_hlle(priml, primr)
    f = f_hlle(priml, primr)
    a = f[2]
    b = -(u[2] + f[1])
    c = u[1]
    q = -0.5 * (b + np.sign(b) * math.sqrt(b**2 - 4.0 * a * c))
    if abs(a) < 1e-10:
        ss = -c / b
    else:
        ss = (-b - math.sqrt(b**2 - 4.0 * a * c)) / (2.0 * a)
        # ls = c / q
    ps = -f[2] * ss + f[1]
    return ss, ps


def ul_star(priml, primr):
    sl, sr = wavespeeds(priml, primr)
    ss, ps = sp_star(priml, primr)
    ul = prim2cons(priml)
    dlam = sl - ss
    dvel = sl - priml[1]
    Estar = (sl * ul[2] - ul[1] + ps * ss) / dlam
    return (ul[0] * dvel / dlam, (Estar + ps) * ss, Estar)


def ur_star(priml, primr):
    sl, sr = wavespeeds(priml, primr)
    ss, ps = sp_star(priml, primr)
    ur = prim2cons(primr)
    dlam = sr - ss
    dvel = sr - primr[1]
    Estar = (sr * ur[2] - ur[1] + ps * ss) / dlam
    return (ur[0] * dvel / dlam, (Estar + ps) * ss, Estar)


def f_hllc(priml, primr, s=0.0):
    fhllc = [0.0, 0.0, 0.0]
    sl, sr = wavespeeds(priml, primr)
    if s <= sl:
        fhllc = prim2flux(priml)
    elif s >= sr:
        fhllc = prim2flux(primr)
    else:
        sstar, pstar = sp_star(priml, primr)
        if s < sstar:  # in left star state
            Dstar, mstar, Estar = ul_star(priml, primr)
        else:  # in right star state
            Dstar, mstar, Estar = ur_star(priml, primr)
        fhllc = Dstar * sstar, mstar * sstar + pstar, mstar
    return fhllc
