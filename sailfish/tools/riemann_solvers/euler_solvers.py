"""
Calculate flux solutions to the Riemann problem for the Euler equations.
Reference: Toro 3rd ed. Chaps. 4, 9, 10
"""
from scipy import optimize
import math

# adiabatic index
g = 7.0 / 5.0
gm1 = g - 1.0
gp1 = g + 1.0
z = gm1 / (2.0 * g)


def cs(prim):
    rho, v, p = prim
    return (g * p / rho) ** (0.5)


def prim2cons(prim):
    rho, v, p = prim
    return rho, rho * v, 0.5 * rho * v**2 + p / gm1


def cons2prim(cons):
    rho, m, E = cons
    return rho, m / rho, (E - 0.5 * m**2 / rho) * gm1


def prim2flux(prim):
    rhop, v, p = prim
    rho, m, E = prim2cons(prim)
    return rho * v, m * v + p, E * v + p * v


def wavespeeds_simple(priml, primr):
    csl = cs(priml)
    csr = cs(primr)
    sl = min(priml[1] - csl, primr[1] - csr)
    sr = max(priml[1] + csl, primr[1] + csr)
    return sl, sr


# Toro eqns. 10.21
def hlle(priml, primr, s=0.0):
    rhol, ul, pl = priml
    rhor, ur, pr = primr

    def wavespeeds(priml, primr):
        def pstar(priml, primr):
            rhobar = 0.5 * (rhol + rhor)
            csbar = 0.5 * (cs(priml) + cs(primr))
            ppvrs = 0.5 * (pl + pr) - 0.5 * (ur - ul) * rhobar * csbar
            return max(0.0, ppvrs)

        ps = pstar(priml, primr)
        csl = cs(priml)
        csr = cs(primr)
        if ps <= pl:
            ql = 1.0
        else:
            ql = math.sqrt(1.0 + gp1 / (2.0 * g) * (ps / pl - 1.0))
        if ps <= pr:
            qr = 1.0
        else:
            qr = math.sqrt(1.0 + gp1 / (2.0 * g) * (ps / pr - 1.0))
        sl = ul - csl * ql
        sr = ur + csr * qr
        return sl, sr

    fhlle = [0.0, 0.0, 0.0]
    sl, sr = wavespeeds(priml, primr)
    if s <= sl:
        fhlle = prim2flux(priml)
    elif s >= sr:
        fhlle = prim2flux(primr)
    else:
        consl = prim2cons(priml)
        fl = prim2flux(priml)
        consr = prim2cons(primr)
        fr = prim2flux(primr)
        for i in range(3):
            fhlle[i] = (sr * fl[i] - sl * fr[i] + sl * sr * (consr[i] - consl[i])) / (
                sr - sl
            )
    return fhlle


def hlle2(priml, primr, s=0.0):
    rhol, ul, pl = priml
    rhor, ur, pr = primr
    csl = cs(priml)
    csr = cs(primr)
    alpham = max(0.0, -(ul - csl), -(ur - csr))
    alphap = max(0.0, ul + csl, ur + csr)

    fhlle = [0.0, 0.0, 0.0]
    consl = prim2cons(priml)
    fl = prim2flux(priml)
    consr = prim2cons(primr)
    fr = prim2flux(primr)
    for i in range(3):
        fhlle[i] = (
            alphap * fl[i] + alpham * fr[i] - alpham * alphap * (consr[i] - consl[i])
        ) / (alphap + alpham)
    return fhlle


# Toro eqn 10.75
def hllc(priml, primr, s=0.0):

    rhol, ul, pl = priml
    rhor, ur, pr = primr

    def pstar(priml, primr):
        rhobar = 0.5 * (rhol + rhor)
        csbar = 0.5 * (cs(priml) + cs(primr))
        ppvrs = 0.5 * (pl + pr) - 0.5 * (ur - ul) * rhobar * csbar
        return max(0.0, ppvrs)

    def wavespeeds1(priml, primr):
        ps = pstar(priml, primr)
        csl = cs(priml)
        csr = cs(primr)
        if ps <= pl:
            ql = 1.0
        else:
            ql = math.sqrt(1.0 + gp1 / (2.0 * g) * (ps / pl - 1.0))
        if ps <= pr:
            qr = 1.0
        else:
            qr = math.sqrt(1.0 + gp1 / (2.0 * g) * (ps / pr - 1.0))
        sl = ul - csl * ql
        sr = ur + csr * qr
        return sl, sr

    def wavespeeds(priml, primr):
        csl = cs(priml)
        csr = cs(primr)
        sl = min(ul - csl, ur - csr)
        sr = max(ul + csl, ur + csr)
        return sl, sr

    def sstar(priml, primr):
        return (pr - pl + rhol * ul * (sl - ul) - rhor * ur * (sr - ur)) / (
            rhol * (sl - ul) - rhor * (sr - ur)
        )

    fhllc = [0.0, 0.0, 0.0]
    sl, sr = wavespeeds(priml, primr)
    if s <= sl:
        fhllc = prim2flux(priml)
    elif s >= sr:
        fhllc = prim2flux(primr)
    else:
        ss = sstar(priml, primr)
        dstar = [0.0, 1.0, ss]
        plr = 0.5 * (
            pl + pr + rhol * (sl - ul) * (ss - ul) + rhor * (sr - ur) * (ss - ur)
        )
        if s < ss:  # in left star state
            consl = prim2cons(priml)
            fl = prim2flux(priml)
            for i in range(3):
                fhllc[i] = (ss * (sl * consl[i] - fl[i]) + sl * plr * dstar[i]) / (
                    sl - ss
                )
        else:  # in right star state
            consr = prim2cons(primr)
            fr = prim2flux(primr)
            for i in range(3):
                fhllc[i] = (ss * (sr * consr[i] - fr[i]) + sr * plr * dstar[i]) / (
                    sr - ss
                )
    return fhllc


def exact(priml, primr, s=0.0):
    # Toro eqns. 4.6, 4.7, 4.37, 4.38
    def fk(p, prim):
        rhok, vk, pk = prim
        csk = cs(prim)
        ak = 2.0 / (gp1 * rhok)
        bk = pk * gm1 / gp1
        fac = math.sqrt(ak / (p + bk))
        if p > pk:  # shock
            return fac * (p - pk)
        else:  # rarefaction
            return 2.0 * csk / gm1 * ((p / pk) ** z - 1.0)

    def fkder(p, prim):
        rhok, vk, pk = prim
        csk = cs(prim)
        ak = 2.0 / (gp1 * rhok)
        bk = gm1 / gp1 * pk
        fac = math.sqrt(ak / (p + bk))
        if p > pk:  # shock
            return fac * (1.0 - (p - pk) / (2.0 * (bk + p)))
        else:  # rarefaction
            return 1.0 / (rhok * csk) * (p / pk) ** (-gp1 / (2.0 * g))

    def fkder2(p, prim):
        rhok, vk, pk = prim
        csk = cs(prim)
        ak = 2.0 / (gp1 * rhok)
        bk = gm1 / gp1 * pk
        fac = math.sqrt(ak / (p + bk))
        if p > pk:  # shock
            return -0.25 * fac * ((4.0 * bk + 3.0 * p + pk) / (bk + p) ** 2.0)
        else:  # rarefaction
            return (
                -gp1
                * csk
                / (2.0 * (g * pk) ** 2)
                * (p / pk) ** (-(3.0 * g + 1.0) / (2.0 * g))
            )

    # Toro eq. 4.5
    def f(p):
        delv = primr[1] - priml[1]
        return fk(p, priml) + fk(p, primr) + delv

    def fder(p):
        return fkder(p, priml) + fkder(p, primr)

    def fder2(p):
        return fkder2(p, priml) + fkder2(p, primr)

    # Pressure guess using two rarefaction (2r) case Toro eqns. 4.46, 9.32
    # For real simulations (as opposed to tests) the two rarefaction case
    # will likely occur somewhere every time step, so it may be most efficient
    # to use the 2r pressure guess, since the 2r iterative solution is most expensive
    # when using other pressure guesses.
    def p2r(priml, primr):
        rhol, vl, pl = priml
        rhor, vr, pr = primr
        csl = cs(priml)
        csr = cs(primr)
        return (
            (csl + csr - 0.5 * gm1 * (vr - vl)) / ((csl / pl**z) + (csr / pr**z))
        ) ** (1 / z)

    pguess = p2r(priml, primr)
    pstar = optimize.newton(f, pguess, fprime=fder, fprime2=fder2)
    rhol, vl, pl = priml
    rhor, vr, pr = primr
    vstar = 0.5 * (vl + vr) + 0.5 * (fk(pstar, primr) - fk(pstar, priml))
    csl = cs(priml)
    csr = cs(primr)
    if s < vstar:  # left of contact wave
        if pstar < pl:  # left wave is rarefaction
            shead = vl - csl  # speed of fan head eq 4.55
            if s <= shead:
                rho = rhol
                v = vl
                p = pl
            else:
                csls = csl * (pstar / pl) ** (gm1 / (2.0 * g))  # eq. 4.54
                stail = vstar - csls  # speed of fan tail eq. 4.55
                if s <= stail:  # inside left rarefaction fan eqns. 4.56
                    rho = rhol * (2.0 / gp1 + gm1 / gp1 / csl * (vl - s)) ** (2.0 / gm1)
                    v = 2.0 / gp1 * (csl + gm1 / 2.0 * vl + s)
                    p = pl * (2.0 / gp1 + gm1 / gp1 / csl * (vl - s)) ** (2.0 * g / gm1)
                else:  # in left star state
                    rho = rhol * (pstar / pl) ** (1.0 / g)
                    v = vstar
                    p = pstar
        else:  # left wave is shock
            al = 2.0 / (gp1 * rhol)
            bl = gm1 / gp1 * pl
            ql = math.sqrt((pstar + bl) / al)  # eq. 4.20
            sl = vl - ql / rhol  # eqn. 4.51
            if s < sl:
                rho = rhol
                v = vl
                p = pl
            else:
                rho = rhol * (
                    ((pstar / pl) + gm1 / gp1) / ((gm1 / gp1) * (pstar / pl) + 1.0)
                )  # eq. 4.50
                v = vstar
                p = pstar
    else:  # to right of contact wave
        if pstar < pr:  # right wave is rarefaction
            shead = vr + csr  # speed of fan head eq 4.62
            if s >= shead:
                rho = rhor
                v = vr
                p = pr
            else:
                csrs = csr * (pstar / pr) ** (gm1 / (2.0 * g))  # eq. 4.61
                stail = vstar + csrs  # speed of fan tail eq. 4.62
                if s >= stail:  # inside right rarefaction fan eqns. 4.63
                    rho = rhor * (2.0 / gp1 - gm1 / gp1 / csr * (vr - s)) ** (2.0 / gm1)
                    v = 2.0 / gp1 * (-csr + gm1 / 2.0 * vr + s)
                    p = pr * (2.0 / gp1 - gm1 / gp1 / csr * (vr - s)) ** (2.0 * g / gm1)
                else:  # in right star state
                    rho = rhor * (pstar / pr) ** (1.0 / g)
                    v = vstar
                    p = pstar
        else:  # right wave is shock
            ar = 2.0 / (gp1 * rhor)
            br = gm1 / gp1 * pr
            qr = math.sqrt((pstar + br) / ar)  # eq. 4.20
            sr = vr + qr / rhor  # eqn. 4.58
            if s > sr:
                rho = rhor
                v = vr
                p = pr
            else:
                rho = rhor * (
                    ((pstar / pr) + gm1 / gp1) / ((gm1 / gp1) * (pstar / pr) + 1.0)
                )  # eq. 4.50
                v = vstar
                p = pstar
    return prim2flux([rho, v, p])


# if __name__ == "__main__":
#    import sys
#    euler_exact(int(sys.argv[1]))
