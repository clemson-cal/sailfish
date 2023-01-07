from .kernels import device


@device
def source_terms_spherical_polar():
    R"""
    DEVICE void source_terms_spherical_polar(
        double r0, double r1,
        double q0, double q1,
        double f0, double f1,
        double *prim,
        double *source)
    {
        // Forumulas below are A8 - A10 from Zhang & MacFadyen (2006), integrated
        // over the cell volume with finite radial and polar extent. The Newtonian
        // limit is obtained by taking h -> 1 and W -> 1.
        //
        // https://iopscience.iop.org/article/10.1086/500792/pdf

        #if RELATIVISTIC == 0
        #define rhoh (dg)
        #elif RELATIVISTIC == 1
        #define rhoh (dg + pg * (1.0 + 1.0 / (ADIABATIC_GAMMA - 1.0)))
        #else
        #error("RELATIVISTIC must be 0 or 1")
        #endif

        double dr2 = r1 * r1 - r0 * r0;
        double df = f1 - f0;

        #if NVECS == 1
        double dcosq = cos(q1) - cos(q0);
        double dg = prim[0];
        double uq = 0.0;
        double uf = 0.0;
        double pg = prim[2];
        double srdot = -0.5 * df * dr2 * dcosq * (rhoh * (uq * uq + uf * uf) + 2 * pg);
        source[0] = 0.0;
        source[1] = srdot;
        source[2] = 0.0;

        #elif NVECS == 2
        double dcosq = cos(q1) - cos(q0);
        double dsinq = sin(q1) - sin(q0);
        double dg = prim[0];
        double ur = prim[1];
        double uq = prim[2];
        double uf = 0.0;
        double pg = prim[3];
        double srdot = -0.5 * df * dr2 * dcosq * (rhoh * (uq * uq + uf * uf) + 2 * pg);
        double sqdot = +0.5 * df * dr2 * (dcosq * rhoh * ur * uq + dsinq * (pg + rhoh * uf * uf));
        source[0] = 0.0;
        source[1] = srdot;
        source[2] = sqdot;
        source[3] = 0.0;

        #elif NVECS == 3
        double dcosq = cos(q1) - cos(q0);
        double dsinq = sin(q1) - sin(q0);
        double dg = prim[0];
        double ur = prim[1];
        double uq = prim[2];
        double uf = prim[3];
        double pg = prim[4];
        double srdot = -0.5 * df * dr2 * dcosq * (rhoh * (uq * uq + uf * uf) + 2 * pg);
        double sqdot = +0.5 * df * dr2 * (dcosq * rhoh * ur * uq + dsinq * (pg + rhoh * uf * uf));
        double sfdot = -0.5 * df * dr2 * rhoh * uf * (uq * dsinq - ur * dcosq);
        source[0] = 0.0;
        source[1] = srdot;
        source[2] = sqdot;
        source[3] = sfdot;
        source[4] = 0.0;
        #else
        #error("NVECS must be 1, 2, or 3")
        #endif
    }
    """


@device
def source_terms_cylindrical_polar():
    R"""
    DEVICE void source_terms_cylindrical_polar(
        double r0, double r1,
        double z0, double z1,
        double f0, double f1,
        double *prim,
        double *source)
    {
        // Forumulas below are A2 - A4 from Zhang & MacFadyen (2006), integrated
        // over the cell volume with finite radial and polar extent. The Newtonian
        // limit is obtained by taking h -> 1 and W -> 1.
        //
        // https://iopscience.iop.org/article/10.1086/500792/pdf

        #if RELATIVISTIC == 0
        #define rhoh (dg)
        #elif RELATIVISTIC == 1
        #define rhoh (dg + pg * (1.0 + 1.0 / (ADIABATIC_GAMMA - 1.0)))
        #else
        #error("RELATIVISTIC must be 0 or 1")
        #endif

        #if NVECS == 1
        double dg = prim[0];
        double uf = 0.0;
        double pg = prim[2];
        double srdot = +(f1 - f0) * (r1 - r0) * (z1 - z0) * (rhoh * uf * uf + pg);
        source[0] = 0.0;
        source[1] = srdot;
        source[2] = 0.0;

        #elif NVECS == 2
        double dg = prim[0];
        double uf = 0.0;
        double pg = prim[3];
        double srdot = +(f1 - f0) * (r1 - r0) * (z1 - z0) * (rhoh * uf * uf + pg);
        double szdot = 0.0;
        source[0] = 0.0;
        source[1] = srdot;
        source[2] = szdot;
        source[3] = 0.0;

        #elif NVECS == 3
        double dg = prim[0];
        double ur = prim[1];
        double uf = prim[3];
        double pg = prim[4];
        double srdot = +(f1 - f0) * (r1 - r0) * (z1 - z0) * (rhoh * uf * uf + pg);
        double szdot = 0.0;
        double sfdot = -(f1 - f0) * (r1 - r0) * (z1 - z0) * (rhoh * ur * uf);
        source[0] = 0.0;
        source[1] = srdot;
        source[2] = szdot;
        source[3] = sfdot;
        source[4] = 0.0;
        #else
        #error("NVECS must be 1, 2, or 3")
        #endif
    }
    """
