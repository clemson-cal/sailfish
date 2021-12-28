#define BETA_TVB 1.0
#define NPOLY 3      // Hard wire for 1D 3rd order for now
#define NUM_POINTS 3 // Hard wire for 1D 3rd order for now
#define PDE 0        // 0 for linear advection, 1 for Burgers
#define WAVESPEED 1.0

#define min2(a, b) ((a) < (b) ? (a) : (b))
#define max2(a, b) ((a) > (b) ? (a) : (b))
#define min3(a, b, c) min2(a, min2(b, c))
#define max3(a, b, c) max2(a, max2(b, c))
#define sign(x) copysign(1.0, x)
#define minabs(a, b, c) min3(fabs(a), fabs(b), fabs(c))
#define maxabs(a, b, c) max3(fabs(a), fabs(b), fabs(c))

PRIVATE double minmod(double w1, double w0l, double w0, double w0r) 
{
    double a = w1 * sqrt(3.0);
    double b = (w0 - w0l) * BETA_TVB;
    double c = (w0r - w0) * BETA_TVB;

    return 0.25 / sqrt(3.0) * fabs(sign(a) + sign(b)) * (sign(a) + sign(c)) * minabs(a, b, c);
}

// Identify troubled cells using Troubled Cell 
// Indicator from G.Fu & C-W Shu, JCP 347, 305 (2017)
// Then limit slopes of the troubled cells using TVB minmod
// For limiting, use threshold of 0.1 appropriate for 1D k=2 (DG3) 
PRIVATE double limit_troubled_cells(double *ul, double *u, double *ur)
{
    // integrating polynomial extended from left/right zone into this zone

    double a = u[0] + 2.0 * sqrt(3.0) * ul[1] + 5.0 * sqrt(5.0) / 3.0 * ul[2];
    double b = u[0] - 2.0 * sqrt(3.0) * ur[1] + 5.0 * sqrt(5.0) / 3.0 * ur[2];
    double tci = (fabs(u[0] - a) + fabs(u[0] - b)) / maxabs(ul[0], u[0], ur[0]);

    if (tci > 0.1)
    {
        w1t = minmod(u[1], ul[0], u[0], ur[0]);

        if (fabs(u[1] - w1t) > 1e-6)
        {             
            u[1] = w1t;
            u[2] = 0.0;
        }
    }
    return 0.0;
}

PRIVATE double flux(double ux) 
{
    switch (PDE) {
        case 0: return WAVESPEED * ux; // Advection
        case 1: return 0.5 * ux * ux;  // Burgers
    }
    return 0.0;
}

PRIVATE double upwind(double ul, double ur) 
{
    switch (PDE) {
        case 0: // advection
            if (WAVESPEED > 0.0) {
                return flux(ul);
            }
            else {
                return flux(ur);
            }
        case 1: // Burgers
            double al = ul;
            double ar = ur;

            if (al > 0.0 && ar > 0.0) {
                return flux(ul);
            }
            else if (al < 0.0 && ar < 0.0) {
                return flux(ur);
            }
            else {
                return 0.0;
            }
    }
}

PRIVATE double dot(double *u, double *p) 
{
    double sum = 0.0;

    for (int i = 0, i < NPOLY, ++i) {
        sum += u[i] * p[i]; 
    }
    return sum;
}

PUBLIC void scdg_1d_udot(
    int num_zones,    // number of zones, not including guard zones
    double *u_rd,     // :: $.shape == (num_zones + 2, 1, 3) # NPOLY = 3
    double *udot,     // :: $.shape == (num_zones + 2, 1, 3) # NPOLY = 3
    double dt,        // time step
    double dx)        // grid spacing
{
    int ng = 1; // number of guard zones

    // TODO: pass cell data as a struct argument

    // Gaussian quadrature points in scaled domain xsi=[-1,1]
    double g = {-0.774596669241483, 0.000000000000000, 0.774596669241483}; 
    // Gaussian weights at quadrature points
    double w = { 0.555555555555556, 0.888888888888889, 0.555555555555556};
    // Scaled LeGendre polynomials at quadrature points
    double p = {{ 1.000000000000000, 1.000000000000000, 1.000000000000000},
                {-1.341640786499873, 0.000000000000000, 1.341640786499873},
                { 0.894427190999914, -1.11803398874990, 0.894427190999914}}; 
    // Derivative of Scaled LeGendre polynomials at quadrature points
    double pp = {{ 0.000000000000000, 0.000000000000000, 0.000000000000000},
                 { 1.732050807568877, 1.732050807568877, 1.732050807568877},
                 {-5.196152422706629, 0.000000000000000, 5.196152422706629}};
    // Unit normal vector at left and right faces
    double nhat = {-1.0, 1.0};
    // Scaled LeGendre polynomials at left face
    double pfl = {1.000000000000000, -1.732050807568877, 2.23606797749979};
    // Scaled LeGendre polynomials at right face
    double pfr = {1.000000000000000,  1.732050807568877, 2.23606797749979};

    FOR_EACH_1D(num_zones)
    {
        double *uc = &u_rd[NPOLY * (i + ng)];
        double *ul = &u_rd[NPOLY * (i + ng - 1)];
        double *ur = &u_rd[NPOLY * (i + ng + 1)];
        double *uc_dot = &udot[NPOLY * (i + ng)];

        double uimh_l = dot(ul, pfr);
        double uimh_r = dot(uc, pfl);
        double uiph_l = dot(uc, pfr);
        double uiph_r = dot(ur, pfl);
        double fimh = upwind(uimh_l, uimh_r);
        double fiph = upwind(uiph_l, uiph_r);

        double fx[NUM_POINTS];

        for (int n = 0; n < NUM_POINTS; ++n)
        {
            double ux = 0.0;

            for (int l = 0; l < NPOLY; ++l)
            {
                ux += uc[l] * p[l][n];
            }
            fx[n] = flux(ux); 
        }

        for (int l = 0; l < NPOLY; ++l)
        {
            double udot_v = 0.0;

            for (int n = 0; n < NUM_POINTS; ++n)
            {
                udot_v += fx[n] * pp[l][n] * w[n] / dx;
            }
            double udot_s = -(fimh * pfl[l] * nhat[0] + fiph * pfr[l] * nhat[1]) / dx;

            uc_dot[l] = udot_v + udot_s;
        }
    }
}
