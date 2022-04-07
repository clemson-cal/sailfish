/*
MODULE: cbdisodg_2d

DESCRIPTION: Isothermal DG solver for a binary accretion problem in 2D planar
  cartesian coordinates.
*/


#define ORDER 3
#define NCONS 3
#define NPOLY 6
#define L_ENDPOINT 99999990
#define R_ENDPOINT 99999991


// ============================ MATH ==========================================
// ============================================================================
#define min2(a, b) ((a) < (b) ? (a) : (b))
#define max2(a, b) ((a) > (b) ? (a) : (b))
#define min3(a, b, c) min2(a, min2(b, c))
#define max3(a, b, c) max2(a, max2(b, c))
#define sign(x) copysign(1.0, x)
#define minabs(a, b, c) min3(fabs(a), fabs(b), fabs(c))


// ============================ INTERNAL STRUCTS ==============================
// ============================================================================
struct PointMass {
    double x;
    double y;
    double vx;
    double vy;
    double mass;
    double softening_length;
    double sink_rate;
    double sink_radius;
    int sink_model;
};

struct PointMassList {
    struct PointMass masses[2];
};

struct KeplerianBuffer {
    double surface_density;
    double surface_pressure;
    double central_mass;
    double driving_rate;
    double outer_radius;
    double onset_width;
    int is_enabled;
};


// ============================ GRAVITY =======================================
// ============================================================================
PRIVATE double gravitational_potential(
    struct PointMassList *mass_list,
    double x1,
    double y1)
{
    double phi = 0.0;

    for (int p = 0; p < 2; ++p)
    {
        if (mass_list->masses[p].mass > 0.0)
        {
            double x0 = mass_list->masses[p].x;
            double y0 = mass_list->masses[p].y;
            double mp = mass_list->masses[p].mass;
            double rs = mass_list->masses[p].softening_length;

            double dx = x1 - x0;
            double dy = y1 - y0;
            double r2 = dx * dx + dy * dy;
            double r2_softened = r2 + rs * rs;

            phi -= mp / sqrt(r2_softened);
        }
    }
    return phi;
}

PRIVATE void point_mass_source_term(
    struct PointMass *mass,
    double x1,
    double y1,
    double dt,
    double *prim,
    double *delta_cons)
{
    double x0 = mass->x;
    double y0 = mass->y;
    double sigma = prim[0];
    double dx = x1 - x0;
    double dy = y1 - y0;
    double r2 = dx * dx + dy * dy;
    double dr = sqrt(r2);
    double r_sink = mass->sink_radius;
    double r_soft = mass->softening_length;

    double fgrav_numerator = sigma * mass->mass * pow(r2 + r_soft * r_soft, -1.5);
    double fx = -fgrav_numerator * dx;
    double fy = -fgrav_numerator * dy;
    double sink_rate = (dr < 4.0 * r_sink) ? mass->sink_rate * exp(-pow(dr / r_sink, 4.0)) : 0.0;
    double mdot = sigma * sink_rate * -1.0;

    switch (mass->sink_model)
    {
        case 1: // acceleration-free
        {
            delta_cons[0] = dt * mdot;
            delta_cons[1] = dt * mdot * prim[1] + dt * fx;
            delta_cons[2] = dt * mdot * prim[2] + dt * fy;
            break;
        }
        case 2: // torque-free
        {
            double vx = prim[1];
            double vy = prim[2];
            double vx0 = mass->vx;
            double vy0 = mass->vy;
            double rhatx = dx / (dr + 1e-12);
            double rhaty = dy / (dr + 1e-12);
            double dvdotrhat = (vx - vx0) * rhatx + (vy - vy0) * rhaty;
            double vxstar = dvdotrhat * rhatx + vx0;
            double vystar = dvdotrhat * rhaty + vy0;
            delta_cons[0] = dt * mdot;
            delta_cons[1] = dt * mdot * vxstar + dt * fx;
            delta_cons[2] = dt * mdot * vystar + dt * fy;
            break;
        }
        case 3: // force-free
        {
            delta_cons[0] = dt * mdot;
            delta_cons[1] = dt * fx;
            delta_cons[2] = dt * fy;
            break;
        }
        default: // sink is inactive
        {
            delta_cons[0] = 0.0;
            delta_cons[1] = 0.0;
            delta_cons[2] = 0.0;
            break;
        }
    }
}

PRIVATE void point_masses_source_term(
    struct PointMassList *mass_list,
    double x1,
    double y1,
    double dt,
    double *prim,
    double *cons)
{
    for (int p = 0; p < 2; ++p)
    {
        double delta_cons[NCONS];
        point_mass_source_term(&mass_list->masses[p], x1, y1, dt, prim, delta_cons);

        for (int q = 0; q < NCONS; ++q)
        {
            cons[q] += delta_cons[q];
        }
    }
}


// ============================ EOS AND BUFFER ================================
// ============================================================================
PRIVATE double sound_speed_squared(
    double cs2,
    double mach_squared,
    int eos_type,
    double x,
    double y,
    struct PointMassList *mass_list)
{
    switch (eos_type)
    {
        case 1: // globally isothermal
            return cs2;
        case 2: // locally Isothermal
            return -gravitational_potential(mass_list, x, y) / mach_squared;
        default:
            return 1.0; // WARNING
    }
}

PRIVATE void buffer_source_term(
    struct KeplerianBuffer *buffer,
    double xc,
    double yc,
    double *cons,
    double *cons_dot)
{
    if (buffer->is_enabled)
    {
        double rc = sqrt(xc * xc + yc * yc);
        double surface_density = buffer->surface_density;
        double central_mass = buffer->central_mass;
        double driving_rate = buffer->driving_rate;
        double outer_radius = buffer->outer_radius;
        double onset_width = buffer->onset_width;
        double onset_radius = outer_radius - onset_width;

        if (rc > onset_radius)
        {
            double pf = surface_density * sqrt(central_mass / rc);
            double px = pf * (-yc / rc);
            double py = pf * ( xc / rc);
            double u0[NCONS] = {surface_density, px, py};

            double omega_outer = sqrt(central_mass * pow(onset_radius, -3.0));
            // double buffer_rate = driving_rate * omega_outer * max2(rc, 1.0);
            double buffer_rate = driving_rate * omega_outer * (rc - onset_radius) / (outer_radius - onset_radius);

            for (int q = 0; q < NCONS; ++q)
            {
                cons_dot[q] -= (cons[q] - u0[q]) * buffer_rate;
            }
        }
    }
}

// ============================ HYDRO =========================================
// ============================================================================
PRIVATE void conserved_to_primitive(
    const double *cons,
    double *prim,
    double velocity_ceiling)
{
    double rho = cons[0];
    double px = cons[1];
    double py = cons[2];
    double vx = sign(px) * min2(fabs(px / rho), velocity_ceiling);
    double vy = sign(py) * min2(fabs(py / rho), velocity_ceiling);

    prim[0] = rho;
    prim[1] = vx;
    prim[2] = vy;
}

PRIVATE double primitive_to_velocity(
    const double *prim,
    int direction)
{
    switch (direction)
    {
        case 0: return prim[1];
        case 1: return prim[2];
        default: return 0.0;
    }
}

PRIVATE void primitive_to_flux(
    const double *prim,
    const double *cons,
    double *flux,
    double cs2,
    int direction)
{
    double vn = primitive_to_velocity(prim, direction);
    double rho = prim[0];
    double pressure = rho * cs2;

    flux[0] = vn * cons[0];
    flux[1] = vn * cons[1] + pressure * (direction == 0);
    flux[2] = vn * cons[2] + pressure * (direction == 1);
}

PRIVATE void primitive_to_outer_wavespeeds(
    const double *prim,
    double *wavespeeds,
    double cs2,
    int direction)
{
    double cs = sqrt(cs2);
    double vn = primitive_to_velocity(prim, direction);
    wavespeeds[0] = vn - cs;
    wavespeeds[1] = vn + cs;
}

PRIVATE double primitive_max_wavespeed(
    const double *prim,
    double cs2)
{
    double cs = sqrt(cs2);
    double vx = prim[1];
    double vy = prim[2];
    double ax = max2(fabs(vx - cs), fabs(vx + cs));
    double ay = max2(fabs(vy - cs), fabs(vy + cs));
    return max2(ax, ay);
}

PRIVATE void riemann_hlle(
    const double *ul,
    const double *ur,
    double *flux,
    double cs2,
    double velocity_ceiling,
    int direction)
{
    double pl[NCONS];
    double pr[NCONS];
    double fl[NCONS];
    double fr[NCONS];
    double al[2];
    double ar[2];

    conserved_to_primitive(ul, pl, velocity_ceiling);
    conserved_to_primitive(ur, pr, velocity_ceiling);
    primitive_to_flux(pl, ul, fl, cs2, direction);
    primitive_to_flux(pr, ur, fr, cs2, direction);
    primitive_to_outer_wavespeeds(pl, al, cs2, direction);
    primitive_to_outer_wavespeeds(pr, ar, cs2, direction);

    const double am = min3(0.0, al[0], ar[0]);
    const double ap = max3(0.0, al[1], ar[1]);

    for (int q = 0; q < NCONS; ++q)
    {
        flux[q] = (fl[q] * ap - fr[q] * am - (ul[q] - ur[q]) * ap * am) / (ap - am);
    }
}

PRIVATE double basis_phi_1d(int i_quad, int m, int deriv)
{
    // Scaled LeGendgre polynomials at the interval endpoints
    static double phi_lface[3] = {+1.000000000000000, -1.732050807568877, +2.23606797749979};
    static double phi_rface[3] = {+1.000000000000000, +1.732050807568877, +2.23606797749979};

    // Scaled LeGendre polynomials at internal quadrature points
    static double phi_vol[3][3] = {
        {+1.000000000000000, +1.00000000000000, +1.000000000000000},
        {-1.341640786499873, +0.00000000000000, +1.341640786499873},
        {+0.894427190999914, -1.11803398874990, +0.894427190999914}
    };

    // Derivative of scaled LeGendre polynomials at 1D quadrature points
    static double phi_deriv[3][3] = {
        {+0.000000000000000, +0.000000000000000, +0.000000000000000},
        {+1.732050807568877, +1.732050807568877, +1.732050807568877},
        {-5.196152422706629, +0.000000000000000, +5.196152422706629}
    };

    if (i_quad == L_ENDPOINT)
    {
        return phi_lface[m];
    }
    else if (i_quad == R_ENDPOINT)
    {
        return phi_rface[m];
    }
    else
    {
        if (deriv == 0)
        {
            return phi_vol[m][i_quad];
        }
        else if (deriv == 1)
        {
            return phi_deriv[m][i_quad];
        }
    }
    printf("WARNING: invalid basis function query\n");
    return 0.0;
}

PRIVATE double basis_phi_2d(int i_quad, int j_quad, int m, int n, int deriv_x, int deriv_y)
{
    return basis_phi_1d(i_quad, m, deriv_x) * basis_phi_1d(j_quad, n, deriv_y);
}

// PRIVATE void reconstruct(int i_quad, int j_quad, double *weights, double *cons)
// {
//     for (int q = 0; q < NCONS; ++q)
//     {
//         cons[q] = 0.0;

//         for (int m = 0; m < ORDER; ++m)
//         {
//             for (int n = 0; n < ORDER; ++n)
//             {
//                 if (m + n < ORDER)
//                 {
//                     cons[q] += (
//                           weights[q * ORDER * ORDER + m * ORDER + n]
//                         * basis_phi_2d(i_quad, j_quad, m, n, 0, 0)
//                     );
//                 }
//             }
//         }
//     }
// }

PRIVATE void reconstruct_2d(int i_quad, int j_quad, double phi[ORDER][ORDER][ORDER][ORDER], double *weights, double *cons)
{
    for (int q = 0; q < NCONS; ++q)
    {
        cons[q] = 0.0;

        for (int m = 0; m < ORDER; ++m)
        {
            for (int n = 0; n < ORDER; ++n)
            {
                if (m + n < ORDER)
                {
                    cons[q] += (
                          weights[q * ORDER * ORDER + m * ORDER + n]
                        * phi[i_quad][j_quad][m][n]
                    );
                }
            }
        }
    }
}

PRIVATE void reconstruct_1d(int quad, double phi[ORDER][ORDER][ORDER], double *weights, double *cons)
{
    for (int q = 0; q < NCONS; ++q)
    {
        cons[q] = 0.0;

        for (int m = 0; m < ORDER; ++m)
        {
            for (int n = 0; n < ORDER; ++n)
            {
                if (m + n < ORDER)
                {
                    cons[q] += (
                          weights[q * ORDER * ORDER + m * ORDER + n]
                        * phi[quad][m][n]
                    );
                }
            }
        }
    }
}

PRIVATE minmod_simple(double w1, double w0l, double w0, double w0r, double dl)
{
    #define sign(x) copysign(1.0, x)

    double beta = 1.0;
    double a = w1;
    double b = (w0 - w0l) * beta / sqrt(3.0);
    double c = (w0r - w0) * beta / sqrt(3.0);

    if (a < 0.0 && b < 0.0 && c < 0.0)
    {
        if (a > b && a > c)
        {
            return a; // no trigger
        }
        if (b > c && b > a)
        {
            return b; // trigger
        }
        if (c > a && c > b)
        {
            return c;
        }
    }
    if (a > 0.0 && b > 0.0 && c > 0.0)
    {
        if (a < b && a < c)
        {
            return a; // no trigger
        }
        if (b < c && b < a)
        {
            return b; // trigger
        }
        if (c < a && c < b)
        {
            return c;
        }
    }
    return 0.0; // trigger
}

PRIVATE minmodTVB(double w1, double w0l, double w0, double w0r, double dl)
{
    double BETA_TVB = 1.0;
    double a = w1 * sqrt(3.0);
    double b = (w0 - w0l) * BETA_TVB;
    double c = (w0r - w0) * BETA_TVB;

    const double M = 10.0; //Cockburn & Shu, JCP 141, 199 (1998) eq. 3.7 suggest M~50.0
    //const double Mtilde = 0.5; //Schaal+
    if (fabs(a) <= M * dl * dl)
    //if (fabs(a) <= Mtilde * dl)
    {
        return w1;
    }
    else
    {
        double x1 = fabs(sign(a) + sign(b)) * (sign(a) + sign(c));
        double x2 = minabs(a, b, c);
        double x = (0.25 / sqrt(3.0)) * x1 * x2;

        return x;
    }
}

// ============================ PUBLIC API ====================================
// ============================================================================
PUBLIC void cbdisodg_2d_slope_limit(
    int ni,
    int nj,
    double patch_xl, // mesh
    double patch_xr,
    double patch_yl,
    double patch_yr,
    double *weights1, // :: $.shape == (ni + 2, nj + 2, 3, 3, 3) # 3, 3, 3 = NCONS, ORDER, ORDER
    double *weights2) // :: $.shape == (ni + 2, nj + 2, 3, 3, 3) # 3, 3, 3 = NCONS, ORDER, ORDER
{
    double dx = (patch_xr - patch_xl) / ni;
    double dy = (patch_yr - patch_yl) / nj;
    int ng = 1; // number of guard zones
    int si = NCONS * ORDER * ORDER * (nj + 2 * ng);
    int sj = NCONS * ORDER * ORDER;

    FOR_EACH_2D(ni, nj)
    {
        // Get the indexes and pointers to neighbor zones
        // --------------------------------------------------------------------
        int ncc = (i     + ng) * si + (j     + ng) * sj;
        int nli = (i - 1 + ng) * si + (j     + ng) * sj;
        int nri = (i + 1 + ng) * si + (j     + ng) * sj;
        int nlj = (i     + ng) * si + (j - 1 + ng) * sj;
        int nrj = (i     + ng) * si + (j + 1 + ng) * sj;

        double *ucc = &weights1[ncc];
        double *uli = &weights1[nli];
        double *uri = &weights1[nri];
        double *ulj = &weights1[nlj];
        double *urj = &weights1[nrj];

        double *w2 = &weights2[ncc];

        for (int q = 0; q < NCONS; ++q)
        {
            int p00 = ORDER * ORDER * q + 0 * ORDER + 0;
            int p01 = ORDER * ORDER * q + 0 * ORDER + 1;
            int p10 = ORDER * ORDER * q + 1 * ORDER + 0;

            double wtilde_x = minmod_simple(ucc[p10], uli[p00], ucc[p00], uri[p00], dx);
            double wtilde_y = minmod_simple(ucc[p01], ulj[p00], ucc[p00], urj[p00], dy);
            
            if (wtilde_x != ucc[p10] || wtilde_y != ucc[p01]) 
            {
                for (int m = 0; m < ORDER; ++m)
                {
                    for (int n = 0; n < ORDER; ++n)
                    {
                        if (m + n > 0)
                        {
                            w2[ORDER * ORDER * q + m * ORDER + n] = 0.0;
                        }
                    }
                }
                w2[p10] = wtilde_x;
                w2[p01] = wtilde_y;
            }
        }
    }
}

//PUBLIC void cbdisodg_2d_slope_limit(
//    int ni,
//    int nj,
//    double patch_xl, // mesh
//    double patch_xr,
//    double patch_yl,
//    double patch_yr,
//    double *weights1, // :: $.shape == (ni + 2, nj + 2, 3, 3, 3) # 3, 3, 3 = NCONS, ORDER, ORDER
//    double *weights2) // :: $.shape == (ni + 2, nj + 2, 3, 3, 3) # 3, 3, 3 = NCONS, ORDER, ORDER
//{
//    #define max2(a, b) (a) > (b) ? (a) : (b)
//    #define max3(a, b, c) max2(a, max2(b, c))
//    #define maxabs5(a, b, c, d, e) max2(max2(fabs(a), fabs(b)), max3(fabs(c), fabs(d), fabs(e)))
//    #define SQRT_THREE sqrt(3.0)
//    #define SQRT_FIVE  sqrt(5.0)
//    #define CK 0.03 // Troubled Cell Indicator G. Fu & C.-W. Shu (JCP, 347, 305 (2017))
//
//    double dx = (patch_xr - patch_xl) / ni;
//    double dy = (patch_yr - patch_yl) / nj;
//    int ng = 1; // number of guard zones
//    int si = NCONS * ORDER * ORDER * (nj + 2 * ng);
//    int sj = NCONS * ORDER * ORDER;
//    double dvol = 4.0; // volume in xsi coordinates [-1,1] x [-1,1]
//
//    FOR_EACH_2D(ni, nj)
//    {
//        // Get the indexes and pointers to neighbor zones
//        // --------------------------------------------------------------------
//        int ncc = (i     + ng) * si + (j     + ng) * sj;
//        int nli = (i - 1 + ng) * si + (j     + ng) * sj;
//        int nri = (i + 1 + ng) * si + (j     + ng) * sj;
//        int nlj = (i     + ng) * si + (j - 1 + ng) * sj;
//        int nrj = (i     + ng) * si + (j + 1 + ng) * sj;
//
//        double *ucc = &weights1[ncc];
//        double *uli = &weights1[nli];
//        double *uri = &weights1[nri];
//        double *ulj = &weights1[nlj];
//        double *urj = &weights1[nrj];
//
//        int qt = 0; // index of conserved variable to test for trouble
//
//        int t00 = ORDER * ORDER * qt + 0 * ORDER + 0;
//        int t01 = ORDER * ORDER * qt + 0 * ORDER + 1;
//        int t10 = ORDER * ORDER * qt + 1 * ORDER + 0;
//        int t02 = ORDER * ORDER * qt + 0 * ORDER + 2;
//        int t20 = ORDER * ORDER * qt + 2 * ORDER + 0;
//
//        double maxpj = maxabs5(ucc[t00], uli[t00], uri[t00], ulj[t00], urj[t00]);
//
//        double a = 4.0 * uli[t00] + 8.0 * SQRT_THREE * uli[t10] + 24.0 * SQRT_FIVE * uli[t20];
//        double b = 4.0 * uri[t00] - 8.0 * SQRT_THREE * uri[t10] + 24.0 * SQRT_FIVE * uri[t20];
//        double c = 4.0 * ulj[t00] + 8.0 * SQRT_THREE * ulj[t01] + 24.0 * SQRT_FIVE * ulj[t02];
//        double d = 4.0 * urj[t00] - 8.0 * SQRT_THREE * urj[t01] + 24.0 * SQRT_FIVE * urj[t02];
//
//        double pbb_li = fabs(ucc[t00] - a / dvol);
//        double pbb_ri = fabs(ucc[t00] - b / dvol);
//        double pbb_lj = fabs(ucc[t00] - c / dvol);
//        double pbb_rj = fabs(ucc[t00] - d / dvol);
//
//        double tci = (pbb_li + pbb_ri + pbb_lj + pbb_rj) / maxpj;
//        
//        if (tci > CK)
//        {
//            double *w2 = &weights2[ncc];
//    
//            for (int q = 0; q < NCONS; ++q)
//            {
//                int p00 = ORDER * ORDER * q + 0 * ORDER + 0;
//                int p01 = ORDER * ORDER * q + 0 * ORDER + 1;
//                int p10 = ORDER * ORDER * q + 1 * ORDER + 0;
//    
//                double wtilde_x = minmod_simple(ucc[p10], uli[p00], ucc[p00], uri[p00], dx);
//                double wtilde_y = minmod_simple(ucc[p01], ulj[p00], ucc[p00], urj[p00], dy);
//                
//                if (wtilde_x != ucc[p10] || wtilde_y != ucc[p01]) 
//                {
//                    for (int m = 0; m < ORDER; ++m)
//                    {
//                        for (int n = 0; n < ORDER; ++n)
//                        {
//                            if (m + n > 0)
//                            {
//                                w2[ORDER * ORDER * q + m * ORDER + n] = 0.0;
//                            }
//                        }
//                    }
//                    w2[p10] = wtilde_x;
//                    w2[p01] = wtilde_y;
//                }
//            }
//        }
//    }
//}

PUBLIC void cbdisodg_2d_advance_rk(
    int ni,
    int nj,
    double patch_xl, // mesh
    double patch_xr,
    double patch_yl,
    double patch_yr,
    double *weights0, // :: $.shape == (ni + 2, nj + 2, 3, 3, 3) # 3, 3, 3 = NCONS, ORDER, ORDER
    double *weights1, // :: $.shape == (ni + 2, nj + 2, 3, 3, 3) # 3, 3, 3 = NCONS, ORDER, ORDER
    double *weights2, // :: $.shape == (ni + 2, nj + 2, 3, 3, 3) # 3, 3, 3 = NCONS, ORDER, ORDER
    double buffer_surface_density,
    double buffer_central_mass,
    double buffer_driving_rate,
    double buffer_outer_radius,
    double buffer_onset_width,
    int buffer_is_enabled,
    double x1, // point mass 1
    double y1,
    double vx1,
    double vy1,
    double mass1,
    double softening_length1,
    double sink_rate1,
    double sink_radius1,
    int sink_model1,
    double x2, // point mass 2
    double y2,
    double vx2,
    double vy2,
    double mass2,
    double softening_length2,
    double sink_rate2,
    double sink_radius2,
    int sink_model2,
    double cs2, // equation of state
    double mach_squared,
    int eos_type,
    double nu, // kinematic viscosity coefficient
    double rk_param, // RK parameter
    double dt, // timestep
    double velocity_ceiling)
{
    // Gaussian weights at quadrature points
    static double gauss_weights_1d[ORDER] = {0.555555555555556, 0.888888888888889, 0.555555555555556};    
    // Gaussian quadrature points in scaled domain xsi=[-1,1]
    double gauss_xsi_1d[ORDER] = {-0.774596669241483, 0.000000000000000, 0.774596669241483};
    double dx = (patch_xr - patch_xl) / ni;
    double dy = (patch_yr - patch_yl) / nj;
    double cell_volume = dx * dy;

    int ng = 1; // number of guard zones
    int si = NCONS * ORDER * ORDER * (nj + 2 * ng);
    int sj = NCONS * ORDER * ORDER;

    struct PointMass m1 = {x1, y1, vx1, vy1, mass1, softening_length1, sink_rate1, sink_radius1, sink_model1};
    struct PointMass m2 = {x2, y2, vx2, vy2, mass2, softening_length2, sink_rate2, sink_radius2, sink_model2};
    struct PointMassList mass_list = {{m1, m2}};

    FOR_EACH_2D(ni, nj)
    {
        // Cache phi and phi gradients
        // --------------------------------------------------------------------
        double phi_volume[ORDER][ORDER][ORDER][ORDER]; // i_quad x j_quad x m x n
        double phi_gradient_x[ORDER][ORDER][ORDER][ORDER];
        double phi_gradient_y[ORDER][ORDER][ORDER][ORDER];
        double phi_face_xl[ORDER][ORDER][ORDER]; // quad x m x n
        double phi_face_xr[ORDER][ORDER][ORDER];
        double phi_face_yl[ORDER][ORDER][ORDER];
        double phi_face_yr[ORDER][ORDER][ORDER];

        for (int i_quad = 0; i_quad < ORDER; ++i_quad)
        {
            for (int j_quad = 0; j_quad < ORDER; ++j_quad)
            {
                for (int m = 0; m < ORDER; ++m)
                {
                    for (int n = 0; n < ORDER; ++n)
                    {
                        phi_volume[i_quad][j_quad][m][n] = basis_phi_2d(i_quad, j_quad, m, n, 0, 0);
                        phi_gradient_x[i_quad][j_quad][m][n] = basis_phi_2d(i_quad, j_quad, m, n, 1, 0);
                        phi_gradient_y[i_quad][j_quad][m][n] = basis_phi_2d(i_quad, j_quad, m, n, 0, 1);
                    }
                }
            }
        }

        for (int quad = 0; quad < ORDER; ++quad)
        {
            for (int m = 0; m < ORDER; ++m)
            {
                for (int n = 0; n < ORDER; ++n)
                {
                    phi_face_xl[quad][m][n] = basis_phi_2d(L_ENDPOINT, quad, m, n, 0, 0);
                    phi_face_xr[quad][m][n] = basis_phi_2d(R_ENDPOINT, quad, m, n, 0, 0);
                    phi_face_yl[quad][m][n] = basis_phi_2d(quad, L_ENDPOINT, m, n, 0, 0);
                    phi_face_yr[quad][m][n] = basis_phi_2d(quad, R_ENDPOINT, m, n, 0, 0);                    
                }
            }
        }

        // Get the indexes and pointers to neighbor zones
        // --------------------------------------------------------------------
        int ncc = (i     + ng) * si + (j     + ng) * sj;
        int nli = (i - 1 + ng) * si + (j     + ng) * sj;
        int nri = (i + 1 + ng) * si + (j     + ng) * sj;
        int nlj = (i     + ng) * si + (j - 1 + ng) * sj;
        int nrj = (i     + ng) * si + (j + 1 + ng) * sj;

        double *ucc = &weights1[ncc];
        double *uli = &weights1[nli];
        double *uri = &weights1[nri];
        double *ulj = &weights1[nlj];
        double *urj = &weights1[nrj];

        // Define and initialize working arrays
        // --------------------------------------------------------------------
        double fhat[NCONS];
        double up[NCONS];
        double um[NCONS];
        double equation_19[NCONS][ORDER][ORDER];
        double equation_20[NCONS][ORDER][ORDER];
        double source_weights[NCONS][ORDER][ORDER];

        for (int q = 0; q < NCONS; ++q)
        {
            for (int m = 0; m < ORDER; ++m)
            {
                for (int n = 0; n < ORDER; ++n)
                {
                    equation_19[q][m][n] = 0.0;
                    equation_20[q][m][n] = 0.0;
                    source_weights[q][m][n] = 0.0;
                }
            }
        }

        // Compute the volume term
        // --------------------------------------------------------------------
        for (int i_quad = 0; i_quad < ORDER; ++i_quad)
        {
            for (int j_quad = 0; j_quad < ORDER; ++j_quad)
            {
                double gw = gauss_weights_1d[i_quad] * gauss_weights_1d[j_quad];
                double cons[NCONS];
                double prim[NCONS];
                double fx[NCONS];
                double fy[NCONS];
                reconstruct_2d(i_quad, j_quad, phi_volume, ucc, cons);
                conserved_to_primitive(cons, prim, velocity_ceiling);
                primitive_to_flux(prim, cons, fx, cs2, 0);
                primitive_to_flux(prim, cons, fy, cs2, 1);

                for (int m = 0; m < ORDER; ++m)
                {
                    for (int n = 0; n < ORDER; ++n)
                    {
                        if (m + n < ORDER)
                        {
                            double dphi_dx = phi_gradient_x[i_quad][j_quad][m][n];
                            double dphi_dy = phi_gradient_y[i_quad][j_quad][m][n];

                            for (int q = 0; q < NCONS; ++q)
                            {
                                equation_19[q][m][n] += dx * fx[q] * dphi_dx * gw;
                                equation_19[q][m][n] += dy * fy[q] * dphi_dy * gw;
                            }                            
                        }
                    }
                }

                // Source terms
                double xc = patch_xl + (i + 0.5) * dx;
                double yc = patch_yl + (j + 0.5) * dy;
                double x = xc + 0.5 * gauss_xsi_1d[i_quad] * dx;
                double y = yc + 0.5 * gauss_xsi_1d[j_quad] * dy;
                double du_source[NCONS];

                for (int q = 0; q < NCONS; ++q)
                {
                    du_source[q] = 0.0;
                }
                point_masses_source_term(&mass_list, x, y, dt, prim, du_source);

                for (int q = 0; q < NCONS; ++q)
                {
                    for (int m = 0; m < ORDER; ++m)
                    {
                        for (int n = 0; n < ORDER; ++n)
                        {
                            if (m + n < ORDER)
                            //if (m == 0 && n == 0)
                            {
                                source_weights[q][m][n] += 0.25 * du_source[q] * phi_volume[i_quad][j_quad][m][n] * gw;
                            }
                        }
                    }
                }
            }
        }

        // Compute the surface term
        // --------------------------------------------------------------------
        for (int quad = 0; quad < ORDER; ++quad)
        {
            reconstruct_1d(quad, phi_face_xl, ucc, up);
            reconstruct_1d(quad, phi_face_xr, uli, um);
            riemann_hlle(um, up, fhat, cs2, velocity_ceiling, 0);

            for (int q = 0; q < NCONS; ++q)
                for (int m = 0; m < ORDER; ++m)
                    for (int n = 0; n < ORDER; ++n)
                        if (m + n < ORDER)
                            equation_20[q][m][n] -=
                                dy * fhat[q] * phi_face_xl[quad][m][n] * gauss_weights_1d[quad];

            reconstruct_1d(quad, phi_face_xl, uri, up);
            reconstruct_1d(quad, phi_face_xr, ucc, um);
            riemann_hlle(um, up, fhat, cs2, velocity_ceiling, 0);

            for (int q = 0; q < NCONS; ++q)
                for (int m = 0; m < ORDER; ++m)
                    for (int n = 0; n < ORDER; ++n)
                        if (m + n < ORDER)
                            equation_20[q][m][n] +=
                                dy * fhat[q] * phi_face_xr[quad][m][n] * gauss_weights_1d[quad];

            reconstruct_1d(quad, phi_face_yl, ucc, up);
            reconstruct_1d(quad, phi_face_yr, ulj, um);
            riemann_hlle(um, up, fhat, cs2, velocity_ceiling, 1);

            for (int q = 0; q < NCONS; ++q)
                for (int m = 0; m < ORDER; ++m)
                    for (int n = 0; n < ORDER; ++n)
                        if (m + n < ORDER)
                            equation_20[q][m][n] -=
                                dx * fhat[q] * phi_face_yl[quad][m][n] * gauss_weights_1d[quad];

            reconstruct_1d(quad, phi_face_yl, urj, up);
            reconstruct_1d(quad, phi_face_yr, ucc, um);
            riemann_hlle(um, up, fhat, cs2, velocity_ceiling, 1);

            for (int q = 0; q < NCONS; ++q)
                for (int m = 0; m < ORDER; ++m)
                    for (int n = 0; n < ORDER; ++n)
                        if (m + n < ORDER)
                            equation_20[q][m][n] +=
                                dx * fhat[q] * phi_face_yr[quad][m][n] * gauss_weights_1d[quad];
        }

        double *w0 = &weights0[ncc];
        double *w1 = &weights1[ncc];
        double *w2 = &weights2[ncc];

        for (int q = 0; q < NCONS; ++q)
        {
            for (int m = 0; m < ORDER; ++m)
            {
                for (int n = 0; n < ORDER; ++n)
                {
                    if (m + n < ORDER)
                    {
                        int k = q * ORDER * ORDER + m * ORDER + n;
                        w2[k] = w1[k] + (equation_19[q][m][n] - equation_20[q][m][n]) * 0.5 * dt / cell_volume + source_weights[q][m][n];
                        w2[k] = (1.0 - rk_param) * w2[k] + rk_param * w0[k];
                    }
                }
            }
        }
    }
}


// PUBLIC void cbdisodg_2d_point_mass_source_term(
//     int ni,
//     int nj,
//     double patch_xl, // mesh
//     double patch_xr,
//     double patch_yl,
//     double patch_yr,
//     double x1, // point mass 1
//     double y1,
//     double vx1,
//     double vy1,
//     double mass1,
//     double softening_length1,
//     double sink_rate1,
//     double sink_radius1,
//     int sink_model1,
//     double x2, // point mass 2
//     double y2,
//     double vx2,
//     double vy2,
//     double mass2,
//     double softening_length2,
//     double sink_rate2,
//     double sink_radius2,
//     int sink_model2,
//     double velocity_ceiling,
//     int which_mass, // :: $ in [1, 2]
//     double *weights, // :: $.shape == (ni + 2, nj + 2, 3, 6)
//     double *cons_rate) // :: $.shape == (ni + 2, nj + 2, 3)
// {
//     struct PointMass m1 = {x1, y1, vx1, vy1, mass1, softening_length1, sink_rate1, sink_radius1, sink_model1};
//     struct PointMass m2 = {x2, y2, vx2, vy2, mass2, softening_length2, sink_rate2, sink_radius2, sink_model2};
//     struct PointMassList mass_list = {{m1, m2}};

//     // Gaussian quadrature points in scaled domain xsi=[-1,1]
//     double g[3] = {-0.774596669241483, 0.000000000000000, 0.774596669241483};
//     // Gaussian weights at quadrature points
//     double w[3] = { 0.555555555555556, 0.888888888888889, 0.555555555555556};
//         // Scaled LeGendre polynomials at quadrature points
//     double p[3][3] = {{ 1.000000000000000, 1.000000000000000, 1.000000000000000},
//                       {-1.341640786499873, 0.000000000000000, 1.341640786499873},
//                       { 0.894427190999914, -1.11803398874990, 0.894427190999914}};

//     int ng = 1; // number of guard zones
//     int si = NCONS * NPOLY * (nj + 2 * ng);
//     int sj = NCONS * NPOLY;

//     double dx = (patch_xr - patch_xl) / ni;
//     double dy = (patch_yr - patch_yl) / nj;

//     FOR_EACH_2D(ni, nj)
//     {
//         int ncc = (i + ng) * si + (j + ng) * sj;
//         double *ucc = &weights[ncc];
//         double *udot = &cons_rate[ncc];
//         double xc = patch_xl + (i + 0.5) * dx;
//         double yc = patch_yl + (j + 0.5) * dy;
//         // double *pc = &primitive[ncc];
//         // double *uc = &cons_rate[ncc];
//         // point_mass_source_term(&mass_list.masses[which_mass - 1], xc, yc, 1.0, pc, uc);

//         double u_dot[NCONS];
//         double u_dot_sum[NCONS];
//         double phi[NPOLY];

//         for (int q = 0; q < NCONS; ++q)
//         {
//             u_dot[q]     = 0.0;
//             u_dot_sum[q] = 0.0;
//         }

//         for (int ic = 0; ic < 3; ++ic)
//         {
//             for (int jc = 0; jc < 3; ++jc)
//             {
//                 double xp = xc + 0.5 * g[ic] * dx;
//                 double yp = yc + 0.5 * g[jc] * dy;

//                 // 2D basis functions phi_l(x,y) = P_m(x) * P_n(y) at cell points
//                 int il = 0;
//                 for (int m = 0; m < 3; ++m)
//                 {
//                     for (int n = 0; n < 3; ++n)
//                     {
//                         if ((n + m) < 3)
//                         {
//                             phi[il]  =  p[m][ic] *  p[n][jc];
//                             il += 1;
//                         }
//                     }
//                 }

//                 double uij[NCONS];
//                 double pij[NCONS];

//                 for (int q = 0; q < NCONS; ++q)
//                 {
//                     uij[q] = 0.0;

//                     for (int l = 0; l < NPOLY; ++l)
//                     {
//                         uij[q] += ucc[NPOLY * q + l] * phi[l];
//                     }
//                 }

//                 conserved_to_primitive(uij, pij, velocity_ceiling);
//                 point_mass_source_term(&mass_list.masses[which_mass - 1], xp, yp, 1.0, pij, u_dot);
//                 for (int q = 0; q < NCONS; ++q)
//                 {
//                     u_dot_sum[q] += w[ic] * w[jc] * u_dot[q];
//                 }
//             }
//         }

//         for (int q = 0; q < NCONS; ++q)
//         {
//             udot[q] = u_dot_sum[q];
//         }
//     }
// }


PUBLIC void cbdisodg_2d_wavespeed(
    int ni, // mesh
    int nj,
    double patch_xl,
    double patch_xr,
    double patch_yl,
    double patch_yr,
    double soundspeed2, // equation of state
    double mach_squared,
    int eos_type,
    double x1, // point mass 1
    double y1,
    double vx1,
    double vy1,
    double mass1,
    double softening_length1,
    double sink_rate1,
    double sink_radius1,
    int sink_model1,
    double x2, // point mass 2
    double y2,
    double vx2,
    double vy2,
    double mass2,
    double softening_length2,
    double sink_rate2,
    double sink_radius2,
    int sink_model2,
    double velocity_ceiling,
    double *weights,   // :: $.shape == (ni + 2, nj + 2, 3, 3, 3)
    double *wavespeed) // :: $.shape == (ni + 2, nj + 2)
{
    struct PointMass m1 = {x1, y1, vx1, vy1, mass1, softening_length1, sink_rate1, sink_radius1, sink_model1};
    struct PointMass m2 = {x2, y2, vx2, vy2, mass2, softening_length2, sink_rate2, sink_radius2, sink_model2};
    struct PointMassList mass_list = {{m1, m2}};

    int ng = 1; // number of guard zones
    int si = NCONS * ORDER * ORDER * (nj + 2 * ng);
    int sj = NCONS * ORDER * ORDER;
    int ti = nj + 2 * ng;
    int tj = 1;
    double dx = (patch_xr - patch_xl) / ni;
    double dy = (patch_yr - patch_yl) / nj;

    FOR_EACH_2D(ni, nj)
    {
        int nu = (i + ng) * si + (j + ng) * sj;
        int na = (i + ng) * ti + (j + ng) * tj;
        double x = patch_xl + (i + 0.5) * dx;
        double y = patch_yl + (j + 0.5) * dy;

        double *ucc = &weights[nu];

        double uij[NCONS];
        double pij[NCONS];

        // use zeroth weights for zone average of conserved variables
        for (int q = 0; q < NCONS; ++q)
        {
            uij[q] = ucc[q * ORDER * ORDER];
        }

        conserved_to_primitive(uij, pij, velocity_ceiling);
        double cs2 = sound_speed_squared(soundspeed2, mach_squared, eos_type, x, y, &mass_list);
        double a = primitive_max_wavespeed(pij, cs2);

        //if (a > 100.0)
        //{
        //    printf("large a! a = %3.2e at position (%+3.2f %+3.2f) prim=(%+3.2e %+3.2e %+3.2e)\n", a, x, y, pij[0], pij[1], pij[2]);
        //}
        wavespeed[na] = a;
    }
}
