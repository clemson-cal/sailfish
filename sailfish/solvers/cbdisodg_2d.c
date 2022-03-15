/*
MODULE: cbdisodg_2d

DESCRIPTION: Isothermal DG solver for a binary accretion problem in 2D planar
  cartesian coordinates.
*/

// ============================ PHYSICS =======================================
// ============================================================================
#define NCONS 3

// ============================ SCHEME =======================================
// ============================================================================
#define NPOLY 3

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
    double *cons_dot)
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
            cons_dot[0] =  mdot;
            cons_dot[1] =  mdot * prim[1] +  fx;
            cons_dot[2] =  mdot * prim[2] +  fy;
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
            cons_dot[0] =  mdot;
            cons_dot[1] =  mdot * vxstar +  fx;
            cons_dot[2] =  mdot * vystar +  fy;
            break;
        }
        case 3: // force-free
        {
            cons_dot[0] =  mdot;
            cons_dot[1] =  fx;
            cons_dot[2] =  fy;
            break;
        }
        default: // sink is inactive
        {
            cons_dot[0] = 0.0;
            cons_dot[1] = 0.0;
            cons_dot[2] = 0.0;
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
    double *cons_dot)
{
    for (int p = 0; p < 2; ++p)
    {
        double cons_dot_single[NCONS];
        point_mass_source_term(&mass_list->masses[p], x1, y1, dt, prim, cons_dot_single);

        for (int q = 0; q < NCONS; ++q)
        {
            cons_dot[q] += cons_dot_single[q];
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
    double dt,
    double *cons,
    double *delta_cons)
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
                // cons[q] -= (cons[q] - u0[q]) * buffer_rate * dt;
                delta_cons[q] = -(cons[q] - u0[q]) * buffer_rate;
            }
        }
    }
}

PRIVATE void shear_strain(
    const double *gx,
    const double *gy,
    double dx,
    double dy,
    double *s)
{
    double sxx = 4.0 / 3.0 * gx[1] / dx - 2.0 / 3.0 * gy[2] / dy;
    double syy =-2.0 / 3.0 * gx[1] / dx + 4.0 / 3.0 * gy[2] / dy;
    double sxy = 1.0 / 1.0 * gx[2] / dx + 1.0 / 1.0 * gy[1] / dy;
    double syx = sxy;
    s[0] = sxx;
    s[1] = sxy;
    s[2] = syx;
    s[3] = syy;
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

PRIVATE void primitive_to_conserved(
    const double *prim,
    double *cons)
{
    double rho = prim[0];
    double vx = prim[1];
    double vy = prim[2];
    double px = vx * rho;
    double py = vy * rho;

    cons[0] = rho;
    cons[1] = px;
    cons[2] = py;
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
    int direction)
{
    double pl[NCONS];
    double pr[NCONS];
    double fl[NCONS];
    double fr[NCONS];
    double al[2];
    double ar[2];

    conserved_to_primitive(ul, pl);
    conserved_to_primitive(ur, pr);
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

PRIVATE double dot(double *u, double *p) 
{
    double sum = 0.0;

    for (int i = 0; i < NPOLY; ++i) {
        sum += u[i] * p[i]; 
    }
    return sum;
}

// ============================ PUBLIC API ====================================
// ============================================================================
PUBLIC void cbdiso_2d_advance_rk(
    int ni,
    int nj,
    double patch_xl, // mesh
    double patch_xr,
    double patch_yl,
    double patch_yr,
    double *conserved_rk, // :: $.shape == (ni + 2, nj + 2, NCONS * NPOLY)
    double *conserved_rd, // :: $.shape == (ni + 2, nj + 2, NCONS * NPOLY)
    double *conserved_wr, // :: $.shape == (ni + 2, nj + 2, NCONS * NPOLY)
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
    double a, // RK parameter
    double dt, // timestep
    double velocity_ceiling)
{
    // Gaussian quadrature points in scaled domain xsi=[-1,1]
    double g[3] = {-0.774596669241483, 0.000000000000000, 0.774596669241483}; 
    // Gaussian weights at quadrature points
    double w[3] = { 0.555555555555556, 0.888888888888889, 0.555555555555556};
    // Scaled LeGendre polynomials at quadrature points
    double p[3][3] = {{ 1.000000000000000, 1.000000000000000, 1.000000000000000},
                      {-1.341640786499873, 0.000000000000000, 1.341640786499873},
                      { 0.894427190999914, -1.11803398874990, 0.894427190999914}}; 
    // Derivative of Scaled LeGendre polynomials at quadrature points
    double pp[3][3] = {{ 0.000000000000000, 0.000000000000000, 0.000000000000000},
                       { 1.732050807568877, 1.732050807568877, 1.732050807568877},
                       {-5.196152422706629, 0.000000000000000, 5.196152422706629}};
    // Unit normal vector at left and right faces
    double nhat[2] = {-1.0, 1.0};
    // Scaled LeGendre polynomials at left face
    double pfl[3] = {1.000000000000000, -1.732050807568877, 2.23606797749979};
    // Scaled LeGendre polynomials at right face
    double pfr[3] = {1.000000000000000,  1.732050807568877, 2.23606797749979};

    struct KeplerianBuffer buffer = {
        buffer_surface_density,
        buffer_central_mass,
        buffer_driving_rate,
        buffer_outer_radius,
        buffer_onset_width,
        buffer_is_enabled
    };
    struct PointMass m1 = {x1, y1, vx1, vy1, mass1, softening_length1, sink_rate1, sink_radius1, sink_model1};
    struct PointMass m2 = {x2, y2, vx2, vy2, mass2, softening_length2, sink_rate2, sink_radius2, sink_model2};
    struct PointMassList mass_list = {{m1, m2}};

    double dx = (patch_xr - patch_xl) / ni;
    double dy = (patch_yr - patch_yl) / nj;

    int ng = 1; // number of guard zones
    int si = NCONS * NPOLY * (nj + 2 * ng);
    int sj = NCONS * NPOLY;

    FOR_EACH_2D(ni, nj)
    {
        double xl = patch_xl + (i + 0.0) * dx;
        double xc = patch_xl + (i + 0.5) * dx;
        double xr = patch_xl + (i + 1.0) * dx;
        double yl = patch_yl + (j + 0.0) * dy;
        double yc = patch_yl + (j + 0.5) * dy;
        double yr = patch_yl + (j + 1.0) * dy;

        // ------------------------------------------------------------------------
        //                 
        //
        //      +-------+-------+-------+
        //      |       |       | x x x |   x(ic, jc) = quadrature points in each zone
        //      |       |  rj   | x x x |
        //      |       |       | x x x |
        //      +-------+-------+-------+
        //      |       |       |       |
        //      |  li  -|+  c  -|+  ri  |  
        //      |       |       |       |
        //      +-------+-------+-------+
        //      |       |       |       |
        //      |       |  lj   |       |
        //      |       |       |       |
        //      +-------+-------+-------+
        //
        //                 
        // ------------------------------------------------------------------------

        int ncc = (i     + ng) * si + (j     + ng) * sj;
        int nli = (i - 1 + ng) * si + (j     + ng) * sj;
        int nri = (i + 1 + ng) * si + (j     + ng) * sj;
        int nlj = (i     + ng) * si + (j - 1 + ng) * sj;
        int nrj = (i     + ng) * si + (j + 1 + ng) * sj;

        double *un  = &conserved_rk[ncc];
        double *ucc = &conserved_rd[ncc];
        double *uli = &conserved_rd[nli];
        double *uri = &conserved_rd[nri];
        double *ulj = &conserved_rd[nlj];
        double *urj = &conserved_rd[nrj];

        double fli[NCONS];
        double fri[NCONS];
        double flj[NCONS];
        double frj[NCONS];
        double pcc[NCONS];
        double pij[NCONS];
        double uij[NCONS];
        double ulim[NCONS];
        double ulip[NCONS];
        double urim[NCONS];
        double urip[NCONS];
        double uljm[NCONS];
        double uljp[NCONS];
        double urjm[NCONS];
        double urjp[NCONS];

        // face node values of basis function phi
        double phili[NPOLY];
        double phiri[NPOLY];   
        double philj[NPOLY];
        double phirj[NPOLY];

        // interior node values of basis function phi and derivatives
        double phiij[NPOLY];
        double dphidx[NPOLY];
        double dphidy[NPOLY];

        double surface_term[NCONS * NPOLY];
        double volume_term[NCONS * NPOLY];

        // Compute viscous fluxes here?

        if (nu > 0.0)
        {
            double sli[4];
            double sri[4];
            double slj[4];
            double srj[4];
            double scc[4];

            shear_strain(gxli, gyli, dx, dy, sli);
            shear_strain(gxri, gyri, dx, dy, sri);
            shear_strain(gxlj, gylj, dx, dy, slj);
            shear_strain(gxrj, gyrj, dx, dy, srj);
            shear_strain(gxcc, gycc, dx, dy, scc);

            fli[1] -= 0.5 * nu * (pli[0] * sli[0] + pcc[0] * scc[0]); // x-x
            fli[2] -= 0.5 * nu * (pli[0] * sli[1] + pcc[0] * scc[1]); // x-y
            fri[1] -= 0.5 * nu * (pcc[0] * scc[0] + pri[0] * sri[0]); // x-x
            fri[2] -= 0.5 * nu * (pcc[0] * scc[1] + pri[0] * sri[1]); // x-y
            flj[1] -= 0.5 * nu * (plj[0] * slj[2] + pcc[0] * scc[2]); // y-x
            flj[2] -= 0.5 * nu * (plj[0] * slj[3] + pcc[0] * scc[3]); // y-y
            frj[1] -= 0.5 * nu * (pcc[0] * scc[2] + prj[0] * srj[2]); // y-x
            frj[2] -= 0.5 * nu * (pcc[0] * scc[3] + prj[0] * srj[3]); // y-y
        }

        // Surface term; loop over face nodes

        for (int ip = 0; ip < 3; ++ip)
        {
            // 2D basis functions phi_l(x,y) = P_m(x) * P_n(y) at face points
            int il = 0;
            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    if ((n + m) < 3)
                    {
                        phili[il] = pfl[m] * p[n][ip];
                        phiri[il] = pfr[m] * p[n][ip];
                        philj[il] = pfl[n] * p[m][ip];
                        phirj[il] = pfr[n] * p[m][ip];
                        il += 1;
                    }
                }
            } 

            double xp = xc + 0.5 * g[ip] * dx;
            double yp = yc + 0.5 * g[ip] * dy;

            double cs2li = sound_speed_squared(cs2, mach_squared, eos_type, xl, yp, &mass_list);
            double cs2ri = sound_speed_squared(cs2, mach_squared, eos_type, xr, yp, &mass_list);            
            double cs2lj = sound_speed_squared(cs2, mach_squared, eos_type, xp, yl, &mass_list);
            double cs2rj = sound_speed_squared(cs2, mach_squared, eos_type, xp, yr, &mass_list);   

            for (int q = 0; q < NCONS; ++q){

                ulim[q] = 0.0;
                ulip[q] = 0.0;
                urim[q] = 0.0;
                urip[q] = 0.0;
                uljm[q] = 0.0;
                uljp[q] = 0.0;
                urjm[q] = 0.0;
                urjp[q] = 0.0;

                for (int l = 0; l < NPOLY; ++l)
                {
                    ulim[q] += uli[NPOLY * q + l] * phiri[l]; // right face of zone i-1 
                    ulip[q] += ucc[NPOLY * q + l] * phili[l]; // left face of zone i                     
                    urim[q] += ucc[NPOLY * q + l] * phiri[l]; // right face of zone i 
                    urip[q] += uri[NPOLY * q + l] * phili[l]; // left face of zone i+1                     
                    uljm[q] += ulj[NPOLY * q + l] * phirj[l]; // top face of zone j-1 
                    uljp[q] += ucc[NPOLY * q + l] * philj[l]; // bottom face of zone j                     
                    urjm[q] += ucc[NPOLY * q + l] * phirj[l]; // top face of zone j 
                    urjp[q] += urj[NPOLY * q + l] * philj[l]; // bottom face of zone j+1                     j
                }
            }

            riemann_hlle(ulim, ulip, fli, cs2li, 0);
            riemann_hlle(urim, urip, fri, cs2ri, 0);
            riemann_hlle(uljm, uljp, flj, cs2lj, 1);
            riemann_hlle(urjm, urjp, frj, cs2rj, 1);

            // Add viscous fluxes here? Use strain averaged across face? 
            
            for (int q = 0; q < NCONS; ++q)
            {
                for (int l = 0; l < NPOLY; ++l)
                {    
                    surface_term[NPOLY * q + l] -= (
                        fli[q] * nhat[0] * phili[l] * w[ip] * dx +
                        fri[q] * nhat[1] * phiri[l] * w[ip] * dx +
                        flj[q] * nhat[0] * philj[l] * w[ip] * dy +
                        frj[q] * nhat[1] * phirj[l] * w[ip] * dy
                        );
                }
            }
        }

        // Volume term including source terms
        for (int ic = 0; ic < 3; ++ic)
        {
            for (int jc = 0; jc < 3; ++jc)
            {
                double xp = xc + 0.5 * g[ic] * dx;
                double yp = yc + 0.5 * g[jc] * dy;

                double cs2ij = sound_speed_squared(cs2, mach_squared, eos_type, xp, yp, &mass_list);

                // 2D basis functions phi_l(x,y) = P_m(x) * P_n(y) and derivatives at cell points
                int il = 0;
                for (int m = 0; m < 3; ++m)
                {
                    for (int n = 0; n < 3; ++n)
                    {
                        if ((n + m) < 3)
                        {
                            phiij[il]  =  p[m][ic] *  p[n][jc];
                            dphidx[il] = pp[m][ic] *  p[n][jc];
                            dphidy[il] =  p[m][ic] * pp[n][jc];
                            il += 1;
                        }
                    }
                }

                double uij[NCONS];
                double pij[NCONS];

                for (int q = 0; q < NCONS; ++q)
                {
                    uij[q] = 0.0;

                    for (int l = 0; l < NPOLY; ++l)
                    {
                        uij[q] += ucc[NPOLY * q + l] * phiij[l]; 
                    }
                }

                conserved_to_primitive(uij, pij, velocity_ceiling);

                double flux_x[NCONS];
                double flux_y[NCONS];

                primitive_to_flux(pij, uij, flux_x, cs2ij, 0);
                primitive_to_flux(pij, uij, flux_y, cs2ij, 1);

                // Add viscous fluxes here? Use point value of strain.

                for (int q = 0; q < NCONS; ++q)
                {
                    for (int l = 0; l < NPOLY; ++l)
                    {
                        volume_term[NPOLY * q + l] += 
                            w[ic] * w[jc] * 
                            (flux_x[q] * dphidx[l] * dx + flux_y[q] * dphidy[l] * dy 
                                + 0.5 * dx * dy * source_term * phiij[l]);
                    }
                }
            }
        }

        primitive_to_conserved(ucc, pcc);
        buffer_source_term(&buffer, xc, yc, dt, ucc);
        point_masses_source_term(&mass_list, xc, yc, dt, pcc, ucc);

        for (int q = 0; q < NCONS; ++q)
        {
            for (int l = 0; l < NK; ++l)
            {
                //ucc[q] -= ((fri[q] - fli[q]) / dx + (frj[q] - flj[q]) / dy) * dt;
                ucc[NPOLY * q + l] += 0.5 * (surface + volume) / (dx * dy);
                ucc[NPOLY * q + l] = (1.0 - a) * ucc[NPOLY * q + l] + a * un[NPOLY * q + l];
            }
        }
        double *pout = &primitive_wr[ncc];
        conserved_to_primitive(ucc, pout, velocity_ceiling);
    }
}

PUBLIC void cbdiso_2d_primitive_to_conserved(
    int ni,
    int nj,
    double *primitive, // :: $.shape == (ni + 4, nj + 4, 3)
    double *conserved) // :: $.shape == (ni + 4, nj + 4, 3)
{
    int ng = 2; // number of guard zones
    int si = NCONS * (nj + 2 * ng);
    int sj = NCONS;

    FOR_EACH_2D(ni, nj)
    {
        int n = (i + ng) * si + (j + ng) * sj;

        double *pc = &primitive[n];
        double *uc = &conserved[n];
        primitive_to_conserved(pc, uc);
    }
}

PUBLIC void cbdiso_2d_point_mass_source_term(
    int ni,
    int nj,
    double patch_xl, // mesh
    double patch_xr,
    double patch_yl,
    double patch_yr,
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
    int which_mass, // :: $ in [1, 2]
    double *primitive, // :: $.shape == (ni + 4, nj + 4, 3)
    double *cons_rate) // :: $.shape == (ni + 4, nj + 4, 3)
{
    struct PointMass m1 = {x1, y1, vx1, vy1, mass1, softening_length1, sink_rate1, sink_radius1, sink_model1};
    struct PointMass m2 = {x2, y2, vx2, vy2, mass2, softening_length2, sink_rate2, sink_radius2, sink_model2};
    struct PointMassList mass_list = {{m1, m2}};

    int ng = 2; // number of guard zones
    int si = NCONS * (nj + 2 * ng);
    int sj = NCONS;

    double dx = (patch_xr - patch_xl) / ni;
    double dy = (patch_yr - patch_yl) / nj;

    FOR_EACH_2D(ni, nj)
    {
        int ncc = (i + ng) * si + (j + ng) * sj;

        double xc = patch_xl + (i + 0.5) * dx;
        double yc = patch_yl + (j + 0.5) * dy;
        double *pc = &primitive[ncc];
        double *uc = &cons_rate[ncc];
        point_mass_source_term(&mass_list.masses[which_mass - 1], xc, yc, 1.0, pc, uc);
    }
}


PUBLIC void cbdiso_2d_wavespeed(
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
    double *primitive, // :: $.shape == (ni + 4, nj + 4, 3)
    double *wavespeed) // :: $.shape == (ni + 4, nj + 4)
{
    struct PointMass m1 = {x1, y1, vx1, vy1, mass1, softening_length1, sink_rate1, sink_radius1, sink_model1};
    struct PointMass m2 = {x2, y2, vx2, vy2, mass2, softening_length2, sink_rate2, sink_radius2, sink_model2};
    struct PointMassList mass_list = {{m1, m2}};

    int ng = 2; // number of guard zones
    int si = NCONS * (nj + 2 * ng);
    int sj = NCONS;
    int ti = nj + 2 * ng;
    int tj = 1;
    double dx = (patch_xr - patch_xl)/ni;
    double dy = (patch_yr - patch_yl)/nj;

    FOR_EACH_2D(ni, nj)
    {
        int np = (i + ng) * si + (j + ng) * sj;
        int na = (i + ng) * ti + (j + ng) * tj;

        double x = patch_xl + (i + 0.5) * dx;
        double y = patch_yl + (j + 0.5) * dy;

        double *pc = &primitive[np];
        double cs2 = sound_speed_squared(soundspeed2, mach_squared, eos_type, x, y, &mass_list);
        double a = primitive_max_wavespeed(pc, cs2);
        wavespeed[na] = a;
    }
}