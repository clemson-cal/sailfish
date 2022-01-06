// TODO: get rid of structs in function signatures
// QUESTION: can we have constant doubles?

// ============================ PHYSICS =======================================
// ============================================================================
#define NCONS 4
#define PLM_THETA 1.5
//#define GAMMA_LAW_INDEX (5.0 / 3.0)


// ============================ MATH ==========================================
// ============================================================================
#define min2(a, b) ((a) < (b) ? (a) : (b))
#define max2(a, b) ((a) > (b) ? (a) : (b))
#define min3(a, b, c) min2(a, min2(b, c))
#define max3(a, b, c) max2(a, max2(b, c))
#define sign(x) copysign(1.0, x)
#define minabs(a, b, c) min3(fabs(a), fabs(b), fabs(c))

PRIVATE double plm_gradient_scalar(double yl, double y0, double yr)
{
    double a = (y0 - yl) * PLM_THETA;
    double b = (yr - yl) * 0.5;
    double c = (yr - y0) * PLM_THETA;
    return 0.25 * fabs(sign(a) + sign(b)) * (sign(a) + sign(c)) * minabs(a, b, c);
}

PRIVATE void plm_gradient(double *yl, double *y0, double *yr, double *g)
{
    for (int q = 0; q < NCONS; ++q)
    {
        g[q] = plm_gradient_scalar(yl[q], y0[q], yr[q]);
    }
}

// ============================ INTERNAL STRUCTS ==============================
// ============================================================================
struct PointMass {
    double x;
    double y;
    double vx;
    double vy;
    double mass;
    double rate;
    double radius;
    int model;
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
    int mode;
};

// ============================ GRAVITY =======================================
// ============================================================================

PRIVATE double disk_height(
    struct PointMassList *mass_list,
    double x1,
    double y1,
    double *prim)
{
    if (mass_list->masses[0].model == 0 && mass_list->masses[1].model == 0)
    {
        return 1.0;
    }

    double omegatilde2 = 0.0;
    for (int p = 0; p < 2; ++p)
    {
        if (mass_list->masses[p].model != 0)
        {
            double x0 = mass_list->masses[p].x;
            double y0 = mass_list->masses[p].y;
            double mp = mass_list->masses[p].mass;

            double dx = x1 - x0;
            double dy = y1 - y0;
            double r2 = dx * dx + dy * dy + 1e-12;
            double r  = sqrt(r2);
            omegatilde2 += mp * pow(r, -3.0);
        }
    }
    double sigma = prim[0];
    double pres  = prim[3];

    return sqrt(pres / sigma) / sqrt(omegatilde2);
}

PRIVATE void point_mass_source_term(
    struct PointMass *mass,
    double x1,
    double y1,
    double dt,
    double *prim,
    double h,
    double *delta_cons,
    int constant_softening,
    double soft_length,
    double gamma_law_index)
{
    double x0 = mass->x;
    double y0 = mass->y;
    double mp = mass->mass;
    double rs = mass->radius;
    double sigma = prim[0];
    double pres  = prim[3];
    double gamma = gamma_law_index;
    double eps = pres / (gamma - 1.0) / sigma;

    double dx = x1 - x0;
    double dy = y1 - y0;
    double r2 = dx * dx + dy * dy;
    double softening_length = constant_softening ? soft_length : 0.5 * h;
    double r2_soft = r2 + pow(softening_length, 2.0);
    double dr = sqrt(r2);
    double mag = sigma * mp * pow(r2_soft, -1.5);
    double fx = -mag * dx;
    double fy = -mag * dy;
    double vx = prim[1];
    double vy = prim[2];
    double sink_rate = 0.0;

    if (dr < 4.0 * rs)
    {
        sink_rate = mass->rate * exp(-pow(dr / rs, 4.0));
    }
    if (!constant_softening)
    {
        if (dr < rs)
        {
            double transition = pow(1.0 - pow(dr / rs, 2.0), 2.0);
            double mod_rs = transition * rs + (1.0 - transition) * 0.5 * h;
            r2_soft = r2 + pow(mod_rs, 2.0);
            mag = sigma * mp / pow(r2_soft, 1.5);
            fx = -mag * dx;
            fy = -mag * dy;
        }
    }
    //if (dr < 1.0 * rs)
    //{
    //    sink_rate = mass->rate * pow(1.0 - pow(dr / rs, 2.0), 2.0);
    //}
    double mdot = sigma * sink_rate * -1.0;

    switch (mass->model) {
        case 1: // AccelerationFree
            delta_cons[0] = dt * mdot;
            delta_cons[1] = dt * mdot * prim[1] + dt * fx;
            delta_cons[2] = dt * mdot * prim[2] + dt * fy;
            delta_cons[3] = dt * (mdot * eps + 0.5 * mdot * (vx * vx + vy * vy)) + dt * (fx * vx + fy * vy);
            break;
        case 2: // TorqueFree
        {
            double vx        = prim[1];
            double vy        = prim[2];
            double vx0       = mass->vx;
            double vy0       = mass->vy;
            double rhatx     = dx / (dr + 1e-12);
            double rhaty     = dy / (dr + 1e-12);
            double dvdotrhat = (vx - vx0) * rhatx + (vy - vy0) * rhaty;
            double vxstar    = dvdotrhat * rhatx + vx0;
            double vystar    = dvdotrhat * rhaty + vy0;
            delta_cons[0] = dt * mdot;
            delta_cons[1] = dt * mdot * vxstar + dt * fx;
            delta_cons[2] = dt * mdot * vystar + dt * fy;
            delta_cons[3] = dt * (mdot * eps + 0.5 * mdot * (vxstar * vxstar + vystar * vystar)) + dt * (fx * vx + fy * vy);
            break;
        }
        case 3: //ForceFree
            delta_cons[0] = dt * mdot;
            delta_cons[1] = dt * fx;
            delta_cons[2] = dt * fy;
            delta_cons[3] = dt * (fx * vx + fy * vy);
            break;
        default:
            delta_cons[0] = 0.0;
            delta_cons[1] = 0.0;
            delta_cons[2] = 0.0;
            delta_cons[3] = 0.0;
            break;
    }
}

PRIVATE void point_masses_source_term(
    struct PointMassList *mass_list,
    double x1,
    double y1,
    double dt,
    double *prim,
    double h,
    double *cons,
    int constant_softening,
    double soft_length,
    double gamma_law_index)
{
    for (int p = 0; p < 2; ++p)
    {
        double delta_cons[NCONS];
        point_mass_source_term(&mass_list->masses[p], x1, y1, dt, prim, h, delta_cons, constant_softening, soft_length, gamma_law_index);

        for (int q = 0; q < NCONS; ++q)
        {
            cons[q] += delta_cons[q];
        }
    }
}

// ============================ EOS AND BUFFER ================================
// ============================================================================
PRIVATE double sound_speed_squared(
    double gamma_law_index,
    double *prim)
{
    return prim[3] / prim[0] * gamma_law_index;
}

PRIVATE void buffer_source_term(
    struct KeplerianBuffer *kb,
    double xc,
    double yc,
    double dt,
    double *cons,
    double gamma_law_index)
{
    switch (kb->mode)
    {
        case 0: // Default
             break;

        case 1: // KeplerianBuffer
        {
            double rc = sqrt(xc * xc + yc * yc);
            double surface_density = kb->surface_density;
            double surface_pressure = kb->surface_pressure;
            double central_mass = kb->central_mass;
            double driving_rate = kb->driving_rate;
            double outer_radius = kb->outer_radius;
            double onset_width = kb->onset_width;
            double onset_radius = outer_radius - onset_width;

            if (rc > onset_radius)
            {
                double pf = surface_density * sqrt(central_mass / rc);
                double px = pf * (-yc / rc);
                double py = pf * ( xc / rc);
                double kinetic_energy = 0.5 * (px * px + py * py) / surface_density;
                double energy = surface_pressure / (gamma_law_index - 1.0) + kinetic_energy;
                double u0[NCONS] = {surface_density, px, py, energy};

                double omega_outer = sqrt(central_mass * pow(onset_radius, -3.0));
                //double buffer_rate = driving_rate * omega_outer * max2(rc, 1.0);
                double buffer_rate = driving_rate * omega_outer * (rc - onset_radius) / (outer_radius - onset_radius);

                for (int q = 0; q < NCONS; ++q)
                {
                    cons[q] -= (cons[q] - u0[q]) * buffer_rate * dt;
                }
            }
            break;
        }
    }
}

PRIVATE void shear_strain(const double *gx, const double *gy, double dx, double dy, double *s)
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
PRIVATE void cooling_term(
    double cooling_coefficient,
    double mach_ceiling,
    double dt,
    double *prim,
    double *cons,
    double gamma_law_index)
{
    double gamma = gamma_law_index;
    double sigma = prim[0];
    double eps = prim[3] / prim[0] / (gamma - 1.0);
    double eps_cooled = eps * pow(1.0 + 3.0 * cooling_coefficient * pow(sigma, -2.0) * pow(eps, 3.0) * dt, -1.0 / 3.0);
    double vx = prim[1];
    double vy = prim[2];

    double ek = 0.5 * (vx * vx + vy * vy);
    eps_cooled = max2(eps_cooled, 2.0 * ek / gamma / (gamma - 1.0) * pow(mach_ceiling, -2.0));

    cons[3] += sigma * (eps_cooled - eps);
}

PRIVATE void conserved_to_primitive(
    const double *cons,
    double *prim,
    double velocity_ceiling,
    double density_floor,
    double pressure_floor,
    double gamma_law_index)
{
    double gamma = gamma_law_index;
    double pres  = max2(pressure_floor, (cons[3] - 0.5 * (cons[1] * cons[1] + cons[2] * cons[2]) / cons[0]) * (gamma - 1.0));
    double vx = sign(cons[1]) * min2(fabs(cons[1] / cons[0]), velocity_ceiling);
    double vy = sign(cons[2]) * min2(fabs(cons[2] / cons[0]), velocity_ceiling);
    double rho = cons[0];

    if (cons[0] < density_floor)
    {
        rho = density_floor;
        vx = 0.0;
        vy = 0.0;
        pres = pressure_floor;
    }

    prim[0] = rho;
    prim[1] = vx;
    prim[2] = vy;
    prim[3] = pres;
}

PRIVATE void primitive_to_conserved(const double *prim, double *cons, double gamma_law_index)
{
    double gamma = gamma_law_index;
    double rho = prim[0];
    double vx = prim[1];
    double vy = prim[2];
    double pres = prim[3];
    double px = vx * rho;
    double py = vy * rho;
    double en = pres / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy);

    cons[0] = rho;
    cons[1] = px;
    cons[2] = py;
    cons[3] = en;
}

PRIVATE double primitive_to_velocity(const double *prim, int direction)
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
    int direction)
{
    double vn = primitive_to_velocity(prim, direction);
    double pressure = prim[3];

    flux[0] = vn * cons[0];
    flux[1] = vn * cons[1] + pressure * (direction == 0);
    flux[2] = vn * cons[2] + pressure * (direction == 1);
    flux[3] = vn * (cons[3] + pressure);
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

PRIVATE double primitive_max_wavespeed(const double *prim, double cs2)
{
    double cs = sqrt(cs2);
    double vx = prim[1];
    double vy = prim[2];
    double ax = max2(fabs(vx - cs), fabs(vx + cs));
    double ay = max2(fabs(vy - cs), fabs(vy + cs));
    return max2(ax, ay);
}

PRIVATE void riemann_hlle(const double *pl, const double *pr, double *flux, double cs2, int direction, double gamma_law_index)
{
    double ul[NCONS];
    double ur[NCONS];
    double fl[NCONS];
    double fr[NCONS];
    double al[2];
    double ar[2];

    primitive_to_conserved(pl, ul, gamma_law_index);
    primitive_to_conserved(pr, ur, gamma_law_index);
    primitive_to_flux(pl, ul, fl, direction);
    primitive_to_flux(pr, ur, fr, direction);
    primitive_to_outer_wavespeeds(pl, al, cs2, direction);
    primitive_to_outer_wavespeeds(pr, ar, cs2, direction);

    const double am = min3(0.0, al[0], ar[0]);
    const double ap = max3(0.0, al[1], ar[1]);

    for (int q = 0; q < NCONS; ++q)
    {
        flux[q] = (fl[q] * ap - fr[q] * am - (ul[q] - ur[q]) * ap * am) / (ap - am);
    }
}




PUBLIC cbdgam_2d_advance_rk(
    int ni,
    int nj,
    double patch_xl, //Mesh
    double patch_xr,
    double patch_yl,
    double patch_yr,
    double *conserved_rk, // :: $.shape == (ni + 4, nj + 4, 4)
    double *primitive_rd, // :: $.shape == (ni + 4, nj + 4, 4)
    double *primitive_wr, // :: $.shape == (ni + 4, nj + 4, 4)
    double gamma_law_index,
    double kb_surface_density, // KeplerianBuffer
    double kb_surface_pressure,
    double kb_central_mass,
    double kb_driving_rate,
    double kb_outer_radius,
    double kb_onset_width,
    int kb_mode,
    double x1, // PointMass*2
    double y1,
    double vx1,
    double vy1,
    double mass1,
    double rate1,
    double radius1,
    int model1,
    double x2,
    double y2,
    double vx2,
    double vy2,
    double mass2,
    double rate2,
    double radius2,
    int model2,
    double alpha, // other
    double a,
    double dt,
    double velocity_ceiling,
    double cooling_coefficient,
    double mach_ceiling,
    double density_floor,
    double pressure_floor,
    int constant_softening,
    double soft_length)
{
    struct KeplerianBuffer kb = {kb_surface_density,
                                 kb_surface_pressure,
                                 kb_central_mass,
                                 kb_driving_rate,
                                 kb_outer_radius,
                                 kb_onset_width,
                                 kb_mode};
    struct PointMass pointmass1 = {x1, y1, vx1, vy1, mass1, rate1, radius1, model1};
    struct PointMass pointmass2 = {x2, y2, vx2, vy2, mass2, rate2, radius2, model2};
    struct PointMassList mass_list = {{pointmass1, pointmass2}};

    double dx = (patch_xr - patch_xl) / ni;
    double dy = (patch_yr - patch_yl) / nj;

    int ng = 2; // number of guard zones
    int si = NCONS * (nj + 2 * ng);
    int sj = NCONS;

    FOR_EACH_2D(ni, nj)
    {
        double xl = patch_xl + (i + 0.0) * dx;
        double xc = patch_xl + (i + 0.5) * dx;
        double xr = patch_xl + (i + 1.0) * dx;
        double yl = patch_yl + (j + 0.0) * dy;
        double yc = patch_yl + (j + 0.5) * dy;
        double yr = patch_yl + (j + 1.0) * dy;

        // ------------------------------------------------------------------------
        //                 tj
        //
        //      +-------+-------+-------+
        //      |       |       |       |
        //      |  lr   |  rj   |   rr  |
        //      |       |       |       |
        //      +-------+-------+-------+
        //      |       |       |       |
        //  ki  |  li  -|+  c  -|+  ri  |  ti
        //      |       |       |       |
        //      +-------+-------+-------+
        //      |       |       |       |
        //      |  ll   |  lj   |   rl  |
        //      |       |       |       |
        //      +-------+-------+-------+
        //
        //                 kj
        // ------------------------------------------------------------------------

        int ncc = (i     + ng) * si + (j     + ng) * sj;
        int nli = (i - 1 + ng) * si + (j     + ng) * sj;
        int nri = (i + 1 + ng) * si + (j     + ng) * sj;
        int nlj = (i     + ng) * si + (j - 1 + ng) * sj;
        int nrj = (i     + ng) * si + (j + 1 + ng) * sj;
        int nki = (i - 2 + ng) * si + (j     + ng) * sj;
        int nti = (i + 2 + ng) * si + (j     + ng) * sj;
        int nkj = (i     + ng) * si + (j - 2 + ng) * sj;
        int ntj = (i     + ng) * si + (j + 2 + ng) * sj;
        int nll = (i - 1 + ng) * si + (j - 1 + ng) * sj;
        int nlr = (i - 1 + ng) * si + (j + 1 + ng) * sj;
        int nrl = (i + 1 + ng) * si + (j - 1 + ng) * sj;
        int nrr = (i + 1 + ng) * si + (j + 1 + ng) * sj;

        double *un = conserved_rk[ncc];
        double *pcc = primitive_rd[ncc];
        double *pli = primitive_rd[nli];
        double *pri = primitive_rd[nri];
        double *plj = primitive_rd[nlj];
        double *prj = primitive_rd[nrj];
        double *pki = primitive_rd[nki];
        double *pti = primitive_rd[nti];
        double *pkj = primitive_rd[nkj];
        double *ptj = primitive_rd[ntj];
        double *pll = primitive_rd[nll];
        double *plr = primitive_rd[nlr];
        double *prl = primitive_rd[nrl];
        double *prr = primitive_rd[nrr];

        double plip[NCONS];
        double plim[NCONS];
        double prip[NCONS];
        double prim[NCONS];
        double pljp[NCONS];
        double pljm[NCONS];
        double prjp[NCONS];
        double prjm[NCONS];

        double gxli[NCONS];
        double gxri[NCONS];
        double gyli[NCONS];
        double gyri[NCONS];
        double gxlj[NCONS];
        double gxrj[NCONS];
        double gylj[NCONS];
        double gyrj[NCONS];
        double gxcc[NCONS];
        double gycc[NCONS];

        plm_gradient(pki, pli, pcc, gxli);
        plm_gradient(pli, pcc, pri, gxcc);
        plm_gradient(pcc, pri, pti, gxri);
        plm_gradient(pkj, plj, pcc, gylj);
        plm_gradient(plj, pcc, prj, gycc);
        plm_gradient(pcc, prj, ptj, gyrj);
        plm_gradient(pll, pli, plr, gyli);
        plm_gradient(prl, pri, prr, gyri);
        plm_gradient(pll, plj, prl, gxlj);
        plm_gradient(plr, prj, prr, gxrj);

        for (int q = 0; q < NCONS; ++q)
        {
            plim[q] = pli[q] + 0.5 * gxli[q];
            plip[q] = pcc[q] - 0.5 * gxcc[q];
            prim[q] = pcc[q] + 0.5 * gxcc[q];
            prip[q] = pri[q] - 0.5 * gxri[q];

            pljm[q] = plj[q] + 0.5 * gylj[q];
            pljp[q] = pcc[q] - 0.5 * gycc[q];
            prjm[q] = pcc[q] + 0.5 * gycc[q];
            prjp[q] = prj[q] - 0.5 * gyrj[q];
        }

        double fli[NCONS];
        double fri[NCONS];
        double flj[NCONS];
        double frj[NCONS];
        double ucc[NCONS];

        double cs2li = sound_speed_squared(gamma_law_index, pli);
        double cs2ri = sound_speed_squared(gamma_law_index, pri);
        double cs2lj = sound_speed_squared(gamma_law_index, plj);
        double cs2rj = sound_speed_squared(gamma_law_index, prj);

        riemann_hlle(plim, plip, fli, cs2li, 0, gamma_law_index);
        riemann_hlle(prim, prip, fri, cs2ri, 0, gamma_law_index);
        riemann_hlle(pljm, pljp, flj, cs2lj, 1, gamma_law_index);
        riemann_hlle(prjm, prjp, frj, cs2rj, 1, gamma_law_index);

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

        double cs2cc = sound_speed_squared(gamma_law_index, pcc);
        double hcc = disk_height(&mass_list, xc, yc, pcc);
        double hli = disk_height(&mass_list, xl, yc, pli);
        double hri = disk_height(&mass_list, xr, yc, pri);
        double hlj = disk_height(&mass_list, xc, yl, plj);
        double hrj = disk_height(&mass_list, xc, yr, prj);

        double nucc = alpha * hcc * sqrt(cs2cc);
        double nuli = alpha * hli * sqrt(cs2li);
        double nuri = alpha * hri * sqrt(cs2ri);
        double nulj = alpha * hlj * sqrt(cs2lj);
        double nurj = alpha * hrj * sqrt(cs2rj);

        fli[1] -= 0.5 * (nuli * pli[0] * sli[0] + nucc * pcc[0] * scc[0]); // x-x
        fli[2] -= 0.5 * (nuli * pli[0] * sli[1] + nucc * pcc[0] * scc[1]); // x-y
        fri[1] -= 0.5 * (nucc * pcc[0] * scc[0] + nuri * pri[0] * sri[0]); // x-x
        fri[2] -= 0.5 * (nucc * pcc[0] * scc[1] + nuri * pri[0] * sri[1]); // x-y
        flj[1] -= 0.5 * (nulj * plj[0] * slj[2] + nucc * pcc[0] * scc[2]); // y-x
        flj[2] -= 0.5 * (nulj * plj[0] * slj[3] + nucc * pcc[0] * scc[3]); // y-y
        frj[1] -= 0.5 * (nucc * pcc[0] * scc[2] + nurj * prj[0] * srj[2]); // y-x
        frj[2] -= 0.5 * (nucc * pcc[0] * scc[3] + nurj * prj[0] * srj[3]); // y-y

        fli[3] -= 0.5 * (nuli * pli[0] * sli[0] * pli[1] + nucc * pcc[0] * scc[0] * pcc[1]); // v^x \tau^x_x
        fri[3] -= 0.5 * (nucc * pcc[0] * scc[0] * pcc[1] + nuri * pri[0] * sri[0] * pri[1]);
        fli[3] -= 0.5 * (nuli * pli[0] * sli[1] * pli[2] + nucc * pcc[0] * scc[1] * pcc[2]); // v^y \tau^x_y
        fri[3] -= 0.5 * (nucc * pcc[0] * scc[1] * pcc[2] + nuri * pri[0] * sri[1] * pri[2]);
        flj[3] -= 0.5 * (nulj * plj[0] * slj[2] * plj[1] + nucc * pcc[0] * scc[2] * pcc[1]); // v^x \tau^y_x
        frj[3] -= 0.5 * (nucc * pcc[0] * scc[2] * pcc[1] + nurj * prj[0] * srj[2] * prj[1]);
        flj[3] -= 0.5 * (nulj * plj[0] * slj[3] * plj[2] + nucc * pcc[0] * scc[3] * pcc[2]); // v^y \tau^y_y
        frj[3] -= 0.5 * (nucc * pcc[0] * scc[3] * pcc[2] + nurj * prj[0] * srj[3] * prj[2]);

        primitive_to_conserved(pcc, ucc, gamma_law_index);
        buffer_source_term(&kb, xc, yc, dt, ucc, gamma_law_index);
        point_masses_source_term(&mass_list, xc, yc, dt, pcc, hcc, ucc, constant_softening, soft_length, gamma_law_index);
        cooling_term(cooling_coefficient, mach_ceiling, dt, pcc, ucc, gamma_law_index);

        for (int q = 0; q < NCONS; ++q)
        {
            ucc[q] -= ((fri[q] - fli[q]) / dx + (frj[q] - flj[q]) / dy) * dt;
            ucc[q] = (1.0 - a) * ucc[q] + a * un[q];
        }
        double *pout = primitive_wr[ncc];
        conserved_to_primitive(ucc, pout, velocity_ceiling, density_floor, pressure_floor, gamma_law_index);

    }
}

PUBLIC cbdgam_2d_wavespeed(
    int ni,
    int nj,
    double *primitive, // :: $.shape == (ni + 4, nj + 4, 4)
    double *wavespeed, // :: $.shape == (ni + 4, nj + 4)
    double gamma_law_index)
{
    int ng = 2; // number of guard zones
    int si = NCONS * (nj + 2 * ng);
    int sj = NCONS;
    int ti = nj + 2 * ng;
    int tj = 1;

    FOR_EACH_2D(ni, nj)
    {
        int np = (i + ng) * si + (j + ng) * sj;
        int na = (i + ng) * ti + (j + ng) * tj;

        double *pc = &primitive[np];
        double cs2 = sound_speed_squared(gamma_law_index, pc);
        double a = primitive_max_wavespeed(pc, cs2);
        wavespeed[na] = a;
    }
}

PUBLIC cbdgam_2d_primitive_to_conserved(
    int ni,
    int nj,
    double *primitive, // :: $.shape == (ni + 4, nj + 4, 4)
    double *conserved, // :: $.shape == (ni + 4, nj + 4, 4)
    double gamma_law_index)
{
    int ng = 2; // number of guard zones
    int si = NCONS * (nj + 2 * ng);
    int sj = NCONS;

    FOR_EACH_2D(ni, nj)
    {
        int n = (i + ng) * si + (j + ng) * sj;

        double *pc = &primitive[n];
        double *uc = &conserved[n];
        primitive_to_conserved(pc, uc, gamma_law_index);
    }
}

PUBLIC cbdgam_2d_point_mass_source_term(
    int ni,
    int nj,
    double patch_xl, //Mesh
    double patch_xr,
    double patch_yl,
    double patch_yr,
    double x1, // PointMass*2
    double y1,
    double vx1,
    double vy1,
    double mass1,
    double rate1,
    double radius1,
    int model1,
    double x2,
    double y2,
    double vx2,
    double vy2,
    double mass2,
    double rate2,
    double radius2,
    int model2,
    int which_mass,
    double *primitive, // :: $.shape == (ni + 4, nj + 4, 4)
    double *cons_rate, // :: $.shape == (ni + 4, nj + 4, 4)
    int constant_softening,
    double soft_length,
    double gamma_law_index)
{
    struct PointMass pointmass1 = {x1, y1, vx1, vy1, mass1, rate1, radius1, model1};
    struct PointMass pointmass2 = {x2, y2, vx2, vy2, mass2, rate2, radius2, model2};
    struct PointMassList mass_list = {{pointmass1, pointmass2}};

    struct Pointmass pointmass;
    if (which_mass == 1) {
        struct p = {x1, y1, vx1, vy1, mass1, rate1, radius1, model1};
        pointmass = p;
    }
    if (which_mass == 2) {
        struct p = {x2, y2, vx2, vy2, mass2, rate2, radius2, model2};
        pointmass = p;
    }

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
        double h = disk_height(&mass_list, xc, yc, pc);
        point_mass_source_term(&pointmass, xc, yc, 1.0, pc, h, uc, constant_softening, soft_length, gamma_law_index);
    }
}
