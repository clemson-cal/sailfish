// TODO: get rid of structs in function signatures
// QUESTION: can we have constant doubles?

// ============================ PHYSICS =======================================
// ============================================================================
#define NCONS 4
#define PLM_THETA 1.5
#define GAMMA_LAW_INDEX (5.0 / 3.0)


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

// ============================ GRAVITY =======================================
// ============================================================================

PRIVATE double disk_height(
    struct PointMassList *mass_list,
    double x1,
    double y1,
    double *prim)
{
    double omegatilde2 = 0.0;
    for (int p = 0; p < mass_list->count; ++p)
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
    double soft_length)
{
    double x0 = mass->x;
    double y0 = mass->y;
    double mp = mass->mass;
    double rs = mass->radius;
    double sigma = prim[0];
    double pres  = prim[3];
    double gamma = GAMMA_LAW_INDEX;
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
        case AccelerationFree:
            delta_cons[0] = dt * mdot;
            delta_cons[1] = dt * mdot * prim[1] + dt * fx;
            delta_cons[2] = dt * mdot * prim[2] + dt * fy;
            delta_cons[3] = dt * (mdot * eps + 0.5 * mdot * (vx * vx + vy * vy)) + dt * (fx * vx + fy * vy);
            break;
        case TorqueFree: {
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
        case ForceFree:
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
    double soft_length)
{
    for (int p = 0; p < mass_list->count; ++p)
    {
        double delta_cons[NCONS];
        point_mass_source_term(&mass_list->masses[p], x1, y1, dt, prim, h, delta_cons, constant_softening, soft_length);

        for (int q = 0; q < NCONS; ++q)
        {
            cons[q] += delta_cons[q];
        }
    }
}

// ============================ EOS AND BUFFER ================================
// ============================================================================
PRIVATE double sound_speed_squared(
    struct EquationOfState *eos,
    double *prim)
{
    switch (eos->type)
    {
        case GammaLaw:
            return prim[3] / prim[0] * GAMMA_LAW_INDEX;
        default:
            return 1.0; // WARNING
    }
}

PRIVATE void buffer_source_term(
    struct BoundaryCondition *bc,
    double xc,
    double yc,
    double dt,
    double *cons)
{
    switch (bc->type)
    {
        case Default:
        case Inflow:
            break;

        case KeplerianBuffer:
        {
            double rc = sqrt(xc * xc + yc * yc);
            double surface_density = bc->keplerian_buffer.surface_density;
            double surface_pressure = bc->keplerian_buffer.surface_pressure;
            double central_mass = bc->keplerian_buffer.central_mass;
            double driving_rate = bc->keplerian_buffer.driving_rate;
            double outer_radius = bc->keplerian_buffer.outer_radius;
            double onset_width = bc->keplerian_buffer.onset_width;
            double onset_radius = outer_radius - onset_width;

            if (rc > onset_radius)
            {
                double pf = surface_density * sqrt(central_mass / rc);
                double px = pf * (-yc / rc);
                double py = pf * ( xc / rc);
                double kinetic_energy = 0.5 * (px * px + py * py) / surface_density;
                double energy = surface_pressure / (GAMMA_LAW_INDEX - 1.0) + kinetic_energy;
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
    double *cons)
{
    double gamma = GAMMA_LAW_INDEX;
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
    double pressure_floor)
{
    double gamma = GAMMA_LAW_INDEX;
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

PRIVATE void primitive_to_conserved(const double *prim, double *cons)
{
    double gamma = GAMMA_LAW_INDEX;
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

PRIVATE void riemann_hlle(const double *pl, const double *pr, double *flux, double cs2, int direction)
{
    double ul[NCONS];
    double ur[NCONS];
    double fl[NCONS];
    double fr[NCONS];
    double al[2];
    double ar[2];

    primitive_to_conserved(pl, ul);
    primitive_to_conserved(pr, ur);
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
        int na = (i + ng) * ti + (j + ng) * sj;

        double *pc = &primitive[np]
        double cs2 = sound_speed_squared(&eos, pc);
        double a = primitive_max_wavespeed(pc, cs2);
        wavespeed[na] = a;
    }
}
