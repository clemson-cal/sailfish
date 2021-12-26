PUBLIC cbdgam_2d_wavespeed(
    int ni,
    int nj,
    double *primitive, // :: $.shape == [ni + 4, nj + 4, 4]
    double *wavespeed, // :: $.shape == [ni + 4, nj + 4]
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

        real *pc = &primitive[np]
        real cs2 = sound_speed_squared(&eos, pc);
        real a = primitive_max_wavespeed(pc, cs2);
        wavespeed[na] = a;
    }
}
