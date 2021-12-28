
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

Identify troubled cells using Troubled Cell 
Indicator from G.Fu & C-W Shu, JCP 347, 305 (2017)
Then limit slopes of the troubled cells using TVB minmod
For limiting, use threshold of 0.1 appropriate for 1D k=2 (DG3) 
PRIVATE double limit_troubled_cells(double *ul, double *u, double *ur)
{
    // integrating polynomial extended from left/right zone into this zone

    double a = u[0] + 2.0 * sqrt(3.0) * ul[1] + 5.0 * sqrt(5.0) / 3.0 * ul[2];
    double b = u[0] - 2.0 * sqrt(3.0) * ur[1] + 5.0 * sqrt(5.0) / 3.0 * ur[2];
    double tci = (fabs(u[0] - a) + fabs(u[0] - b)) / maxabs(ul[0], u[0], ur[0]);

    if (tci > 0.1)
    {
        double w1t = minmod(u[1], ul[0], u[0], ur[0]);

        if (fabs(u[1] - w1t) > 1e-6)
        {             
            u[1] = w1t;
            u[2] = 0.0;
        }
    }
    return 0.0;
}
