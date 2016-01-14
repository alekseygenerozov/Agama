#include "math_base.h"
#include "math_specfunc.h"
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <gsl/gsl_sf_gegenbauer.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_ellint.h>
#include <gsl/gsl_sf_bessel.h>

/* Most of the functions here are implemented by calling corresponding routines from GSL,
   but having library-independent wrappers makes it possible to switch the back-end if necessary */

namespace math {

void trigMultiAngle(const double phi, const unsigned int m, const bool needSine, double* outputArray)
{
    // we assume that m>=1 but don't check it;
    // also note that the recurrence relation below is not suitable for large m due to loss of accuracy,
    // but for our purposes this should suffice;
    // a more accurate expression is given in section 5.4 of Num.Rec.3rd ed.
    double cosphi = cos(phi);
    outputArray[0] = cosphi;
    for(unsigned int k=1; k<m; k++)
        outputArray[k] = 2 * cosphi * outputArray[k-1] - (k>1 ? outputArray[k-2] : 1);
    if(!needSine)
        return;
    outputArray[m] = sin(phi);
    for(unsigned int k=m+1; k<m*2; k++)
        outputArray[k] = 2 * cosphi * outputArray[k-1] - (k>m+1 ? outputArray[k-2] : 0);
}

double gegenbauer(const int n, double lambda, double x) {
    return gsl_sf_gegenpoly_n(n, lambda, x);
}

void gegenbauerArray(const int nmax, double lambda, double x, double* result_array) {
    gsl_sf_gegenpoly_array(nmax, lambda, x, result_array);
}

double erfinv(const double x)
{
    if(x < -1 || x > 1)
        return NAN;
    if(x == 0)
        return 0;
    double z;   // first approximation
    if(fabs(x)<=0.7) {
        double x2 = x*x;
        z  = x * (((-0.140543331 * x2 + 0.914624893) * x2 - 1.645349621) * x2 + 0.886226899) /
            (1 + (((0.012229801 * x2 - 0.329097515) * x2 + 1.442710462) * x2 - 2.118377725) * x2);
    }
    else {
        double y = sqrt(-log((1-fabs(x))/2));
        z = (((1.641345311 * y + 3.429567803) * y - 1.62490649) * y - 1.970840454) / 
            (1 + (1.6370678 * y + 3.5438892) * y);
        if(x<0) z = -z;
    }
    // improve by Halley iteration
    double f = erf(z) - x, fp = 2/M_SQRTPI * exp(-z*z), fpp = -2*z*fp;
    z -= f / (fp - f*fpp/2/fp);
    return z;
}

double hypergeom2F1(const double a, const double b, const double c, const double x)
{
    if (-1.<=x and x<1.)
        return gsl_sf_hyperg_2F1(a, b, c, x);
    // extension for 2F1 into the range x<-1 which is not provided by GSL; code from Heiko Bauke
    if (x<-1.) {
        if (c-a<0)
            return pow(1.-x, -a) * gsl_sf_hyperg_2F1(a, c-b, c, x/(x-1.));
        if (c-b<0)
            return pow(1.-x, -b) * gsl_sf_hyperg_2F1(c-a, c, c, x/(x-1.));
        // choose one of two equivalent formulas which is expected to be more accurate
        if (a*(c-b)<(c-a)*b)
            return pow(1.-x, -a) * gsl_sf_hyperg_2F1(a, c-b, c, x/(x-1.));
        else
            return pow(1.-x, -b) * gsl_sf_hyperg_2F1(c-a, b, c, x/(x-1.));
    }
    return NAN;  // not defined for x>=1
}

double factorial(const unsigned int n) {
    return gsl_sf_fact(n);
}

double lnfactorial(const unsigned int n) {
    return gsl_sf_lnfact(n);
}

double dfactorial(const unsigned int n) {
    return gsl_sf_doublefact(n);
}

double gamma(const double x) {
    return gsl_sf_gamma(x);
}

double lngamma(const double x) {
    return gsl_sf_lngamma(x);
}

double digamma(const double x) {
    return gsl_sf_psi(x);
}

double digamma(const int x) {
    return gsl_sf_psi_int(x);
}

double ellintK(const double k) {
    return gsl_sf_ellint_Kcomp(k, GSL_PREC_SINGLE);
}

double ellintE(const double k) {
    return gsl_sf_ellint_Ecomp(k, GSL_PREC_SINGLE);
}

double ellintF(const double phi, const double k) {
    return gsl_sf_ellint_F(phi, k, GSL_PREC_SINGLE);
}

double ellintE(const double phi, const double k) {
    return gsl_sf_ellint_E(phi, k, GSL_PREC_SINGLE);
}

double ellintP(const double phi, const double k, const double n) {
    return gsl_sf_ellint_P(phi, k, n, GSL_PREC_SINGLE);
}

double besselJ(const int n, const double x) {
    return gsl_sf_bessel_Jn(n, x);
}

double besselY(const int n, const double x) {
    return gsl_sf_bessel_Yn(n, x);
}

double besselI(const int n, const double x) {
    return gsl_sf_bessel_In(n, x);
}

double besselK(const int n, const double x) {
    return gsl_sf_bessel_Kn(n, x);
}

}  // namespace