/** \file   math_specfunc.h
    \brief  various special functions
    \date   2015
    \author Eugene Vasiliev
*/
#pragma once

namespace math {

/*  Compute the values of cosines and optionally sines of an arithmetic progression of angles:
    cos(phi), cos(2 phi), ..., cos(m phi), [ sin(phi), sin(2 phi), ..., sin(m phi) ].
    \param[in]  phi - the angle;
    \param[in[  m   - the number of multiples of this angle to process, must be >=1 (not checked);
    \param[in]  needSine - whether to compute sines as well (if false then only cosines are computed);
    \param[out] outputArray - pointer to an existing array of length m (if needSine==false)
                or 2m (if needSine==true) that will store the output values.
*/
void trigMultiAngle(const double phi, const unsigned int m, const bool needSine, double* outputArray);

/** Gegenbauer (ultraspherical) polynomial:  \f$ C_n^{(\lambda)}(x) \f$ */
double gegenbauer(const int n, double lambda, double x);

/** Array of Gegenbauer (ultraspherical) polynomials for n=0,1,...,nmax */
void gegenbauerArray(const int nmax, double lambda, double x, double* result_array);

/** Inverse error function (defined for -1<x<1) */
double erfinv(const double x);

/** Gauss's hypergeometric function 2F1(a, b; c; x) */
double hypergeom2F1(const double a, const double b, const double c, const double x);

/** Factorial of an integer number */
double factorial(const unsigned int n);

/** Logarithm of factorial of an integer number (doesn't overflow quite that easy) */
double lnfactorial(const unsigned int n);

/** Gamma function */
double gamma(const double x);

/** Logarithm of gamma function (doesn't overflow quite that easy) */
double lngamma(const double n);

/** Psi (digamma) function */
double digamma(const double x);

/** Psi (digamma) function for integer argument */
double digamma(const int x);

/** Complete elliptic integrals of the first kind K(k) = F(pi/2, k) */
double ellintK(const double k);

/** Complete elliptic integrals of the second kind K(k) = E(pi/2, k) */
double ellintE(const double k);

/** Incomplete elliptic integrals of the first kind:
    \f$  F(\phi,k) = \int_0^\phi d t \, \frac{1}{\sqrt{1-k^2\sin^2 t}}  \f$  */
double ellintF(const double phi, const double k);

/** Incomplete elliptic integrals of the second kind:
    \f$  E(\phi,k) = \int_0^\phi d t \, \sqrt{1-k^2\sin^2 t}  \f$  */
double ellintE(const double phi, const double k);

/** Incomplete elliptic integrals of the third kind:
    \f$  \Pi(\phi,k,n) = \int_0^\phi d t \, \frac{1}{(1+n\sin^2 t)\sqrt{1-k^2\sin^2 t}}  \f$  */
double ellintP(const double phi, const double k, const double n);

/** Bessel J_n(x) function (regular at x=0) */
double besselJ(const int n, const double x);

/** Bessel Y_n(x) function (singular at x=0) */
double besselY(const int n, const double x);

/** Modified Bessel function I_n(x) (regular) */
double besselI(const int n, const double x);

/** Modified Bessel function K_n(x) (singular) */
double besselK(const int n, const double x);

}  // namespace
