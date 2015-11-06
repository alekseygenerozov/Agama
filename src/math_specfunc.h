/** \file   math_specfunc.h
    \brief  various special functions
    \date   2015
    \author Eugene Vasiliev
*/
#pragma once
#include "math_base.h"

namespace math {

/** Associated Legendre polynomial (or, rather, function) of the first kind:
    \f$  P_l^m(x)  \f$.
    These functions are used in spherical-harmonic expansion as follows: 
    \f$  Y_l^m = \sqrt{\frac{ (2l+1) (l-m)! }{ 4\pi (l+m)! }}
         P_l^m(\cos(\theta)) * \{\sin,\cos\}(m\phi)  \f$
    (this routine returns un-normalized P).
*/
double legendrePoly(const int l, const int m, const double x);

/** Array of un-normalized associate Legendre polynomials P_l^m(x) and their derivatives.
    The output arrays contain values of P and optionally dP/dx  for l=m,m+1,...,lmax;
*/
void legendrePolyArray(const int lmax, const int m, const double x,
    double* result_array, double* deriv_array=0);

/** Array of normalized associate Legendre polynomials W and their derivatives for l=m..lmax
    (theta-dependent factors in spherical-harmonic expansion):
    \f$  Y_l^m(\theta, \phi) = W_l^m(\theta) \{\sin,\cos\}(m\phi) ,
         W_l^m = \sqrt{\frac{ (2l+1) (l-m)! }{ 4\pi (l+m)! }} P_l^m(\cos(\theta))  \f$,
    where P are un-normalized associated Legendre functions.
    The output arrays contain values of W, dW/dtheta, d^2W/dtheta^2  for l=m,m+1,...,lmax;
    if either deriv_array or deriv2_array = NULL, the corresponding thing is not computed
    (note that if deriv2_array is not NULL, deriv_array must not be NULL too).
    This routine differs from `legendrePolyArray` in the following:
    (1) it takes theta rather than cos(theta) as argument;
    (2) returns normalized functions directly suitable for Y_l^m;
    (3) returns derivatives w.r.t. theta, not cos(theta), and may compute the 2nd derivative;
    (4) accurately handles values of theta close to 0 or pi.
*/
void sphHarmonicArray(const int lmax, const int m, const double theta,
    double* result_array, double* deriv_array=0, double* deriv2_array=0);

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
