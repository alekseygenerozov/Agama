#include "math_sphharm.h"
#include "math_core.h"
#include "math_specfunc.h"
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <gsl/gsl_sf_legendre.h>

namespace math{

//** WILL BE REMOVED **//
double legendrePoly(const int l, const int m, const double x) {
    return gsl_sf_legendre_Plm(l, m, x);
}


/** Calculate P_m^m(theta) from the analytic result:
    P_m^m(theta) = (-1)^m (2m-1)!! (sin(theta))^m , m > 0 ;
                 = 1 , m = 0 .
    store the pre-factor sqrt[ (2*m+1) / (4 pi (2m)!) ] in prefact,
    the value of Pmm in val, and optionally its first/second derivative w.r.t theta
    in der/der2 if they are not NULL.
*/
static inline void legendrePmm(int m, double costheta, double sintheta, 
    double& prefact, double* value, double* der, double* der2)
{
    prefact = 0.5/M_SQRTPI;
    if(m == 0) {
        if(der)
            *der = 0;
        if(der2)
            *der2= 0;
        *value   = prefact;
        return;
    }
    prefact *= sqrt( (2*m+1) / factorial(2*m) );
    if(m == 1) {
        if(der)
            *der = -costheta * prefact;
        if(der2)
            *der2=  sintheta * prefact;
        *value   = -sintheta * prefact;
        return;
    }
    double coef  = prefact * dfactorial(2*m-1) * (m%2 == 1 ? -1 : 1);
    double sinm2 = powInt(sintheta, m-2);
    if(der)
        *der = m * coef * sinm2 * sintheta * costheta;
    if(der2)
        *der2= m * coef * sinm2 * (m * pow_2(costheta) - 1);
    *value   =     coef * sinm2 * pow_2(sintheta);
}

void sphHarmonicArray(const int lmax, const int m, const double theta,
    double* resultArray, double* derivArray, double* deriv2Array)
{
    if(m<0 || m>lmax || /* x<-1 || x>1 */ resultArray == NULL)
        throw std::domain_error("Invalid parameters in sphHarmArray");
    if(lmax==0) {
        resultArray[0] = 0.5/M_SQRTPI;
        if(derivArray)
            derivArray[0] = 0;
        if(deriv2Array)
            deriv2Array[0] = 0;
        return;
    }
    double x = cos(theta), y = abs(x)<1-1e-6 ? sqrt(1-x*x) : sin(theta);
    double prefact;
    legendrePmm(m, x, y, prefact, resultArray, derivArray, deriv2Array);
    if(lmax == m)
        return;

    const double EPS = 1e-8;  // threshold in y for applying asymptotic expressions for derivatives
    // values of two previous un-normalized polynomials needed in the recurrent relation
    double Plm1 = resultArray[0] / prefact, Plm = x * (2*m+1) * Plm1, Plm2 = 0;
    // values of 2nd derivatives of un-normalized polynomials needed for the special case
    // m==1 and y<<1, since we need another recurrent relation for computing 2nd derivative
    double d2Plm1 = y, d2Plm2 = 0, d2Plm = 12 * x * y;

    for(int l=m+1; l<=lmax; l++) {
        int ind = l-m;  // index in the output array
        if(l>m+1)  // skip first iteration which was assigned above
            Plm = (x * (2*l-1) * Plm1 - (l+m-1) * Plm2) / (l-m);  // use recurrence for the rest
        prefact *= sqrt( (2*l+1.) / (2*l-1.) * (l-m) / (l+m) );
        resultArray[ind] = Plm * prefact;
        if(derivArray) {
            double dPlm = 0;
            if(y >= EPS)
                dPlm = (l * x * Plm - (l+m) * Plm1) / y;
            else if(m==0)
                dPlm = -l*(l+1)/2 * y * (x>0 || l%2==1 ? 1 : -1);
            else if(m==1)
                dPlm = -l*(l+1)/2 * (x>0 || l%2==0 ? 1 : -1);
            else if(m==2)
                dPlm = l*(l+1)*(l+2)*(l-1)/4 * y * (x>0 || l%2==1 ? 1 : -1);
            derivArray[ind] = prefact * dPlm;
        }
        if(deriv2Array!=NULL) {
            if(y >= EPS)
                deriv2Array[ind] = x * derivArray[ind] / (-y) - (l*(l+1)-pow_2(m/y)) * resultArray[ind];
            else if(m==0)
                deriv2Array[ind] = -l*(l+1)/2 * prefact * (x>0 || l%2==0 ? 1 : -1);
            else if(m==1) {
                if(l>m+1) {
                    double twodPlm1 = -l*(l-1) * (x>0 || l%2==1 ? 1 : -1);
                    d2Plm = ( (2*l-1) * (x * (d2Plm1 - Plm1) - y * twodPlm1) - l * d2Plm2) / (l-1);
                }
                deriv2Array[ind] = prefact * d2Plm;
                d2Plm2 = d2Plm1;
                d2Plm1 = d2Plm;
            }
            else if(m==2)
                deriv2Array[ind] = l*(l+1)*(l+2)*(l-1)/4 * prefact * (x>0 || l%2==0 ? 1 : -1);
        }
        Plm2 = Plm1;
        Plm1 = Plm;
    }
}

SphHarmIndices::SphHarmIndices(int _lmax, int _lstep, int _mmin, int _mmax, int _mstep) :
    lmax(_lmax), lstep(lmax>0 ? _lstep : 2), mmin(_mmin), mmax(_mmax), mstep(mmax>0||_mstep!=0 ? _mstep : 2)
{
    if(lmax<0 || mmax<0 || mmax>lmax || (mmin!=0 && mmin!=-mmax) ||
        (lstep!=1 && lstep!=2) || (mstep!=1 && mstep!=2))
        throw std::invalid_argument("SphHarmIndices: incorrect indexing scheme requested");
}

int SphHarmIndices::index_l(unsigned int c) 
{
    return sqrt(c);
}

int SphHarmIndices::index_m(unsigned int c)
{
    int l=index_l(c);
    return c-l*(l+1);
}

// ------ class for performing many transformations with identical setup ------ //

SphHarmTransformForward::SphHarmTransformForward(const SphHarmIndices& _ind):
    ind(_ind),
    nnodth (ind.lmax+1),  // number of nodes in Gauss-Legendre grid in cos(theta)
    nlegfn (nnodth * (ind.mmax+1)),  // number of Legendre functions for each theta-node
    nnodphi(ind.mmax+1),  // number of nodes in uniform grid in phi
    ntrigfn(ind.mmax*2+1) // number of trig functions for each phi-node
{
    // nodes and weights of Gauss-Legendre quadrature of degree lmax+1 on [-1:1]
    std::vector<double> tmpnodes(nnodth), weights(nnodth);
    prepareIntegrationTableGL(-1, 1, nnodth, &tmpnodes.front(), &weights.front());

    // compute the values of associated Legendre functions at nodes of theta-grid
    // (there is a factor 2 redundancy here, as we don't take into account
    // their symmetry properties w.r.t. change of sign of cos(theta), but it's not a big deal).
    legFnc.resize(nnodth * nlegfn);
    thnodes.resize(nnodth);
    for(int j=0; j<nnodth; j++) {  // loop over nodes of theta-grid
        // reorder nodes of theta grid so that even-indexed nodes correspond to cos(theta)>=0
        // and odd-indexed - to cos(theta)<0, the latter will not be used later if ind.lstep==2
        int tmpi = nnodth/2 + (j%2 ? -(j+1)/2 : j/2);
        thnodes[j] = acos(tmpnodes[tmpi]);
        // loop over m and compute all functions of order up to lmax for each m
        for(int m=0; m<=ind.mmax; m++)
            sphHarmonicArray(ind.lmax, m, thnodes[j], &legFnc[ j * nlegfn + m * (ind.lmax+1) ]);
        // multiply the values of all Legendre functions at theta[i]
        // by the weight of this node in GL quadrature and by additional prefactor
        for(int k=0; k<nlegfn; k++)
            legFnc[ j * nlegfn + k ] *= weights[tmpi] * 0.5 / M_SQRTPI;
    }

    // compute the values of trigonometric functions at nodes of phi-grid for all mmin<=m<=mmax:
    // cos(m phi_k), and optionally sin(m phi_k) if terms with m<0 are non-trivial (i.e. mmin!=0)
    const bool useSine = ind.mmin!=0;
    // weight of a single value in uniform integration over phi, times additional factor 1/sqrt{2}
    double weight = M_PI / (ind.mmax+0.5) * M_SQRT2;
    trigFnc.resize(nnodphi * ntrigfn);
    std::vector<double> tmptrig(2*ind.mmax+1);
    tmptrig[0] = 1./M_SQRT2; // * cos(0*phi[k])
    for(int k=0; k<nnodphi; k++) {
        if(ind.mmax>0)
            trigMultiAngle(phi(k), ind.mmax, useSine, &tmptrig[1]);
        // rearrange trig fncs so that sine terms come before cosines, 
        // multiply them by uniform weights for integration over phi
        for(int m=ind.mmin; m<=ind.mmax; m++)
            trigFnc[indTrig(k, m)] = tmptrig[m>=0 ? m : ind.mmax-m] * weight;
    }
}

unsigned int SphHarmTransformForward::index(int j, int k) const
{
    if(j<0 || j>ind.lmax || k<-ind.mmax || k>ind.mmax)
        throw std::range_error("SphHarmTransformForward: index out of range");
    return j * (2*ind.mmax+1) + k+ind.mmax;
}

/// return the coordinate of j-th node for theta on (0:pi), 0 <= j <= ind.lmax
double SphHarmTransformForward::theta(int j) const
{
    if(j<0 || j>ind.lmax)
        throw std::range_error("SphHarmTransformForward: index out of range");
    return thnodes.at(j);
}

/// return the coordinate of k-th node for phi on (-pi:pi), ind.mmin <= k <= ind.mmax
double SphHarmTransformForward::phi(int k) const
{
    if(k<-ind.mmax || k>ind.mmax)
        throw std::range_error("SphHarmTransformForward: index out of range");
    return k * M_PI / (ind.mmax+0.5);
}

unsigned int SphHarmTransformForward::indLeg(int j, int l, int m) const
{
    int absm = abs(m);
    return j * nlegfn + absm * (ind.lmax+1) + l-absm;
}

unsigned int SphHarmTransformForward::indTrig(int k, int m) const
{
    return abs(k) * ntrigfn + m+ind.mmax;
}

unsigned int SphHarmTransformForward::indFour(int j, int m) const
{
    return j * ntrigfn + m+ind.mmax;
}

void SphHarmTransformForward::transform(const double values[], double coefs[]) const
{
    for(unsigned int t=0; t<ind.size(); t++)
        coefs[t] = 0;
    // tmp storage: azimuthal Fourier coefficients F_jm for each value of theta_j and m.
    // indexing scheme: val_m[ indFour(j,m) ] = F_jm, 0<=j<=lmax, -mmax<=m<=mmax.
    std::vector<double> val_m( nnodth * ntrigfn );

    // first step: perform integration in phi for each value of theta, using DFT
    for(int j=0; j<nnodth; j++) {
        // if have mirror symmetry w.r.t. change of sign in z (indicated by lstep==2),
        // then use only input values for 0<=cos(theta)<1,
        // i.e. take the value at pi-theta_j if theta_j>pi/2
        int jv = ind.lstep==2 && j%2==1 ? j+1-2*(ind.lmax%2) : j;  // index of theta_j in input array
        for(int k=-ind.mmax; k<=ind.mmax; k++) {
            // if have mirror symmetry w.r.t. change of sign in y (indicated by mmin==0),
            // then use only values for 0<=phi<pi, i.e. take the input value at -phi_k if phi_k<0
            int kv = ind.mmin==0 && k<0 ? -k : k;  // index of phi_k in input array
            double val = values[index(jv, kv)];    // value of input fnc at theta_j, phi_k
            for(int m=-ind.mmax; m<=ind.mmax; m++) {
                // if k<0 (phi_k<0), use the values of trig fnc for -phi_k and change sign of sine terms
                double trig = trigFnc[indTrig(k, m)];
                if(k<0 && m<0)  // m<0 corresponds to sine terms, sin(-phi_k) = -sin(phi_k)
                    trig *= -1;
                val_m[indFour(j, m)] += val * trig;
            }
        }
    }

    // second step: perform integration in theta for each m
    for(int m=-ind.mmax; m<=ind.mmax; m++) {
        for(int l=abs(m); l<=ind.lmax; l++) {
            for(int j=0; j<nnodth; j++)
                coefs[ind.index(l, m)] += legFnc[indLeg(j, l, m)] * val_m[indFour(j, m)];
        }
    }
}

void eliminateNearZeros(std::vector<double>& vec, double threshold)
{
    double mag=0;
    for(unsigned int t=0; t<vec.size(); t++)
        mag+=fabs(vec[t]);
    mag *= threshold;
    for(unsigned int t=0; t<vec.size(); t++)
        if(fabs(vec[t]) < mag)
            vec[t]=0;
}

double sphHarmTransformInverse(const SphHarmIndices& ind, const double coefs[],
    const double theta, const double phi, double* tmptrig)
{
    const bool useSine = ind.mmin!=0;
    if(ind.mmax>0)
        trigMultiAngle(phi, ind.mmax, useSine, tmptrig);
    double* tmpleg = &tmptrig[2*ind.mmax];
    double result = 0;
    for(int m=ind.mmin; m<=ind.mmax; m+=ind.mstep) {
        int absm = abs(m);
        double trig = m==0 ? 1. : m>0 ? tmptrig[m]*M_SQRT2 : tmptrig[ind.mmax-m]*M_SQRT2;
        sphHarmonicArray(ind.lmax, absm, theta, tmpleg);
        // if lstep is even and m is odd, start from next even number greater than m (???)
        int lmin = ind.lstep==2 ? (absm+1)/2*2 : absm;
        for(int l=lmin; l<=ind.lmax; l+=ind.lstep) {
            double leg = tmpleg[l-absm] * 2*M_SQRTPI;
            result += coefs[ind.index(l, m)] * leg * trig;
        }
    }
    return result;
}

};  // namespace math