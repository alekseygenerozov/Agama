#include "math_sphharm.h"
#include "math_core.h"
#include "math_specfunc.h"
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <gsl/gsl_sf_legendre.h>

namespace math{

double legendrePoly(const int l, const int m, const double x) {
    return gsl_sf_legendre_Plm(l, m, x);
}

void legendrePolyArray(const int lmax, const int m, const double x,
    double* result_array, double* deriv_array)
{
    assert(result_array!=NULL);
#if GSL_MAJOR_VERSION < 2
    if(deriv_array)
        gsl_sf_legendre_Plm_deriv_array(lmax, m, x, result_array, deriv_array);
    else
        gsl_sf_legendre_Plm_array(lmax, m, x, result_array);
#else
    if(m!=0)
        throw std::runtime_error("m!=0 is not supported anymore in GSL Legendre API");
    if(deriv_array)
        gsl_sf_legendre_Pl_deriv_array(lmax, x, result_array, deriv_array);
    else
        gsl_sf_legendre_Pl_array(lmax, x, result_array);
#endif
}

void sphHarmonicArray(const int lmax, const int m, const double theta,
    double* result_array, double* deriv_array, double* deriv2_array)
{
    assert(result_array!=NULL);
    double costheta = cos(theta), sintheta=0;
    // compute unnormalized polynomials and then normalize manually, which is faster than computing normalized ones.
    // This is not suitable for large l,m (when overflow may occur), but in our application we aren't going to have such large values.
#if GSL_MAJOR_VERSION < 2
    if(deriv_array) {
        gsl_sf_legendre_Plm_deriv_array(lmax, m, costheta, result_array, deriv_array);
        sintheta = sin(theta);
    }
    else
        gsl_sf_legendre_Plm_array(lmax, m, costheta, result_array);
#else
    if(m!=0)
        throw std::runtime_error("m!=0 is not supported anymore in GSL Legendre API");
    if(deriv_array) {
        gsl_sf_legendre_Pl_deriv_array(lmax, costheta, result_array, deriv_array);
        sintheta = sin(theta);
    }
    else
        gsl_sf_legendre_Pl_array(lmax, costheta, result_array);
#endif    
    double prefact = 0.5/sqrt(M_PI*factorial(2*m));
    for(int l=m; l<=lmax; l++) {
        double prefactl=sqrt(2*l+1.)*prefact;
        result_array[l-m] *= prefactl;
        if(deriv_array)
            deriv_array[l-m] *= prefactl;
        prefact *= sqrt((l-m+1.)/(l+m+1.));
    }
    if(deriv2_array) {
        assert(deriv_array!=NULL);
        for(int l=m; l<=lmax; l++) {
            // accurate treatment of asymptotic values to avoid NaN
            if(m==0)
                deriv2_array[l-m] = costheta * deriv_array[l-m] - l*(l+1) * result_array[l-m];
            else if(costheta>=1-1e-6)
                deriv2_array[l-m] = deriv_array[l-m] * (costheta - 2*(l*(l+1)*(costheta-1)/m + m/(costheta+1)) );
            else if(costheta<=-1+1e-6)
                deriv2_array[l-m] = deriv_array[l-m] * (costheta - 2*(l*(l+1)*(costheta+1)/m + m/(costheta-1)) );
            else
                deriv2_array[l-m] = costheta * deriv_array[l-m] - (l*(l+1)-pow_2(m/sintheta)) * result_array[l-m];
        }
    }
    if(deriv_array) {
        for(int l=0; l<=lmax-m; l++)
            deriv_array[l] *= -sintheta;
    }
}

// ------ class for performing many transformations with identical setup ------ //

LegendreTransform::LegendreTransform(unsigned int _lmax):
    lmax(_lmax)
{
    nodes.resize(lmax+1);
    weights.resize(lmax+1);
    prepareIntegrationTableGL(-1, 1, lmax+1, &nodes.front(), &weights.front());
    legPoly.resize(pow_2(lmax+1));
    for(unsigned int i=0; i<=lmax; i++)
        legendrePolyArray(lmax, 0, nodes[i], &legPoly[ i * (lmax+1) ]);
}

void LegendreTransform::forward(const double values[], double coefs[]) const
{
    for(unsigned int l=0; l<=lmax; l++) {
        coefs[l]=0;
        for(unsigned int i=0; i<=lmax; i++)
            coefs[l] += values[i] * weights[i] * legPoly[ i * (lmax+1) + l ];
    }
}

void LegendreTransform::inverse(const double coefs[], double values[]) const
{
    for(unsigned int i=0; i<=lmax; i++) {
        values[i]=0;
        for(unsigned int l=0; l<=lmax; l++)
            values[i] += coefs[l] * legPoly[ i * (lmax+1) + l ] * (1+2*l)/2;
    }
}


SphHarmIndices::SphHarmIndices(int _lmax, int _lstep, int _mmin, int _mmax, int _mstep) :
    lmax(_lmax), lstep(_lstep), mmin(_mmin), mmax(_mmax), mstep(mmax>0 ? _mstep : 1)
{
    if(lmax<0 || mmax<0 || mmax>lmax || (mmin!=0 && mmin!=-mmax) ||
        (lstep!=1 && lstep!=2) || (mstep!=1 && mstep!=2))
        throw std::invalid_argument("SphHarmIndices: incorrect indexing scheme requested");
}

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
        // reorder nodes of theta grid so that even-indexed nodes correspond to costheta>=0
        // and odd-indexed - to costheta<0, the latter will not be used later if ind.lstep==2
        int tmpi = nnodth/2 + (j%2 ? -(j+1)/2 : j/2);
        thnodes[j] = tmpnodes[tmpi];
        // loop over m and compute all functions of order up to lmax for each m
        for(int m=0; m<=ind.mmax; m++)
            sphHarmonicArray(ind.lmax, m, acos(thnodes[j]), &legFnc[ j * nlegfn + m * (ind.lmax+1) ]);
        // multiply the values of all Legendre functions at theta[i]
        // by the weight of this node in GL quadrature and by additional prefactor
        for(int k=0; k<nlegfn; k++)
            legFnc[ j * nlegfn + k ] *= weights[tmpi] * 0.5 / M_SQRTPI;
    }
    //!!! optimize space usage for legFnc, eliminate redundant ones, switch to legendrePolyArray

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

/// return the coordinate of j-th node for cos(theta) on (-1:1), 0 <= j <= ind.lmax
double SphHarmTransformForward::costheta(int j) const
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

    // polishing: zero out coefs with extremely small magnitude
    double mag=0;
    for(unsigned int t=0; t<ind.size(); t++)
        mag+=fabs(coefs[t]);
    mag*=1e-14;
    for(unsigned int t=0; t<ind.size(); t++)
        if(fabs(coefs[t])<mag)
            coefs[t]=0;
}

double sphHarmTransformInverse(const SphHarmIndices& ind, const double coefs[],
    const double theta, const double phi)
{
    //!!! get rid of dynamic memory allocation
    const bool useSine = ind.mmin!=0;
    std::vector<double> tmptrig(2*ind.mmax+1);
    if(ind.mmax>0)
        trigMultiAngle(phi, ind.mmax, useSine, &tmptrig[1]);
    std::vector<double> tmpleg(ind.lmax+1);
    double result = 0;
    for(int m=ind.mmin; m<=ind.mmax; m+=ind.mstep) {
        int absm = abs(m);
        double trig = m==0 ? 1. : m>0 ? tmptrig[m]*M_SQRT2 : tmptrig[ind.mmax-m]*M_SQRT2;
        sphHarmonicArray(ind.lmax, absm, theta, &tmpleg.front());
        // if lstep is even and m is odd, start from next even number greater than m
        int lmin = ind.lstep==2 ? (absm+1)/2*2 : absm;
        for(int l=lmin; l<=ind.lmax; l+=ind.lstep) {
            double leg = tmpleg[l-absm] * 2*M_SQRTPI;
            result += coefs[ind.index(l, m)] * leg * trig;
        }
    }
    return result;
}

};  // namespace math