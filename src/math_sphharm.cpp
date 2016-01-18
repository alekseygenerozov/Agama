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

void sphHarmArray(const unsigned int lmax, const unsigned int m, const double theta,
    double* resultArray, double* derivArray, double* deriv2Array)
{
    if(m>lmax || resultArray==NULL || (deriv2Array!=NULL && derivArray==NULL))
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
    double prefact; // will be initialized by legendrePmm
    legendrePmm(m, x, y, prefact, resultArray, derivArray, deriv2Array);
    if(lmax == m)
        return;

    const double EPS = 1e-8;  // threshold in y for applying asymptotic expressions for derivatives
    // values of two previous un-normalized polynomials needed in the recurrent relation
    double Plm1 = resultArray[0] / prefact, Plm = x * (2*m+1) * Plm1, Plm2 = 0;
    // values of 2nd derivatives of un-normalized polynomials needed for the special case
    // m==1 and y<<1, since we need another recurrent relation for computing 2nd derivative -
    // the usual formula suffers from cancellation
    double d2Plm1 = y, d2Plm2 = 0, d2Plm = 12 * x * y;

    for(int l=m+1; l<=(int)lmax; l++) {
        unsigned int ind = l-m;  // index in the output array
        if(l>(int)m+1)  // skip first iteration which was assigned above
            Plm = (x * (2*l-1) * Plm1 - (l+m-1) * Plm2) / (l-m);  // use recurrence for the rest
        prefact *= sqrt( (2*l+1.) / (2*l-1.) * (l-m) / (l+m) );
        resultArray[ind] = Plm * prefact;
        if(derivArray) {
            double dPlm = 0;
            if(y >= EPS || (m>2 && y>0))
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
            if(y >= EPS || (m>2 && y>0))
                deriv2Array[ind] = x * derivArray[ind] / (-y) - (l*(l+1)-pow_2(m/y)) * resultArray[ind];
            else if(m==0)
                deriv2Array[ind] = -l*(l+1)/2 * prefact * (x>0 || l%2==0 ? 1 : -1);
            else if(m==1) {
                if(l>(int)m+1) {
                    double twodPlm1 = -l*(l-1) * (x>0 || l%2==1 ? 1 : -1);
                    d2Plm = ( (2*l-1) * (x * (d2Plm1 - Plm1) - y * twodPlm1) - l * d2Plm2) / (l-1);
                }
                deriv2Array[ind] = prefact * d2Plm;
                d2Plm2 = d2Plm1;
                d2Plm1 = d2Plm;
            }
            else if(m==2)
                deriv2Array[ind] = l*(l+1)*(l+2)*(l-1)/4 * prefact * (x>0 || l%2==0 ? 1 : -1);
            else
                deriv2Array[ind] = 0;
        }
        Plm2 = Plm1;
        Plm1 = Plm;
    }
}

void trigMultiAngle(const double phi, const unsigned int m, const bool needSine, double* outputArray)
{
    if(m<1)
        return;
    // note that the recurrence relation below is not suitable for large m due to loss of accuracy,
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
    
// ------ indexing scheme for spherical harmonics, encoding its symmetry properties ------ //

SphHarmIndices::SphHarmIndices(int _lmax, int _mmax, coord::SymmetryType _sym) :
    lmax(_lmax), mmax(_mmax),
    step((_sym & coord::ST_ZREFLECTION) == coord::ST_ZREFLECTION ||
         (_sym & coord::ST_REFLECTION ) == coord::ST_REFLECTION ? 2 : 1),
    sym(_sym)
{
    if(lmax<0 || mmax<0 || mmax>lmax)
        throw std::invalid_argument("SphHarmIndices: incorrect indexing scheme requested");
    // consistency check: if three plane symmetries are present, mirror symmetry is implied
    if((sym & (coord::ST_XREFLECTION | coord::ST_YREFLECTION | coord::ST_ZREFLECTION)) ==
       (coord::ST_XREFLECTION | coord::ST_YREFLECTION | coord::ST_ZREFLECTION) &&
       (sym & coord::ST_REFLECTION) != coord::ST_REFLECTION)
        throw std::invalid_argument("SphHarmIndices: invalid symmetry requested");
    if(mmax==0)
        sym = static_cast<coord::SymmetryType>
            (sym | coord::ST_ZROTATION | coord::ST_XREFLECTION | coord::ST_YREFLECTION);
    if(lmax==0)
        sym = static_cast<coord::SymmetryType>
            (sym | coord::ST_ROTATION | coord::ST_ZREFLECTION | coord::ST_REFLECTION);
    // fill the lmin array
    lmin_arr.resize(2*mmax+1);
    for(int m=-mmax; m<=mmax; m++) {
        int lminm = abs(m);   // by default start from the very first coefficient
        if((sym & coord::ST_REFLECTION) == coord::ST_REFLECTION && m%2!=0)
            lminm = abs(m)+1; // in this case start from the next even l, because step in l is 2
        if( ((sym & coord::ST_YREFLECTION)  == coord::ST_YREFLECTION  &&  m<0) ||
            ((sym & coord::ST_XREFLECTION)  == coord::ST_XREFLECTION  && (m<0 ^ m%2!=0) ) || 
            ((sym & coord::ST_XYREFLECTION) == coord::ST_XYREFLECTION &&  m%2!=0) )
            lminm = lmax+1;  // don't consider this m at all
        lmin_arr[m+mmax] = lminm;
    }
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

SphHarmIndices getIndicesFromCoefs(const std::vector<double> &C)
{
    int lmax = sqrt(C.size())-1;
    if(lmax<0 || (int)C.size() != pow_2(lmax+1))
        throw std::invalid_argument("getIndicesFromCoefs: invalid size of coefs array");
    int sym  = coord::ST_SPHERICAL;
    int mmax = 0;
    for(unsigned int c=0; c<C.size(); c++) {
        if(!isFinite(C[c]))
            throw std::domain_error("getIndicesFromCoefs: coefficient not finite");
        if(C[c]!=0) {  // nonzero coefficient may break some of the symmetries, depending on l,m
            int l = SphHarmIndices::index_l(c);
            int m = SphHarmIndices::index_m(c);
            if(l%2 == 1)
                sym &= ~coord::ST_REFLECTION;
            if(m<0)
                sym &= ~coord::ST_YREFLECTION;
            if((l+m)%2 == 1)
                sym &= ~coord::ST_ZREFLECTION;
            if((m<0) ^ (m%2 != 0))
                sym &= ~coord::ST_XREFLECTION;
            if(m!=0) {
                sym &= ~coord::ST_ZROTATION;
                mmax = std::max<int>(abs(m), mmax);
            }
            if(l>0)
                sym &= ~coord::ST_ROTATION;
        }
    }
    return math::SphHarmIndices(lmax, mmax, static_cast<coord::SymmetryType>(sym));
}

std::vector<int> getIndicesAzimuthal(int mmax, coord::SymmetryType sym)
{
    std::vector<int> result(1, 0);  // m=0 is always present
    if((sym & coord::ST_ZROTATION) == coord::ST_ZROTATION)
        return result;  // in this case all m!=0 indices are zero
    for(int m=1; m<=mmax; m++) {
        // odd-m indices are excluded under the combination of z-reflection and mirror symmetry 
        if( (sym & coord::ST_XYREFLECTION) == coord::ST_XYREFLECTION && m%2 != 0)
            continue;
        bool addplusm = true, addminusm = true;
        // in case of y-reflection, only m>=0 indices are present
        if((sym & coord::ST_YREFLECTION) == coord::ST_YREFLECTION)
            addminusm = false;
        // in case of x-reflection, negative-even and positive-odd indices are zero
        if((sym & coord::ST_XREFLECTION) == coord::ST_XREFLECTION) {
            if(m%2==1)
                addplusm = false;
            else
                addminusm = false;
        }
        if(addminusm)
            result.push_back(-m);
        if(addplusm)
            result.push_back(m);        
    }
    return result;
}

void eliminateNearZeros(std::vector<double>& vec, double threshold)
{
    double mag=0;
    for(unsigned int t=0; t<vec.size(); t++)
        mag+=fabs(vec[t]);
    mag *= threshold;
    for(unsigned int t=0; t<vec.size(); t++)
        if(fabs(vec[t]) <= mag)
            vec[t]=0;
}

bool allZeros(const std::vector<double>& vec)
{
    for(unsigned int i=0; i<vec.size(); i++)
        if(vec[i]!=0)
            return false;
    return true;
}

// ------ classes for performing many transformations with identical setup ------ //

FourierTransformForward::FourierTransformForward(int _mmax, bool _useSine) :
    mmax(_mmax), useSine(_useSine)
{
    if(mmax<0)
        throw std::invalid_argument("FourierTransformForward: mmax must be non-negative");
    const int nphi = mmax+1;  // number of nodes in uniform grid in phi
    const int nfnc = useSine ? mmax*2+1 : mmax+1;  // number of trig functions for each phi-node
    trigFnc.resize(nphi * nfnc);
    // weight of a single value in uniform integration over phi
    double weight = M_PI / (mmax+0.5);
    // compute the values of trigonometric functions at nodes of phi-grid for all 0<=m<=mmax:
    // cos(m phi_k), and optionally sin(m phi_k) if terms with m<0 are non-trivial
    for(int k=0; k<nphi; k++) {
        trigFnc[k*nfnc] = 1.;  // cos(0*phi[k])
        if(mmax>0)
            trigMultiAngle(phi(k), mmax, useSine, &trigFnc[k*nfnc+1]);
        // if not using sines, then the grid in phi is 0 = phi_0 < ... < phi_{mmax} < pi,
        // so that all nodes except 0th should count twice.
        for(int m=0; m<nfnc; m++)
            trigFnc[k*nfnc+m] *= weight * (useSine || k==0 ? 1 : 2);
    }
}

void FourierTransformForward::transform(const double values[], double coefs[]) const
{
    const int nfnc = useSine ? mmax*2+1 : mmax+1;  // number of trig functions for each phi-node
    for(int mm=0; mm<nfnc; mm++) {  // index in the output array
        coefs[mm] = 0;
        int m = useSine ? mm-mmax : mm;  // if use sines, m runs from -mmax to mmax
        for(int k=0; k<nfnc; k++) {
            int indphi = k<=mmax ? k : 2*mmax+1-k;  // index of angle phi_k is between 0 and mmax
            int indfnc = m>=0 ? m : mmax-m;  // index of trig function is between 0 and mmax or 2mmax
            double fnc = trigFnc[indphi*nfnc + indfnc];
            if(m<0 && k>mmax)  // sin(2pi-phi) = -sin(phi)
                fnc*=-1;
            coefs[mm] += fnc * values[k];
        }
    }
}
    
// index of Legendre function P_{lm}(theta_j) in the `legFnc` array
static inline unsigned int indLeg(const SphHarmIndices& ind, int j, int l, int m)
{
    int ntheta = ind.lmax/2+1;
    int nlegfn = (ind.lmax+1) * (ind.mmax+1);
    int absm = abs(m);
    int absj = j<ntheta ? j : ind.lmax-j;
    return absj * nlegfn + absm * (ind.lmax+1) + l;
}

SphHarmTransformForward::SphHarmTransformForward(const SphHarmIndices& _ind):
    ind(_ind),
    fourier(ind.mmax, ind.mmin()<0)
{
    int ngrid  = ind.lmax+1;    // # of nodes of GL grid on [-1:1]
    int ntheta = ind.lmax/2+1;  // # of theta values to compute Plm
    int nlegfn = (ind.lmax+1) * (ind.mmax+1);  // # of Legendre functions for each theta-node
    
    legFnc.resize(ntheta * nlegfn);
    thnodes.resize(ngrid);
    // obtain nodes and weights of Gauss-Legendre quadrature of degree lmax+1 on [-1:1] for cos(theta)
    std::vector<double> nodes(ngrid), weights(ngrid);
    prepareIntegrationTableGL(-1, 1, ngrid, &nodes.front(), &weights.front());
    // compute the values of associated Legendre functions at nodes of theta-grid
    for(int j=0; j<ngrid; j++) {  // loop over nodes of theta-grid
        thnodes[j] = acos(nodes[ngrid-1-j]);
        if(j>=ntheta)  // don't consider nodes with theta>pi/2, 
            continue;  // as the values of Plm for them are known from symmetry properties
        // loop over m and compute all functions of order up to lmax for each m
        for(int m=0; m<=ind.mmax; m++) {
            sphHarmArray(ind.lmax, m, thnodes[j], &legFnc[indLeg(ind, j, m, m)]);
            // multiply the values of all Legendre functions at theta[i]
            // by the weight of this node in GL quadrature and by additional prefactor
            for(int l=m; l<=ind.lmax; l++)
                legFnc[indLeg(ind, j, l, m)] *= weights[j] * 0.5 / M_SQRTPI * (m>0 ? M_SQRT2 : 1);
        }
    }
}

void SphHarmTransformForward::transform(const double values[], double coefs[]) const
{
    for(unsigned int c=0; c<ind.size(); c++)
        coefs[c] = 0;
    int mmin  = ind.mmin();
    assert(mmin == 0 || mmin == -ind.mmax);
    int ngrid = ind.lmax+1;      // # of nodes of GL grid for integration in theta on (0:pi)
    int nfour = ind.mmax-mmin+1; // # of Fourier harmonic terms - either mmax+1 or 2*mmax+1
    int nsamp = thetasize();     // # of samples taken in theta (is less than ngrid in case of z-symmetry)
    // tmp storage: azimuthal Fourier coefficients F_jm for each value of theta_j and m.
    // indexing scheme: val_m[ j * nfour + m+ind.mmax ] = F_jm, 0<=j<nsamp, -mmax<=m<=mmax.
    std::vector<double> val_m( thetasize() * nfour );

    // first step: perform integration in phi for each value of theta, using Fourier transform
    for(unsigned int j=0; j<thetasize(); j++)
        fourier.transform( &values[j*fourier.size()], &val_m[j*nfour] );

    // second step: perform integration in theta for each m
    for(int m=ind.mmin(); m<=ind.mmax; m++) {
        for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m);
            for(int j=0; j<ngrid; j++) {
                // take the sample at |z| if z<0 and have z-reflection symmetry
                unsigned int jsamp = j<nsamp ? j : ngrid-1-j;
                double Plm = legFnc[indLeg(ind, j, l, m)];
                if( (l+m)%2 == 1 && j>ind.lmax/2 )
                    Plm *= -1;   // Plm(-x) = (-1)^{l+m) Plm(x), here x=cos(theta) and theta > pi/2
                coefs[c] += Plm * val_m[ jsamp * nfour + m-mmin ];
            }
        }
    }
}

double sphHarmTransformInverse(const SphHarmIndices& ind, const double coefs[],
    const double theta, const double phi, double* tmptrig)
{
    const bool useSine = ind.mmin()<0;
    if(ind.mmax>0)
        trigMultiAngle(phi, ind.mmax, useSine, tmptrig);
    double* tmpleg = &tmptrig[2*ind.mmax];
    double result = 0;
    for(int m=ind.mmin(); m<=ind.mmax; m++) {
        int lmin = ind.lmin(m);
        if(lmin>ind.lmax)
            continue;  // empty m-harmonic
        int absm = abs(m);
        double trig = m==0 ?       2*M_SQRTPI : // extra numerical factors from the definition of sph.harm.
            m>0 ? tmptrig[m-1]   * 2*M_SQRTPI * M_SQRT2 :
            tmptrig[ind.mmax-m-1]* 2*M_SQRTPI * M_SQRT2;
        sphHarmArray(ind.lmax, absm, theta, tmpleg);
        for(int l=lmin; l<=ind.lmax; l+=ind.step) {
            double leg = tmpleg[l-absm];
            result += coefs[ind.index(l, m)] * leg * trig;
        }
    }
    return result;
}

};  // namespace math