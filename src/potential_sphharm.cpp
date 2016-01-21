#include "potential_sphharm.h"
#include "math_core.h"
#include "math_specfunc.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <stdexcept>

#ifdef VERBOSE_REPORT
#include <iostream>
#endif

namespace potential {

// internal definitions
namespace{

/// max number of basis-function expansion members (radial and angular).
const unsigned int MAX_NCOEFS_ANGULAR = 101;
const unsigned int MAX_NCOEFS_RADIAL  = 100;

/// minimum number of terms in sph.-harm. expansion used to compute coefficients
/// of a non-spherical density or potential model (it may be larger than
/// the requested number of output terms, to improve the accuracy of integration)    
const unsigned int LMIN_SPHHARM = 16;

/// max number of function evaluations in multidimensional integration
const unsigned int MAX_NUM_EVAL = 4096;

/// relative accuracy of potential computation (integration tolerance parameter)
const double EPSREL_POTENTIAL_INT = 1e-6;

/// relative accuracy in auxiliary root-finding routines
const double ACCURACY_ROOT = 1e-6;

/** helper class for integrating the density weighted with spherical harmonics over 3d volume;
    angular part is shared between BasisSetExp and SplineExp, 
    which further define additional functions for radial multiplication factor. 
    The integration is carried over  scaled r  and  cos(theta). */
class DensitySphHarmIntegrand: public math::IFunctionNdim {
public:
    DensitySphHarmIntegrand(const BaseDensity& _dens, int _l, int _m, 
        const math::IFunction& _radialMultFactor, double _rscale) :
        dens(_dens), l(_l), m(_m), radialMultFactor(_radialMultFactor), rscale(_rscale),
        mult( 0.5*sqrt((2*l+1.) * math::factorial(l-math::abs(m)) / math::factorial(l+math::abs(m))) )
    {};

    /// evaluate the m-th azimuthal harmonic of density at a point in (scaled r, cos(theta)) plane
    virtual void eval(const double vars[], double values[]) const 
    {   // input array is [scaled coordinate r, cos(theta)]
        const double scaled_r = vars[0], costheta = vars[1];
        if(scaled_r == 1) {
            values[0] = 0;  // we're at infinity
            return;
        }
        const double r = rscale * scaled_r / (1-scaled_r);
        const double R = r * sqrt(1-pow_2(costheta));
        const double z = r * costheta;
        const double Plm = math::legendrePoly(l, math::abs(m), costheta);
        double val = computeRho_m(dens, R, z, m) * Plm;
        if((dens.symmetry() & coord::ST_TRIAXIAL) == coord::ST_TRIAXIAL)   // symmetric w.r.t. change of sign in z
            val *= (l%2==0 ? 2 : 0);  // only even-l terms survive
        else
            val += computeRho_m(dens, R, -z, m) * Plm * (l%2==0 ? 1 : -1);
        values[0] = val * mult *
            r*r *                      // jacobian of transformation to spherical coordinates
            rscale/pow_2(1-scaled_r) * // un-scaling the radial coordinate
            radialMultFactor(r);       // additional radius-dependent factor
    }
    /// return the scaled radial variable (useful for determining the integration interval)
    double scaledr(double r) const {
        return r==INFINITY ? 1. : r/(r+rscale); }
    /// dimension of space to integrate over (R,theta)
    virtual unsigned int numVars() const { return 2; }
    /// integrate a single function at a time
    virtual unsigned int numValues() const { return 1; }
protected:
    const BaseDensity& dens;                  ///< input density to integrate
    const int l, m;                           ///< multipole indices
    const math::IFunction& radialMultFactor;  ///< additional radius-dependent multiplier
    const double rscale;                      ///< scaling factor for integration in radius
    const double mult;                        ///< constant multiplicative factor in Y_l^m
};

}  // internal namespace

//----------------------------------------------------------------------------//

// BasePotentialSphericalHarmonic -- parent class for all potentials 
// using angular expansion in spherical harmonics
void SphericalHarmonicCoefSet::setSymmetry(coord::SymmetryType sym)
{
    mysymmetry = sym;
    lmax = (mysymmetry & coord::ST_ROTATION)  ==coord::ST_ROTATION   ? 0 :     // if spherical model, use only l=0,m=0 term
        static_cast<int>(std::min<unsigned int>(Ncoefs_angular, MAX_NCOEFS_ANGULAR-1));
    lstep= (mysymmetry & coord::ST_REFLECTION)==coord::ST_REFLECTION ? 2 : 1;  // if reflection symmetry, use only even l
    mmax = (mysymmetry & coord::ST_ZROTATION) ==coord::ST_ZROTATION  ? 0 : 1;  // if axisymmetric model, use only m=0 terms, otherwise all terms up to l (1 is the multiplying factor)
    mmin = (mysymmetry & coord::ST_TRIAXIAL)  ==coord::ST_TRIAXIAL   ? 0 :-1;  // if triaxial symmetry, do not use sine terms which correspond to m<0
    mstep= (mysymmetry & coord::ST_TRIAXIAL)  ==coord::ST_TRIAXIAL   ? 2 : 1;  // if triaxial symmetry, use only even m
}

void BasePotentialSphericalHarmonic::evalSph(const coord::PosSph &pos,
    double* potential, coord::GradSph* grad, coord::HessSph* hess) const
{
    double result = 0;
    if(grad!=NULL)
        grad->dr = grad->dtheta = grad->dphi = 0;
    if(hess!=NULL)
        hess->dr2 = hess->dtheta2 = hess->dphi2 = hess->drdtheta = hess->dthetadphi = hess->drdphi = 0;
    // arrays where angular expansion coefficients will be accumulated by calling computeSHcoefs() for derived classes
    double coefsF[MAX_NCOEFS_ANGULAR*MAX_NCOEFS_ANGULAR];      // F(theta,phi)
    double coefsdFdr[MAX_NCOEFS_ANGULAR*MAX_NCOEFS_ANGULAR];   // dF(theta,phi)/dr
    double coefsd2Fdr2[MAX_NCOEFS_ANGULAR*MAX_NCOEFS_ANGULAR]; // d2F(theta,phi)/dr2
    computeSHCoefs(pos.r, coefsF, (grad!=NULL || hess!=NULL)? coefsdFdr : NULL, 
        hess!=NULL? coefsd2Fdr2 : NULL);  // implemented in derived classes
    double legendre_array[MAX_NCOEFS_ANGULAR];
    double legendre_deriv_array[MAX_NCOEFS_ANGULAR];
    double legendre_deriv2_array[MAX_NCOEFS_ANGULAR];
    for(int m=0; m<=mmax*lmax; m+=mstep) {
        math::sphHarmArray(lmax, m, pos.theta, legendre_array, 
            grad!=NULL||hess!=NULL ? legendre_deriv_array : NULL, 
            hess!=NULL ? legendre_deriv2_array : NULL);
        double cosmphi = (m==0 ? 1 : cos(m*pos.phi)*M_SQRT2) * (2*M_SQRTPI);   // factor \sqrt{4\pi} from the definition of spherical function Y_l^m absorbed into this term
        double sinmphi = (sin(m*pos.phi)*M_SQRT2) * (2*M_SQRTPI);
        int lmin = lstep==2 ? (m+1)/2*2 : m;   // if lstep is even and m is odd, start from next even number greater than m
        for(int l=lmin; l<=lmax; l+=lstep) {
            int indx=l*(l+1)+m;
            result += coefsF[indx] * legendre_array[l-m] * cosmphi;
            if(grad!=NULL) {
                grad->dr +=  coefsdFdr[indx] * legendre_array[l-m] * cosmphi;
                grad->dtheta += coefsF[indx] * legendre_deriv_array[l-m] * cosmphi;
                grad->dphi   += coefsF[indx] * legendre_array[l-m] * (-m)*sinmphi;
            }
            if(hess!=NULL) {
                hess->dr2 +=  coefsd2Fdr2[indx] * legendre_array[l-m] * cosmphi;
                hess->drdtheta+=coefsdFdr[indx] * legendre_deriv_array[l-m] * cosmphi;
                hess->drdphi  +=coefsdFdr[indx] * legendre_array[l-m] * (-m)*sinmphi;
                hess->dtheta2   += coefsF[indx] * legendre_deriv2_array[l-m] * cosmphi;
                hess->dthetadphi+= coefsF[indx] * legendre_deriv_array[l-m] * (-m)*sinmphi;
                hess->dphi2     += coefsF[indx] * legendre_array[l-m] * cosmphi * -m*m;
            }
            if(mmin<0 && m>0) {
                indx=l*(l+1)-m;
                result += coefsF[indx] * legendre_array[l-m] * sinmphi;
                if(grad!=NULL) {
                    grad->dr +=  coefsdFdr[indx] * legendre_array[l-m] * sinmphi;
                    grad->dtheta += coefsF[indx] * legendre_deriv_array[l-m] * sinmphi;
                    grad->dphi   += coefsF[indx] * legendre_array[l-m] * m*cosmphi;
                }
                if(hess!=NULL) {
                    hess->dr2 +=  coefsd2Fdr2[indx] * legendre_array[l-m] * sinmphi;
                    hess->drdtheta+=coefsdFdr[indx] * legendre_deriv_array[l-m] * sinmphi;
                    hess->drdphi  +=coefsdFdr[indx] * legendre_array[l-m] * m*cosmphi;
                    hess->dtheta2   += coefsF[indx] * legendre_deriv2_array[l-m] * sinmphi;
                    hess->dthetadphi+= coefsF[indx] * legendre_deriv_array[l-m] * m*cosmphi;
                    hess->dphi2     += coefsF[indx] * legendre_array[l-m] * sinmphi * -m*m;
                }
            }
        }
    }
    if(potential!=NULL)
        *potential = result;
}

//----------------------------------------------------------------------------//
// Basis-set expansion for arbitrary potential (using Zhao(1995) basis set)

BasisSetExp::BasisSetExp(
    double _Alpha, unsigned int _Ncoefs_radial, unsigned int _Ncoefs_angular, 
                         const particles::PointMassArray<coord::PosSph> &points, coord::SymmetryType _sym):
    BasePotentialSphericalHarmonic(_Ncoefs_angular), 
    Ncoefs_radial(std::min<unsigned int>(MAX_NCOEFS_RADIAL-1, _Ncoefs_radial)),
    Alpha(_Alpha)
{
    setSymmetry(_sym);
    if(points.size()==0)
        throw std::invalid_argument("BasisSetExp: input particle set is empty");
    prepareCoefsDiscrete(points);
    checkSymmetry();
}

BasisSetExp::BasisSetExp(double _Alpha, const std::vector< std::vector<double> > &coefs):
    BasePotentialSphericalHarmonic(coefs.size()>0 ? static_cast<unsigned int>(sqrt(coefs[0].size()*1.0)-1) : 0), 
    Ncoefs_radial(std::min<unsigned int>(MAX_NCOEFS_RADIAL-1, static_cast<unsigned int>(coefs.size()-1))),
    Alpha(_Alpha)  // here Alpha!=0 - no autodetect
{
    if(_Alpha<0.5) 
        throw std::invalid_argument("BasisSetExp: invalid parameter Alpha");
    for(unsigned int n=0; n<coefs.size(); n++)
        if(coefs[n].size()!=pow_2(Ncoefs_angular+1))
            throw std::invalid_argument("BasisSetExp: incorrect size of coefficients array");
    SHcoefs = coefs;
    checkSymmetry();
}

BasisSetExp::BasisSetExp(double _Alpha, unsigned int _Ncoefs_radial, unsigned int _Ncoefs_angular, 
    const BaseDensity& srcdensity):    // init potential from analytic mass model
    BasePotentialSphericalHarmonic(_Ncoefs_angular), 
    Ncoefs_radial(std::min<unsigned int>(MAX_NCOEFS_RADIAL-1, _Ncoefs_radial)),
    Alpha(_Alpha)
{
    setSymmetry(srcdensity.symmetry());
    prepareCoefsAnalytic(srcdensity);
    checkSymmetry();
}

void BasisSetExp::checkSymmetry()
{
    coord::SymmetryType sym=coord::ST_SPHERICAL;  // too optimistic:))
    const double MINCOEF = 1e-8 * fabs(SHcoefs[0][0]);
    for(unsigned int n=0; n<=Ncoefs_radial; n++) {
        for(int l=0; l<=(int)Ncoefs_angular; l++)
            for(int m=-l; m<=l; m++)
                if(fabs(SHcoefs[n][l*(l+1)+m])>MINCOEF) 
                {   // nonzero coef.: check if that breaks any symmetry
                    if(l%2==1)  sym = (coord::SymmetryType)(sym & ~coord::ST_REFLECTION);
                    if(m<0 || m%2==1)  sym = (coord::SymmetryType)(sym & ~coord::ST_TRIAXIAL);
                    if(m!=0) sym = (coord::SymmetryType)(sym & ~coord::ST_ZROTATION);
                    if(l>0) sym = (coord::SymmetryType)(sym & ~coord::ST_ROTATION);
                }
    }
    // now set all coefs excluded by the inferred symmetry  to zero
    for(size_t n=0; n<=Ncoefs_radial; n++) {
        for(int l=0; l<=(int)Ncoefs_angular; l++)
            for(int m=-l; m<=l; m++)
                if( (l>0 && (sym & coord::ST_ROTATION)) ||
                   (m!=0 && (sym & coord::ST_ZROTATION)) ||
                   ((m<0 || m%2==1) && (sym & coord::ST_TRIAXIAL)) ||
                   (l%2==1 && (sym & coord::ST_REFLECTION)) ) 
                        SHcoefs[n][l*(l+1)+m] = 0;
    }
    setSymmetry(sym);
}

/// radius-dependent multiplication factor for density integration in BasisSetExp potential
class BasisSetExpRadialMult: public math::IFunctionNoDeriv {
public:
    BasisSetExpRadialMult(int _n, int _l, double _alpha) :
        n(_n), l(_l), alpha(_alpha), w((2*l+1)*alpha+0.5) {};
    virtual double value(double r) const {
        const double r1alpha = pow(r, 1./alpha);
        const double xi = (r1alpha-1)/(r1alpha+1);
        return math::gegenbauer(n, w, xi) * math::powInt(r, l) * pow(1+r1alpha, -(2*l+1)*alpha) * 4*M_PI;
    }
private:
    const int n, l;
    const double alpha, w;
};

void BasisSetExp::prepareCoefsAnalytic(const BaseDensity& srcdensity)
{
    if(Alpha<0.5)
        Alpha = 1.;
    SHcoefs.resize(Ncoefs_radial+1);
    for(size_t n=0; n<=Ncoefs_radial; n++)
        SHcoefs[n].assign(pow_2(Ncoefs_angular+1), 0);
    const double rscale = 1.0;
    for(unsigned int n=0; n<=Ncoefs_radial; n++)
        for(int l=0; l<=lmax; l+=lstep) {
            double w=(2*l+1)*Alpha+0.5;
            double Knl = (4*pow_2(n+w)-1)/8/pow_2(Alpha);
            double Inl = Knl * 4*M_PI*Alpha * 
                exp( math::lngamma(n+2*w) - 2*math::lngamma(w) - math::lnfactorial(n) - 4*w*log(2.0)) / (n+w);
            for(int m=l*mmin; m<=l*mmax; m+=mstep) {
                BasisSetExpRadialMult rmult(n, l, Alpha);
                DensitySphHarmIntegrand fnc(srcdensity, l, m, rmult, rscale);
                double xlower[2] = {fnc.scaledr(0), 0};
                double xupper[2] = {fnc.scaledr(INFINITY), 1};
                double result, error;
                int numEval;
                math::integrateNdim(fnc, 
                    xlower, xupper, EPSREL_POTENTIAL_INT, MAX_NUM_EVAL, &result, &error, &numEval);
                SHcoefs[n][l*(l+1)+m] = result * (m==0 ? 1 : M_SQRT2) / Inl;
            }
        }
}

void BasisSetExp::prepareCoefsDiscrete(const particles::PointMassArray<coord::PosSph> &points)
{
    SHcoefs.resize(Ncoefs_radial+1);
    for(unsigned int n=0; n<=Ncoefs_radial; n++)
        SHcoefs[n].assign(pow_2(1+Ncoefs_angular), 0);
    unsigned int npoints=points.size();
    if(Alpha<0.5)
        Alpha=1.;
    double legendre_array[MAX_NCOEFS_ANGULAR][MAX_NCOEFS_ANGULAR-1];
    double gegenpoly_array[MAX_NCOEFS_RADIAL];
    double Inl[MAX_NCOEFS_RADIAL][MAX_NCOEFS_ANGULAR];
    for(int l=0; l<=lmax; l+=lstep)
    {   // pre-compute coefficients
        double w=(2*l+1)*Alpha+0.5;
        for(unsigned int n=0; n<=Ncoefs_radial; n++)
            Inl[n][l] = 4*M_PI*Alpha * 
              exp( math::lngamma(n+2*w) - 2*math::lngamma(w) - math::lnfactorial(n) - 4*w*log(2.0)) /
              (n+w) * (4*(n+w)*(n+w)-1)/(8*Alpha*Alpha);
    }
    for(unsigned int i=0; i<npoints; i++) {
        const coord::PosSph& point = points.point(i);
        double massi = points.mass(i);
        double ralpha=pow(point.r, 1/Alpha);
        double xi=(ralpha-1)/(ralpha+1);
        for(int m=0; m<=lmax; m+=mstep)
            math::sphHarmArray(lmax, m, point.theta, legendre_array[m]);
        for(int l=0; l<=lmax; l+=lstep) {
            double w=(2*l+1)*Alpha+0.5;
            double phil=pow(point.r, l) * pow(1+ralpha, -(2*l+1)*Alpha);
            math::gegenbauerArray(Ncoefs_radial, w, xi, gegenpoly_array);
            for(size_t n=0; n<=Ncoefs_radial; n++) {
                double mult= massi * gegenpoly_array[n] * phil * (2*M_SQRTPI) / Inl[n][l];
                for(int m=0; m<=l*mmax; m+=mstep)
                    SHcoefs[n][l*(l+1)+m] += mult * legendre_array[m][l-m] * cos(m*point.phi) * (m==0 ? 1 : M_SQRT2);
                if(mmin)
                    for(int m=mmin*l; m<0; m+=mstep)
                        SHcoefs[n][l*(l+1)+m] += mult * legendre_array[-m][l+m] * sin(-m*point.phi) * M_SQRT2;
            }
        }
    }
}

double BasisSetExp::enclosedMass(const double r) const
{
    if(r<=0) return 0;
    double ralpha=pow(r, 1/Alpha);
    double xi=(ralpha-1)/(ralpha+1);
    double gegenpoly_array[MAX_NCOEFS_RADIAL];
    math::gegenbauerArray(Ncoefs_radial, Alpha+0.5, xi, gegenpoly_array);
    double multr = pow(1+ralpha, -Alpha);
    double multdr= -ralpha/((ralpha+1)*r);
    double result=0;
    for(int n=0; n<=static_cast<int>(Ncoefs_radial); n++) {
        double dGdr=(n>0 ? (-n*xi*gegenpoly_array[n] + (n+2*Alpha)*gegenpoly_array[n-1])/(2*Alpha*r) : 0);
        result += SHcoefs[n][0] * multr * (multdr * gegenpoly_array[n] + dGdr);
    }
    return -result * r*r;   // d Phi(r)/d r = G M(r) / r^2
}

void BasisSetExp::computeSHCoefs(const double r, double coefsF[], double coefsdFdr[], double coefsd2Fdr2[]) const
{
    double ralpha=pow(r, 1/Alpha);
    double xi=(ralpha-1)/(ralpha+1);
    double gegenpoly_array[MAX_NCOEFS_RADIAL];
    if(coefsF)      for(size_t k=0; k<pow_2(Ncoefs_angular+1); k++) coefsF     [k] = 0;
    if(coefsdFdr)   for(size_t k=0; k<pow_2(Ncoefs_angular+1); k++) coefsdFdr  [k] = 0;
    if(coefsd2Fdr2) for(size_t k=0; k<pow_2(Ncoefs_angular+1); k++) coefsd2Fdr2[k] = 0;
    for(int l=0; l<=lmax; l+=lstep) {
        double w=(2*l+1)*Alpha+0.5;
        math::gegenbauerArray(Ncoefs_radial, w, xi, gegenpoly_array);
        double multr = -pow(r, l) * pow(1+ralpha, -(2*l+1)*Alpha);
        double multdr= (l-(l+1)*ralpha)/((ralpha+1)*r);
        for(unsigned int n=0; n<=Ncoefs_radial; n++) {
            double multdFdr = 0, multd2Fdr2 = 0;
            if(coefsdFdr!=NULL) {
                double dGdr = (n>0 ? (-xi*n*gegenpoly_array[n] + (n+2*w-1)*gegenpoly_array[n-1])/(2*Alpha*r) : 0);
                multdFdr = multdr * gegenpoly_array[n] + dGdr;
                if(coefsd2Fdr2!=NULL)
                    multd2Fdr2 = ( (l+1)*(l+2)*pow_2(ralpha) + 
                                   ( (1-2*l*(l+1)) - (2*n+1)*(2*l+1)/Alpha - n*(n+1)/pow_2(Alpha))*ralpha + 
                                   l*(l-1) 
                                 ) / pow_2( (1+ralpha)*r ) * gegenpoly_array[n] - dGdr*2/r;
            }
            for(int m=l*mmin; m<=l*mmax; m+=mstep) {
                int indx=l*(l+1)+m;
                double coef = SHcoefs[n][indx] * multr;
                if(coefsF)      coefsF     [indx] += coef * gegenpoly_array[n];
                if(coefsdFdr)   coefsdFdr  [indx] += coef * multdFdr;
                if(coefsd2Fdr2) coefsd2Fdr2[indx] += coef * multd2Fdr2;
            }
        }
    }
}

//----------------------------------------------------------------------------//
// Spherical-harmonic expansion of arbitrary potential, radial part is spline interpolated on a grid

// init coefs from point mass set
SplineExp::SplineExp(unsigned int _Ncoefs_radial, unsigned int _Ncoefs_angular, 
                     const particles::PointMassArray<coord::PosSph> &points, coord::SymmetryType _sym, 
    double smoothfactor, double Rmin, double Rmax):
    BasePotentialSphericalHarmonic(_Ncoefs_angular),
    Ncoefs_radial(std::max<size_t>(5,_Ncoefs_radial))
{
    setSymmetry(_sym);
    prepareCoefsDiscrete(points, smoothfactor, Rmin, Rmax);
}

// init from existing coefs
SplineExp::SplineExp(
    const std::vector<double> &_gridradii, const std::vector< std::vector<double> > &_coefs):
    BasePotentialSphericalHarmonic(_coefs.size()>0 ? static_cast<size_t>(sqrt(_coefs[0].size()*1.0)-1) : 0), 
    Ncoefs_radial(std::min<size_t>(MAX_NCOEFS_RADIAL-1, _coefs.size()-1))
{
    for(unsigned int n=0; n<_coefs.size(); n++)
        if(_coefs[n].size()!=pow_2(Ncoefs_angular+1))
            throw std::invalid_argument("SplineExp: incorrect size of coefficients array");
    initSpline(_gridradii, _coefs);
}

// init potential from analytic mass model
SplineExp::SplineExp(unsigned int _Ncoefs_radial, unsigned int _Ncoefs_angular, 
    const BaseDensity& srcdensity, double Rmin, double Rmax):
    BasePotentialSphericalHarmonic(_Ncoefs_angular), 
    Ncoefs_radial(std::max<unsigned int>(5,_Ncoefs_radial))
{
    setSymmetry(srcdensity.symmetry());
    prepareCoefsAnalytic(srcdensity, Rmin, Rmax);
}

/// radius-dependent multiplication factor for density integration in SplineExp potential
class SplineExpRadialMult: public math::IFunctionNoDeriv {
public:
    SplineExpRadialMult(int _n) : n(_n) {};
    virtual double value(double r) const {
        return math::powInt(r, n);
    }
private:
    const int n;
};

void SplineExp::prepareCoefsAnalytic(const BaseDensity& srcdensity, double Rmin, double Rmax)
{
    // find inner/outermost radius if they were not provided
    if(Rmin<0 || Rmax<0 || (Rmax>0 && Rmax<=Rmin*Ncoefs_radial))
        throw std::invalid_argument("SplineExp: invalid choice of min/max grid radii");
    double totalmass = srcdensity.totalMass();
    if(!math::isFinite(totalmass))
        throw std::invalid_argument("SplineExp: source density model has infinite mass");
    if(Rmax==0) {
        // how far should be the outer node (leave out this fraction of mass)
        double epsout = 0.1/sqrt(pow_2(Ncoefs_radial)+0.01*pow(Ncoefs_radial*1.0,4.0));
        Rmax = getRadiusByMass(srcdensity, totalmass*(1-epsout));
    }
    if(Rmin==0) {
        // how close can we get to zero, in terms of innermost grid node
        double epsin = 5.0/pow(Ncoefs_radial*1.0,3.0);
        Rmin  = getRadiusByMass(srcdensity, totalmass*epsin*0.1);
    }
    std::vector<double> radii = //math::createNonuniformGrid(Ncoefs_radial+1, Rmin, Rmax, true);
    math::createExpGrid(Ncoefs_radial, Rmin, Rmax);
    radii.insert(radii.begin(),0);
    std::vector< std::vector<double> > coefsArray(Ncoefs_radial+1);  // SHE coefficients to pass to initspline routine
    for(unsigned int i=0; i<=Ncoefs_radial; i++)
        coefsArray[i].assign(pow_2(1+Ncoefs_angular), 0);
    const double rscale = getRadiusByMass(srcdensity, 0.5*totalmass);  // scaling factor for integration in radius
    std::vector<double> coefsInner, coefsOuter;
    const double SPLINE_MIN_RADIUS = 1e-10;
    radii.front() = SPLINE_MIN_RADIUS*radii[1];  // to prevent log divergence for gamma=2 potentials
    for(int l=0; l<=lmax; l+=lstep) {
        for(int m=l*mmin; m<=l*mmax; m+=mstep) {
            // first precompute inner and outer density integrals at each radial grid point, summing contributions from each interval of radial grid
            coefsInner.assign(Ncoefs_radial+1, 0);
            coefsOuter.assign(Ncoefs_radial+1, 0);
            // loop over inner intervals
            double result, error;
            for(size_t c=0; c<Ncoefs_radial; c++) {
                SplineExpRadialMult rmult(l);
                DensitySphHarmIntegrand fnc(srcdensity, l, m, rmult, rscale);
                double xlower[2] = {fnc.scaledr(radii[c]), 0};
                double xupper[2] = {fnc.scaledr(radii[c+1]), 1};
                int numEval;
                math::integrateNdim(fnc, 
                    xlower, xupper, EPSREL_POTENTIAL_INT, MAX_NUM_EVAL, &result, &error, &numEval);
                coefsInner[c+1] = result + coefsInner[c];
            }
            // loop over outer intervals, starting from infinity backwards
            for(size_t c=Ncoefs_radial+1; c>(l==0 ? 0 : 1u); c--) {
                SplineExpRadialMult rmult(-l-1);
                DensitySphHarmIntegrand fnc(srcdensity, l, m, rmult, rscale);
                double xlower[2] = {fnc.scaledr(radii[c-1]), 0};
                double xupper[2] = {fnc.scaledr(c>Ncoefs_radial ? INFINITY : radii[c]), 1};
                int numEval;
                math::integrateNdim(fnc,
                    xlower, xupper, EPSREL_POTENTIAL_INT, MAX_NUM_EVAL, &result, &error, &numEval);
                coefsOuter[c-1] = result + (c>Ncoefs_radial?0:coefsOuter[c]);
            }
            // now compute the coefs of potential expansion themselves
            for(size_t c=0; c<=Ncoefs_radial; c++) {
                coefsArray[c][l*(l+1) + m] = ((c>0 ? coefsInner[c]*pow(radii[c], -l-1.0) : 0) + coefsOuter[c]*pow(radii[c], l*1.0)) *
                    -4*M_PI/(2*l+1) * (m==0 ? 1 : M_SQRT2);
#if 0 //#ifdef VERBOSE_REPORT
                std::cout << l<<'\t' << radii[c] << '\t' << 
                (sqrt(2*l+1)*coefsArray[c][l*(l+1) + m]/(m==0 ? 1 : M_SQRT2)) << '\t' << 
                (4*M_PI*sqrt(2*l+1)*coefsInner[c]*pow(radii[c],-l-1.0)/(m==0 ? 1 : M_SQRT2)) << '\t' << 
                (4*M_PI*sqrt(2*l+1)*coefsOuter[c]*pow(radii[c], l*1.0)/(m==0 ? 1 : M_SQRT2)) << '\n';
#endif
            }
#ifdef DEBUGPRINT
            my_message(FUNCNAME, "l="+convertToString(l)+",m="+convertToString(m));
#endif
        }
    }
    radii.front()=0;
    initSpline(radii, coefsArray);
}

/// \cond INTERNAL_DOCS
inline bool compareParticleSph(
    const particles::PointMassArray<coord::PosSph>::ElemType& val1, 
    const particles::PointMassArray<coord::PosSph>::ElemType& val2) {
    return val1.first.r < val2.first.r;  }
/// \endcond

void SplineExp::computeCoefsFromPoints(const particles::PointMassArray<coord::PosSph> &srcPoints, 
    std::vector<double>& outputRadii, std::vector< std::vector<double> >& outputCoefs)
{
    double legendre_array[MAX_NCOEFS_ANGULAR][MAX_NCOEFS_ANGULAR-1];
    size_t npoints = srcPoints.size();
    for(size_t i=0; i<npoints; i++) {
        if(srcPoints.point(i).r<=0)
            throw std::invalid_argument("SplineExp: particles at r=0 are not allowed");
        if(srcPoints.mass(i)<0) 
            throw std::invalid_argument("SplineExp: input particles have negative mass");
    }

    // make a copy of input array to allow it to be sorted
    particles::PointMassArray<coord::PosSph> points(srcPoints);
    std::sort(points.data.begin(), points.data.end(), compareParticleSph);

    // having sorted particles in radius, may now initialize coefs
    outputRadii.resize(npoints);
    for(size_t i=0; i<npoints; i++)
        outputRadii[i] = points.point(i).r;

    // we need two intermediate arrays of inner and outer coefficients for each particle,
    // and in the end we output one array of 'final' coefficients for each particle.
    // We can use a trick to save memory, by allocating only one temporary array, 
    // and using the output array as the second intermediate one.
    std::vector< std::vector<double> > coefsInner(pow_2(Ncoefs_angular+1));  // this is the 1st temp array
    outputCoefs.resize(pow_2(Ncoefs_angular+1));  // this will be the final array
    // instead of allocating 2nd temp array, we use a reference to the already existing array
    std::vector< std::vector<double> >& coefsOuter = outputCoefs;
    // reserve memory only for those coefficients that are actually needed
    for(int l=0; l<=lmax; l+=lstep)
        for(int m=l*mmin; m<=l*mmax; m+=mstep)
            coefsOuter[l*(l+1)+m].assign(npoints, 0);  // reserve memory only for those coefs that will be used
    for(int l=0; l<=lmax; l+=lstep)
        for(int m=l*mmin; m<=l*mmax; m+=mstep)
            coefsInner[l*(l+1)+m].assign(npoints, 0);  // yes do it separately from the above, to allow contiguous block of memory to be freed after deleting CoefsInner

    // initialize SH expansion coefs at each point's location
    for(size_t i=0; i<npoints; i++) {
        for(int m=0; m<=lmax; m+=mstep)
            math::sphHarmArray(lmax, m, points.point(i).theta, legendre_array[m]);
        for(int l=0; l<=lmax; l+=lstep)
            for(int m=l*mmin; m<=l*mmax; m+=mstep) {
                int coefind = l*(l+1)+m;
                int absm = math::abs(m);  // negative m correspond to sine, positive - to cosine
                double mult = -sqrt(4*M_PI)/(2*l+1) * (m==0 ? 1 : M_SQRT2) * points.mass(i) *
                    legendre_array[absm][l-absm] * 
                    (m>=0 ? cos(m*points.point(i).phi) : sin(-m*points.point(i).phi));
                coefsOuter[coefind][i] = mult * pow(points.point(i).r, -(1+l));
                coefsInner[coefind][i] = mult * pow(points.point(i).r, l);
            }
    }

    // sum inner coefs interior and outer coefs exterior to each point's location
    for(int l=0; l<=lmax; l+=lstep)
        for(int m=l*mmin; m<=l*mmax; m+=mstep)
        {
            int coefind=l*(l+1)+m;
            for(size_t i=1; i<npoints; i++)
                coefsInner[coefind][i] += coefsInner[coefind][i-1];
            for(size_t i=npoints-1; i>0; i--)
                coefsOuter[coefind][i-1] += coefsOuter[coefind][i];
        }

    // initialize potential expansion coefs by multiplying 
    // inner and outer coefs by r^(-1-l) and r^l, correspondingly
    for(size_t i=0; i<npoints; i++) {
        for(int l=0; l<=lmax; l+=lstep)
            for(int m=l*mmin; m<=l*mmax; m+=mstep) {
                int coefind = l*(l+1)+m;
                // note that here we are destroying the values of CoefsOuter, because this array
                // is aliased with outcoefs; but we do it from inside out, and for each i-th point
                // the coefficients from i+1 till the end of array are still valid.
                outputCoefs[coefind][i] = 
                    (i>0 ? coefsInner[coefind][i-1] * pow(points.point(i).r, -(1+l)) : 0) + 
                    coefsOuter[coefind][i] * pow(points.point(i).r, l);
            }
    }
    // local variable coefsInner will be automatically freed, but outputCoefs will remain
}

/** obtain the value of scaling radius for non-spherical harmonic coefficients `ascale`
    from the radial dependence of the l=0 coefficient, by finding the radius at which
    the value of this coefficient equals half of its value at r=0 */
double get_ascale(const std::vector<double>& radii, const std::vector<std::vector<double> >& coefsArray)
{
    assert(radii.size() == coefsArray.size());
    double targetVal = fabs(coefsArray[0][0])*0.5;
    double targetRad = NAN;
    for(size_t i=1; i<radii.size() && targetRad!=targetRad; i++) 
        if(fabs(coefsArray[i][0]) < targetVal && fabs(coefsArray[i-1][0]) >= targetVal) {
            // linearly interpolate
            targetRad = radii[i-1] + (radii[i]-radii[i-1]) *
                (targetVal-fabs(coefsArray[i-1][0])) / (fabs(coefsArray[i][0])-fabs(coefsArray[i-1][0]));
        }
    if(targetRad!=targetRad)  // shouldn't occur, but if it does, return some sensible value
        targetRad = radii[radii.size()/2];
    return targetRad;
}

void SplineExp::prepareCoefsDiscrete(const particles::PointMassArray<coord::PosSph> &points, 
    double smoothfactor, double innerBinRadius, double outerBinRadius)
{
    if(points.size() <= Ncoefs_radial*10)
        throw std::invalid_argument("SplineExp: number of particles is too small");
    if(innerBinRadius<0 || outerBinRadius<0 ||
        (outerBinRadius>0 && outerBinRadius<=innerBinRadius*Ncoefs_radial))
        throw std::invalid_argument("SplineExp: invalid choice of min/max grid radii");
    // radii of each point in ascending order
    std::vector<double> pointRadii;
    // note that array indexing is swapped w.r.t. coefsArray (i.e. pointCoefs[coefIndex][pointIndex])
    // to save memory by not initializing unnecessary coefs
    std::vector< std::vector<double> > pointCoefs;
    computeCoefsFromPoints(points, pointRadii, pointCoefs);

    // choose the radial grid parameters if they were not provided:
    // innermost cell contains minBinMass and outermost radial node should encompass cutoffMass.
    // spline definition region extends up to outerRadius which is 
    // ~several times larger than outermost radial node, 
    // however coefficients at that radius are not used in the potential computation later
    const size_t npoints = pointRadii.size();
    const size_t minBinPoints = 10;
    size_t npointsMargin = static_cast<size_t>(sqrt(npoints*0.1));
    // number of points within 1st grid radius
    size_t npointsInnerGrid = std::max<size_t>(minBinPoints, npointsMargin);
    // number of points within outermost grid radius
    size_t npointsOuterGrid = npoints - std::max<size_t>(minBinPoints, npointsMargin);
    if(innerBinRadius < pointRadii[0])
        innerBinRadius = pointRadii[npointsInnerGrid];
    if(outerBinRadius == 0 || outerBinRadius > pointRadii.back())
        outerBinRadius = pointRadii[npointsOuterGrid];
    std::vector<double> radii =     // radii of grid nodes to pass to initspline routine
        math::createNonuniformGrid(Ncoefs_radial+1, innerBinRadius, outerBinRadius, true);

    // find index of the inner- and outermost points which are used in b-spline fitting
    size_t npointsInnerSpline = 0;
    while(pointRadii[npointsInnerSpline]<radii[1])
        npointsInnerSpline++;
    npointsInnerSpline = std::min<size_t>(npointsInnerSpline-2,
        std::max<size_t>(minBinPoints, npointsInnerSpline/2));
    // index of last point used in b-spline fitting 
    // (it is beyond outer grid radius, since b-spline definition region
    // is larger than the range of radii for which the spline approximation
    // will eventually be constructed) -
    // roughly equally logarithmically spaced from the last two points
    double outerRadiusSpline = pow_2(radii[Ncoefs_radial])/radii[Ncoefs_radial-1];
    size_t npointsOuterSpline = npoints-1;
    while(pointRadii[npointsOuterSpline]>outerRadiusSpline)
        npointsOuterSpline--;
    //!!!FIXME: what happens if the outermost pointRadius is < outerBinRadius ?

    // outer and inner points are ignored
    size_t numPointsUsed = npointsOuterSpline-npointsInnerSpline;
    // including zero and outermost point; only interior nodes are actually used
    // for computing best-fit coefs (except l=0 coef, for which r=0 is also used)
    size_t numBSplineKnots = Ncoefs_radial+2;
    // transformed x- and y- values of original data points
    // which will be approximated by a spline regression
    std::vector<double> scaledPointRadii(numPointsUsed), scaledPointCoefs(numPointsUsed);
    // transformed x- and y- values of regression spline knots
    std::vector<double> scaledKnotRadii(numBSplineKnots), scaledSplineValues;

    // SHE coefficients to pass to initspline routine
    std::vector< std::vector<double> > coefsArray(Ncoefs_radial+1);
    for(size_t i=0; i<=Ncoefs_radial; i++)
        coefsArray[i].assign(pow_2(Ncoefs_angular+1), 0);

    // open block so that temp variable "appr" will be destroyed upon closing this block
    {
        // first construct spline for zero-order term (radial dependence)
        potcenter=pointCoefs[0][0];  // value of potential at origin (times 1/2\sqrt{\pi} ?)
        for(size_t p=1; p<=Ncoefs_radial; p++)
            scaledKnotRadii[p] = log(radii[p]);
        scaledKnotRadii[Ncoefs_radial+1] = log(outerRadiusSpline);
        scaledKnotRadii[0] = log(pointRadii[npointsInnerSpline]);
        for(size_t i=0; i<numPointsUsed; i++)
        {
            scaledPointRadii[i] = log(pointRadii[i+npointsInnerSpline]);
            scaledPointCoefs[i] = log(1/(1/potcenter - 1/pointCoefs[0][i+npointsInnerSpline]));
        }
        math::SplineApprox appr(scaledPointRadii, scaledKnotRadii);
//        if(appr.isSingular())
//            my_message(FUNCNAME, 
//                "Warning, in Spline potential initialization: singular matrix for least-square fitting; fallback to a slow algorithm");
        double derivLeft, derivRight;
        appr.fitData(scaledPointCoefs, 0, scaledSplineValues, derivLeft, derivRight);
        // now store fitted values in coefsArray to pass to initspline routine
        coefsArray[0][0] = potcenter;
        for(size_t c=1; c<=Ncoefs_radial; c++)
            coefsArray[c][0] = -1./(exp(-scaledSplineValues[c])-1/potcenter);
    }
    if(lmax>0) {  // construct splines for all l>0 spherical-harmonic terms separately
        // first estimate the asymptotic power-law slope of coefficients at r=0 and r=infinity
        double gammaInner = 2-log((coefsArray[1][0]-potcenter)/(coefsArray[2][0]-potcenter))/log(radii[1]/radii[2]);
        if(gammaInner<0) gammaInner=0; 
        if(gammaInner>2) gammaInner=2;
        // this was the estimate of density slope. Now we need to convert it to the estimate of power-law slope of l>0 coefs
        gammaInner = 2.0-gammaInner;   // the same recipe as used later in initSpline
        double gammaOuter = -1.0;      // don't freak out, assume default value
        // init x-coordinates from scaling transformation
        ascale = get_ascale(radii, coefsArray);  // this uses only the l=0 term
        for(size_t p=0; p<=Ncoefs_radial; p++)
            scaledKnotRadii[p] = log(ascale+radii[p]);
        scaledKnotRadii[Ncoefs_radial+1] = log(ascale+outerRadiusSpline);
        for(size_t i=0; i<numPointsUsed; i++)
            scaledPointRadii[i] = log(ascale+pointRadii[i+npointsInnerSpline]);
        math::SplineApprox appr(scaledPointRadii, scaledKnotRadii);
//        if(appr.status()==CSplineApprox::AS_SINGULAR)
//            my_message(FUNCNAME, 
//                "Warning, in Spline potential initialization: singular matrix for least-square fitting; fallback to a slow algorithm without smoothing");
        // loop over l,m
        for(int l=lstep; l<=lmax; l+=lstep)
        {
            for(int m=l*mmin; m<=l*mmax; m+=mstep)
            {
                int coefind=l*(l+1) + m;
                // init matrix of values to fit
                for(size_t i=0; i<numPointsUsed; i++)
                    scaledPointCoefs[i] = pointCoefs[coefind][i+npointsInnerSpline]/pointCoefs[0][i+npointsInnerSpline];
                double derivLeft, derivRight;
                double edf=0;  // equivalent number of free parameters in the fit; if it is ~2, fit is oversmoothed to death (i.e. to a linear regression, which means we should ignore it)
                appr.fitDataOversmooth(scaledPointCoefs, smoothfactor, scaledSplineValues, derivLeft, derivRight, NULL, &edf);
                if(edf<3.0)   // in case of error or an oversmoothed fit fallback to zero values
                    scaledSplineValues.assign(Ncoefs_radial+1, 0);
                // now store fitted values in coefsArray to pass to initspline routine
                coefsArray[0][coefind] = 0;  // unused
                for(size_t c=1; c<=Ncoefs_radial; c++)
                    coefsArray[c][coefind] = scaledSplineValues[c] * coefsArray[c][0];  // scale back (multiply by l=0,m=0 coefficient)
                // correction to avoid fluctuation at first and last grid radius
                if( coefsArray[1][coefind] * coefsArray[2][coefind] < 0 || coefsArray[1][coefind]/coefsArray[2][coefind] > pow(radii[1]/radii[2], gammaInner))
                    coefsArray[1][coefind] = coefsArray[2][coefind] * pow(radii[1]/radii[2], gammaInner);   // make the smooth curve drop to zero at least as fast as gammaInner'th power of radius
                if( coefsArray[Ncoefs_radial][coefind] * coefsArray[Ncoefs_radial-1][coefind] < 0 || 
                    coefsArray[Ncoefs_radial][coefind] / coefsArray[Ncoefs_radial-1][coefind] > pow(radii[Ncoefs_radial]/radii[Ncoefs_radial-1], gammaOuter))
                    coefsArray[Ncoefs_radial][coefind] = coefsArray[Ncoefs_radial-1][coefind] * pow(radii[Ncoefs_radial]/radii[Ncoefs_radial-1], gammaOuter);
            }
        }
    }
    initSpline(radii, coefsArray);
}

void SplineExp::checkSymmetry(const std::vector< std::vector<double> > &coefsArray)
{
    coord::SymmetryType sym=coord::ST_SPHERICAL;  // too optimistic:))
    // if ALL coefs of a certain subset of indices are below this value, assume some symmetry
    const double MINCOEF = 1e-8 * fabs(coefsArray[0][0]);
    for(size_t n=0; n<=Ncoefs_radial; n++)
    {
        for(int l=0; l<=(int)Ncoefs_angular; l++)
            for(int m=-l; m<=l; m++)
                if(fabs(coefsArray[n][l*(l+1)+m])>MINCOEF) 
                {   // nonzero coef.: check if that breaks any symmetry
                    if(l%2==1)  sym = (coord::SymmetryType)(sym & ~coord::ST_REFLECTION);
                    if(m<0 || m%2==1)  sym = (coord::SymmetryType)(sym & ~coord::ST_TRIAXIAL);
                    if(m!=0) sym = (coord::SymmetryType)(sym & ~coord::ST_ZROTATION);
                    if(l>0) sym = (coord::SymmetryType)(sym & ~coord::ST_ROTATION);
                }
    }
    setSymmetry(sym); 
}

/// \cond INTERNAL_DOCS
// auxiliary functions to find outer slope and 1st order correction to inner power-law slope for potential
class FindGammaOut: public math::IFunctionNoDeriv {
private:
    double r1,r2,r3,K;
public:
    FindGammaOut(double _r1, double _r2, double _r3, double _K) :
        r1(_r1), r2(_r2), r3(_r3), K(_K) {};
    virtual double value(double y) const {
        return( pow(r2, 3-y) - pow(r1, 3-y))/( pow(r3, 3-y) - pow(r2, 3-y)) - K;
    }
};
class FindBcorrIn: public math::IFunctionNoDeriv {
private:
    double r1,r2,r3,K2,K3;
public:
    FindBcorrIn(double _r1, double _r2, double _r3, double _K2, double _K3) :
        r1(_r1), r2(_r2), r3(_r3), K2(_K2), K3(_K3) {};
    virtual double value(double B) const {
        return (K2 - log( (1-B*r2)/(1-B*r1) ))*log(r3/r1) - (K3 - log( (1-B*r3)/(1-B*r1) ))*log(r2/r1);
    }
};
/// \endcond

void SplineExp::initSpline(const std::vector<double> &_radii, const std::vector< std::vector<double> > &_coefsArray)
{
    if(_radii[0]!=0)  // check if the innermost node is at origin
        throw std::invalid_argument("SplineExp: radii[0] != 0");
    if(_radii.size()!=Ncoefs_radial+1 || _coefsArray.size()!=Ncoefs_radial+1)
        throw std::invalid_argument("SplineExp: coefArray length != Ncoefs_radial+1");
    potcenter=_coefsArray[0][0];
    // safety measure: if zero-order coefs are the same as potcenter for r>0, skip these elements
    std::vector<double> newRadii;
    std::vector< std::vector<double> > newCoefsArray;
    size_t nskip=0;
    while(nskip+1<_coefsArray.size() && _coefsArray[nskip+1][0]==potcenter)
        nskip++;   // values of potential at r>0 should be strictly larger than at r=0
    if(nskip>0) {  // skip some elements
        newRadii=_radii;
        newRadii.erase(newRadii.begin()+1, newRadii.begin()+nskip+1);
        Ncoefs_radial=newRadii.size()-1;
        newCoefsArray=_coefsArray;
        newCoefsArray.erase(newCoefsArray.begin()+1, newCoefsArray.begin()+nskip+1);
        if(newRadii.size()<5) 
            throw std::invalid_argument("SplineExp: too few radial points");
    }
    const std::vector<double> &radii = nskip==0 ? _radii : newRadii;
    const std::vector< std::vector<double> > &coefsArray = nskip==0 ? _coefsArray : newCoefsArray;
    checkSymmetry(coefsArray);   // assign nontrivial symmetry class if some of coefs are equal or close to zero
    gridradii=radii;    // copy real radii
    minr=gridradii[1];
    maxr=gridradii.back();
    std::vector<double> spnodes(Ncoefs_radial);  // scaled radii
    std::vector<double> spvalues(Ncoefs_radial);
    splines.resize (pow_2(Ncoefs_angular+1));
    slopein. assign(pow_2(Ncoefs_angular+1), 1.);
    slopeout.assign(pow_2(Ncoefs_angular+1), -1.);

    // estimate outermost slope  (needed for accurate extrapolation beyond last grid point)
    const double Kout =
        ( coefsArray[Ncoefs_radial  ][0] * radii[Ncoefs_radial] - 
          coefsArray[Ncoefs_radial-1][0] * radii[Ncoefs_radial-1] ) / 
        ( coefsArray[Ncoefs_radial-1][0] * radii[Ncoefs_radial-1] - 
          coefsArray[Ncoefs_radial-2][0] * radii[Ncoefs_radial-2] );
    if(math::isFinite(Kout)) {
        FindGammaOut fout(radii[Ncoefs_radial], radii[Ncoefs_radial-1], radii[Ncoefs_radial-2], Kout);
        gammaout = math::findRoot(fout, 3.01, 10., ACCURACY_ROOT);
        if(gammaout != gammaout)
            gammaout = 4.0;
        coefout = fmax(0,
            (1 - coefsArray[Ncoefs_radial-1][0] * radii[Ncoefs_radial-1] /
                (coefsArray[Ncoefs_radial  ][0] * radii[Ncoefs_radial]) ) / 
            (pow(radii[Ncoefs_radial-1] / radii[Ncoefs_radial], 3-gammaout) - 1) );
    } else {
        gammaout=10.0;
        coefout=0;
    }

    // estimate innermost slope with 1st order correction for non-zero radii 
    const double K2 = log((coefsArray[2][0]-potcenter)/(coefsArray[1][0]-potcenter));
    const double K3 = log((coefsArray[3][0]-potcenter)/(coefsArray[1][0]-potcenter));
    FindBcorrIn fin(radii[1], radii[2], radii[3], K2, K3);
    double B = math::findRoot(fin, 0, 0.9/radii[3], ACCURACY_ROOT);
    if(B!=B)
        B = 0.;
    gammain = 2. - ( log((coefsArray[2][0]-potcenter)/(coefsArray[1][0]-potcenter)) - 
                     log((1-B*radii[2])/(1-B*radii[1])) ) / log(radii[2]/radii[1]);
    double gammainuncorr = 2. - log((coefsArray[2][0]-potcenter)/(coefsArray[1][0]-potcenter)) / 
        log(radii[2]/radii[1]);
    if(gammain>=1) gammain=gammainuncorr;
    if(gammain<0) gammain=0; 
    if(gammain>2) gammain=2;
    coefin = (1-coefsArray[1][0]/potcenter) / pow(radii[1], 2-gammain);
#ifdef DEBUGPRINT
    my_message(FUNCNAME, "gammain="+convertToString(gammain)+
        " ("+convertToString(gammainuncorr)+");  gammaout="+convertToString(gammaout));
#endif

    potmax  = coefsArray.back()[0];
    potminr = coefsArray[1][0];
    // first init l=0 spline which has radial scaling "log(r)" and nontrivial transformation 1/(1/phi-1/phi0)
    for(size_t i=0; i<Ncoefs_radial; i++)
    {
        spnodes[i] = log(gridradii[i+1]);
        spvalues[i]= log(1/ (1/potcenter - 1/coefsArray[i+1][0]));
    }
    double derivLeft  = -(2-gammain)*potcenter/coefsArray[1][0];   // derivative at leftmost node
    double derivRight = - (1+coefout*(3-gammaout))/(1 - potmax/potcenter);  // derivative at rightmost node
    splines[0] = math::CubicSpline(spnodes, spvalues, derivLeft, derivRight);
    coef0(maxr, NULL, NULL, &der2out);

    // next init all higher-order splines which have radial scaling log(ascale+r) and value scaled to l=0,m=0 coefficient
    ascale = get_ascale(radii, coefsArray);
    for(size_t i=0; i<Ncoefs_radial; i++)
        spnodes[i] = log(ascale+gridradii[i+1]);
    double C00val, C00der;
    coef0(minr, &C00val, &C00der, NULL);
    for(int l=lstep; l<=lmax; l+=lstep)
    {
        for(int m=l*mmin; m<=l*mmax; m+=mstep)
        {
            int coefind=l*(l+1)+m;
            for(size_t i=0; i<Ncoefs_radial; i++)
                spvalues[i] = coefsArray[i+1][coefind]/coefsArray[i+1][0];
            slopein[coefind] = log(coefsArray[2][coefind]/coefsArray[1][coefind]) / log(gridradii[2]/gridradii[1]);   // estimate power-law slope of Clm(r) at r->0
            if(!math::isFinite(slopein[coefind]))
                slopein[coefind]=1.0;  // default
            slopein[coefind] = std::max<double>(slopein[coefind], std::min<double>(l, 2-gammain));  // the asymptotic power-law behaviour of the coefficient expected for power-law density profile
            derivLeft = spvalues[0] * (1+ascale/minr) * (slopein[coefind] - minr*C00der/C00val);   // derivative at innermost node
            slopeout[coefind] = log(coefsArray[Ncoefs_radial][coefind]/coefsArray[Ncoefs_radial-1][coefind]) / log(gridradii[Ncoefs_radial]/gridradii[Ncoefs_radial-1]) + 1;   // estimate slope of Clm(r)/C00(r) at r->infinity (+1 is added because C00(r) ~ 1/r at large r)
            if(!math::isFinite(slopeout[coefind]))
                slopeout[coefind]=-1.0;  // default
            slopeout[coefind] = std::min<double>(slopeout[coefind], std::max<double>(-l, 3-gammaout));
            derivRight = spvalues[Ncoefs_radial-1] * (1+ascale/maxr) * slopeout[coefind];   // derivative at outermost node
            splines[coefind] = math::CubicSpline(spnodes, spvalues, derivLeft, derivRight);
#ifdef DEBUGPRINT
            my_message(FUNCNAME, "l="+convertToString(l)+", m="+convertToString(m)+
                " - inner="+convertToString(slopein[coefind])+", outer="+convertToString(slopeout[coefind]));
#endif
        }
    }
#if 0
    bool densityNonzero = checkDensityNonzero();
    bool massMonotonic  = checkMassMonotonic();
    if(!massMonotonic || !densityNonzero) 
        my_message(FUNCNAME, "Warning, " + 
        std::string(!massMonotonic ? "mass does not monotonically increase with radius" : "") +
        std::string(!massMonotonic && !densityNonzero ? " and " : "") + 
        std::string(!densityNonzero ? "density drops to zero at a finite radius" : "") + "!");
#endif
}

void SplineExp::getCoefs(
    std::vector<double> &radii, std::vector< std::vector<double> > &coefsArray) const
{
    radii.resize(Ncoefs_radial+1);
    for(size_t i=0; i<=Ncoefs_radial; i++)
        radii[i] = gridradii[i];
    coefsArray.resize(Ncoefs_radial+1);
    for(size_t i=0; i<=Ncoefs_radial; i++) {
        double rad = radii[i];
        double Coef00;
        coef0(rad, &Coef00, NULL, NULL);
        coefsArray[i].assign(pow_2(Ncoefs_angular+1), 0);
        coefsArray[i][0] = Coef00;
        double xi = log(ascale+rad);
        for(int l=lstep; l<=lmax; l+=lstep)
            for(int m=l*mmin; m<=l*mmax; m+=mstep) {
                int coefind=l*(l+1)+m;
                coeflm(coefind, rad, xi, &(coefsArray[i][l*(l+1)+m]), NULL, NULL, Coef00);
            }
    }
}

void SplineExp::coeflm(unsigned int lm, double r, double xi, double *val, double *der, double *der2, double c0val, double c0der, double c0der2) const  // works only for l2>0
{
    double cval=0, cder=0, cder2=0;   // value and derivatives of \tilde Clm = Clm(r)/C00(r)
    if(r < maxr)
    {
        if(r > minr)  // normal interpolation
        {
            if(der==NULL) {
                splines[lm].evalDeriv(xi, &cval, NULL, NULL);
            } else if(der2==NULL) {
                splines[lm].evalDeriv(xi, &cval, &cder, NULL);
                cder /= r+ascale;
            } else {
                splines[lm].evalDeriv(xi, &cval, &cder, &cder2);
                cder /= r+ascale;
                cder2 = (cder2/(r+ascale) - cder)/(r+ascale);
            }
        }
        else  // power-law asymptotics at r<minr
        {
            cval = splines[lm](splines[lm].xmin()) * potminr;
            if(val!=NULL)  *val = cval * pow(r/minr, slopein[lm]);
            if(der!=NULL){ *der = (*val) * slopein[lm]/r;
            if(der2!=NULL) *der2= (*der) * (slopein[lm]-1)/r; }
            return;   // for r<minr, Clm is not scaled by C00
        }
    }
    else  // power-law asymptotics for r>maxr
    {     // god knows what happens here...
        double ximax = splines[lm].xmax();
        double mval, mder, mder2;
        splines[lm].evalDeriv(ximax, &mval, &mder, &mder2);
        cval = mval * pow(r/maxr, slopeout[lm]);
        cder = cval * slopeout[lm]/r;
        cder2= cder * (slopeout[lm]-1)/r;
        double der2left = (mder2 - mder)/pow_2(r+ascale);
        double der2right = mval*slopeout[lm]*(slopeout[lm]-1)/pow_2(maxr);
        double acorr = (der2left-der2right)*0.5;
        double slopecorr = slopeout[lm]-4;
        double powcorr = pow(r/maxr, slopecorr);
        cval += acorr*powcorr*pow_2(r-maxr);
        cder += acorr*powcorr*(r-maxr)*(2 + slopecorr*(r-maxr)/r);
        cder2+= acorr*powcorr*(2 + 4*slopecorr*(r-maxr)/r + pow_2(1-maxr/r)*slopecorr*(slopecorr-1));
    }
    // scale by C00
    if(val!=NULL)  *val = cval*c0val;
    if(der!=NULL)  *der = cder*c0val + cval*c0der;
    if(der2!=NULL) *der2= cder2*c0val + 2*cder*c0der + cval*c0der2;
}

void SplineExp::coef0(double r, double *val, double *der, double *der2) const  // works only for l=0
{
    if(r<=maxr) {
        double logr=log(r);
        double sval, sder, sder2;
        if(r<minr) {
            double ratio = 1-coefin*pow(r, 2-gammain);  // C00(r)/C00(0)
            sval = log(-potcenter/coefin) - (2-gammain)*logr + log(ratio);
            sder = -(2-gammain)/ratio;
            sder2= -pow_2(sder)*(1-ratio);
        } else {
            splines[0].evalDeriv(logr, &sval, &sder, &sder2);
        }
        double sexp = (r>0? exp(-sval) : 0);
        double vval = 1./(sexp-1/potcenter);
        if(val!=NULL)  *val = -vval;
        if(der!=NULL)  *der = -vval*vval*sexp/r * sder;  // this would not work for r=0 anyway...
        if(der2!=NULL) *der2= -pow_2(vval/r)*sexp * (sder2 - sder + sder*sder*(2*vval*sexp-1) );
    } else {
        double r_over_maxr_g=pow(r/maxr, 3-gammaout);
        double der2right = -2*potmax*maxr/pow(r,3) * (1 - coefout*(r_over_maxr_g*(gammaout-1)*(gammaout-2)/2 - 1));
        double slopecorr = -gammaout-4;
        double acorr = 0*pow(r/maxr, slopecorr) * (der2out-der2right)*0.5;   // apparently unused, but why?
        if(val!=NULL)  *val = -(-potmax*maxr/r * (1 - coefout*(r_over_maxr_g-1))  + acorr*pow_2(r-maxr) );
        if(der!=NULL)  *der = -( potmax*maxr/r/r * (1 - coefout*(r_over_maxr_g*(gammaout-2) - 1))  + acorr*(r-maxr)*(2 + slopecorr*(r-maxr)/r) );
        if(der2!=NULL) *der2= -(der2right  + acorr*(2 + 4*slopecorr*(r-maxr)/r + pow_2(1-maxr/r)*slopecorr*(slopecorr-1)) );
    }
}

void SplineExp::computeSHCoefs(const double r, double coefsF[], double coefsdFdr[], double coefsd2Fdr2[]) const
{
    double xi = log(r+ascale);
    double val00, der00, der200;
    coef0(r, &val00, &der00, &der200);  // compute value and two derivatives of l=0,m=0 spline
    if(coefsF)    coefsF[0]    = val00;
    if(coefsdFdr) coefsdFdr[0] = der00;
    if(coefsd2Fdr2) coefsd2Fdr2[0] = der200;
    for(int l=lstep; l<=lmax; l+=lstep)
    {
        for(int m=l*mmin; m<=l*mmax; m+=mstep)
        {
            int coefind=l*(l+1)+m;
            coeflm(coefind, r, xi, 
                coefsF!=NULL ? coefsF+coefind : NULL, 
                coefsdFdr!=NULL ? coefsdFdr+coefind : NULL, 
                coefsd2Fdr2!=NULL ? coefsd2Fdr2+coefind : NULL, 
                val00, der00, der200);
        }
    }
}

double SplineExp::enclosedMass(const double r) const
{
    if(r<=0) return 0;
    double der;
    coef0(r, NULL, &der, NULL);
    return der * r*r;   // d Phi(r)/d r = G M(r) / r^2
}

///////////-------- NEW IMPLEMENTATIONS -----------/////////////

namespace {  // internal routines

// Helper function to deduce symmetry from the list of non-zero coefficients;
// combine the array of coefficients at different radii into a single array
// and then call the corresponding routine from math::.
// This routine is templated on the number of arrays that it handles:
// each one should have identical number of elements (SH coefs at different radii),
// and each element of each array should have the same dimension (number of SH coefs).
template<int N>
static math::SphHarmIndices getIndicesFromCoefs(const std::vector< std::vector<double> >* C[N])
{
    unsigned int numRadii=0, numCoefs=0;
    bool correct=true;
    for(int n=0; n<N; n++) {
        if(n==0) {
            numRadii = C[n]->size();
            if(numRadii>0)
                numCoefs = C[n]->at(0).size();
        } else
            correct &= C[n]->size() == numRadii;
    }
    std::vector<double> D(numCoefs);
    for(unsigned int k=0; correct && k<numRadii; k++) {
        for(int n=0; n<N; n++) {
            if(C[n]->at(k).size() == numCoefs)
                // if any of the elements is non-zero,
                // then the combined array will have non-zero c-th element too.
                for(unsigned int c=0; c<numCoefs; c++)
                    D[c] += fabs(C[n]->at(k)[c]);
            else
                correct = false;
        }
    }
    if(!correct)
        throw std::invalid_argument("Error in SphHarmIndices: invalid size of input arrays");
    return math::getIndicesFromCoefs(D);
}

static math::SphHarmIndices getIndicesFromCoefs(const std::vector< std::vector<double> > &C)
{
    const std::vector< std::vector<double> >* A = &C;
    return getIndicesFromCoefs<1>(&A);
}
static math::SphHarmIndices getIndicesFromCoefs(
    const std::vector< std::vector<double> > &C1, const std::vector< std::vector<double> > &C2)
{
    const std::vector< std::vector<double> >* A[2] = {&C1, &C2};
    return getIndicesFromCoefs<2>(A);
}
    
// ------- Spherical-harmonic expansion of density or potential ------- //
// The routine `computeSphHarmCoefs` can work with both density and potential classes,
// computes the sph-harm expansion for either density (in the first case),
// or potential and its r-derivative (in the second case).
// To avoid code duplication, the function that actually retrieves the relevant quantity
// is separated into a dedicated routine `storeValue`, which stores either one or two
// values for each input point. The `computeSphHarmCoefsSph` routine is templated on both
// the type of input data and the number of quantities stored for each point.

template<class BaseDensityOrPotential>
void storeValue(const BaseDensityOrPotential& src,
    const coord::PosSph& pos, double values[], int arraySize);

template<>
inline void storeValue(const BaseDensity& src,
    const coord::PosSph& pos, double values[], int) {
    *values = src.density(pos);
}

template<>
inline void storeValue(const BasePotential& src,
    const coord::PosSph& pos, double values[], int arraySize) {
    coord::GradSph grad;
    src.eval(pos, values, &grad);
    values[arraySize] = grad.dr;
}

template<class BaseDensityOrPotential, int NQuantities>
void computeSphHarmCoefs(const BaseDensityOrPotential& src, 
    const math::SphHarmIndices& ind, const std::vector<double>& radii,
    std::vector< std::vector<double> > * coefs[])
{
    unsigned int numPointsRadius = radii.size();
    if(numPointsRadius<1)
        throw std::invalid_argument("computeSphHarmCoefs: radial grid size too small");
    //  initialize sph-harm transform
    math::SphHarmTransformForward trans(ind);

    // 1st step: collect the values of input quantities at a 2d grid in (r,theta);
    // loop over radii and angular directions, using a combined index variable for better load balancing.
    int numSamplesAngles = trans.size();  // size of array of density values at each r
    int numSamplesTotal  = numSamplesAngles * numPointsRadius;
    std::vector<double> values(numSamplesTotal * NQuantities);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int n=0; n<numSamplesTotal; n++) {
        int indR     = n / numSamplesAngles;  // index in radial grid
        int indA     = n % numSamplesAngles;  // combined index in angular direction (theta,phi)
        double rad   = radii[indR];
        double theta = trans.theta(indA);
        double phi   = trans.phi(indA);
        storeValue(src, coord::PosSph(rad, theta, phi),
            &values[indR * numSamplesAngles + indA], numSamplesTotal);
    }

    // 2nd step: transform these values to spherical-harmonic expansion coefficients at each radius
    for(int q=0; q<NQuantities; q++) {
        coefs[q]->resize(numPointsRadius);
        for(unsigned int indR=0; indR<numPointsRadius; indR++) {
            coefs[q]->at(indR).assign(ind.size(), 0);
            trans.transform(&values[indR * numSamplesAngles + q * numSamplesTotal],
                &coefs[q]->at(indR).front());
            math::eliminateNearZeros(coefs[q]->at(indR));
        }
    }
}

}  // internal namespace

// driver functions that call the above templated transformation routine
void computeDensityCoefsSph(const BaseDensity& src, 
    const math::SphHarmIndices& ind, const std::vector<double>& gridRadii,
    std::vector< std::vector<double> > &output)
{
    std::vector< std::vector<double> > *coefs = &output;
    computeSphHarmCoefs<BaseDensity, 1>(src, ind, gridRadii, &coefs);
}

void computePotentialCoefsSph(const BasePotential& src,
    const math::SphHarmIndices& ind, const std::vector<double> &gridRadii,
    std::vector< std::vector<double> > &Phi, std::vector< std::vector<double> > &dPhi)
{
    std::vector< std::vector<double> > *coefs[2] = {&Phi, &dPhi};
    computeSphHarmCoefs<BasePotential, 2>(src, ind, gridRadii, coefs);    
}


// Core function to solve Poisson equation in spherical harmonics for a smooth density profile
void computePotentialCoefsSph(const BaseDensity& dens, 
    const math::SphHarmIndices& ind, const std::vector<double>& gridRadii,
    std::vector< std::vector<double> >& Phi, std::vector< std::vector<double> >& dPhi)
{
    int gridSizeR = gridRadii.size();
    if(gridSizeR<2)
        throw std::invalid_argument("computePotentialCoefs: radial grid size too small");
    for(int k=0; k<gridSizeR; k++)
        if(gridRadii[k] <= (k==0 ? 0 : gridRadii[k-1]))
            throw std::invalid_argument("computePotentialCoefs: "
                "radii of grid points must be positive and sorted in increasing order");

    // several intermediate arrays are aliased with the output arrays,
    // but are denoted by different names to clearly mark their identity
    std::vector< std::vector<double> >& Qint = Phi;
    std::vector< std::vector<double> >& Qext = dPhi;
    std::vector< std::vector<double> >& Pint = Phi;
    std::vector< std::vector<double> >& Pext = dPhi;
    Phi .resize(gridSizeR);
    dPhi.resize(gridSizeR);
    for(int k=0; k<gridSizeR; k++) {
        Phi [k].assign(ind.size(), 0);
        dPhi[k].assign(ind.size(), 0);
    }

    // prepare tables for (non-adaptive) integration over radius
    const int Nsub = 15;  // number of sub-steps in each radial bin
    std::vector<double> glx(Nsub), glw(Nsub);  // Gauss-Legendre nodes and weights
    math::prepareIntegrationTableGL(0, 1, Nsub, &glx.front(), &glw.front());

    // prepare SH transformation
    math::SphHarmTransformForward trans(ind);

    // Loop over radial grid segments and compute integrals of rho_lm(r) times powers of radius,
    // for each interval of radii in the input grid (0 <= k < Nr):
    //   Qint[k][l,m] = \int_{r_{k-1}}^{r_k} \rho_{l,m}(r) (r/r_k)^{l+2} dr,  with r_{-1} = 0;
    //   Qext[k][l,m] = \int_{r_k}^{r_{k+1}} \rho_{l,m}(r) (r/r_k)^{1-l} dr,  with r_{Nr} = \infty.
    // Here \rho_{l,m}(r) are the sph.-harm. coefs for density at each radius.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int k=0; k<=gridSizeR; k++) {
        std::vector<double> densValues(trans.size());
        std::vector<double> tmpCoefs(ind.size());
        double rkminus1 = (k>0 ? gridRadii[k-1] : 0);
        double deltaGridR = k<gridSizeR ?
            gridRadii[k] - rkminus1 :  // length of k-th radial segment
            gridRadii.back();          // last grid segment extends to infinity

        // loop over Nsub nodes of GL quadrature for each radial grid segment
        for(int s=0; s<Nsub; s++) {
            double r = k<gridSizeR ?
                rkminus1 + glx[s] * deltaGridR :  // radius inside ordinary k-th segment
                // special treatment for the last segment which extends to infinity:
                // the integration variable is t = r_{Nr-1} / r
                gridRadii.back() / glx[s];

            // collect the values of density at all points of angular grid at the given radius
            for(unsigned int i=0; i<densValues.size(); i++)
                densValues[i] = dens.density(coord::PosSph(r, trans.theta(i), trans.phi(i)));

            // compute density SH coefs
            trans.transform(&densValues.front(), &tmpCoefs.front());
            math::eliminateNearZeros(tmpCoefs);

            // accumulate integrals over density times radius in the Qint and Qext arrays
            for(int m=ind.mmin(); m<=ind.mmax; m++) {
                for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
                    unsigned int c = ind.index(l, m);
                    if(k<gridSizeR)
                        // accumulate Qint for all segments except the one extending to infinity
                        Qint[k][c] += tmpCoefs[c] * glw[s] * deltaGridR *
                            math::powInt(r / gridRadii[k], l+2);
                    if(k>0)
                        // accumulate Qext for all segments except the innermost one
                        // (which starts from zero), with a special treatment for last segment
                        // that extends to infinity and has a different integration variable
                        Qext[k-1][c] += glw[s] * tmpCoefs[c] * deltaGridR *
                            (k==gridSizeR ? 1 / pow_2(glx[s]) : 1) * // jacobian of 1/r transform
                            math::powInt(r / gridRadii[k-1], 1-l);
                }
            }
        }
    }

    // Run the summation loop, replacing the intermediate values Qint, Qext
    // with the interior and exterior potential coefficients (stored in the same arrays):
    //   Pint_{l,m}(r) = r^{-l-1} \int_0^r \rho_{l,m}(s) s^{l+2} ds ,
    //   Pext_{l,m}(r) = r^l \int_r^\infty \rho_{l,m}(s) s^{1-l} ds ,
    // In doing so, we use a recurrent relation that avoids over/underflows when
    // dealing with large powers of r, by replacing r^n with (r/r_prev)^n.
    // Finally, compute the total potential and its radial derivative for each SH term.
    for(int m=ind.mmin(); m<=ind.mmax; m++) {
        for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m);

            // Compute Pint by summing from inside out, using the recurrent relation
            // Pint(r_{k+1}) r_{k+1}^{l+1} = Pint(r_k) r_k^{l+1} + Qint[k] * r_k^{l+2}
            double val = 0;
            for(int k=0; k<gridSizeR; k++) {
                if(k>0)
                    val *= math::powInt(gridRadii[k-1] / gridRadii[k], l+1);
                val += gridRadii[k] * Qint[k][c];
                Pint[k][c] = val;
            }

            // Compute Pext by summing from outside in, using the recurrent relation
            // Pext(r_k) r_k^{-l} = Pext(r_{k+1}) r_{k+1}^{-l} + Qext[k] * r_k^{1-l}
            val = 0;
            for(int k=gridSizeR-1; k>=0; k--) {
                if(k<gridSizeR-1)
                    val *= math::powInt(gridRadii[k] / gridRadii[k+1], l);
                val += gridRadii[k] * Qext[k][c];
                Pext[k][c] = val;
            }

            // Finally, put together the interior and exterior coefs to compute 
            // the potential and its radial derivative for each spherical-harmonic term
            double mul = -4*M_PI / (2*l+1);
            for(int k=0; k<gridSizeR; k++) {
                double tmpPhi = mul * (Pint[k][c] + Pext[k][c]);
                dPhi[k][c]    = mul * (-(l+1)*Pint[k][c] + l*Pext[k][c]) / gridRadii[k];
                // extra step needed because Phi/dPhi and Pint/Pext are aliased
                Phi[k][c]     = tmpPhi;
            }
        }
    }

    // polishing: zero out coefs with small magnitude at each radius
    for(int k=0; k<gridSizeR; k++) {
        math::eliminateNearZeros(Phi[k]);
        math::eliminateNearZeros(dPhi[k]);
    }
}

namespace {  // internal routines

/** helper function to compute the second derivative of a function f(x) at x=x1,
    given the values f and first derivatives df of this function at three points x0,x1,x2.
*/
static double der2f(double f0, double f1, double f2,
    double df0, double df1, double df2, double x0, double x1, double x2)
{
    // construct a divided difference table to evaluate 2nd derivative via Hermite interpolation
    double dx10 = x1-x0, dx21 = x2-x1, dx20 = x2-x0;
    double df10 = (f1   - f0  ) / dx10;
    double df21 = (f2   - f1  ) / dx21;
    double dd10 = (df10 - df0 ) / dx10;
    double dd11 = (df1  - df10) / dx10;
    double dd21 = (df21 - df1 ) / dx21;
    double dd22 = (df2  - df21) / dx21;
    return ( -2 * (pow_2(dx21)*(dd10-2*dd11) + pow_2(dx10)*(dd22-2*dd21)) +
        4*dx10*dx21 * (dx10*dd21 + dx21*dd11) / dx20 ) / pow_2(dx20);
}

/** helper function to determine the coefficients for potential extrapolation:
    assuming that 
        Phi(r) = W * (r/r1)^v + U * (r/r1)^s              if s!=v, or
        Phi(r) = W * (r/r1)^v + U * (r/r1)^s * ln(r/r1)   if s==v,
    and given v and the values of Phi and its radial derivatives
    at three points r0<r1<r2, determine the coefficients s, U and W.
    Here v = l for the inward and v = -l-1 for the outward extrapolation.
    This corresponds to the density profile extrapolated as rho ~ r^(s-2).
*/
static void computeExtrapolationCoefs(double Phi0, double Phi1, double Phi2,
    double dPhi0, double dPhi1, double dPhi2, double r0, double r1, double r2,
    int v, double& s, double& U, double& W)
{
    // the routine below operates on a function Phi(log(r)), thus we transform dPhi/dr -> dPhi/d(ln r)
    dPhi0 *= r0; dPhi1 *= r1; dPhi2 *= r2;
    double d2Phi1 = der2f(Phi0, Phi1, Phi2, dPhi0, dPhi1, dPhi2, log(r0), log(r1), log(r2));
    s = (d2Phi1 - v*dPhi1) / (dPhi1 - v*Phi1);
    // safeguard against weird slope determination
    if(v>=0 && (!math::isFinite(s) || s<=-1))
        s = 2;  // results in a constant-density core for the inward extrapolation
    if(v<0  && (!math::isFinite(s) || s>=0))
        s = -2; // results in a r^-4 falloff for the outward extrapolation
    if(s != v) {
        U = (dPhi1 - v*Phi1) / (s-v);
        W = (dPhi1 - s*Phi1) / (v-s);
    } else {
        U = dPhi1 - v*Phi1;
        W = Phi1;
    }
}

static PtrPotential initAsympt(const double radii[3],
    const std::vector<double> Phi[3],
    const std::vector<double> dPhi[3], bool inner)
{
    unsigned int nc = Phi[0].size();
    // limit the number of terms to consider
    const unsigned int lmax = 8;
    nc = std::min<unsigned int>(nc, pow_2(lmax+1));
    std::vector<double> S(nc), U(nc), W(nc);

    // determine the coefficients for potential extrapolation at small and large radii
    for(unsigned int c=0; c<nc; c++) {
        int l = math::SphHarmIndices::index_l(c);
        computeExtrapolationCoefs(
            Phi [0][c], Phi [1][c], Phi [2][c],
            dPhi[0][c], dPhi[1][c], dPhi[2][c],
            radii[0],   radii[1],   radii[2],  inner ? l : -l-1,
            /*output*/ S[c], U[c], W[c]);
    }
    return PtrPotential(new PowerLawMultipole(radii[1], inner, S, U, W));
}


/// transform SH coefs to the Fourier components in the meridional plane
static void sphHarmTransformPolar(const double theta,
    const math::SphHarmIndices& ind,
    const double C_lm[], const double dC_lm[], const double d2C_lm[],
    double* C_m, coord::GradSph* dC_m, coord::HessSph* d2C_m)
{
    double P_lm[MAX_NCOEFS_ANGULAR];
    double dP_lm_arr [MAX_NCOEFS_ANGULAR];
    double d2P_lm_arr[MAX_NCOEFS_ANGULAR];
    double* dP_lm = dC_m!=NULL || d2C_m!=NULL ? dP_lm_arr : NULL;
    double* d2P_lm = d2C_m!=NULL ? d2P_lm_arr : NULL;
    for(int m=ind.mmin(); m<=ind.mmax; m++) {
        int lmin = ind.lmin(m);
        if(lmin > ind.lmax)
            continue;
        int absm = math::abs(m);
        unsigned int mm = m+ind.mmax;
        double mul = m==0 ? 2*M_SQRTPI : 2*M_SQRTPI*M_SQRT2;  // extra factor sqrt{2} for m!=0 trig fncs
        if(C_m)
            C_m[mm] = 0;
        if(dC_m)
            dC_m[mm].dr = dC_m[mm].dtheta = 0;
        if(d2C_m)
            d2C_m[mm].dr2 = d2C_m[mm].dtheta2 = d2C_m[mm].drdtheta = 0;
        math::sphHarmArray(ind.lmax, absm, theta, P_lm, dP_lm, d2P_lm);
        for(int l=lmin; l<=ind.lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m), p = l-absm;
            if(C_m)
                C_m[mm] += P_lm[p] * C_lm[c] * mul;
            if(dC_m) {
                dC_m[mm].dr     +=  P_lm[p] * dC_lm[c] * mul;
                dC_m[mm].dtheta += dP_lm[p] *  C_lm[c] * mul;
            }
            if(d2C_m) {
                d2C_m[mm].dr2     +=   P_lm[p] * d2C_lm[c] * mul;
                d2C_m[mm].dtheta2 += d2P_lm[p] *   C_lm[c] * mul;
                d2C_m[mm].drdtheta+=  dP_lm[p] *  dC_lm[c] * mul;
            }
        }
    }
}

/// transform Fourier components C_m(r, theta) and their derivs to the actual potential
static void fourierTransformAzimuth(const double phi,
    const math::SphHarmIndices& ind,
    const double C_m[], const coord::GradSph dC_m[], const coord::HessSph d2C_m[],
    double* val, coord::GradSph* grad, coord::HessSph* hess)
{
    const bool useSine = ind.mmin()<0 || grad!=NULL || hess!=NULL;
    double trig_m[2*MAX_NCOEFS_ANGULAR];
    if(ind.mmax>0)
        math::trigMultiAngle(phi, ind.mmax, useSine, trig_m);
    if(val)
        *val = 0;
    if(grad)
        grad->dr = grad->dtheta = grad->dphi = 0;
    if(hess)
        hess->dr2 = hess->dtheta2 = hess->dphi2 =
        hess->drdtheta = hess->drdphi = hess->dthetadphi = 0;
    for(int m=ind.mmin(); m<=ind.mmax; m++) {
        if(ind.lmin(m)>ind.lmax)
            continue;  // empty harmonic
        unsigned int mm = m+ind.mmax;  // index of element in C_m array
        double trig  = m==0 ? 1. : m>0 ? trig_m[m-1] : trig_m[ind.mmax-m-1];  // cos or sin
        double dtrig = m==0 ? 0. : m>0 ? -m*trig_m[ind.mmax+m-1] : -m*trig_m[-m-1];
        double d2trig = -m*m*trig;
        if(val)
            *val += C_m[mm] * trig;
        if(grad) {
            grad->dr     += dC_m[mm].dr     *  trig;
            grad->dtheta += dC_m[mm].dtheta *  trig;
            grad->dphi   +=  C_m[mm]        * dtrig;
        }
        if(hess) {
            hess->dr2       += d2C_m[mm].dr2      *   trig;
            hess->dtheta2   += d2C_m[mm].dtheta2  *   trig;
            hess->drdtheta  += d2C_m[mm].drdtheta *   trig;
            hess->drdphi    +=  dC_m[mm].dr       *  dtrig;
            hess->dthetadphi+=  dC_m[mm].dtheta   *  dtrig;
            hess->dphi2     +=   C_m[mm]          * d2trig;
        }
    }
}

// transform potential derivatives from ln(r) to r
static inline void transformRadialDerivs(double r, coord::GradSph* grad, coord::HessSph* hess)
{
    if(hess) {
        hess->dr2 = (hess->dr2 - grad->dr) / pow_2(r);
        hess->drdtheta /= r;
        hess->drdphi   /= r;
    }
    if(grad)
        grad->dr /= r;
}

}  // end internal namespace

//------ Spherical-harmonic expansion of density ------//

PtrDensity DensitySphericalHarmonic::create(
    const BaseDensity& dens, int lmax, int mmax, 
    unsigned int gridSizeR, double rmin, double rmax)
{
    if(gridSizeR<2 || rmin<0 || rmax<=rmin)
        throw std::invalid_argument("DensitySphericalHarmonic: invalid choice of min/max grid radii");
    // to improve accuracy of SH coefficient computation, we may increase the order of expansion
    // that determines the number of integration points in angles
    int lmax_tmp =     isSpherical(dens) ? 0 : std::max<int>(lmax, LMIN_SPHHARM);
    int mmax_tmp = isZRotSymmetric(dens) ? 0 : std::max<int>(mmax, LMIN_SPHHARM);
    std::vector<std::vector<double> > coefs;
    std::vector<double> gridRadii = math::createExpGrid(gridSizeR, rmin, rmax);
    computeDensityCoefsSph(dens,
        math::SphHarmIndices(lmax_tmp, mmax_tmp, dens.symmetry()),
        gridRadii, coefs);
    // now resize the coefficients back to the requested order
    for(unsigned int k=0; k<gridSizeR; k++)
        coefs[k].resize(pow_2(lmax+1), 0);
    return PtrDensity(new DensitySphericalHarmonic(gridRadii, coefs));
}

DensitySphericalHarmonic::DensitySphericalHarmonic(const std::vector<double> &gridRadii,
    const std::vector< std::vector<double> > &coefs) :
    BaseDensity(), ind(getIndicesFromCoefs(coefs))
{
    unsigned int gridSizeR = gridRadii.size();
    if(gridSizeR < 2 || gridSizeR != coefs.size())
        throw std::invalid_argument("DensitySphericalHarmonic: input arrays are empty");
    for(unsigned int n=0; n<gridSizeR; n++)
        if(coefs[n].size() != ind.size())
            throw std::invalid_argument("DensitySphericalHarmonic: incorrect size of coefficients array");

    // We check (and correct if necessary) the logarithmic slopes of multipole components
    // at the innermost and outermost grid radii, to ensure correctly behaving extrapolation.
    // slope = (1/rho) d(rho)/d(logr), is usually negative (at least at large radii);
    // put constraints on min inner and max outer slopes:
    const double minDerivInner=-2.8, maxDerivOuter=-2.2;
    const double maxDerivInner=20.0, minDerivOuter=-20.;
    // Note that the inner slope less than -2 leads to a divergent potential at origin,
    // but the enclosed mass is still finite if slope is greater than -3;
    // similarly, outer slope greater than -3 leads to a divergent total mass,
    // but the potential tends to a finite limit as long as the slope is less than -2.
    // Both these 'dangerous' semi-infinite regimes are allowed here, but likely may result
    // in problems elsewhere.
    // The l>0 components must not have steeper/shallower slopes than the l=0 component.
    innerSlope.assign(ind.size(), minDerivInner);
    outerSlope.assign(ind.size(), maxDerivOuter);
    spl.resize(ind.size());
    std::vector<double> tmparr(gridSizeR);

    // set up 1d splines in radius for each non-trivial (l,m) coefficient
    for(int m=ind.mmin(); m<=ind.mmax; m++) {
        if(ind.lmin(m) > ind.lmax)
            continue;
        for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m);
            //  determine asymptotic slopes of density profile at large and small r
            double derivInner = log(coefs[1][c] / coefs[0][c]) / log(gridRadii[1] / gridRadii[0]);
            double derivOuter = log(coefs[gridSizeR-1][c] / coefs[gridSizeR-2][c]) /
                log(gridRadii[gridSizeR-1] / gridRadii[gridSizeR-2]);
            if( derivInner > maxDerivInner)
                derivInner = maxDerivInner;
            if( derivOuter < minDerivOuter)
                derivOuter = minDerivOuter;
            if(coefs.front()[c] == 0)
                derivInner = 0;
            if(coefs.back() [c] == 0)
                derivOuter = 0;
            if(!math::isFinite(derivInner) || derivInner < innerSlope[0])
                derivInner = innerSlope[0];  // works even for l==0 since we have set it up in advance
            if(!math::isFinite(derivOuter) || derivOuter > outerSlope[0])
                derivOuter = outerSlope[0];
            innerSlope[c] = derivInner;
            outerSlope[c] = derivOuter;
                for(unsigned int k=0; k<gridSizeR; k++)
                    tmparr[k] = coefs[k][c];
            spl[c] = math::CubicSpline(gridRadii, tmparr, 
                derivInner / gridRadii.front() * coefs.front()[c],
                derivOuter / gridRadii.back()  * coefs.back() [c]);
        }
    }
}

void DensitySphericalHarmonic::getCoefs(
    std::vector<double> &radii, std::vector< std::vector<double> > &coefs) const
{
    radii = spl[0].xvalues();
    computeDensityCoefsSph(*this, ind, radii, coefs);
}

double DensitySphericalHarmonic::densitySph(const coord::PosSph &pos) const
{
    double coefs[MAX_NCOEFS_ANGULAR*MAX_NCOEFS_ANGULAR];
    double tmparr[3*MAX_NCOEFS_ANGULAR];
    double rmin=spl[0].xmin(), rmax=spl[0].xmax();
    for(int m=ind.mmin(); m<=ind.mmax; m++)
        for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m);
            if(pos.r<rmin || pos.r>rmax) {
                double r0  = pos.r<rmin ? rmin : rmax;
                double val = spl[c](r0);
                coefs[c] = val==0 ? 0 : val * pow(pos.r/r0, pos.r<rmin ? innerSlope[c] : outerSlope[c]);
            } else
                coefs[c] = spl[c](pos.r);
        }
    return math::sphHarmTransformInverse(ind, coefs, pos.theta, pos.phi, tmparr);
}


//----- declarations of two multipole potential interpolators -----//

class MultipoleInterp1d: public BasePotentialSph {
public:
    /** construct interpolating splines from the values and derivatives of harmonic coefficients */
    MultipoleInterp1d(
        const std::vector<double> &radii,
        const std::vector<std::vector<double> > &Phi,
        const std::vector<std::vector<double> > &dPhi);
    virtual coord::SymmetryType symmetry() const { return ind.symmetry(); }
    virtual const char* name() const { return "MultipoleInterp1d"; };
private:
    /// indexing scheme for sph.-harm. coefficients
    math::SphHarmIndices ind;
    /// interpolation splines in log(r) for each {l,m} sph.-harm. component of potential
    std::vector<math::QuinticSpline> spl;
    
    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const;
};

class MultipoleInterp2d: public BasePotentialSph {
public:
    /** construct interpolating splines from the values and derivatives of harmonic coefficients */
    MultipoleInterp2d(
        const std::vector<double> &radii,
        const std::vector<std::vector<double> > &Phi,
        const std::vector<std::vector<double> > &dPhi);
    virtual coord::SymmetryType symmetry() const { return ind.symmetry(); }
    virtual const char* name() const { return "MultipoleInterp2d"; }
private:
    /// indexing scheme for sph.-harm. coefficients
    math::SphHarmIndices ind;
    /// 2d interpolation splines in meridional plane for each azimuthal harmonic (m) component
    std::vector<math::QuinticSpline2d> spl;

    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const;
};

//------ the wrapper class for multipole potential ------//

template<class BaseDensityOrPotential>
static PtrPotential createMultipole(
    const BaseDensityOrPotential& src,
    int lmax, int mmax,
    unsigned int gridSizeR, double rmin, double rmax)
{
    if(gridSizeR<=2 || rmin<=0 || rmax<=rmin)
        throw std::invalid_argument("Error in Multipole: invalid grid parameters");
    std::vector<double> gridRadii = math::createExpGrid(gridSizeR, rmin, rmax);
    // to improve accuracy of SH coefficient computation, we may increase the order of expansion
    // that determines the number of integration points in angles
    int lmax_tmp =     isSpherical(src) ? 0 : std::max<int>(lmax, LMIN_SPHHARM);
    int mmax_tmp = isZRotSymmetric(src) ? 0 : std::max<int>(mmax, LMIN_SPHHARM);
    std::vector<std::vector<double> > Phi, dPhi;
    computePotentialCoefsSph(src,
        math::SphHarmIndices(lmax_tmp, mmax_tmp, src.symmetry()),
        gridRadii, Phi, dPhi);
    // now resize the coefficients back to the requested order
    for(unsigned int k=0; k<gridSizeR; k++) {
        Phi [k].resize(pow_2(lmax+1), 0);
        dPhi[k].resize(pow_2(lmax+1), 0);
    }
    return PtrPotential(new Multipole(gridRadii, Phi, dPhi));
}

PtrPotential Multipole::create(
    const BaseDensity& src, int lmax, int mmax,
    unsigned int gridSizeR, double rmin, double rmax)
{ return createMultipole(src, lmax, mmax, gridSizeR, rmin, rmax); }

PtrPotential Multipole::create(
    const BasePotential& src, int lmax, int mmax,
    unsigned int gridSizeR, double rmin, double rmax)
{ return createMultipole(src, lmax, mmax, gridSizeR, rmin, rmax); }

Multipole::Multipole(
    const std::vector<double> &_gridRadii,
    const std::vector<std::vector<double> > &Phi,
    const std::vector<std::vector<double> > &dPhi) :
    gridRadii(_gridRadii), ind(getIndicesFromCoefs(Phi, dPhi))
{
    unsigned int gridSizeR = gridRadii.size();
    bool correct = gridSizeR > 2 && gridSizeR == Phi.size() && gridSizeR == dPhi.size();
    for(unsigned int k=1; correct && k<gridSizeR; k++)
        correct &= gridRadii[k] > gridRadii[k-1];
    if(!correct)
        throw std::invalid_argument("Error in Multipole: invalid radial grid");

    // construct the interpolating splines
    impl = ind.lmax<=2 ?   // choose between 1d or 2d splines, depending on the expected efficiency
        PtrPotential(new MultipoleInterp1d(gridRadii, Phi, dPhi)) :
        PtrPotential(new MultipoleInterp2d(gridRadii, Phi, dPhi));

    // determine asymptotic behaviour at small and large radii
    asymptInner = initAsympt(&gridRadii[0], &Phi[0], &dPhi[0], true);
    asymptOuter = initAsympt(&gridRadii[gridSizeR-3], &Phi[gridSizeR-3], &dPhi[gridSizeR-3], false);
}

void Multipole::getCoefs(
    std::vector<double> &radii,
    std::vector<std::vector<double> > &Phi,
    std::vector<std::vector<double> > &dPhi) const
{
    radii = gridRadii;
    // use the fact that the spherical-harmonic transform is invertible to machine precision:
    // take the values and derivatives of potential at grid nodes and apply forward transform
    // to obtain the coefficients.
    computePotentialCoefsSph(*impl, ind, radii, Phi, dPhi);
}

void Multipole::evalSph(const coord::PosSph &pos,
    double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const
{
    if(pos.r < gridRadii[1])
        asymptInner->eval(pos, potential, deriv, deriv2);
    else if(pos.r > gridRadii[gridRadii.size()-3])
        asymptOuter->eval(pos, potential, deriv, deriv2);
    else
        impl->eval(pos, potential, deriv, deriv2);
}

// ------- Implementations of multipole potential interpolators ------- //

// TODO: redesign this common block without the use of preprocessor.
// declare temporary arrays for storing coefficients and use only the needed ones
#define EVALSPH_HEADER \
    double   Phi_lm_arr[MAX_NCOEFS_ANGULAR*MAX_NCOEFS_ANGULAR]; \
    double  dPhi_lm_arr[MAX_NCOEFS_ANGULAR*MAX_NCOEFS_ANGULAR]; \
    double d2Phi_lm_arr[MAX_NCOEFS_ANGULAR*MAX_NCOEFS_ANGULAR]; \
    double           Phi_m_arr[2*MAX_NCOEFS_ANGULAR-1]; \
    coord::GradSph  dPhi_m_arr[2*MAX_NCOEFS_ANGULAR-1]; \
    coord::HessSph d2Phi_m_arr[2*MAX_NCOEFS_ANGULAR-1]; \
    bool needPhi  = true; \
    bool needGrad = grad!=NULL || hess!=NULL; \
    bool needHess = hess!=NULL; \
    double *  Phi_lm = needPhi  ?   Phi_lm_arr : NULL; \
    double * dPhi_lm = needGrad ?  dPhi_lm_arr : NULL; \
    double *d2Phi_lm = needHess ? d2Phi_lm_arr : NULL; \
    double *  Phi_m  = needPhi  ?   Phi_m_arr  : NULL; \
    coord::GradSph *dPhi_m  = needGrad ? dPhi_m_arr  : NULL; \
    coord::HessSph *d2Phi_m = needHess ? d2Phi_m_arr : NULL;

// ------- PowerLawPotential ------- //

PowerLawMultipole::PowerLawMultipole(double _r0, bool _inner,
    const std::vector<double>& _S,
    const std::vector<double>& _U,
    const std::vector<double>& _W) :
    ind(math::getIndicesFromCoefs(_U)), r0(_r0), inner(_inner), S(_S), U(_U), W(_W) 
{
    // safeguard against errors in slope determination - 
    // ensure that all harmonics with l>0 do not asymptotically overtake the principal one (l=0)
    for(unsigned int c=1; c<S.size(); c++)
        if(U[c]!=0 && ((inner && S[c] < S[0]) || (!inner && S[c] > S[0])) )
            S[c] = S[0];
}
    
void PowerLawMultipole::evalSph(const coord::PosSph &pos,
    double* potential, coord::GradSph* grad, coord::HessSph* hess) const
{
    EVALSPH_HEADER
    // define {v=l, r0=rmin} for the inner or {v=-l-1, r0=rmax} for the outer extrapolation;
    // Phi_{l,m}(r) = U_{l,m} * (r/r0)^{s_{l,m}}            + W_{l,m} * (r/r0)^v   if s!=v,
    // Phi_{l,m}(r) = U_{l,m} * (r/r0)^{s_{l,m}} * ln(r/r0) + W_{l,m} * (r/r0)^v   if s==v.
    double dlogr = log(pos.r / r0);
    for(int m=ind.mmin(); m<=ind.mmax; m++)
        for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m);
            double s=S[c], u=U[c], w=W[c], v = inner ? l : -l-1;
            double rv  = v!=0 ? exp( dlogr * v ) : 1;                // (r/r0)^v
            double rs  = s!=v ? (s!=0 ? exp( dlogr * s ) : 1) : rv;  // (r/r0)^s
            double urs = u * rs * (s!=v || u==0 ? 1 : dlogr);  // if s==v, multiply by ln(r/r0)
            double wrv = w * rv;
            if(needPhi)
                Phi_lm[c] = urs + wrv;
            if(needGrad)
                dPhi_lm[c] = urs*s + wrv*v + (s!=v ? 0 : u*rs);
            if(needHess)
                d2Phi_lm[c] = urs*s*s + wrv*v*v + (s!=v ? 0 : 2*s*u*rs);
        }
    sphHarmTransformPolar(pos.theta, ind, Phi_lm, dPhi_lm, d2Phi_lm, Phi_m, dPhi_m, d2Phi_m);
    fourierTransformAzimuth(pos.phi, ind, Phi_m, dPhi_m, d2Phi_m, potential, grad, hess);
    transformRadialDerivs(pos.r, grad, hess);
}

// ------- Multipole potential with 1d interpolating splines for each SH harmonic ------- //

MultipoleInterp1d::MultipoleInterp1d(
    const std::vector<double> &radii,
    const std::vector< std::vector<double> > &Phi,
    const std::vector< std::vector<double> > &dPhi) :
    ind(getIndicesFromCoefs(Phi, dPhi))
{
    unsigned int gridSizeR = radii.size();
    assert(gridSizeR >= 2 && gridSizeR == Phi.size() && gridSizeR == dPhi.size() &&
        Phi[0].size() == ind.size() && ind.lmax >= 0 && ind.mmax <= ind.lmax);
    
    // set up a logarithmic radial grid
    std::vector<double> gridR(gridSizeR);
    for(unsigned int k=0; k<gridSizeR; k++)
        gridR[k] = log(radii[k]);
    std::vector<double> Phi_lm(gridSizeR), dPhi_lm(gridSizeR);  // temp.arrays

    // set up 1d quintic splines in radius for each non-trivial (l,m) coefficient
    spl.resize(ind.size());
    for(int m=ind.mmin(); m<=ind.mmax; m++)
        for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m);
            for(unsigned int k=0; k<gridSizeR; k++) {
                Phi_lm[k] = Phi[k][c];
                dPhi_lm[k]=dPhi[k][c] * radii[k];
            }
            spl[c] = math::QuinticSpline(gridR, Phi_lm, dPhi_lm);
        }
}
    
void MultipoleInterp1d::evalSph(const coord::PosSph &pos,
    double* potential, coord::GradSph* grad, coord::HessSph* hess) const
{
    EVALSPH_HEADER
    // compute spherical-harmonic coefs
    double logr = log(pos.r);
    for(int m=ind.mmin(); m<=ind.mmax; m++)
        for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m);
            spl[c].evalDeriv(logr,
                needPhi ?   &Phi_lm[c] : NULL,
                needGrad?  &dPhi_lm[c] : NULL,
                needHess? &d2Phi_lm[c] : NULL);
        }
    sphHarmTransformPolar(pos.theta, ind, Phi_lm, dPhi_lm, d2Phi_lm, Phi_m, dPhi_m, d2Phi_m);
    fourierTransformAzimuth(pos.phi, ind, Phi_m, dPhi_m, d2Phi_m, potential, grad, hess);
    transformRadialDerivs(pos.r, grad, hess);
}

// ------- Multipole potential with 2d interpolating splines for each azimuthal harmonic ------- //

/** Set up non-uniform grid in cos(theta), with denser spacing close to z-axis.
    We want (some of) the nodes of the grid to coincide with the nodes of Gauss-Legendre
    quadrature on the interval -1 <= cos(theta) <= 1, which ensures that the values
    of 2d spline at these angles exactly equals the input values, thereby making
    the forward and reverse Legendre transformation invertible to machine precision.
    So we first compute these nodes for the given order of sph.-harm. expansion lmax,
    and then take only the non-negative half of them for the spline in cos(theta),
    plus one at theta=0.
    To achieve better accuracy in approximating the Legendre polynomials by quintic
    splines, we insert additional nodes in between the original ones.
*/
static std::vector<double> createGridInTheta(unsigned int lmax)
{
    unsigned int numPointsGL = lmax+1;
    std::vector<double> theta(numPointsGL+2), dummy(numPointsGL);
    math::prepareIntegrationTableGL(-1, 1, numPointsGL, &theta[1], &dummy.front());
    // convert GL nodes (cos theta) to theta
    for(unsigned int iGL=1; iGL<=numPointsGL; iGL++)
        theta[iGL] = acos(theta[iGL]);
    // add points at the ends of original interval (GL nodes are all interior)
    theta.back() = 0.;
    theta.front()= M_PI;
    // split each interval between two successive GL nodes (or the ends of original interval)
    // into this many grid points (accuracy of Legendre function approximation is better than 1e-6)
    unsigned int oversampleFactor = 3;
    // number of grid points for spline in 0 <= theta) <= pi
    unsigned int gridSizeT = (numPointsGL+1) * oversampleFactor + 1;
    std::vector<double> gridT(gridSizeT);
    for(unsigned int iGL=0; iGL<=numPointsGL; iGL++)
        for(unsigned int iover=0; iover<oversampleFactor; iover++) {
            gridT[gridT.size() - 1 - (iGL * oversampleFactor + iover)] =
                (theta[iGL] * (oversampleFactor-iover) + theta[iGL+1] * iover) / oversampleFactor;
        }
    return gridT;
}

MultipoleInterp2d::MultipoleInterp2d(
    const std::vector<double> &radii,
    const std::vector< std::vector<double> > &Phi,
    const std::vector< std::vector<double> > &dPhi) :
    ind(getIndicesFromCoefs(Phi, dPhi))
{
    unsigned int gridSizeR = radii.size();
    assert(gridSizeR > 2 && gridSizeR == Phi.size() && gridSizeR == dPhi.size() &&
        Phi[0].size() == ind.size() && ind.lmax >= 0 && ind.mmax <= ind.lmax);
    
    // set up a 2D grid in ln(r) & theta:
    std::vector<double> gridR(gridSizeR);
    for(unsigned int k=0; k<gridSizeR; k++)
        gridR[k] = log(radii[k]);
    std::vector<double> gridT = createGridInTheta(ind.lmax);
    unsigned int gridSizeT = gridT.size();

    // allocate temporary arrays for initialization of 2d splines
    math::Matrix<double> Phi_val(gridSizeR, gridSizeT);
    math::Matrix<double> Phi_dR (gridSizeR, gridSizeT);
    math::Matrix<double> Phi_dT (gridSizeR, gridSizeT);
    std::vector<double>  Plm(ind.lmax+1), dPlm(ind.lmax+1);

    // loop over azimuthal harmonic indices (m)
    spl.resize(2*ind.mmax+1);
    for(int m=ind.mmin(); m<=ind.mmax; m++) {
        int lmin = ind.lmin(m);
        if(lmin > ind.lmax)
            continue;
        int absm = math::abs(m);
        double mul = m==0 ? 2*M_SQRTPI : 2*M_SQRTPI*M_SQRT2;
        // assign Phi_m, dPhi_m/d(ln r) & dPhi_m/d(cos(theta)) at each node of 2d grid (r_k, theta_j)
        for(unsigned int j=0; j<gridSizeT; j++) {
            math::sphHarmArray(ind.lmax, absm, gridT[j], &Plm.front(), &dPlm.front());
            for(unsigned int k=0; k<gridSizeR; k++) {            
                double val=0, dR=0, dT=0;
                for(int l=lmin; l<=ind.lmax; l+=ind.step) {
                    unsigned int c = ind.index(l, m);
                    val += Phi [k][c] *  Plm[l-absm];   // Phi_{l,m}(r)
                    dR  += dPhi[k][c] *  Plm[l-absm];   // d Phi / d r
                    dT  += Phi [k][c] * dPlm[l-absm];   // d Phi / d theta
                }
                Phi_val(k, j) = val * mul;
                Phi_dR (k, j) = dR  * mul * radii[k];  // transform to d Phi / d ln(r)
                Phi_dT (k, j) = dT  * mul;
            }
        }
        // establish 2D quintic spline for Phi_m(ln(r), theta)
        spl[m+ind.mmax] = math::QuinticSpline2d(gridR, gridT, Phi_val, Phi_dR, Phi_dT);
    }
}

void MultipoleInterp2d::evalSph(const coord::PosSph &pos,
    double* potential, coord::GradSph* grad, coord::HessSph* hess) const
{
    // temporary arrays for storing coefficients
    double Phi_m          [2*MAX_NCOEFS_ANGULAR-1];
    coord::GradSph dPhi_m [2*MAX_NCOEFS_ANGULAR-1];
    coord::HessSph d2Phi_m[2*MAX_NCOEFS_ANGULAR-1];
    
    // only compute those quantities that will be needed in output
    bool needPhi  = true;
    bool needGrad = grad!=NULL || hess!=NULL;
    bool needHess = hess!=NULL;
    
    // compute azimuthal harmonics
    for(int m=ind.mmin(); m<=ind.mmax; m++) {
        if(ind.lmin(m) > ind.lmax)
            continue;
        unsigned int mm = m+ind.mmax;
        spl[mm].evalDeriv(log(pos.r), pos.theta, 
            needPhi  ?   &Phi_m[mm]          : NULL, 
            needGrad ?  &dPhi_m[mm].dr       : NULL,
            needGrad ?  &dPhi_m[mm].dtheta   : NULL,
            needHess ? &d2Phi_m[mm].dr2      : NULL,
            needHess ? &d2Phi_m[mm].drdtheta : NULL,
            needHess ? &d2Phi_m[mm].dtheta2  : NULL);
    }
    fourierTransformAzimuth(pos.phi, ind, Phi_m, dPhi_m, d2Phi_m, potential, grad, hess);
    transformRadialDerivs(pos.r, grad, hess);
}

}; // namespace
