/*
Copyright Walter Dehnen, 1996-2004 
e-mail:   walter.dehnen@astro.le.ac.uk 
address:  Department of Physics and Astronomy, University of Leicester 
          University Road, Leicester LE1 7RH, United Kingdom 

------------------------------------------------------------------------
Version 0.0    15. July      1997 
Version 0.1    24. March     1998 
Version 0.2    22. September 1998 
Version 0.3    07. June      2001 
Version 0.4    22. April     2002 
Version 0.5    05. December  2002 
Version 0.6    05. February  2003 
Version 0.7    23. September 2004
Version 0.8    24. June      2005

----------------------------------------------------------------------
Modifications by Eugene Vasiliev, 2015 
(so extensive that almost nothing of the original code remains)

*/
#include "potential_galpot.h"
#include "potential_composite.h"
#include "potential_sphharm.h"
#include "math_core.h"
#include "math_specfunc.h"
#include <cmath>
#include <stdexcept>
#include <cassert>

#ifdef VERBOSE_REPORT
#include <iostream>
#endif

namespace potential{

static const int MAX_NCOEFS_ANGULAR=201;///< 1 + maximum l for the Multipole expansion 
static const int    GALPOT_LMAX=16;     ///< DEFAULT order (lmax) for the Multipole expansion 
static const int    GALPOT_NRAD=201;    ///< DEFAULT number of radial points in Multipole 
static const double GALPOT_RMIN=1.e-4,  ///< DEFAULT min radius of logarithmic radial grid in Multipole
                    GALPOT_RMAX=1.e4;   ///< DEFAULT max radius of logarithmic radial grid

//----- disk density and potential -----//

/** simple exponential radial density profile without inner hole or wiggles */
class DiskDensityRadialExp: public math::IFunction {
public:
    DiskDensityRadialExp(double _surfaceDensity, double _scaleRadius): 
        surfaceDensity(_surfaceDensity), scaleRadius(_scaleRadius) {};
private:
    const double surfaceDensity, scaleRadius;
    /**  evaluate  f(R) and optionally its two derivatives, if these arguments are not NULL  */
    virtual void evalDeriv(double R, double* f=NULL, double* fprime=NULL, double* fpprime=NULL) const {
        double val = surfaceDensity * exp(-R/scaleRadius);
        if(f)
            *f = val;
        if(fprime)
            *fprime = -val/scaleRadius;
        if(fpprime)
            *fpprime = val/pow_2(scaleRadius);
    }
    virtual unsigned int numDerivs() const { return 2; }
};

/** more convoluted radial density profile - exponential with possible inner hole and modulation */
class DiskDensityRadialRichExp: public math::IFunction {
public:
    DiskDensityRadialRichExp(const DiskParam& _params): params(_params) {};
private:
    const DiskParam params;
    /**  evaluate  f(R) and optionally its two derivatives, if these arguments are not NULL  */
    virtual void evalDeriv(double R, double* f=NULL, double* fprime=NULL, double* fpprime=NULL) const {
        if(params.innerCutoffRadius && R==0.) {
            if(f) *f=0;
            if(fprime)  *fprime=0;
            if(fpprime) *fpprime=0;
            return;
        }
        const double Rrel = R/params.scaleRadius;
        const double cr = params.modulationAmplitude ? params.modulationAmplitude*cos(Rrel) : 0;
        const double sr = params.modulationAmplitude ? params.modulationAmplitude*sin(Rrel) : 0;
        if(R==0 && params.innerCutoffRadius==0)
            R = 1e-100;  // avoid 0/0 indeterminacy
        double val = params.surfaceDensity * exp(-params.innerCutoffRadius/R - Rrel + cr);
        double fp  = params.innerCutoffRadius/(R*R) - (1+sr)/params.scaleRadius;
        if(fpprime)
            *fpprime = val ? (fp*fp - 2*params.innerCutoffRadius/(R*R*R)
                - cr/pow_2(params.scaleRadius)) * val : 0;  // if val==0, the bracket could be NaN
        if(fprime)
            *fprime  = val ? fp*val : 0;
        if(f) 
            *f = val;
    }
    virtual unsigned int numDerivs() const { return 2; }
};

/** exponential vertical disk density profile */
class DiskDensityVerticalExp: public math::IFunction {
public:
    DiskDensityVerticalExp(double _scaleHeight): scaleHeight(_scaleHeight) {};
private:
    const double scaleHeight;
    /**  evaluate  H(z) and optionally its two derivatives, if these arguments are not NULL  */
    virtual void evalDeriv(double z, double* H=NULL, double* Hprime=NULL, double* Hpprime=NULL) const {
        double      x        = fabs(z/scaleHeight);
        double      h        = exp(-x);
        if(H)       *H       = 0.5 * scaleHeight * (h-1+x);
        if(Hprime)  *Hprime  = 0.5 * math::sign(z) * (1.-h);
        if(Hpprime) *Hpprime = h / (2*scaleHeight);
    }
    virtual unsigned int numDerivs() const { return 2; }
};

/** isothermal (sech^2) vertical disk density profile */
class DiskDensityVerticalIsothermal: public math::IFunction {
public:
    DiskDensityVerticalIsothermal(double _scaleHeight): scaleHeight(_scaleHeight) {};
private:
    const double scaleHeight;
    /**  evaluate  H(z) and optionally its two derivatives, if these arguments are not NULL  */
    virtual void evalDeriv(double z, double* H=NULL, double* Hprime=NULL, double* Hpprime=NULL) const {
        double      x        = fabs(z/scaleHeight);
        double      h        = exp(-x);
        double      sh1      = 1 + h;
        if(H)       *H       = scaleHeight * (0.5*x + log(0.5*sh1));
        if(Hprime)  *Hprime  = 0.5 * math::sign(z) * (1.-h) / sh1;
        if(Hpprime) *Hpprime = h / (sh1*sh1*scaleHeight);
    }
    virtual unsigned int numDerivs() const { return 2; }
};

/** vertically thin disk profile */
class DiskDensityVerticalThin: public math::IFunction {
public:
    DiskDensityVerticalThin() {};
private:
    /**  evaluate  H(z) and optionally its two derivatives, if these arguments are not NULL  */
    virtual void evalDeriv(double z, double* H=NULL, double* Hprime=NULL, double* Hpprime=NULL) const {
        if(H)       *H       = fabs(z)/2;
        if(Hprime)  *Hprime  = math::sign(z)/2;
        if(Hpprime) *Hpprime = 0;
    }
    virtual unsigned int numDerivs() const { return 2; }
};

/** helper routine to create an instance of radial density function */
math::PtrFunction createRadialDiskFnc(const DiskParam& params) {
    if(params.scaleRadius<=0)
        throw std::invalid_argument("Disk scale radius cannot be <=0");
    if(params.innerCutoffRadius<0)
        throw std::invalid_argument("Disk inner cutoff radius cannot be <0");
    if(params.innerCutoffRadius==0 && params.modulationAmplitude==0)
        return math::PtrFunction(new DiskDensityRadialExp(params.surfaceDensity, params.scaleRadius));
    else
        return math::PtrFunction(new DiskDensityRadialRichExp(params));
}

/** helper routine to create an instance of vertical density function */
math::PtrFunction createVerticalDiskFnc(const DiskParam& params) {
    if(params.scaleHeight>0)
        return math::PtrFunction(new DiskDensityVerticalExp(params.scaleHeight));
    if(params.scaleHeight<0)
        return math::PtrFunction(new DiskDensityVerticalIsothermal(-params.scaleHeight));
    else
        return math::PtrFunction(new DiskDensityVerticalThin());
}

double DiskDensity::densityCyl(const coord::PosCyl &pos) const
{
    double h;
    verticalFnc->evalDeriv(pos.z, NULL, NULL, &h);
    return radialFnc->value(pos.R) * h;
}

double DiskAnsatz::densityCyl(const coord::PosCyl &pos) const
{
    double h, H, Hp, f, fp, fpp, r=hypot(pos.R, pos.z);
    verticalFnc->evalDeriv(pos.z, &H, &Hp, &h);
    radialFnc  ->evalDeriv(r, &f, &fp, &fpp);
    return f*h + (pos.z!=0 ? 2*fp*(H+pos.z*Hp)/r : 0) + fpp*H;
}

void DiskAnsatz::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const
{
    double r=hypot(pos.R, pos.z);
    double h, H, Hp, f, fp, fpp;
    bool deriv1 = deriv!=NULL || deriv2!=NULL;  // compute 1st derivative of f and H only if necessary
    verticalFnc->evalDeriv(pos.z, &H, deriv1? &Hp : NULL, deriv2? &h : NULL);
    radialFnc  ->evalDeriv(r,     &f, deriv1? &fp : NULL, deriv2? &fpp : NULL);
    f*=4*M_PI; fp*=4*M_PI; fpp*=4*M_PI;
    double Rr=pos.R/r, zr=pos.z/r;
    if(r==0) { Rr=0; zr=0; r=1e-100; }
    if(potential) {
        *potential = f*H;
    }
    if(deriv) {
        deriv->dR = H*fp*Rr;
        deriv->dz = H*fp*zr + Hp*f;
        deriv->dphi=0;
    }
    if(deriv2) {
        deriv2->dR2 = H*(fpp*Rr*Rr + fp*zr*zr/r);
        deriv2->dz2 = H*(fpp*zr*zr + fp*Rr*Rr/r) + 2*fp*Hp*zr + f*h;
        deriv2->dRdz= H*Rr*zr*(fpp - fp/r) + fp*Hp*Rr;
        deriv2->dRdphi=deriv2->dzdphi=deriv2->dphi2=0;
    }
}

//----- spheroid density -----//
SpheroidDensity::SpheroidDensity (const SphrParam &_params) :
    BaseDensity(), params(_params)
{
    if(params.scaleRadius<=0)
        throw std::invalid_argument("Spheroid scale radius cannot be <=0");
    if(params.axisRatio<=0)
        throw std::invalid_argument("Spheroid axis ratio cannot be <=0");
    if(params.outerCutoffRadius<0)
        throw std::invalid_argument("Spheroid outer cutoff radius cannot be <0");
    if(params.gamma>=3)
        throw std::invalid_argument("Spheroid inner slope gamma must be less than 3");
    if(params.beta<=2 && params.outerCutoffRadius==0)
        throw std::invalid_argument("Spheroid outer slope beta must be greater than 2, "
            "or a positive cutoff radius must be provided");
};

double SpheroidDensity::densityCyl(const coord::PosCyl &pos) const
{
    double m   = hypot(pos.R, pos.z/params.axisRatio);
    double m0  = m/params.scaleRadius;
    double rho = params.densityNorm;
    if(params.gamma==0.5)rho /= sqrt(m0); else
    if(params.gamma==1.) rho /= m0;       else 
    if(params.gamma==2.) rho /= m0*m0;    else
    if(params.gamma!=0.) rho /= pow(m0, params.gamma);
    m0 += 1;
    const double beg = params.beta-params.gamma;
    if(beg==1.) rho /= m0;       else
    if(beg==2.) rho /= m0*m0;    else
    if(beg==3.) rho /= m0*m0*m0; else
    rho /= pow(m0,beg);
    if(params.outerCutoffRadius)
        rho *= exp(-pow_2(m/params.outerCutoffRadius));
    return rho;
}

//----- multipole potential -----//

Multipole::Multipole(const BaseDensity& sourceDensity,
    double rmin, double rmax, unsigned int gridSizeR, unsigned int numCoefsAngular)
{
    if(gridSizeR<=2 || numCoefsAngular<0 || rmin<=0 || rmax<=rmin)
        throw std::invalid_argument("Error in Multipole: invalid grid parameters");
    if(!isAxisymmetric(sourceDensity))
        throw std::invalid_argument("Error in Multipole: source density must be axisymmetric");
    numCoefsAngular = std::min<unsigned int>(numCoefsAngular, MAX_NCOEFS_ANGULAR-1);
    // number of multipoles (only even-l terms are used)
    const unsigned int numMultipoles = numCoefsAngular/2+1;

    // set up radial grid
    std::vector<double> r = math::createExpGrid(gridSizeR, rmin, rmax);
    
    // check if the input density is of a spherical-harmonic type already...
    bool useExistingSphHarm = sourceDensity.name() == DensitySphericalHarmonic::myName();
    // ...and construct a fresh spherical-harmonic expansion of density if it wasn't such.
    PtrDensity mySphHarm(useExistingSphHarm ? NULL :
        new DensitySphericalHarmonic(gridSizeR, numCoefsAngular, sourceDensity, rmin, rmax));
    // for convenience, this reference points to either the original or internally created SH density
    const DensitySphericalHarmonic& densh = dynamic_cast<const DensitySphericalHarmonic&>(
        useExistingSphHarm ? sourceDensity : *mySphHarm);

    // accumulate the 'inner' (Pint) and 'outer' (Pext) parts of the potential at radial grid points:
    // Pint_l(r) = r^{-l-1} \int_0^r \rho_l(s) s^{l+2} ds,
    // Pext_l(r) = r^l \int_r^\infty \rho_l(s) s^{1-l} ds.
    // For each l, we compute Pint(r) by looping over sub-intervals from inside out,
    // and Pext(r) - from outside in. In doing so, we use the recurrent relation
    // Pint(r_{i+1}) r_{i+1}^{l+1} = r_i^{l+1} Pint(r_i) + \int_{r_i}^{r_{i+1}} \rho_l(s) s^{l+2} ds,
    // and a similar relation for Pext. This is further rewritten so that the integration over 
    // density  \rho_l(s)  uses  (s/r_{i+1})^{l+2}  as the radius-dependent weight function -
    // this avoids over/underflows when both l and r are large or small.
    // Finally, \Phi_l(r) is computed as -4\pi ( Pint_l(r) + Pext_l(r) ), and similarly its derivative.
    std::vector<double> Pint(gridSizeR), Pext(gridSizeR);
    std::vector<std::vector<double> > Phil(gridSizeR), dPhil(gridSizeR);
    for(unsigned int k=0; k<gridSizeR; k++) {
        Phil[k].assign(numMultipoles, 0);
        dPhil[k].assign(numMultipoles, 0);
    }
    for(unsigned int il=0; il<numMultipoles; il++) {
        int l = il*2;
        // inner part
        Pint[0] = densh.integrate(0, r[0], l, l+2, r[0]) * r[0];
        if(!math::isFinite(Pint[0]))
            throw std::runtime_error("Error in Multipole: mass is divergent at origin");
        for(unsigned int k=1; k<gridSizeR; k++) {
            double s = densh.integrate(r[k-1], r[k], l, l+2, r[k]);
            Pint[k] = Pint[k-1] * math::powInt(r[k-1]/r[k], l+1) + s * r[k];
        }

        // outer part
        Pext[gridSizeR-1] = densh.integrate(r[gridSizeR-1], INFINITY,
            l, 1-l, r[gridSizeR-1]) * r[gridSizeR-1];
        if(!math::isFinite(Pext[gridSizeR-1]))
            throw std::runtime_error("Error in Multipole: potential is divergent at infinity");
        for(unsigned int k=gridSizeR-1; k>0; k--) {
            double s = densh.integrate(r[k-1], r[k], l, 1-l, r[k-1]);
            Pext[k-1] = Pext[k] * math::powInt(r[k-1]/r[k], l) + s * r[k-1];
        }

        // put together inner and outer parts to compute the potential and its radial derivative,
        // for each spherical-harmonic term
        for(unsigned int k=0; k<gridSizeR; k++) {
            Phil [k][il] = -4*M_PI * (Pint[k] + Pext[k]);           // Phi_l
            dPhil[k][il] =  4*M_PI * ( (l+1)*Pint[k] - l*Pext[k]);  // dPhi_l/dlogr
            if(!math::isFinite(Phil[k][l/2]))
                throw std::runtime_error("Error in Multipole: bad value of potential");
        }
    }
    // finish initialization by constructing the 2d spline and determining extrapolation coefs
    initSpline(r, Phil, dPhil);
}

Multipole::Multipole(const std::vector<double> &radii,
    const std::vector<std::vector<double> > &Phi,
    const std::vector<std::vector<double> > &dPhi)
{
    unsigned int gridSizeR = radii.size();
    bool correct = gridSizeR > 2 && gridSizeR == Phi.size() && gridSizeR == dPhi.size();
    unsigned int numCoefs = 0;
    for(unsigned int k=0; correct && k<radii.size(); k++) {
        if(k==0)
            numCoefs = Phi[0].size();
        else
            correct &= radii[k] > radii[k-1];
        correct &= Phi[k].size() == numCoefs && dPhi[k].size() == numCoefs;
    }
    if(!correct)
        throw std::invalid_argument("Error in Multipole: invalid size of input arrays");
    initSpline(radii, Phi, dPhi);
}

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
    and given v and the values of Phi and its derivatives w.r.t. ln(r)
    at three points r0<r1<r2, determine the coefficients s, U and W.
    Here v = l for the inward and v = -l-1 for the outward extrapolation.
    This corresponds to the density profile extrapolated as rho ~ r^(s-2).
*/
static void computeExtrapolationCoefs(double Phi0, double Phi1, double Phi2,
    double dPhi0, double dPhi1, double dPhi2, double lnr0, double lnr1, double lnr2,
    int v, double& s, double& U, double& W)
{
    double d2Phi1 = der2f(Phi0, Phi1, Phi2, dPhi0, dPhi1, dPhi2, lnr0, lnr1, lnr2);
    s = (d2Phi1 - v*dPhi1) / (dPhi1 - v*Phi1);
    int signv = v>=0 ? 1 : -1;
    if(!math::isFinite(s) || s * signv <= -v-1)  // safeguard against weird slope determination
        // results in a constant-density core for the inward or r^-4 falloff for the outward extrapolation
        s = 2 * signv;
    if(s != v) {
        U = (dPhi1 - v*Phi1) / (s-v);
        W = (dPhi1 - s*Phi1) / (v-s);
    } else {
        U = dPhi1 - v*Phi1;
        W = Phi1;
    }
}

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
static std::vector<double> createGridInCosTheta(unsigned int lmax)
{
    unsigned int numPointsGL = lmax+1;  // lmax is even, numPointsGL is odd
    std::vector<double> theta(numPointsGL+1), dummy(numPointsGL);
    math::prepareIntegrationTableGL(-1, 1, numPointsGL, &theta.front(), &dummy.front());
    // convert GL nodes (cos theta) to theta (only the upper half of the original interval)
    for(unsigned int iGL=numPointsGL/2; iGL<numPointsGL; iGL++)
        theta[iGL] = acos(theta[iGL]);
    theta.back() = 0.;  // add point at the end of original interval (GL nodes are all interior)
    // use this number of grid points for each original GL node (accuracy better than 1e-6)
    unsigned int oversampleFactor = 3;
    // number of grid points for spline in 0 <= cos(theta) <= 1
    unsigned int gridSizeC = (lmax/2+1) * oversampleFactor + 1;
    std::vector<double> gridC(gridSizeC);
    for(unsigned int iGL = numPointsGL/2; iGL<numPointsGL; iGL++)  // covers lmax/2+1 GL nodes
        for(unsigned int iover=0; iover<oversampleFactor; iover++) {
            gridC[(iGL-numPointsGL/2) * oversampleFactor + iover] = cos(
                (theta[iGL] * (oversampleFactor-iover) + theta[iGL+1] * iover) / oversampleFactor);
        }
    gridC.front() = 0.;
    gridC.back()  = 1.;
    return gridC;
}

void Multipole::initSpline(const std::vector<double> &radii,
    const std::vector<std::vector<double> > &Phil, const std::vector<std::vector<double> > &dPhil)
{
    unsigned int gridSizeR = radii.size();
    assert(gridSizeR > 2 && gridSizeR == Phil.size() &&
        gridSizeR == dPhil.size() && Phil[0].size() >= 1);

    // determine the indices of grid radii at which the extrapolation values are computed
    unsigned int imin = 1;
    unsigned int imax = gridSizeR-2;
    unsigned int numMultipoles = Phil[0].size();
    unsigned int lmax = 2*(numMultipoles-1);

    // set up a 2D grid in log[r] & cos[theta]:
    std::vector<double> gridR(gridSizeR);
    for(unsigned int i=0; i<gridSizeR; i++)
        gridR[i] = log(radii[i]);
    logrmin = gridR[imin];
    logrmax = gridR[imax];
    std::vector<double> gridC = createGridInCosTheta(lmax);
    unsigned int gridSizeC = gridC.size();

    innerSlope.assign(numMultipoles, 0);
    innerCoefU.assign(numMultipoles, 0);
    innerCoefW.assign(numMultipoles, 0);
    outerSlope.assign(numMultipoles, 0);
    outerCoefU.assign(numMultipoles, 0);
    outerCoefW.assign(numMultipoles, 0);
    isSpherical = true;

    for(unsigned int il=0; il<numMultipoles; il++) {
        int l = il*2;
        if(l>0)  // check if non-spherical components are identically zero
            for(unsigned int k=0; k<gridSizeR; k++)
                isSpherical &= Phil[k][il]==0 && dPhil[k][il]==0;

        // determine the coefficients for potential extrapolation at small and large radii
        computeExtrapolationCoefs(
            Phil [imin-1][il],  Phil[imin][il],  Phil[imin+1][il],
            dPhil[imin-1][il], dPhil[imin][il], dPhil[imin+1][il],
            gridR[imin-1],     gridR[imin],     gridR[imin+1],   l,
            /*output*/ innerSlope[il], innerCoefU[il], innerCoefW[il]);
        computeExtrapolationCoefs(
            Phil [imax-1][il],  Phil[imax][il],  Phil[imax+1][il],
            dPhil[imax-1][il], dPhil[imax][il], dPhil[imax+1][il],
            gridR[imax-1],     gridR[imax],     gridR[imax+1], -l-1,
            /*output*/ outerSlope[il], outerCoefU[il], outerCoefW[il]);
#ifdef VERBOSE_REPORT
        std::cout << "Multipole: for l=" << l << 
            ": inner density slope=" << (innerSlope[il]-2) <<
            ", outer density slope=" << (outerSlope[il]-2) << "\n";
#endif
    }
    
    // assign Phi, dPhi/dlogr & dPhi/dcos[theta] 
    math::Matrix<double> Phi_val(gridSizeR, gridSizeC);
    math::Matrix<double> Phi_dR (gridSizeR, gridSizeC);
    math::Matrix<double> Phi_dC (gridSizeR, gridSizeC);
    std::vector<double>  Pl(lmax+1), dPl(lmax+1);
    for(unsigned int i=0; i<gridSizeC; i++) {
        math::legendrePolyArray(lmax, 0, gridC[i], &Pl.front(), &dPl.front());
        for(unsigned int k=0; k<gridSizeR; k++) {
            double val=0, dR=0, dC=0;
            for(unsigned int il=0; il<numMultipoles; il++) {
                val += Phil [k][il] *  Pl[il*2];   // Phi
                dR  += dPhil[k][il] *  Pl[il*2];   // d Phi / d log(r)
                dC  += Phil [k][il] * dPl[il*2];   // d Phi / d cos(theta)
            }
            Phi_val(k, i) = val;
            Phi_dR (k, i) = dR;
            Phi_dC (k, i) = dC;
        }
    }

    // establish 2D quintic spline of Phi in log[r] & cos[theta]
    spl = math::QuinticSpline2d(gridR, gridC, Phi_val, Phi_dR, Phi_dC);
}

void Multipole::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const
{
    double r=hypot(pos.R, pos.z);
    double ct=pos.z/r, st=pos.R/r;  // cos(theta), sin(theta)
    if(r==0 || r==INFINITY) {ct=st=0;}
    double logr  = log(r);
    double absct = fabs(ct);
    // value and derivatives of spline by its arguments (ln r, z/r)
    double Phi=0, PhiR=0, PhiC=0, PhiRR=0, PhiRC=0, PhiCC=0;

    if(logr < logrmin || logr > logrmax) {  // extrapolation at small or large radii
        // use statically-sized arrays for Legendre polynomials, to avoid dynamic memory allocation
        double Pl[MAX_NCOEFS_ANGULAR], dPl[MAX_NCOEFS_ANGULAR];
        math::legendrePolyArray(innerSlope.size()*2-2, 0, ct, Pl, dPl);

        // define {v=l, r0=rmin} for the inner or {v=-l-1, r0=rmax} for the outer extrapolation;
        // Phi_l(r) = U_l * (r/r0)^s            + W_l * (r/r0)^v   if s!=v,
        // Phi_l(r) = U_l * (r/r0)^s * ln(r/r0) + W_l * (r/r0)^v   if s==v.
        for(unsigned int l2=0; l2<innerSlope.size(); l2++) {
            int l  = l2*2;
            double s, v, U, W, dr;
            if(logr < logrmin) {
                s = innerSlope[l2];
                v = l;
                U = innerCoefU[l2];
                W = innerCoefW[l2];
                dr= logr - logrmin;
            } else {
                s = outerSlope[l2];
                v = -l-1;
                U = outerCoefU[l2];
                W = outerCoefW[l2];
                dr= logr - logrmax;
            }
            double rv  = v!=0 ? exp( dr * v ) : 1;                // (r/r0)^v
            double rs  = s!=v ? (s!=0 ? exp( dr * s ) : 1) : rv;  // (r/r0)^s
            double Urs = U * rs * (s!=v || U==0 ? 1 : dr);  // if s==v, multiply by ln(r/r0)
            double Wrv = W * rv;
            Phi  +=  Pl[l] * (Urs + Wrv);  // Phi
            PhiC += dPl[l] * (Urs + Wrv);  // d Phi / d cos(theta)
            double der = Urs*s + Wrv*v + (s!=v ? 0 : U*rs);
            PhiR +=  Pl[l] * der;          // d Phi / d ln(r)
            if(deriv2) {
                PhiRR +=  Pl[l] * (Urs*s*s + Wrv*v*v + (s!=v ? 0 : 2*s*U*rs));
                PhiRC += dPl[l] * der;
                double d2Pl = absct<1 ? 
                    (2*ct*dPl[l] - l*(l+1)*Pl[l]) / (1-pow_2(ct)) :
                    (l-1)*l*(l+1)*(l+2) / 8.;  // limiting value for |cos(theta)|==1
                PhiCC += d2Pl * (Urs + Wrv);
            }
        }
    } else {  // normal interpolation within the radial range of the spline
        if(deriv2)
            spl.evalDeriv(logr, absct, &Phi, &PhiR, &PhiC, &PhiRR, &PhiRC, &PhiCC);
        else if(deriv)
            spl.evalDeriv(logr, absct, &Phi, &PhiR, &PhiC);
        else
            spl.evalDeriv(logr, absct, &Phi);
        PhiC  *= math::sign(ct);
        PhiRC *= math::sign(ct);
    }
    if(potential)
        *potential = Phi;
    if(r==0) r = 1e-100;  // safety measure to avoid 0/0
    if(deriv) {
        deriv->dR   = (PhiR - PhiC*ct) * st / r;
        deriv->dz   = (PhiR*ct + PhiC*st*st) / r;
        deriv->dphi = 0;
    }
    if(deriv2) {
        double z2=ct*ct, R2=st*st;
        deriv2->dR2 = (PhiRR*R2 + PhiCC*R2*z2 - PhiRC*2*R2*ct
            + PhiR*(z2-R2) + PhiC*(2*R2-z2)*ct) / (r*r);  // d2/dR2
        deriv2->dz2 = (PhiRR*z2 + PhiCC*R2*R2 + PhiRC*2*R2*ct
            + PhiR*(R2-z2) - PhiC*3*R2*ct) / (r*r);       // d2/dz2
        deriv2->dRdz= (PhiRR*ct*st - PhiCC*ct*st*R2 + PhiRC*st*(R2-z2)
            - PhiR*2*ct*st + PhiC*st*(2*z2-R2)) / (r*r);  // d2/dRdz
        deriv2->dRdphi = deriv2->dzdphi = deriv2->dphi2 = 0;
    }
}

void Multipole::getCoefs(std::vector<double> &radii,
    std::vector<std::vector<double> > &Phi,
    std::vector<std::vector<double> > &dPhi) const
{
    // we compute the values and radial derivatives of potential at values of cos(theta)
    // corresponding to nodes of Gauss-Legendre quadrature of order lmax+1 on the interval [-1:1]
    unsigned int lmax = innerSlope.size()*2-2;
    LegendreTransform transf(lmax);
    unsigned int numPointsGL = lmax+1;
    std::vector<double> values(numPointsGL), derivs(numPointsGL), output(numPointsGL);
    radii = spl.xvalues();  // get log-radii of spline grid nodes
    Phi. resize(radii.size());
    dPhi.resize(radii.size());
    for(unsigned int k=0; k<radii.size(); k++) {
        for(unsigned int i=numPointsGL/2; i<numPointsGL; i++) {
            double val, der;  // Phi and dPhi/d(ln r)
            spl.evalDeriv(radii[k], transf.x(i), &val, &der);
            values[i] = values[numPointsGL-1-i] = val;
            derivs[i] = derivs[numPointsGL-1-i] = der;
        }
        Phi[k].resize(lmax/2+1);
        transf.forward(&values.front(), &output.front());
        for(unsigned int l=0; l<=lmax; l+=2)
            Phi[k][l/2] = output[l] * (2*l+1)/2;
        dPhi[k].resize(lmax/2+1);
        transf.forward(&derivs.front(), &output.front());
        for(unsigned int l=0; l<=lmax; l+=2)
            dPhi[k][l/2] = output[l] * (2*l+1)/2;
        radii[k] = exp(radii[k]);
    }
}

//----- GalaxyPotential refactored into a Composite potential -----//
std::vector<PtrPotential> createGalaxyPotentialComponents(
    const std::vector<DiskParam>& diskParams, 
    const std::vector<SphrParam>& sphrParams)
{
    // keep track of length scales of all components
    double lengthMin=INFINITY, lengthMax=0;
    
    // assemble the set of density components for the multipole
    // (all spheroids and residual part of disks),
    // and the complementary set of potential components
    // (the flattened part of disks, eventually to be joined by the multipole itself)
    std::vector<PtrDensity> componentsDens;
    std::vector<PtrPotential> componentsPot;
    for(unsigned int i=0; i<diskParams.size(); i++) {
        // the two parts of disk profile: DiskAnsatz goes to the list of potentials...
        componentsPot.push_back(PtrPotential(new DiskAnsatz(diskParams[i])));
        // ...and gets subtracted from the entire DiskDensity for the list of density components
        componentsDens.push_back(PtrDensity(new DiskDensity(diskParams[i])));
        DiskParam negDisk(diskParams[i]);
        negDisk.surfaceDensity *= -1;  // subtract the density of DiskAnsatz
        componentsDens.push_back(PtrDensity(new DiskAnsatz(negDisk)));
        // keep track of characteristic lengths
        lengthMin = fmin(lengthMin, diskParams[i].scaleRadius);
        lengthMax = fmax(lengthMax, diskParams[i].scaleRadius);
        if(diskParams[i].innerCutoffRadius>0)
            lengthMin = fmin(lengthMin, diskParams[i].innerCutoffRadius);
    }
    for(unsigned int i=0; i<sphrParams.size(); i++) {
        componentsDens.push_back(PtrDensity(new SpheroidDensity(sphrParams[i])));
        lengthMin = fmin(lengthMin, sphrParams[i].scaleRadius);
        lengthMax = fmax(lengthMax, sphrParams[i].scaleRadius);
        if(sphrParams[i].outerCutoffRadius) 
            lengthMax = fmax(lengthMax, sphrParams[i].outerCutoffRadius);
    }
    if(componentsDens.size()==0)
        throw std::invalid_argument("Empty parameters in GalPot");
    // create an composite density object to be passed to Multipole;
    const CompositeDensity dens(componentsDens);

    // create multipole potential from this combined density
    double rmin = GALPOT_RMIN * lengthMin;
    double rmax = GALPOT_RMAX * lengthMax;
    componentsPot.push_back(PtrPotential(
        new Multipole(dens, rmin, rmax, GALPOT_NRAD, GALPOT_LMAX)));

    // the array of components to be passed to the constructor of CompositeCyl potential:
    // the multipole and the non-residual parts of disk potential
    return componentsPot;
}

PtrPotential createGalaxyPotential(
    const std::vector<DiskParam>& DiskParams,
    const std::vector<SphrParam>& SphrParams)
{
    return PtrPotential(new CompositeCyl(createGalaxyPotentialComponents(DiskParams, SphrParams)));
}

} // namespace
