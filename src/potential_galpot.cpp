/*

galpot.cc 

C++ code 

Copyright Walter Dehnen, 1996-2004 
e-mail:   walter.dehnen@astro.le.ac.uk 
address:  Department of Physics and Astronomy, University of Leicester 
          University Road, Leicester LE1 7RH, United Kingdom 

Modifications by Eugene Vasiliev, June 2015 

------------------------------------------------------------------------

source code for class GalaxyPotential and dependencies 

Version 0.0    15. July      1997 
Version 0.1    24. March     1998 
Version 0.2    22. September 1998 
Version 0.3    07. June      2001 
Version 0.4    22. April     2002 
Version 0.5    05. December  2002 
Version 0.6    05. February  2003 
Version 0.7    23. September 2004
Version 0.8    24. June      2005

----------------------------------------------------------------------*/
#include "potential_galpot.h"
#include "potential_composite.h"
#include "potential_sphharm.h"
#include "math_core.h"
#include "math_specfunc.h"
#include <cmath>
#include <stdexcept>

#ifdef VERBOSE_REPORT
#include <iostream>
#endif

namespace potential{

static const int MAX_NCOEFS_ANGULAR=201;///< 1 + maximum l for the Multipole expansion 
static const int    GALPOT_LMAX=64;     ///< DEFAULT order (lmax) for the Multipole expansion 
static const int    GALPOT_NRAD=201;    ///< DEFAULT number of radial points in Multipole 
static const double GALPOT_RMIN=1.e-4,  ///< DEFAULT min radius of logarithmic radial grid in Multipole
                    GALPOT_RMAX=1.e3;   ///< DEFAULT max radius of logarithmic radial grid

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

double DiskResidual::densityCyl(const coord::PosCyl &pos) const
{
    if(pos.z==0) return 0;
    double h, H, Hp, F, f, fp, fpp, r=hypot(pos.R, pos.z);
    verticalFnc->evalDeriv(pos.z, &H, &Hp, &h);
    radialFnc  ->evalDeriv(r, &f, &fp, &fpp);
    radialFnc  ->evalDeriv(pos.R, &F);
    return (F-f)*h - 2*fp*(H+pos.z*Hp)/r - fpp*H;
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
    if(params.gamma<0 || params.gamma>=3)
        throw std::invalid_argument("Spheroid inner slope gamma must be between 0 and 3");
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

Multipole::Multipole(const BaseDensity& source_density,
                     double Rmin, double Rmax,
                     int gridSizeR, int numCoefsAngular)
{
    if(gridSizeR<=2 || numCoefsAngular<0 || Rmin<=0 || Rmax<=Rmin)
        throw std::invalid_argument("Error in Multipole expansion: invalid grid parameters");
    if(!isAxisymmetric(source_density))
        throw std::invalid_argument("Error in Multipole expansion: source density must be axisymmetric");
    numCoefsAngular = std::min<int>(numCoefsAngular, MAX_NCOEFS_ANGULAR-1);
    const int numMultipoles = numCoefsAngular/2+1;   // number of multipoles (only even-l terms are used)
    const int gridSizeC     = 2*numCoefsAngular+1;   // number of grid points for theta in [0,pi/2]

    // set up radial grid
    std::vector<double> r = math::createExpGrid(gridSizeR, Rmin, Rmax);
    
    // check if the input density is of a spherical-harmonic type already...
    bool useExistingSphHarm = source_density.name() == DensitySphericalHarmonic::myName();
    // ...and construct a fresh spherical-harmonic expansion of density if it wasn't such.
    PtrDensity mySphHarm(useExistingSphHarm ? NULL :
        new DensitySphericalHarmonic(gridSizeR*2, numCoefsAngular, source_density, Rmin, Rmax));
    // for convenience, this reference points to either the original or internally created SH density
    const DensitySphericalHarmonic& densh = static_cast<const DensitySphericalHarmonic&>(
        useExistingSphHarm ? source_density : *mySphHarm);

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
    math::Matrix<double> Phil(gridSizeR, numMultipoles), dPhil(gridSizeR, numMultipoles);
    std::vector<double>  Pint(gridSizeR), Pext(gridSizeR);
    innerSlopes.assign(numMultipoles, 0);
    outerSlopes.assign(numMultipoles, 0);
    innerValues.assign(numMultipoles, 0);
    outerValues.assign(numMultipoles, 0);

    for(int l=0; l<=numCoefsAngular; l+=2) {
        // inner part
        Pint[0] = densh.integrate(0, Rmin, l, l+2, Rmin) * Rmin;
        for(int k=1; k<gridSizeR; k++) {
            double s = densh.integrate(r[k-1], r[k], l, l+2, r[k]);
            Pint[k] = Pint[k-1] * math::powInt(r[k-1]/r[k], l+1) + s * r[k];
        }

        // outer part
        Pext[gridSizeR-1] = densh.integrate(Rmax, INFINITY, l, 1-l, Rmax) * Rmax;
        if(!math::isFinite(Pext[gridSizeR-1]))
            Pext[gridSizeR-1] = 0;
        for(int k=gridSizeR-2; k>=0; k--) {
            double s = densh.integrate(r[k], r[k+1], l, 1-l, r[k]);
            Pext[k] = Pext[k+1] * math::powInt(r[k]/r[k+1], l) + s * r[k];
        }

        // put together inner and outer parts to compute the potential and its radial derivative,
        // for each spherical-harmonic term
        for(int k=0; k<gridSizeR; k++) {
            Phil (k, l/2) = -4*M_PI * (Pint[k] + Pext[k]);           // Phi_l
            dPhil(k, l/2) =  4*M_PI * ( (l+1)*Pint[k] - l*Pext[k]);  // dPhi_l/dlogr
#ifdef VERBOSE_REPORT
            //std::cout << l << '\t' << r[k] << '\t' << Phil(k,l/2) << '\t' <<
            //(4*M_PI*Pint[k]) << '\t' << (4*M_PI*Pext[k]) << '\n';
#endif
        }

        // store the values and derivatives of multipole terms
        // used for extrapolating to large and small radii (beyond the grid definition region)
        if(l==0) {
            // determine the central value of potential
            innerValues[0] = 4*M_PI * (densh.integrate(0, r[0], 0, 1) - Pint[0]);
            Phi0 = Phil(0, 0) - innerValues[0];   // innerValues[0] is Phi(rmin)-Phi(0)
            // if the central value is finite, we derive the value of inner density slope gamma
            // assuming that Phi(r) = Phi(0) + (Phi(rmin) - Phi(0)) * (r/rmin)^(2-gamma)
            if(math::isFinite(innerValues[0]) && innerValues[0]!=0)
                innerSlopes[0] = dPhil(0, 0) / innerValues[0];
            else {
                // otherwise we take the ratio between enclosed mass within rmin, given by Pint(rmin),
                // and the value of density at this radius, assuming that it varies as 
                // rho(r) = rho(rmin) * (r/rmin)^(-gamma)  for r<rmin
                innerSlopes[0] = densh.rho_l(Rmin, 0) * pow_2(Rmin) / Pint.front() - 1.;
                innerValues[0];
            }
            if(!math::isFinite(innerSlopes[0]) || innerSlopes[0] < -0.5)
                innerSlopes[0] = 0;  // rather arbitrary..

            // determine the outer slope of density profile rho ~ r^-beta
            double densRmax = densh.rho_l(Rmax, 0);
            double betaminustwo = densRmax * pow_2(Rmax) / Pext.back();
            if(!math::isFinite(betaminustwo) || fabs(Pext.back()) < fabs(densRmax) * pow_2(Rmax) * 0.01) {
                betaminustwo = 0;    // discard all remaining density outside Rmax,
                Pext.back() = 0;       // if the density profile seems to be too steeply falling
                densRmax = 0;
            } else if(betaminustwo<0.2 || fabs(betaminustwo-1.)<0.01)
                betaminustwo = 1.1;  // beta=3 would need a different treatment of the asymptotic law
            //!!! FIXME: leads to crazy results if density is negative at Rmax and falling off too slowly
            outerValueMain = -4*M_PI * (Pint.back() + Pext.back() * betaminustwo / (betaminustwo-1));
            outerValues[0] =  4*M_PI * Pext.back() / (betaminustwo-1);
            outerSlopes[0] = -betaminustwo;
#ifdef VERBOSE_REPORT
            std::cout << "Multipole: inner density profile "
                "rho=" << densh.rho_l(Rmin, 0) << "*r^" << (innerSlopes[0]-2) <<
                ", outer rho=" << densRmax << "*r^" << (-betaminustwo-2) << "\n";
#endif
        } else {
            innerValues[l/2] = Phil(0, l/2);
            if(Phil(0, l/2) != 0)
                innerSlopes[l/2] = dPhil(0, l/2) / Phil(0, l/2);
            outerValues[l/2] = Phil(gridSizeR-1, l/2);
            if(Phil(gridSizeR-1, l/2) != 0)
                outerSlopes[l/2] = fmin(-1, dPhil(gridSizeR-1, l/2) / Phil(gridSizeR-1, l/2));
        }
    }

    // Put potential and its derivatives on a 2D grid in log[r] & cos[theta]
    //
    // set linear grid in theta, i.e. non-uniform in cos(theta), with denser spacing close to z-axis
    //
    std::vector<double> gridR(gridSizeR), gridC(gridSizeC);
    for(int i=0; i<gridSizeR; i++)
        gridR[i] = log(r[i]);
    for(int i=0; i<gridSizeC-1; i++) 
        gridC[i] = sin(M_PI_2 * i / (gridSizeC-1));
    gridC[gridSizeC-1] = 1.0;
    //
    // set dPhi/dlogr & dPhi/dcos[theta] 
    //
    math::Matrix<double> Phi_val(gridSizeR, gridSizeC);
    math::Matrix<double> Phi_dR (gridSizeR, gridSizeC);
    math::Matrix<double> Phi_dC (gridSizeR, gridSizeC);
    std::vector<double>  Pl(numCoefsAngular+1), dPl(numCoefsAngular+1);
    for(int i=0; i<gridSizeC; i++) {
        math::legendrePolyArray(numCoefsAngular, 0, gridC[i], &Pl.front(), &dPl.front());
        for(int k=0; k<gridSizeR; k++) {
            double val=0, dR=0, dC=0;
            for(int l=0; l<numMultipoles; l+=2) {
                val += Phil (k, l/2) *  Pl[l];   // Phi
                dR  += dPhil(k, l/2) *  Pl[l];   // d Phi / d log(r)
                dC  += Phil (k, l/2) * dPl[l];   // d Phi / d cos(theta)
            }
            Phi_val(k, i) = val;
            Phi_dR (k, i) = dR;
            Phi_dC (k, i) = dC;
        }
    }

    //
    // establish 2D quintic spline of Phi in log[r] & cos[theta]
    //
    spl = math::QuinticSpline2d(gridR, gridC, Phi_val, Phi_dR, Phi_dC);

    // determine if the potential is spherically-symmetric
    // (could determine this explicitly by analyzing the angular dependence of Phi(r,theta),
    // but for now simply ask the source density model
    isSpherical = (source_density.symmetry() & ST_SPHERICAL) == ST_SPHERICAL;
}

void Multipole::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const
{
    double r=hypot(pos.R, pos.z);
    double ct=pos.z/r, st=pos.R/r;  // cos(theta), sin(theta)
    if(r==0 || r==INFINITY) {ct=st=0;}
    double logr = log(r);
    // coordinates on the scaled grid
    const double logRmin = spl.xmin(), logRmax = spl.xmax();
    double valR = fmin(logRmax, fmax(logRmin, logr));
    double valC = fabs(ct);
    // value and derivatives of spline by its arguments (log r, z/r)
    double Phi=0, PhiR=0, PhiC=0, PhiRR=0, PhiRC=0, PhiCC=0;

    if(logr > logRmax) {  // extrapolation at large radii
        double Pl[MAX_NCOEFS_ANGULAR], dPl[MAX_NCOEFS_ANGULAR];
        math::legendrePolyArray(outerValues.size()*2-2, 0, ct, Pl, dPl);
        // special treatment for l==0: we assume that the density falls off
        // as a power law in radius, and potential then behaves as 
        // V * (rmax/r) + \sum_{l=0}^{lmax} V_l * (r/rmax)^{\alpha_l}
        // here V_l and \alpha_l are 'outerValues' and 'outerSlopes'
        double V = outerValueMain * exp(logRmax-logr);
        Phi   += V;
        PhiR  -= V;
        PhiRR += V;
        for(unsigned int l2=0; l2<outerValues.size(); l2++) {
            int l = l2*2;
            double alpha = outerSlopes[l2];
            double rfact = outerValues[l2] * exp( (logr-logRmax) * alpha );
            Phi  += rfact * Pl[l];          // Phi
            PhiR += rfact * Pl[l] * alpha;  // d Phi / d log(r)
            PhiC += rfact * dPl[l];         // d Phi / d cos(theta)
            if(deriv2) {
                PhiRR += rfact * Pl[l] * pow_2(alpha);
                PhiRC += rfact * dPl[l] * alpha;
                double d2Pl = valC<1 ? 
                    (2*ct*dPl[l] - l*(l+1)*Pl[l]) / (1-pow_2(ct)) :
                    (l-1)*l*(l+1)*(l+2) / 8.;  // limiting value for |cos(theta)|==1
                PhiCC += rfact * d2Pl;
            }
        }
    } else {
        if(deriv2)
            spl.evalDeriv(valR, valC, &Phi, &PhiR, &PhiC, &PhiRR, &PhiRC, &PhiCC);
        else if(deriv)
            spl.evalDeriv(valR, valC, &Phi, &PhiR, &PhiC);
        else
            spl.evalDeriv(valR, valC, &Phi);
        PhiC  *= math::sign(ct);
        PhiRC *= math::sign(ct);
        if(logr < logRmin) {  // extrapolation at small radii -- NEEDS UPDATE!!!
            if(innerSlopes[0]>0) {
                double coef = exp(innerSlopes[0]*(logr-valR));
                Phi   = (Phi-Phi0)*coef;
                PhiR  = innerSlopes[0]*Phi;
                PhiRR = pow_2(innerSlopes[0])*Phi;
                Phi  += Phi0;
                PhiC *= coef;
                PhiCC*= coef;
            } else if(innerSlopes[0]==0) {
                PhiR  = Phi/logRmin;
                PhiRR = 0.;
                Phi  *= logr/logRmin;
            } else {
                Phi  *= exp(innerSlopes[0]*(logr-valR));
                PhiR  = innerSlopes[0]*Phi;
                PhiRR = pow_2(innerSlopes[0])*Phi;
            }
        }
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

//----- GalaxyPotential refactored into a Composite potential -----//
std::vector<PtrPotential> createGalaxyPotentialComponents(
    const std::vector<DiskParam>& DiskParams, 
    const std::vector<SphrParam>& SphrParams)
{
    // keep track of length scales of all components
    double lengthMin=1e100, lengthMax=0;
    
    // assemble the set of density components for the multipole
    // (all spheroids and residual part of disks),
    // and the complementary set of potential components
    // (the flattened part of disks, eventually to be joined by the multipole itself)
    std::vector<PtrDensity> componentsDens;
    std::vector<PtrPotential> componentsPot;
    for(unsigned int i=0; i<DiskParams.size(); i++) {
        // the two 1d functions shared between DiskAnsatz and DiskResidual
        math::PtrFunction radialFnc(createRadialDiskFnc(DiskParams[i]));
        math::PtrFunction verticalFnc(createVerticalDiskFnc(DiskParams[i]));
        // the two parts of disk profile
        componentsPot.push_back(PtrPotential(new DiskAnsatz(DiskParams[i])));
        componentsDens.push_back(PtrDensity(new DiskResidual(DiskParams[i])));
        // keep track of characteristic lengths
        lengthMin = fmin(lengthMin, DiskParams[i].scaleRadius);
        lengthMax = fmax(lengthMax, DiskParams[i].scaleRadius);
        if(DiskParams[i].innerCutoffRadius>0)
            lengthMin = fmin(lengthMin, DiskParams[i].innerCutoffRadius);
    }
    for(unsigned int i=0; i<SphrParams.size(); i++) {
        componentsDens.push_back(PtrDensity(new SpheroidDensity(SphrParams[i])));
        lengthMin = fmin(lengthMin, SphrParams[i].scaleRadius);
        lengthMax = fmax(lengthMax, SphrParams[i].scaleRadius);
        if(SphrParams[i].outerCutoffRadius) 
            lengthMax = fmax(lengthMax, SphrParams[i].outerCutoffRadius);
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
