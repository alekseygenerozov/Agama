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
Modifications by Eugene Vasiliev, 2015-2016
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

namespace potential{

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

/** integrand for computing the total mass:  2pi R Sigma(R); x=R/scaleRadius */
class DiskDensityRadialRichExpIntegrand: public math::IFunctionNoDeriv {
public:
    DiskDensityRadialRichExpIntegrand(const DiskParam& _params): params(_params) {};
private:
    const DiskParam params;
    virtual double value(double x) const {
        if(x==0 || x==1) return 0;
        double Rrel = x/(1-x);
        return x / pow_3(1-x) *
            exp(-params.innerCutoffRadius/params.scaleRadius/Rrel - Rrel
                +params.modulationAmplitude*cos(Rrel));
    }
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

double DiskParam::mass() const
{
    if(modulationAmplitude==0) {  // have an analytic expression
        if(innerCutoffRadius==0)
            return 2*M_PI * pow_2(scaleRadius) * surfaceDensity;
        else {
            double p = sqrt(innerCutoffRadius / scaleRadius);
            return 4*M_PI * pow_2(scaleRadius) * surfaceDensity *
            p * (p * math::besselK(0, 2*p) + math::besselK(1, 2*p));
        }
    }
    return 2*M_PI * pow_2(scaleRadius) * surfaceDensity *
        math::integrate(DiskDensityRadialRichExpIntegrand(*this), 0, 1, 1e-4);
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

/** integrand for computing the total mass:  4pi r^2 rho(r), x=r/scaleRadius */
class SpheroidDensityIntegrand: public math::IFunctionNoDeriv {
public:
    SpheroidDensityIntegrand(const SphrParam& _params): params(_params) {};
private:
    const SphrParam params;
    virtual double value(double x) const {
        if(x==0 || x==1) return 0;
        double rrel = x/(1-x);
        return 
            pow_2(x/pow_2(1-x)) * pow(rrel, -params.gamma) * pow(1+rrel, params.gamma-params.beta) *
            exp(-pow_2(rrel * params.scaleRadius / params.outerCutoffRadius));
    }
};

double SphrParam::mass() const
{
    if(outerCutoffRadius==0)   // have an analytic expression
        return 4*M_PI * densityNorm * pow_3(scaleRadius) * axisRatio *
            math::gamma(beta-3) * math::gamma(3-gamma) / math::gamma(beta-gamma);
    return 4*M_PI * axisRatio * densityNorm * pow_3(scaleRadius) *
        math::integrate(SpheroidDensityIntegrand(*this), 0, 1, 1e-4);
}

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
}

double SpheroidDensity::densityCyl(const coord::PosCyl &pos) const
{
    double m   = hypot(pos.R, pos.z/params.axisRatio);
    double m0  = m/params.scaleRadius;
    double rho = params.densityNorm;
    if(params.gamma==1.) rho /= m0;       else 
    if(params.gamma==2.) rho /= m0*m0;    else
    if(params.gamma==0.5)rho /= sqrt(m0); else
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

//----- GalaxyPotential refactored into a Composite potential -----//
std::vector<PtrPotential> createGalaxyPotentialComponents(
    const std::vector<DiskParam>& diskParams, 
    const std::vector<SphrParam>& sphrParams)
{
    double lengthMin=INFINITY, lengthMax=0;  // keep track of length scales of all components
    bool isSpherical=diskParams.size()==0;   // whether there are any non-spherical components

    // assemble the set of density components for the multipole
    // (all spheroids and residual part of disks),
    // and the complementary set of potential components
    // (the flattened part of disks, eventually to be joined by the multipole itself)
    std::vector<PtrDensity> componentsDens;
    std::vector<PtrPotential> componentsPot;
    for(unsigned int i=0; i<diskParams.size(); i++) {
        if(diskParams[i].surfaceDensity == 0)
            continue;
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
        if(sphrParams[i].densityNorm == 0)
            continue;
        componentsDens.push_back(PtrDensity(new SpheroidDensity(sphrParams[i])));
        lengthMin = fmin(lengthMin, sphrParams[i].scaleRadius);
        lengthMax = fmax(lengthMax, sphrParams[i].scaleRadius);
        if(sphrParams[i].outerCutoffRadius) 
            lengthMax = fmax(lengthMax, sphrParams[i].outerCutoffRadius);
        isSpherical &= sphrParams[i].axisRatio == 1;
    }
    if(componentsDens.size()==0)
        throw std::invalid_argument("Empty parameters in GalPot");
    // create an composite density object to be passed to Multipole;
    const CompositeDensity dens(componentsDens);

    // create multipole potential from this combined density
    double rmin = GALPOT_RMIN * lengthMin;
    double rmax = GALPOT_RMAX * lengthMax;
    int lmax = isSpherical ? 0 : GALPOT_LMAX;
    componentsPot.push_back(Multipole::create(dens, lmax, 0, GALPOT_NRAD, rmin, rmax));

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
