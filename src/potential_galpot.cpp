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

namespace potential{

inline double sign(double x) { return x>0?1.:x<0?-1.:0; }

const int    GALPOT_LMAX=78;     ///< maximum l for the Multipole expansion 
const int    GALPOT_NRAD=201;    ///< DEFAULT number of radial points in Multipole 
const double GALPOT_RMIN=1.e-4,  ///< DEFAULT min radius of logarithmic radial grid in Multipole
             GALPOT_RMAX=1.e3;   ///< DEFAULT max radius of logarithmic radial grid

//----- disk density and potential -----//

/** simple exponential radial density profile without inner hole or wiggles */
class DiskDensityRadialExp: public math::IFunction {
public:
    DiskDensityRadialExp(double _surfaceDensity, double _scaleLength): 
        surfaceDensity(_surfaceDensity), scaleLength(_scaleLength) {};
private:
    const double surfaceDensity, scaleLength;
    /**  evaluate  f(R) and optionally its two derivatives, if these arguments are not NULL  */
    virtual void evalDeriv(double R, double* f=NULL, double* fprime=NULL, double* fpprime=NULL) const {
        double val = surfaceDensity * exp(-R/scaleLength);
        if(f)
            *f = val;
        if(fprime)
            *fprime = -val/scaleLength;
        if(fpprime)
            *fpprime = val/pow_2(scaleLength);
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
        double val = params.surfaceDensity * exp(-params.innerCutoffRadius/R-Rrel+cr);
        double fp  = params.innerCutoffRadius/(R*R)-(1+sr)/params.scaleRadius;
        if(fpprime)
            *fpprime = val ? (fp*fp-2*params.innerCutoffRadius/(R*R*R)-cr/pow_2(params.scaleRadius))*val : 0;
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
        if(H)       *H       = scaleHeight/2*(h-1+x);
        if(Hprime)  *Hprime  = sign(z)*(1.-h)/2;
        if(Hpprime) *Hpprime = h/(2*scaleHeight);
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
        double      sh1      = 1.+h;
        if(H)       *H       = scaleHeight*(0.5*x+log(0.5*sh1));
        if(Hprime)  *Hprime  = 0.5*sign(z)*(1.-h)/sh1;
        if(Hpprime) *Hpprime = h/(sh1*sh1*scaleHeight);
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
        if(Hprime)  *Hprime  = sign(z)/2;
        if(Hpprime) *Hpprime = 0;
    }
    virtual unsigned int numDerivs() const { return 2; }
};

/** helper routine to create an instance of radial density function */
const math::IFunction* createRadialDiskFnc(const DiskParam& params) {
    if(params.scaleRadius<=0)
        throw std::invalid_argument("Disk scale radius cannot be <=0");
    if(params.innerCutoffRadius<0)
        throw std::invalid_argument("Disk inner cutoff radius cannot be <0");
    if(params.innerCutoffRadius==0 && params.modulationAmplitude==0)
        return new DiskDensityRadialExp(params.surfaceDensity, params.scaleRadius);
    else
        return new DiskDensityRadialRichExp(params);
}

/** helper routine to create an instance of vertical density function */
const math::IFunction* createVerticalDiskFnc(const DiskParam& params) {
    if(params.scaleHeight>0)
        return new DiskDensityVerticalExp(params.scaleHeight);
    if(params.scaleHeight<0)
        return new DiskDensityVerticalIsothermal(-params.scaleHeight);
    else
        return new DiskDensityVerticalThin();
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
    verticalFnc->evalDeriv(pos.z, &H, &Hp, &h);
    radialFnc  ->evalDeriv(r, &f, &fp, &fpp);
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
                     const double Rmin, const double Rmax,
                     const int gridSizeR, const int numCoefsAngular)
{
    if(!isAxisymmetric(source_density))
        throw std::invalid_argument("Error in Multipole expansion: source density must be axially symmetric");
    const int numMultipoles = numCoefsAngular/2+1;   // number of multipoles (only even-l terms are used)
    const int gridSizeC     = 2*numCoefsAngular+1;   // number of grid points for theta in [0,pi/2]

    // set up radial grid
    std::vector<double> gridR(gridSizeR), gridC(gridSizeC);
    std::vector<double> r(gridSizeR);
    const double
        lRmin = log(Rmin),
        lRmax = log(Rmax), 
        dlr = (lRmax-lRmin)/double(gridSizeR-1);
    for(int k=0; k<gridSizeR; k++) {
        gridR[k] = k<gridSizeR-1? lRmin+dlr*k : lRmax;
        r[k]     = exp(gridR[k]);
    }

    // construct spherical-harmonic expansion of density
    const DensitySphericalHarmonic* densh = NULL;
    if(source_density.name() == DensitySphericalHarmonic::myName())   // use existing sph.-harm. expansion
        densh = static_cast<const DensitySphericalHarmonic*>(&source_density);
    else    // create new spherical-harmonic expansion
        densh = new DensitySphericalHarmonic(gridSizeR, numCoefsAngular, source_density, Rmin, Rmax);

    // accumulate the 'inner' (P1) and 'outer' (P2) parts of the potential at radial grid points:
    // P1_l(r) = r^{-l-1} \int_0^r \rho_l(s) s^{l+2} ds,
    // P2_l(r) = r^l \int_r^\infty \rho_l(s) s^{1-l} ds.
    // For each l, we compute P1(r) by looping over sub-intervals from inside out,
    // and P2(r) - from outside in. In doing so, we use the recurrent relation
    // P1(r_{i+1}) r_{i+1}^{l+1} = r_i^{l+1} P1(r_i) + \int_{r_i}^{r_{i+1}} \rho_l(s) s^{l+2} ds,
    // and a similar relation for P2. This is further rewritten so that the integration over 
    // density  \rho_l(s)  uses  (s/r_{i+1})^{l+2}  as the radius-dependent weight function -
    // this avoids over/underflows when both l and r are large or small.
    // Finally, \Phi_l(r) is computed as -4\pi ( P1_l(r) + P2_l(r) ), and similarly its derivative.
    math::Matrix<double> Phil(gridSizeR, numMultipoles), dPhil(gridSizeR, numMultipoles);
    std::vector<double>  P1(gridSizeR), P2(gridSizeR);
    for(int l=0; l<=numCoefsAngular; l+=2) {
        // inner part
        P1[0] = densh->integrate(0, Rmin, l, l+2, Rmin) * Rmin;
        for(int k=1; k<gridSizeR; k++) {
            double s = densh->integrate(r[k-1], r[k], l, l+2, r[k]);
            P1[k] = P1[k-1] * math::powInt(r[k-1]/r[k], l+1) + s * r[k];
        }
        // outer part
        P2[gridSizeR-1] = densh->integrate(Rmax, INFINITY, l, 1-l, Rmax) * Rmax;
        if(!math::isFinite(P2[gridSizeR-1]))
            P2[gridSizeR-1] = 0;
        for(int k=gridSizeR-2; k>=0; k--) {
            double s = densh->integrate(r[k], r[k+1], l, 1-l, r[k]);
            P2[k] = P2[k+1] * math::powInt(r[k]/r[k+1], l) + s * r[k];
        }
        // put together inner and outer parts to compute the potential and its radial derivative,
        // for each spherical-harmonic term
        for(int k=0; k<gridSizeR; k++) {
            Phil (k, l/2) = -4*M_PI * (P1[k] + P2[k]);           // Phi_l
            dPhil(k, l/2) =  4*M_PI * ( (l+1)*P1[k] - l*P2[k]);  // dPhi_l/dlogr
        }
        // analyze the behaviour of potential at small radii
        if(l==0) {
            // determine the central value of potential
            double deltaPhi = 4*M_PI * (densh->integrate(0, r[0], 0, 1) - P1[0]);
            Phi0 = Phil(0, 0) - deltaPhi;  // deltaPhi is Phi(rmin)-Phi(0)
            // if the central value is finite, we derive the value of inner density slope gamma
            // assuming that Phi(r) = Phi(0) + (Phi(rmin) - Phi(0)) * (r/rmin)^(2-gamma)
            if(math::isFinite(deltaPhi) && deltaPhi!=0)
                twominusgamma = dPhil(0, 0) / deltaPhi;
            else
                // otherwise we take the ratio between enclosed mass within rmin, given by P1(rmin),
                // and the value of density at this radius, assuming that it varies as 
                // rho(r) = rho(rmin) * (r/rmin)^(-gamma)  for r<rmin
                twominusgamma = densh->rho_l(Rmin, 0) * pow_2(Rmin) / P1[0] - 1;
            if(!math::isFinite(twominusgamma) || twominusgamma<-0.5)
                twominusgamma=-0.5;  // rather arbitrary..
        }
    }

    if(source_density.name() != DensitySphericalHarmonic::myName())
        delete densh;   // destroy temporarily created spherical-harmonic expansion of the density

    // Put potential and its derivatives on a 2D grid in log[r] & cos[theta]
    //
    // set linear grid in theta, i.e. non-uniform in cos(theta), with denser spacing close to z-axis
    //
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
    double lr = log(r);
    // coordinates on the scaled grid
    const double lRmin = spl.xmin(), lRmax = spl.xmax();
    double valR = fmin(lRmax,fmax(lRmin,lr));
    double valC = fabs(ct);
    // value and derivatives of spline by its arguments (log r, z/r)
    double Phi=0, PhiR=0, PhiC=0, PhiRR=0, PhiRC=0, PhiCC=0;
    if(deriv2)
        spl.evalDeriv(valR, valC, &Phi, &PhiR, &PhiC, &PhiRR, &PhiRC, &PhiCC);
    else if(deriv)
        spl.evalDeriv(valR, valC, &Phi, &PhiR, &PhiC);
    else
        spl.evalDeriv(valR, valC, &Phi);
    PhiC  *= sign(ct);
    PhiRC *= sign(ct);
    if(lr < lRmin) {  // extrapolation at small radii
        if(twominusgamma>0) {
            Phi   = (Phi-Phi0)*exp(twominusgamma*(lr-valR));
            PhiR  = twominusgamma*Phi;
            PhiRR = pow_2(twominusgamma)*Phi;
            Phi  += Phi0;
        } else if(twominusgamma==0) {
            PhiR  = Phi/lRmin;
            PhiRR = 0.;
            Phi  *= lr/lRmin;
        } else {
            Phi  *= exp(twominusgamma*(lr-valR));
            PhiR  = twominusgamma*Phi;
            PhiRR = pow_2(twominusgamma)*Phi;
        }
    } else if(lr > lRmax)  // extrapolation at large radii
    {  // use monopole+quadrupole terms; more accurate than in the original code
        double Rmax = exp(lRmax);
        double M = (1.5*Phi + 0.5*PhiR) * Rmax/r;      // compute them from the spline value
        double Q = -0.5*(Phi + PhiR) * pow_3(Rmax/r);  // and derivative at r=Rmax
        Phi   = M+Q;
        PhiR  = -M-3*Q;
        PhiRR = M+9*Q;
    }
    if(potential)
        *potential = Phi;
    if(r==0) r = 1e-100;  // safety measure to avoid 0/0
    if(deriv) {
        deriv->dR   = (PhiR-PhiC*ct)*st/r;
        deriv->dz   = (PhiR*ct+PhiC*st*st)/r;
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
        deriv2->dRdphi=deriv2->dzdphi=deriv2->dphi2=0;
    }
}

//----- GalaxyPotential refactored into a Composite potential -----//
std::vector<const BasePotential*> createGalaxyPotentialComponents(
    const std::vector<DiskParam>& DiskParams, 
    const std::vector<SphrParam>& SphrParams)
{
    // keep track of length scales of all components
    double lengthMin=1e100, lengthMax=0;
    
    // first create a set of density components for the multipole
    // (all spheroids and residual part of disks)
    std::vector<const BaseDensity*> componentsDens;
    for(unsigned int i=0; i<DiskParams.size(); i++) {
        componentsDens.push_back(new DiskResidual(DiskParams[i]));
        lengthMin = fmin(lengthMin, DiskParams[i].scaleRadius);
        lengthMax = fmax(lengthMax, DiskParams[i].scaleRadius);
        if(DiskParams[i].innerCutoffRadius>0)
            lengthMin = fmin(lengthMin, DiskParams[i].innerCutoffRadius);
    }
    for(unsigned int i=0; i<SphrParams.size(); i++) {
        componentsDens.push_back(new SpheroidDensity(SphrParams[i]));
        lengthMin = fmin(lengthMin, SphrParams[i].scaleRadius);
        lengthMax = fmax(lengthMax, SphrParams[i].scaleRadius);
        if(SphrParams[i].outerCutoffRadius) 
            lengthMax = fmax(lengthMax, SphrParams[i].outerCutoffRadius);
    }
    if(componentsDens.size()==0)
        throw std::invalid_argument("Empty parameters in GalPot");
    // create an composite density object to be passed to Multipole;
    // the temporary density objects will be destroyed once this gets out of scope
    const CompositeDensity dens(componentsDens);

    // create multipole potential from this combined density
    double rmin = GALPOT_RMIN * lengthMin;
    double rmax = GALPOT_RMAX * lengthMax;
    const BasePotential* mult=new Multipole(dens, rmin, rmax, GALPOT_NRAD, GALPOT_LMAX);

    // now create a composite potential from the multipole and non-residual part of disk potential
    std::vector<const BasePotential*> componentsPot;
    componentsPot.push_back(mult);
    for(unsigned int i=0; i<DiskParams.size(); i++)  // note that we create another class of objects than above
        componentsPot.push_back(new DiskAnsatz(DiskParams[i]));
    // this array should be passed to the constructor of CompositeCyl potential;
    // instances of potential components created here are taken over by the composite
    // and will be deallocated when the latter is destroyed, so we don't do it here
    return componentsPot;
}

const potential::BasePotential* createGalaxyPotential(
    const std::vector<DiskParam>& DiskParams,
    const std::vector<SphrParam>& SphrParams)
{
    return new CompositeCyl(createGalaxyPotentialComponents(DiskParams, SphrParams));
}

} // namespace
