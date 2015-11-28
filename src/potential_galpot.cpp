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

/** helper function to determine the coefficients for potential extrapolation:
    assuming that 
        Phi(r) = W * (r/r0)^v + U * (r/r0)^s              if s!=v, or
        Phi(r) = W * (r/r0)^v + U * (r/r0)^s * ln(r/r0)   if s==v,
    compute the values of U and W from the following identities at r=r0:
        Phi            = -(P1 + P2)
        d Phi / d ln r = -(P1 * v - P2 * (v+1) )
*/
static void computeUW(double s, double v, double P1, double P2, double& U, double& W)
{
    if(s != v) {
        U = (2*v+1) / (s-v) * P2;
        W = (s+v+1) / (v-s) * P2 - P1;
    } else {
        U = (2*v+1) * P2;
        W = -(P1+P2);
    }
}

Multipole::Multipole(const BaseDensity& sourceDensity,
    double rmin, double rmax, int gridSizeR, int numCoefsAngular)
{
    if(gridSizeR<=2 || numCoefsAngular<0 || rmin<=0 || rmax<=rmin)
        throw std::invalid_argument("Error in Multipole expansion: invalid grid parameters");
    if(!isAxisymmetric(sourceDensity))
        throw std::invalid_argument("Error in Multipole expansion: source density must be axisymmetric");
    numCoefsAngular = std::min<int>(numCoefsAngular, MAX_NCOEFS_ANGULAR-1);
    const int numMultipoles = numCoefsAngular/2+1;   // number of multipoles (only even-l terms are used)
    const int gridSizeC     = 2*numCoefsAngular+1;   // number of grid points for theta in [0,pi/2]

    // set up radial grid
    std::vector<double> r = math::createExpGrid(gridSizeR, rmin, rmax);
    // add extra cells at the upper end:
    // the potential is computed and the spline initialized over the enlarged grid,
    // but the coefficients for extrapolation to large r are computed at the original rmax,
    // i.e. the original upper boundary, and extrapolation will be used for all r>{original rmax};
    // this decreases the influence of boundary effects in the 2d spline
    unsigned int imin = 0, imax=gridSizeR-1;
    r.push_back(rmax*1.1);
    r.push_back(rmax*1.2);
    gridSizeR = r.size();
    
    // check if the input density is of a spherical-harmonic type already...
    bool useExistingSphHarm = sourceDensity.name() == DensitySphericalHarmonic::myName();
    // ...and construct a fresh spherical-harmonic expansion of density if it wasn't such.
    PtrDensity mySphHarm(useExistingSphHarm ? NULL :
        new DensitySphericalHarmonic(gridSizeR, numCoefsAngular, sourceDensity, r.front(), r.back()));
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
    math::Matrix<double> Phil(gridSizeR, numMultipoles), dPhil(gridSizeR, numMultipoles);
    std::vector<double>  Pint(gridSizeR), Pext(gridSizeR);
    innerSlope.assign(numMultipoles, 0);
    innerCoefU.assign(numMultipoles, 0);
    innerCoefW.assign(numMultipoles, 0);
    outerSlope.assign(numMultipoles, 0);
    outerCoefU.assign(numMultipoles, 0);
    outerCoefW.assign(numMultipoles, 0);

    for(int l=0; l<=numCoefsAngular; l+=2) {
        // inner part
        Pint[0] = densh.integrate(0, r[0], l, l+2, r[0]) * r[0];
        if(!math::isFinite(Pint[0]))
            throw std::runtime_error("Error in Multipole: mass is divergent at origin");
        for(int k=1; k<gridSizeR; k++) {
            double s = densh.integrate(r[k-1], r[k], l, l+2, r[k]);
            Pint[k] = Pint[k-1] * math::powInt(r[k-1]/r[k], l+1) + s * r[k];
        }

        // outer part
        Pext[gridSizeR-1] = densh.integrate(r[gridSizeR-1], INFINITY,
            l, 1-l, r[gridSizeR-1]) * r[gridSizeR-1];
        if(!math::isFinite(Pext[gridSizeR-1]))
            throw std::runtime_error("Error in Multipole: potential is divergent at infinity");
        for(int k=gridSizeR-2; k>=0; k--) {
            double s = densh.integrate(r[k], r[k+1], l, 1-l, r[k]);
            Pext[k] = Pext[k+1] * math::powInt(r[k]/r[k+1], l) + s * r[k];
        }

        // put together inner and outer parts to compute the potential and its radial derivative,
        // for each spherical-harmonic term
        for(int k=0; k<gridSizeR; k++) {
            Phil (k, l/2) = -4*M_PI * (Pint[k] + Pext[k]);           // Phi_l
            dPhil(k, l/2) =  4*M_PI * ( (l+1)*Pint[k] - l*Pext[k]);  // dPhi_l/dlogr
            if(!math::isFinite(Phil(k,l/2)))
                throw std::runtime_error("Error in Multipole: bad value of potential");
#ifdef VERBOSE_REPORT
            //std::cout << l << '\t' << r[k] << '\t' << Phil(k,l/2) << '\t' << 
            //(4*M_PI*Pint[k]) << '\t' << (4*M_PI*Pext[k]) << '\n';
#endif
        }

        // determine the density slope at small and large radii:
        // [r<rmin] assume that rho(r) = rho(rmin) * (r/rmin)^{s-2}  with some slope s
        double densrmin = densh.rho_l(r[imin], l);
        // try to derive the 'mean' density slope inside rmin, using the integral over density
        innerSlope[l/2] =  densrmin * pow_2(r[imin]) / Pint[imin] - l-1;
        if(!math::isFinite(innerSlope[l/2]) || innerSlope[l/2] <= -l-1)
            innerSlope[l/2] = 2;  // safe value: for l=0 it corresponds to a constant-density core
        //if(fabs(innerSlope[l/2])<0.001) innerSlope[l/2]=0;

        // [r>rmax], same exercise to determine the outer slope (one node behind the end of grid)
        double densrmax = densh.rho_l(r[imax], l);
        outerSlope[l/2] = -densrmax * pow_2(r[imax]) / Pext[imax] + l;
        if(!math::isFinite(outerSlope[l/2]) || outerSlope[l/2] >= l)
            outerSlope[l/2] = -2;  // safe value: for l=0 it corresponds to r^-4 falloff

        // determine the coefficients U and W for extrapolating to large and small radii
        // (beyond the definition region of 2d interpolating spline), see comments in evalCyl
        computeUW(innerSlope[l/2], l,    4*M_PI*Pext[imin], 4*M_PI*Pint[imin],
            innerCoefU[l/2], innerCoefW[l/2]);
        computeUW(outerSlope[l/2], -l-1, 4*M_PI*Pint[imax], 4*M_PI*Pext[imax],
            outerCoefU[l/2], outerCoefW[l/2]);

#ifdef VERBOSE_REPORT
        std::cout << "Multipole: density profile for l=" << l << 
            ": inner rho=" << densrmin << "*r^" << (innerSlope[l/2]-2) <<
            ", outer rho=" << densrmax << "*r^" << (outerSlope[l/2]-2) << "\n";
#endif
    }

    // Put potential and its derivatives on a 2D grid in log[r] & cos[theta]:
    // set up linear grid in theta, i.e. non-uniform in cos(theta), with denser spacing close to z-axis
    std::vector<double> gridR(gridSizeR), gridC(gridSizeC);
    for(int i=0; i<gridSizeR; i++)
        gridR[i] = log(r[i]);
    logrmin = gridR[imin];
    logrmax = gridR[imax];  // revert back to the original upper bound (rmax) without extra cells
    for(int i=0; i<gridSizeC-1; i++) 
        gridC[i] = sin(M_PI_2 * i / (gridSizeC-1));
    gridC[gridSizeC-1] = 1.0;

    // assign Phi, dPhi/dlogr & dPhi/dcos[theta] 
    math::Matrix<double> Phi_val(gridSizeR, gridSizeC);
    math::Matrix<double> Phi_dR (gridSizeR, gridSizeC);
    math::Matrix<double> Phi_dC (gridSizeR, gridSizeC);
    std::vector<double>  Pl(numCoefsAngular+1), dPl(numCoefsAngular+1);
    for(int i=0; i<gridSizeC; i++) {
        math::legendrePolyArray(numCoefsAngular, 0, gridC[i], &Pl.front(), &dPl.front());
        for(int k=0; k<gridSizeR; k++) {
            double val=0, dR=0, dC=0;
            for(int l=0; l<=numCoefsAngular; l+=2) {
                val += Phil (k, l/2) *  Pl[l];   // Phi
                dR  += dPhil(k, l/2) *  Pl[l];   // d Phi / d log(r)
                dC  += Phil (k, l/2) * dPl[l];   // d Phi / d cos(theta)
            }
            Phi_val(k, i) = val;
            Phi_dR (k, i) = dR;
            Phi_dC (k, i) = dC;
        }
    }

    // establish 2D quintic spline of Phi in log[r] & cos[theta]
    spl = math::QuinticSpline2d(gridR, gridC, Phi_val, Phi_dR, Phi_dC);

    // determine if the potential is spherically-symmetric
    // (could determine this explicitly by analyzing the angular dependence of Phi(r,theta),
    // but for now simply ask the source density model
    isSpherical = (sourceDensity.symmetry() & ST_SPHERICAL) == ST_SPHERICAL;
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
