#include "df_spherical.h"
#include "math_core.h"
#include "math_specfunc.h"
#include <stdexcept>
#include <cmath>

#ifdef VERBOSE_OUTPUT
#include <fstream>
#include <iomanip>
#endif

namespace df {
namespace{

/// relative accuracy in computing the integrals for phase-space volume
static const double ACCURACY = 1e-10;

/// default grid spacing in log radius or log phase volume
static const double DELTALOG = 0.125;

// required tolerance on the 2nd deriv to declare the asymptotic limit
static const double EPS2DER  = 1e-6;

/// helper function to find a root of fnc(x)=val  (TODO: augment math::findRoot with this feature!)
class RootFinder: public math::IFunction {
    const math::IFunction& fnc;
    double val;
public:
    RootFinder(const math::IFunction& _fnc, double _val) : fnc(_fnc), val(_val) {}
    virtual void evalDeriv(const double x, double *v, double *d, double *dd) const {
        fnc.evalDeriv(x, v, d, dd);
        if(v)
            *v -= val;
    }
    virtual unsigned int numDerivs() const { return fnc.numDerivs(); }
};
    
/// integrand for computing phase volume and density of states:
/// int_0^{rmax} dr r^2 v^n(E,r)  is transformed into
/// rmax^3 * int_0^1 ds rs(s)^2 v^n(E,rmax*rs) drs/ds, where rs=(3-2s)s^2;
/// this improves accuracy by regularizing the integrand at endpoints.
class PotIntegrand: public math::IFunctionNoDeriv {
    const math::IFunction& pot;
    double rmax, n, E;
public:
    PotIntegrand(const math::IFunction& _pot, double _rmax, double _n) :
        pot(_pot), rmax(_rmax), n(_n), E(pot(rmax)) {}
    virtual double value(const double s) const {
        double rs = s*s*(3-2*s);
        return 6*s*(1-s) * pow_2(rs) * pow(fmax(0, E - pot(rmax*rs)), n);
    }
};

/// different regimes for calculation of various integrals involving f(h)
typedef enum { MODE_INTF, MODE_INTFG, MODE_INTFH, MODE_INTJ1, MODE_INTJ3 } Operation;

/// integrand for computing the product of f(h) and some other function, depending on the mode
template<Operation mode>
class DFIntegrand: public math::IFunctionNoDeriv {
    const math::IFunction &df;
    const PhaseVolume &pv;
    const double Phi;
public:
    DFIntegrand(const math::IFunction& _df, const PhaseVolume& _pv, double _Phi=0) :
        df(_df), pv(_pv), Phi(_Phi) {}
    virtual double value(const double logh) const {
        double h = exp(logh), g, E = pv.E(h, &g);
        double val =   // the value of weighting function in the integrand
            mode==MODE_INTF  ? 1 :
            mode==MODE_INTFG ? g :
            mode==MODE_INTFH ? h :
            mode==MODE_INTJ1 ? sqrt(1-E/Phi) :
            mode==MODE_INTJ3 ? pow_3(sqrt(1-E/Phi)) : NAN;
        // the original integrals are formulated in terms of \int f(E) val(E) dE,
        // and we replace dE by d(log h) * [ dh / d(log h) ] / [ dh / dE ],
        // that's why there are extra factors h and 1/g below.
        return df(h) * h / g * val;
    }
};

/// scaling transformations for energy: the input energy ranges from Phi0 to 0,
/// the output scaled variable - from -inf to +inf. Here Phi0=Phi(0) may be finite or -inf.
static inline double scaledE(const double E, const double Phi0) {
    return log(1/Phi0 - 1/E);
}

/// inverse scaling transformation for energy or potential
static inline double unscaledE(const double scaledE, const double Phi0) {
    return 1 / (1/Phi0 - exp(scaledE));
}

/// derivative of scaling transformation: dE/d{scaledE}
static inline double scaledEder(const double E, const double Phi0) {
    return E * (E/Phi0 - 1);
}

/// linearly extrapolate a two-dimensional spline (or just compute it if the point is inside its domain)
static inline double evalExtrapolate(const math::BaseInterpolator2d& interp, double x, double y)
{
    double xx = fmin(fmax(x, interp.xmin()), interp.xmax());
    double yy = fmin(fmax(y, interp.ymin()), interp.ymax());
#if 0
    double val, xder, yder;
    interp.evalDeriv(xx, yy, &val, xx!=x ? &xder : NULL, yy!=y ? &yder : NULL);
    if(xx!=x)
        val += xder * (x-xx);
    if(yy!=y)
        val += yder * (y-yy);
    return val;
#else
    // in the present implementation, derivatives are assumed to be zero at endpoints
    return interp.value(xx, yy);
#endif
}

}  // internal namespace

//---- Correspondence between h and E ----//

PhaseVolume::PhaseVolume(const math::IFunction& pot)
{
    Phi0 = pot(0);
    if(!(Phi0<0))
        throw std::invalid_argument("PhaseVolume: invalid value of Phi(r=0)");

    // TODO: make the choice of initial radius more scale-invariant!
    const double logRinit = 0; // initial value of log radius (rather arbitrary but doesn't matter)
    const int NUM_ARRAYS = 3;  // 1d arrays of various quantities:
    std::vector<double> grids[NUM_ARRAYS];
    std::vector<double>   // assign a proper name to each of these arrays:
    &gridE  =grids[0],    // scaled E=Phi(r)
    &gridH  =grids[1],    // log(h(E))
    &gridG  =grids[2];    // log(g(E)), where g=dh/dE

    double logR = logRinit;
    int   stage = 0;   // 0 means scan inward, 1 - outward, 2 - done
    while(stage<2) {   // first scan inward in radius, then outward, then stop
        double R = exp(logR);
        double E = pot(R);
        double G = math::integrate(PotIntegrand(pot, R, 0.5), 0, 1, ACCURACY);
        double H = math::integrate(PotIntegrand(pot, R, 1.5), 0, 1, ACCURACY);
        gridE.push_back(scaledE(E, Phi0));
        gridH.push_back(log(H) + log(16*M_PI*M_PI/3*2*M_SQRT2) + 3*logR);
        gridG.push_back(1.5*G/H * scaledEder(E, Phi0));

        // check if we have reached an asymptotic regime,
        // by examining the curvature (2nd derivative) of relation between scaled H and E.
        unsigned int np = gridE.size();
        double dlogR = DELTALOG;
        if(np>=3 && fabs(logR - logRinit)>=2) {
            double der2 = math::deriv2(gridE[np-3], gridE[np-2], gridE[np-1],
                gridH[np-3], gridH[np-2], gridH[np-1], gridG[np-3], gridG[np-2], gridG[np-1]);
            // check if converged, or if the covered range of radii is too large (>1e6)
            if(fabs(der2) < EPS2DER || fabs(logR - logRinit)>=15) {
                if(stage==0) {   // we've been assembling the arrays inward, now need to reverse them
                    for(int i=0; i<NUM_ARRAYS; i++)
                        std::reverse(grids[i].begin(), grids[i].end());
                }
                logR = logRinit;  // restart from the middle
                ++stage;          // switch direction in scanning, or finish
            } else {
                // if we are close to the asymptotic regime but not yet there, we may afford to increase
                // the spacing between grid nodes without deteriorating the accuracy of interpolation
                if(fabs(der2) < EPS2DER*10)
                    dlogR *= 4;
                else if(fabs(der2) < EPS2DER*100)
                    dlogR *= 2;
            }
        }
        if(stage==0)
            logR -= dlogR;
        else
            logR += dlogR;
    }
    HofE = math::QuinticSpline(gridE, gridH, gridG);
    // inverse relation between E and H - the derivative is reciprocal
    for(unsigned int i=0; i<gridG.size(); i++)
        gridG[i] = 1/gridG[i];
    EofH = math::QuinticSpline(gridH, gridE, gridG);
}

void PhaseVolume::evalDeriv(const double E, double* h, double* g, double*) const
{
    // out-of-bounds value of energy returns 0 or infinity, but not NAN
    if(E<=Phi0) {
        if(h) *h=0;
        if(g) *g=0;
        return;
    }
    if(E>=0) {
        if(h) *h=INFINITY;
        if(g) *g=INFINITY;
        return;
    }
    double val;
    HofE.evalDeriv(scaledE(E, Phi0), &val, g);
    val = exp(val);
    if(h)
        *h = val;
    if(g)
        *g *= val / scaledEder(E, Phi0);
}

double PhaseVolume::E(const double h, double* g) const
{
    if(h==0) {
        if(g) *g=0;
        return Phi0;
    }
    if(h==INFINITY) {
        if(g) *g=INFINITY;
        return 0;
    }
    double e;
    EofH.evalDeriv(log(h), &e, g);
    e = unscaledE(e, Phi0);
    if(g)
        *g = h / *g / scaledEder(e, Phi0);
    return e;
}


//---- Distribution functions f(h) ----//

SphericalIsotropic::SphericalIsotropic(const std::vector<double>& gridh, const std::vector<double>& gridf,
    double slopeIn, double slopeOut)
{
    if(gridh.size() != gridf.size())
        throw std::invalid_argument("SphericalIsotropic: array lengths are not equal");
    std::vector<double> sh(gridh.size()), sf(gridf.size());
    for(unsigned int i=0; i<gridh.size(); i++) {
        sh[i] = log(gridh[i]);
        sf[i] = log(gridf[i]);
        if((i>0 && sh[i]<=sh[i-1]) || !math::isFinite(sf[i]+sh[i]))
            throw std::invalid_argument("SphericalIsotropic: incorrect input data");
    }
    // construct the spline, optionally with the user-provided endpoint derivatives:
    // f(h) ~ h^slopeIn, df/dh = slopeIn * f / h = d(log f) / d(log h) * f / h
    spl = math::CubicSpline(sh, sf, slopeIn, slopeOut);
    // check correctness of asymptotic behaviour
    double der;
    spl.evalDeriv(sh.front(), NULL, &der);
    if(!(der > -1))
        throw std::runtime_error("SphericalIsotropic: f(h) rises too steeply as h-->0");
    spl.evalDeriv(sh.back(), NULL, &der);
    if(!(der < -1))
        throw std::runtime_error("SphericalIsotropic: f(h) falls too slowly as h-->infinity");
}

double SphericalIsotropic::value(const double h) const
{
    if(!(h>0))
        return 0;
    return exp(spl(log(h)));
}


SphericalIsotropic makeEddingtonDF(const math::IFunction& /*density*/, const math::IFunction& /*potential*/)
{
    throw std::runtime_error("makeEddingtonDF not implemented");
}

SphericalIsotropic fitSphericalDF(
    const std::vector<double>& hvalues, const std::vector<double>& masses, unsigned int gridSize)
{
    const unsigned int nbody = hvalues.size();
    if(masses.size() != nbody)
        throw std::invalid_argument("fitSphericalDF: array sizes are not equal");
    const int minParticlesPerBin  = std::max(1, static_cast<int>(log(nbody+1)/log(2)));
    std::vector<double> logh(nbody);
    for(unsigned int i=0; i<nbody; i++) {
        logh[i] = log(hvalues[i]);
        if(!math::isFinite(logh[i]+masses[i]) || masses[i]<0)
            throw std::invalid_argument("fitSphericalDF: incorrect input data");
    }
    std::vector<double> gridh = math::createAlmostUniformGrid(gridSize, logh, minParticlesPerBin);
    math::CubicSpline fitfnc(gridh,
        math::splineLogDensity<3>(gridh, logh, masses,
        math::FitOptions(math::FO_INFINITE_LEFT | math::FO_INFINITE_RIGHT | math::FO_PENALTY_3RD_DERIV)));
    double slopeIn, slopeOut;
    fitfnc.evalDeriv(gridh.front(), NULL, &slopeIn);
    fitfnc.evalDeriv(gridh.back(),  NULL, &slopeOut);
    assert(slopeIn > 0 && slopeOut < 0);  // a condition for a valid fit (total mass should be finite)
    std::vector<double> gridf(gridh.size());
    for(unsigned int i=0; i<gridh.size(); i++) {
        double h = exp(gridh[i]);
        // the fit provides log( dM/d(log h) ) = log( h dM/dh ) = log( h f(h) )
        gridf[i] = exp(fitfnc(gridh[i])) / h;
        gridh[i] = h;
    }
    // construct an interpolating spline that matches exactly our fitfnc (it's also a cubic spline
    // in the same scaled variables), including the correct slopes for extrapolation outside the grid
    return SphericalIsotropic(gridh, gridf, slopeIn-1, slopeOut-1);
}


DiffusionCoefs::DiffusionCoefs(const PhaseVolume& _phasevol, const math::IFunction& df) :
    phasevol(_phasevol)
{
    // 1. determine the range of h that covers the region of interest
    // and construct the grid in X = log[h(Phi)] and Y = log[h(E)/h(Phi)]
    const double logHmin         = phasevol.logHmin(),  logHmax = phasevol.logHmax();
    const unsigned int npoints   = static_cast<unsigned int>(fmax(100, (logHmax-logHmin)/0.5));
    std::vector<double> gridLogH = math::createUniformGrid(npoints, logHmin, logHmax);
    const unsigned int npointsY  = 100;
    const double mindeltaY       = fmin(0.1, (logHmax-logHmin)/npointsY);
    std::vector<double> gridY    = math::createNonuniformGrid(npointsY, mindeltaY, logHmax-logHmin, true);
    
    // 2. store the values of f, g, h at grid nodes
    std::vector<double> gridF(npoints), gridG(npoints), gridH(npoints);
    std::vector<double> gridFint(npoints), gridFGint(npoints), gridFHint(npoints);
    for(unsigned int i=0; i<npoints; i++) {
        gridH[i] = exp(gridLogH[i]);
        gridF[i] = df(gridH[i]);
        phasevol.E(gridH[i], &gridG[i]);
    }

    // 3a. determine the asymptotic behaviour of f(h):
    // f(h) ~ h^outerFslope as h-->inf  or  h^innerFslope as h-->0
    double innerFslope = log(gridF[1] / gridF[0]) / (gridLogH[1] - gridLogH[0]);
    double outerFslope = log(gridF[npoints-1] / gridF[npoints-2]) /
        (gridLogH[npoints-1] - gridLogH[npoints-2]);
    if(!(innerFslope > -1))
        throw std::runtime_error("DiffusionCoefs: f(h) rises too rapidly as h-->0");
    if(!(outerFslope < -1))
        throw std::runtime_error("DiffusionCoefs: f(h) falls off too slowly as h-->infinity");

    // 3b. determine the asymptotic behaviour of h(E), or rather, g(h) = dh/dE:
    // -E ~ h^outerEslope  and  g(h) ~ h^(1-outerEslope)  as  h-->inf,
    // and in the nearly Keplerian potential at large radii outerEslope should be ~ -2/3.
    // -E ~ h^innerEslope + const  and  g(h) ~ h^(1-innerEslope)  as  h-->0:
    // if innerEslope<0, Phi(r) --> -inf as r-->0, and we assume that |innerE| >> const;
    // otherwise Phi(0) is finite, and we assume that  innerE-Phi(0) << |Phi(0)|.
    // in general, if Phi ~ r^n + const at small r, then innerEslope = 2n / (6+3n);
    // innerEslope ranges from -2/3 for a Kepler potential to ~0 for a logarithmic potential,
    // to +1/3 for a harmonic (constant-density) core.
    double Phi0   = phasevol.E(0);  // Phi(r=0), may be -inf
    double innerE = phasevol.E(gridH.front());
    double outerE = phasevol.E(gridH.back());
    if(!(Phi0 < innerE && innerE < outerE && outerE < 0))
        throw std::runtime_error("DiffusionCoefs: weird behaviour of potential");
    if(Phi0 != -INFINITY)   // determination of inner slope depends on whether the potential is finite
        innerE -= Phi0;
    double innerEslope = gridH.front() / gridG.front() / innerE;
    double outerEslope = gridH.back()  / gridG.back()  / outerE;
    double outerRatio  = outerFslope  / outerEslope;
    if(!(outerRatio > 0 && innerEslope + innerFslope > -1))
        throw std::runtime_error("DiffusionCoefs: weird asymptotic behaviour of phase volume");

    // 4. construct 1d interpolating splines for integrals of f(E) dE, f(E) g(E) dE, and f(E) h(E) dE

    // 4a. integral of f(h) dE = f(h) / g(h) dh -- compute from outside in,
    // summing contributions from all intervals of h above its current value
    DFIntegrand<MODE_INTF> dfint(df, phasevol);
    // the outermost segment from h_max to infinity is integrated analytically
    gridFint.back() = -gridF.back() * outerE / (1 + outerRatio);
    for(int i=npoints-1; i>=1; i--) {
        gridFint[i-1] = gridFint[i] + math::integrate(dfint, gridLogH[i-1], gridLogH[i], ACCURACY);
    }
    
    // 4b. integrands of f*g dE  and  f*h dE;  note that g = dh/dE.
    // compute from inside out, summing contributions from all previous intervals of h
    DFIntegrand<MODE_INTFG> dfgint(df, phasevol);
    DFIntegrand<MODE_INTFH> dfhint(df, phasevol);
    // integrals over the first segment (0..gridH[0]) are computed analytically
    gridFGint[0] = gridF[0] * gridH[0] / (1 + innerFslope);
    gridFHint[0] = gridF[0] * pow_2(gridH[0]) / gridG[0] / (1 + innerEslope + innerFslope);
    for(unsigned int i=1; i<npoints; i++) {
        gridFGint[i] = gridFGint[i-1] + math::integrate(dfgint, gridLogH[i-1], gridLogH[i], ACCURACY);
        gridFHint[i] = gridFHint[i-1] + math::integrate(dfhint, gridLogH[i-1], gridLogH[i], ACCURACY);
    }
    // add the contribution of integrals from the last grid point up to infinity (very small anyway)
    gridFGint.back() -= gridF.back() * gridH.back() / (1 + outerFslope);
    gridFHint.back() -= gridF.back() * pow_2(gridH.back()) / gridG.back() / (1 + outerEslope + outerFslope);

    // 4c. log-scale the computed values and prepare derivatives for quintic spline
    std::vector<double> gridFder(npoints), gridFGder(npoints), gridFHder(npoints);
    for(unsigned int i=0; i<npoints; i++) {
        gridFder[i]  = gridH[i] / -gridFint[i] * gridF[i] / gridG[i];
        gridFGder[i] = gridH[i] / gridFGint[i] * gridF[i];
        gridFHder[i] = gridH[i] / gridFHint[i] * gridF[i] * gridH[i] / gridG[i];
        gridFint[i]  = log(gridFint[i]);
        gridFGint[i] = log(gridFGint[i]);
        gridFHint[i] = log(gridFHint[i]);
        if(!(gridFder[i]<=0 && gridFGder[i]>=0 && gridFHder[i]>=0 && 
            math::isFinite(gridFint[i] + gridFGint[i] + gridFHint[i])))
            throw std::runtime_error("DiffusionCoefs: cannot construct valid interpolators");
    }
    // integrals of f*g and f*h have finite limit as h-->inf;
    // extrapolate them as constants beyond the last grid point
    gridFGder.back() = gridFHder.back() = 0;

    // 4d. initialize splines for log-scaled integrals
    intf  = math::QuinticSpline(gridLogH, gridFint,  gridFder);
    intfg = math::QuinticSpline(gridLogH, gridFGint, gridFGder);
    intfh = math::QuinticSpline(gridLogH, gridFHint, gridFHder);

    // 5. construct 2d interpolating splines for J1 and J3 as functions of Phi and E

    // 5a. asymptotic values for J1/J0 and J3/J0 as Phi --> 0 and (E/Phi) --> 0
    double outerJ1 = 0.5*M_SQRTPI * math::gamma(2 + outerRatio) / math::gamma(2.5 + outerRatio);
    double outerJ3 = outerJ1 * 3 / (5 + 2*outerRatio);

    // 5b. compute the values of J1/J0 and J3/J0 at nodes of 2d grid in X=log(h(Phi)), Y=log(h(E)/h(Phi))
    math::Matrix<double> gridv2par(npoints, npointsY), gridv2per(npoints, npointsY);
    for(unsigned int i=0; i<npoints; i++) {
        double Phi = phasevol.E(gridH[i]);
        double I0  = exp(intf(gridLogH[i]));
        double J1overI0 = 0, J3overI0 = 0;
        DFIntegrand<MODE_INTJ1> intJ1(df, phasevol, Phi);
        DFIntegrand<MODE_INTJ3> intJ3(df, phasevol, Phi);
        gridv2par(i, 0) = log(2./5);  // analytic limiting values for Phi=E
        gridv2per(i, 0) = log(8./5);
        for(unsigned int j=1; j<npointsY; j++) {
            double EoverPhi=0, J0overI0=1;
            if(i==npoints-1) {
                // last row: analytic limiting values for Phi-->0 and any E/Phi
                EoverPhi = exp(gridY[j] * outerEslope);
                double oneMinusJ0overI0 = pow(EoverPhi, 1+outerRatio);
                J0overI0 = 1 - oneMinusJ0overI0;
                J1overI0 = outerJ1 - oneMinusJ0overI0 * (j<npointsY-1 ?
                    math::hypergeom2F1(-0.5, 1+outerRatio, 2+outerRatio, EoverPhi) : 0);
                J3overI0 = outerJ3 - oneMinusJ0overI0 * (j<npointsY-1 ?
                    math::hypergeom2F1(-1.5, 1+outerRatio, 2+outerRatio, EoverPhi) : 0);
            } else {
                double logHprev = gridLogH[i] + gridY[j-1];
                double logHcurr = gridLogH[i] + gridY[j];
                /*if(j==npointsY-1)
                    logHcurr = fmax(logHcurr, gridLogH.back());*/
                double hcurr = exp(logHcurr);
                EoverPhi = phasevol.E(hcurr) / Phi;  // <=1
                J0overI0 = 1 - exp(intf(logHcurr)) / I0;
                if(j==1) {
                    J1overI0 = math::integrate(math::ScaledIntegrandEndpointSing(
                        intJ1, logHprev, logHcurr), 0, 1, ACCURACY) / I0;
                    J3overI0 = math::integrate(math::ScaledIntegrandEndpointSing(
                        intJ3, logHprev, logHcurr), 0, 1, ACCURACY) / I0;
                } else {
                    J1overI0 += math::integrate(intJ1, logHprev, logHcurr, ACCURACY) / I0;
                    J3overI0 += math::integrate(intJ3, logHprev, logHcurr, ACCURACY) / I0;
                }
                /*if(j==npointsY-1) {
                    double mult = -Phi * EoverPhi * df(hcurr) / (1+outerRatio) / I0;
                    J1overI0 += mult * math::hypergeom2F1(-0.5, 1+outerRatio, 2+outerRatio, EoverPhi);
                    J3overI0 += mult * math::hypergeom2F1(-1.5, 1+outerRatio, 2+outerRatio, EoverPhi);
                    EoverPhi = 0;
                    J0overI0 = 1;
                }*/
            }
            double J1 = J1overI0 / J0overI0 / sqrt(1-EoverPhi);
            double J3 = J3overI0 / J0overI0 / pow_3(sqrt(1-EoverPhi));
            gridv2par(i, j) = log(J3);
            gridv2per(i, j) = log(3 * J1 - J3);
        }
    }

#ifdef VERBOSE_OUTPUT
    ///!!! debugging
    std::ofstream strm("bla");
    for(unsigned int i=0; i<npoints; i++) {
        double Phi = phasevol.E(gridH[i]);
        for(unsigned int j=0; j<npointsY; j++) {
            double E = phasevol.E(exp(gridLogH[i] + gridY[j]));
            strm << gridLogH[i] << ' ' << gridY[j] << '\t' << Phi << ' ' << E << '\t' <<
            exp(gridv2par(i, j)) << ' ' << exp(gridv2per(i, j)) << '\n';
        }
        strm << '\n';
    }
    strm.close();
#endif

    // 5c. construct the 2d splines
    intv2par = math::CubicSpline2d(gridLogH, gridY, gridv2par/*, 0, 0, NAN, 0*/);
    intv2per = math::CubicSpline2d(gridLogH, gridY, gridv2per/*, 0, 0, NAN, 0*/);
}

void DiffusionCoefs::evalOrbitAvg(double E, double &DE, double &DEE) const
{
    double h, g;
    phasevol.evalDeriv(E, &h, &g);
    double
    logh = log(h),
    mass = exp(intfg(intfg.xmax())),
    IF   = exp(intf(logh)),
    IFG  = exp(intfg(logh)),
    IFH  = exp(intfh(logh));
    DE   = 16*M_PI*M_PI * mass * (IF - IFG / g);
    DEE  = 32*M_PI*M_PI * mass * (IF * h + IFH) / g;
}

void DiffusionCoefs::evalLocal(double Phi, double E, double &dvpar, double &dv2par, double &dv2per) const
{
    double Ei   = fmin(E, 0);   // if E>0, evaluate the interpolants for E=0, and then apply a correction
    double hPhi = phasevol(Phi), loghPhi = log(hPhi);
    double hEi  = phasevol(Ei),  loghEi  = log(hEi);
    if(!(Phi<0 && loghEi >= loghPhi))
        throw std::invalid_argument("DiffusionCoefs: incompatible values of E and Phi");
    
    double logmass = intfg(intfg.xmax());   // log(total mass)
    double I0 = exp(intf(loghEi) + logmass);
    double J0 = exp(intf(loghPhi)+ logmass) - I0;
    double v2par = exp(evalExtrapolate(intv2par, loghPhi, loghEi-loghPhi)) * J0;
    double v2per = exp(evalExtrapolate(intv2per, loghPhi, loghEi-loghPhi)) * J0;
    if(E>0) {  // in this case, the coefficients were computed for Ei=0, need to scale them to E>0
        double J1 = (v2par + v2per) / 3;
        double corr = 1 / sqrt(1 - E / Phi);  // correction factor <1
        J1    *= corr;
        v2par *= pow_3(corr);
        v2per  = 3 * J1 - v2par;
    }
    double mult = 32*M_PI*M_PI/3;
    dvpar  = -mult * (v2par + v2per);
    dv2par =  mult * (v2par + I0);
    dv2per =  mult * (v2per + I0 * 2);
}

double DiffusionCoefs::cumulMass(const double h) const
{
    return exp(intfg(log(h)));
}

double DiffusionCoefs::findh(const double cm) const
{
    // solve the relation intfg(log(h)) = log(cm)  to find h for the given cm (cumulative mass)
    if(cm==0)
        return 0;
    double logcm = log(cm), loghmin = intfg.xmin(), loghmax = intfg.xmax();
    if(logcm > intfg(loghmax))
        return INFINITY;
    double valmin, dermin;
    intfg.evalDeriv(loghmin, &valmin, &dermin);
    if(logcm <= valmin) {
        // find the root (logh) using linear extrapolation:
        // log(cm) = intfg(logh) = valmin + dermin * (logh - loghmin)
        return exp((logcm - valmin) / dermin + loghmin);
    }
    return exp(findRoot(RootFinder(intfg, logcm), loghmin, loghmax, ACCURACY));
}


std::vector<double> sampleSphericalDF(const DiffusionCoefs& model, unsigned int npoints)
{
    std::vector<double> result(npoints);
    double totalMass = model.cumulMass();
    for(unsigned int i=0; i<npoints; i++)
        result[i] = model.findh(totalMass * math::random());
    return result;
}

}; // namespace
