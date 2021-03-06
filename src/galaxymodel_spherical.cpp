#include "galaxymodel_spherical.h"
#include "math_core.h"
#include "math_specfunc.h"
#include "math_sample.h"
#include "utils.h"
#include "potential_multipole.h"  // used in Fokker-Planck
#include "potential_composite.h"  // used in Fokker-Planck
#include <cmath>
#include <algorithm>
#include <cassert>
#include <stdexcept>
//debugging output
#include <fstream>

namespace galaxymodel{

namespace{

/// default grid spacing in log radius or log phase volume
static const double DELTALOG = 0.125;

/// required tolerance for the root-finder
static const double EPSROOT  = 1e-6;

/// tolerance on the 2nd derivative of a function of phase volume for grid generation
static const double EPSDER2  = 1e-6;

/// fixed order of Gauss-Legendre quadrature
static const int GLORDER = 8;

/// lower limit on the value of density or DF to be considered seriously
static const double MIN_VALUE = 1e-100;

/** helper function for finding the slope of asymptotic power-law behaviour of a certain function:
    if  f(x) ~ f0 * (1 + A * x^B)  as  x --> 0  or  x --> infinity,  then the slope B is given by
    solving the equation  [x0^B - x1^B] / [x1^B - x2^B] = [f(x0) - f(x1)] / [f(x1) - f(x2)],
    where x0, x1 and x2 are three consecutive points near the end of the interval.
    The arrays of log(x) and corresponding f(x) are passed as parameters to this function,
    and its value() method is used in the root-finding routine.
*/
class SlopeFinder: public math::IFunctionNoDeriv {
    const double logx0, logx1, logx2, ratio;
public:
    SlopeFinder(double _logx0, double _logx1, double _logx2, double f0, double f1, double f2) :
        logx0(_logx0), logx1(_logx1), logx2(_logx2), ratio( (f0-f1) / (f1-f2) ) {}

    virtual double value(const double B) const {
        if(B==0)
            return (logx0 - logx1) / (logx1 - logx2) - ratio;
        if(B==INFINITY || B==-INFINITY)  // in either case, assume that the ratio of (x0/x1)^B vanishes
            return -ratio;
        double x0B = exp(B*logx0), x1B = exp(B*logx1), x2B = exp(B*logx2);
        return (x0B - x1B) / (x1B - x2B) - ratio;
    }
};

/// helper class for interpolating the density as a function of phase volume,
/// used in the Eddington inversion routine,
/// optionally with an asymptotic expansion for a constant-density core.
class DensityInterp: public math::IFunction {
    const std::vector<double> &logh, &logrho;
    const math::CubicSpline spl;
    double logrho0, A, B, loghmin;

    // evaluate log(rho) and its derivatives w.r.t. log(h) using the asymptotic expansion
    void asympt(const double logh, /*output*/ double* logrho, double* dlogrho, double* d2logrho) const
    {
        double AhB = A * exp(B * logh);
        if(logrho)
            *logrho   = logrho0 + log(1 + AhB);
        if(dlogrho)
            *dlogrho  = B / (1 + 1 / AhB);
        if(d2logrho)
            *d2logrho = pow_2(B / (1 + AhB)) * AhB;
    }

public:
    // initialize the spline for log(rho) as a function of log(h),
    // and check if the density is approaching a constant value as h-->0;
    /// if it does, then determines the coefficients of its asymptotic expansion
    /// rho(h) = rho0 * (1 + A * h^B)
    DensityInterp(const std::vector<double>& _logh, const std::vector<double>& _logrho) :
        logh(_logh), logrho(_logrho),
        spl(logh, logrho),
        logrho0(-INFINITY), A(0), B(0), loghmin(-INFINITY)
    {
        if(logh.size()<4)
            return;
        // try determining the exponent B from the first three grid points
        double B0 = math::findRoot(
            SlopeFinder(logh[0], logh[1], logh[2], logrho[0], logrho[1], logrho[2]),
            0, INFINITY, EPSROOT);
        if(!(B0 > 0.05 && B0 < 20))
            return;
        // now try the same using three points shifted by one
        double B1 = math::findRoot(
            SlopeFinder(logh[1], logh[2], logh[3], logrho[1], logrho[2], logrho[3]),
            0, INFINITY, EPSROOT);
        if(!(B1 > 0.05 && B1 < 20))
            return;
        // consistency check - if the two values differ significantly, we're getting nonsense
        if(B0 < B1*0.95 || B1 < B0*0.95)
            return;
        B = B0;
        A = (logrho[0] - logrho[1]) / (exp(B*logh[0]) - exp(B*logh[1]));
        logrho0 = logrho[0] - A * exp(B*logh[0]);
        if(!isFinite(logrho0) || fabs(logrho[0]-logrho0) > 0.1)
            return;

        // now need to determine the critical value of log(h)
        // below which we will use the asymptotic expansion
        for(unsigned int i=1; i<logh.size(); i++) {
            loghmin = logh[i];
            double corelogrho, coredlogrho, cored2logrho;  // values returned by the asymptotic expansion
            asympt(logh[i], &corelogrho, &coredlogrho, &cored2logrho);
            if(!(fabs((corelogrho-logrho[i]) / (corelogrho-logrho0)) < 0.01)) {
                utils::msg(utils::VL_VERBOSE, "makeEddingtonDF",
                    "Density core: rho="+utils::toString(exp(logrho0))+"*(1"+(A>0?"+":"")+
                    utils::toString(A)+"*h^"+utils::toString(B)+") at h<"+utils::toString(exp(loghmin)));
                break;   // TODO: come up with a more rigorous method...
            }
        }
    }

    /// returns the interpolated value for log(rho) as a function of log(h),
    /// together with its two derivatives.
    /// It automatically switches to the asymptotic expansion if necessary.
    virtual void evalDeriv(const double logh,
        /*output*/ double* logrho, double* dlogrho, double* d2logrho) const
    {
        if(logh < loghmin)
            asympt(logh, logrho, dlogrho, d2logrho);
        else
            spl.evalDeriv(logh, logrho, dlogrho, d2logrho);
    }

    virtual unsigned int numDerivs() const { return 2; }
};

/// integrand for computing the product of f(E) and a weight function (E-Phi)^{P/2}
template<int P>
class DFIntegrand: public math::IFunctionNoDeriv {
    const math::IFunction &df;
    const potential::PhaseVolume &pv;
    const double logh0;
public:
    DFIntegrand(const math::IFunction& _df, const potential::PhaseVolume& _pv, double _logh0) :
        df(_df), pv(_pv), logh0(_logh0) {}

    virtual double value(const double logh) const
    {
        double h = exp(logh), g, w = sqrt(pv.deltaE(logh, logh0, &g));
        // the original integrals are formulated in terms of  \int f(E) weight(E) dE,
        // and we replace  dE  by  d(log h) * [ dh / d(log h) ] / [ dh / dE ],
        // that's why there are extra factors h and 1/g below.
        return df(h) * h / g * math::powInt(w, P);
    }
};


/** helper class for integrating or sampling the isotropic DF in a given spherical potential */
class DFSphericalIntegrand: public math::IFunctionNdim {
    const math::IFunction& pot;
    const math::IFunction& df;
    const potential::PhaseVolume pv;
public:
    DFSphericalIntegrand(const math::IFunction& _pot, const math::IFunction& _df) :
        pot(_pot), df(_df), pv(pot) {}
    
    /// un-scale r and v and return the jacobian of this transformation
    double unscalerv(double scaledr, double scaledv, double& r, double& v, double& Phi) const { 
        r   = exp( 1/(1-scaledr) - 1/scaledr );
        Phi = pot(r);
        double vesc = sqrt(-2*Phi);
        v   = scaledv * vesc;
        return pow_2(4*M_PI) * pow_3(r*vesc) * (1/pow_2(1-scaledr) + 1/pow_2(scaledr)) * pow_2(scaledv);
    }
    virtual void eval(const double vars[], double values[]) const
    {
        double r, v, Phi;
        double jac = unscalerv(vars[0], vars[1], r, v, Phi);
        values[0] = 0;
        if(isFinite(jac) && jac>1e-100 && jac<1e100) {
            double f = df(pv(Phi + 0.5 * v*v));
            if(isFinite(f))
                values[0] = f * jac;
        }
    }
    virtual unsigned int numVars()   const { return 2; }
    virtual unsigned int numValues() const { return 1; }
};


/** helper class for setting up a grid for log(rho) in log(h) used in the Eddington inversion */
class LogRhoOfLogH: public math::IFunction {
    const math::IFunction& density;
    const potential::PhaseVolume& phasevol;
    const math::IFunction& pot;
public:
    LogRhoOfLogH(const math::IFunction& _density,
        const potential::PhaseVolume& _phasevol, const math::IFunction& _pot) :
        density(_density), phasevol(_phasevol), pot(_pot) {}

    virtual void evalDeriv(double logh, double* logrho, double* der, double* der2) const
    {
        double h = exp(logh), g, dgdh;
        double E = phasevol.E(h, &g, &dgdh);
        double r = R_max(potential::FunctionToPotentialWrapper(pot), E);
        double dPhidr, d2Phidr2;
        pot.evalDeriv(r, NULL, &dPhidr, &d2Phidr2);
        math::PointNeighborhood rho(density, r);
        // now we have the full transformation chain h -> E -> r -> rho,
        // with two derivatives at each stage, and will combine them to obtain
        // log(rho) and its derivatives w.r.t. log(h)
        double drhodh = rho.fder / dPhidr / g;
        if(logrho)
            *logrho = log(rho.f0);
        if(der)
            *der  = drhodh * h / rho.f0;
        if(der2) {
            double d2rhodh2 = rho.fder2 / pow_2(dPhidr * g) -
                rho.fder / (g*g * dPhidr) * (d2Phidr2 / pow_2(dPhidr) + dgdh);
            *der2 = (d2rhodh2 * h + (1 - drhodh * h / rho.f0) * drhodh) * h / rho.f0;
        }
    }

    virtual unsigned int numDerivs()   const { return 2; }
};

}  // internal namespace

//---- Eddington inversion ----//

void makeEddingtonDF(const math::IFunction& density, const math::IFunction& potential,
    /* input/output */ std::vector<double>& gridh, /* output */ std::vector<double>& gridf)
{
    // 1. construct a phase-volume interpolator
    potential::PhaseVolume phasevol(potential);

    // 2. prepare grids
    std::vector<double> gridlogh, gridPhi, gridlogrho;
    if(gridh.empty()) {   // no input: estimate the grid extent
        gridlogh = math::createInterpolationGrid(
            LogRhoOfLogH(density, phasevol, potential), EPSDER2);
    } else {              // input grid in h was provided
        gridlogh.resize(gridh.size());
        std::transform(gridh.begin(), gridh.end(), gridlogh.begin(), log);
    }

    // 2b. store the values of h, Phi and rho at the nodes of a grid
    potential::FunctionToPotentialWrapper potw(potential);
    for(unsigned int i=0; i<gridlogh.size();) {
        double Phi = phasevol.E(exp(gridlogh[i]));
        double R   = R_max(potw, Phi);
        double rho = density(R);
        if(rho > MIN_VALUE) {
            gridPhi.push_back(Phi);
            gridlogrho.push_back(log(rho));
            i++;
        } else {
            gridlogh.erase(gridlogh.begin()+i);
        }
    }
    unsigned int gridsize = gridlogh.size();
    if(gridsize < 3)
        throw std::runtime_error("makeEddingtonDF: invalid grid size");
    gridf.resize(gridsize);   // f(h_i) = int_{h[i]}^{infinity}    
    gridh.resize(gridsize);
    std::transform(gridlogh.begin(), gridlogh.end(), gridh.begin(), exp);
    
    // 3. construct a spline for log(rho) as a function of log(h),
    // optionally with an asymptotic expansion in the case of a constant-density core
    DensityInterp densityInterp(gridlogh, gridlogrho);

    // 4. compute the integrals for f(h) at each point in the h-grid.
    // The conventional Eddington inversion formula is
    // f(E) = 1/(\sqrt{8}\pi^2)  \int_{E(h)}^0  d\Phi  (d^2\rho / d\Phi^2)  /  \sqrt{\Phi-E} ,
    // we replace the integration variable by \ln h, and the density is also expressed as
    // \rho(h), or, rather, \ln \rho ( \ln h ), in the form of a spline constructed above.
    // Therefore, the expression for the Eddington integral changes to a somewhat lengthier one:
    // f(h) = 1/(\sqrt{8}\pi^2)  \int_{\ln h}^\infty  d \ln h'  /  \sqrt{\Phi(h') - \Phi(h)} *
    // ( d2rho(h') +  drho(h') * ( (drho(h') - 1) g(h') / h'  + dg(h')/dh') ) * \rho(h'),
    // where  drho = d(\ln\rho) / d(\ln h), d2rho = d^2(\ln\rho) / d(\ln h)^2, computed at h'.
    // These integrals for each value of h on the grid are computed by summing up
    // contributions from all segments of h-grid above the current one,
    // and on each segment we use a Gauss-Legendre quadrature with GLORDER points.
    // We start from the last interval from h_max to infinity, where the integrals are computed
    // analytically, and then proceed to lower values of h, each time accumulating the contribution
    // of the current segment to all integrals which contain this segment.
    // As a final remark, the integration variable on each segment is not \ln h', but rather
    // an auxiliary variable y, defined such that \ln h = \ln h[i-1] + (\ln h[i] - \ln h[i-1]) y^2.
    // This eliminates the singularity in the term  1 / \sqrt{\Phi(h')-\Phi(h)}  when h=h[i-1]. 

    // 4a. integrate from log(h_max) = logh[gridsize-1] to infinity, or equivalently, 
    // from Phi(r_max) to 0, assuming that Phi(r) ~ -1/r and h ~ r^(3/2) in the asymptotic regime.
    // First we find the slope d(logrho)/d(logh) at h_max
    double logrho, dlogrho, d2logrho;
    densityInterp.evalDeriv(gridlogh.back(), &logrho, &dlogrho, &d2logrho);
    double slope = -1.5*dlogrho;  // density at large radii is ~ r^(-slope)
    utils::msg(utils::VL_VERBOSE, "makeEddingtonDF",
        "Density is ~r^"+utils::toString(-slope)+" at large r");
    double mult  = 0.5 / M_SQRT2 / M_PI / M_PI;
    // next we compute analytically the values of all integrals for f(h_j) on the segment h_max..infinity
    for(unsigned int j=0; j<gridsize; j++) {
        double factor = j==gridsize-1 ?
            (slope-1) * M_SQRTPI * math::gamma(slope-1) / math::gamma(slope-0.5) :
            math::hypergeom2F1(0.5, slope-1, slope, gridPhi.back() / gridPhi[j]);
        if(isFinite(factor))
            gridf[j] = mult * slope * exp(logrho) / -gridPhi.back() * factor / sqrt(-gridPhi[j]);
        // do nothing (keep f=0) in case of arithmetic overflow (outer slope too steep)
    }

    // 4b. prepare integration tables
    double glnodes[GLORDER], glweights[GLORDER];
    math::prepareIntegrationTableGL(0, 1, GLORDER, glnodes, glweights);

    // 4c. integrate from logh[i-1] to logh[i] for all i=1..gridsize-1
    for(unsigned int i=gridsize-1; i>0; i--)
    {
        for(int k=0; k<GLORDER; k++)
        {
            // node of Gauss-Legendre quadrature within the current segment (logh[i-1] .. logh[i]);
            // the integration variable y ranges from 0 to 1, and logh(y) is defined below
            double y = glnodes[k];
            double logh = gridlogh[i-1] + (gridlogh[i]-gridlogh[i-1]) * y*y;
            // compute E, g, dg/dh at the current point h (GL node)
            double h = exp(logh), g, dgdh;
            double E = phasevol.E(h, &g, &dgdh);
            // compute log[rho](log[h]) and its derivatives
            densityInterp.evalDeriv(logh, &logrho, &dlogrho, &d2logrho);
            // GL weight -- contribution of this point to each integral on the current segment,
            // taking into account the transformation of variable y -> logh
            double weight = glweights[k] * 2*y * (gridlogh[i]-gridlogh[i-1]);
            // common factor for all integrals - the derivative d^2\rho / d\Phi^2,
            // expressed in scaled and transformed variables
            double factor = (d2logrho * g / h + dlogrho * ( (dlogrho-1) * g / h + dgdh) ) * exp(logrho);
            // now add a contribution to the integral expressing f(h_j) for all h_j <= h[i-1]
            for(unsigned int j=0; j<i; j++) {
                double denom2 = E - gridPhi[j];
                if(denom2 < fabs(gridPhi[i]) * 1e-8)              // loss of precision is possible:
                    denom2 = phasevol.deltaE(logh, gridlogh[j]);  // use a more accurate expression
                gridf[j] += mult * weight * factor / sqrt(denom2);
            }
        }
    }

    // 5. check validity and remove negative values
    bool hasNegativeF = false;
    for(unsigned int i=0; i<gridf.size();) {
        if(gridf[i] <= MIN_VALUE) {
            hasNegativeF |= gridf[i]<0;
            gridf.erase(gridf.begin() + i);
            gridh.erase(gridh.begin() + i);
        } else
            i++;
    }
    if(hasNegativeF)
        utils::msg(utils::VL_WARNING, "makeEddingtonDF", "Distribution function is negative");
    if(gridf.size() < 2)
        throw std::runtime_error("makeEddingtonDF: could not construct a valid non-negative DF");

    // also check that the remaining positive values describe a DF that doesn't grow too fast as h-->0
    // and decays fast enough as h-->infinity;  this still does not guarantee that it's reasonable,
    // but at least allows to construct a valid interpolator
    do {
        math::LogLogSpline spl(gridh, gridf);
        double derIn, derOut;
        spl.evalDeriv(gridh.front(), NULL, &derIn);
        spl.evalDeriv(gridh.back(),  NULL, &derOut);
        double slopeIn  = gridh.front() / gridf.front() * derIn;
        double slopeOut = gridh.back()  / gridf.back()  * derOut;
        if(slopeIn > -1 && slopeOut < -1) {
            utils::msg(utils::VL_VERBOSE, "makeEddingtonDF",
                "f(h) ~ h^"+utils::toString(slopeIn)+
                " at small h and ~ h^"+utils::toString(slopeOut)+" at large h");
            // results are returned in the two arrays, gridh and gridf
            return;
        }
        // otherwise remove the innermost and/or outermost point and repeat
        if(slopeIn < -1) {
            gridf.erase(gridf.begin());
            gridh.erase(gridh.begin());
        }
        if(slopeOut > -1) {
            gridf.erase(gridf.end()-1);
            gridh.erase(gridh.end()-1);
        }
    } while(gridf.size() > 2);
    throw std::runtime_error("makeEddingtonDF: could not construct a valid non-negative DF");
}


//---- Construction of f(h) from an N-body snapshot ----//

math::LogLogSpline fitSphericalDF(
    const std::vector<double>& hvalues, const std::vector<double>& masses, unsigned int gridSize)
{
    const unsigned int nbody = hvalues.size();
    if(masses.size() != nbody)
        throw std::invalid_argument("fitSphericalDF: array sizes are not equal");

    // 1. collect the log-scaled values of phase volume
    std::vector<double> logh(nbody);
    for(unsigned int i=0; i<nbody; i++) {
        logh[i] = log(hvalues[i]);
        if(!isFinite(logh[i]+masses[i]) || masses[i]<0)
            throw std::invalid_argument("fitSphericalDF: incorrect input data");
    }

    // 2. create a reasonable grid in log(h), with almost uniform spacing subject to the condition
    // that each segment contains at least "a few" particles (weakly depending on their total number)
    const int minParticlesPerBin  = std::max(1, static_cast<int>(log(nbody+1)/log(2)));
    std::vector<double> gridh = math::createAlmostUniformGrid(gridSize+2, logh, minParticlesPerBin);
    utils::msg(utils::VL_DEBUG, "fitSphericalDF",
        "Grid in h=["+utils::toString(exp(gridh[1]))+":"+utils::toString(exp(gridh[gridh.size()-2]))+"]"
        ", particles span h=["+utils::toString(exp(gridh[0]))+":"+utils::toString(exp(gridh.back()))+"]");
    gridh.erase(gridh.begin());
    gridh.pop_back();

    // 3a. perform spline log-density fit, and
    // 3b. initialize a cubic spline for log(f) as a function of log(h)
    math::CubicSpline fitfnc(gridh,
        math::splineLogDensity<3>(gridh, logh, masses,
        math::FitOptions(math::FO_INFINITE_LEFT | math::FO_INFINITE_RIGHT | math::FO_PENALTY_3RD_DERIV)));

    // 4. store the values of cubic spline at grid nodes, together with two endpoint derivatives --
    // this data is sufficient to reconstruct it exactly later in the LogLogSpline constructor
    double derLeft, derRight;
    fitfnc.evalDeriv(gridh.front(), NULL, &derLeft);
    fitfnc.evalDeriv(gridh.back(),  NULL, &derRight);
    assert(derLeft > 0 && derRight < 0);  // a condition for a valid fit (total mass should be finite)
    std::vector<double> gridf(gridh.size());
    for(unsigned int i=0; i<gridh.size(); i++) {
        double h = exp(gridh[i]);
        // the fit provides log( dM/d(log h) ) = log( h dM/dh ) = log( h f(h) )
        gridf[i] = exp(fitfnc(gridh[i])) / h;
        gridh[i] = h;
    }
    // endpoint derivatives of fitfnc are  d [log (h f(h) ) ] / d [log h] = 1 + d [log f] / d [log h],
    // thus the derivatives provided to LogLogSpline are df/dh = (f/h) d[log f] / d[log h]
    derLeft  = (derLeft -1) * gridf.front() / gridh.front();
    derRight = (derRight-1) * gridf.back () / gridh.back ();

    // 5. construct an interpolating spline that matches exactly our fitfnc (it's also a cubic spline
    // in the same scaled variables), including the correct slopes for extrapolation outside the grid
    return math::LogLogSpline(gridh, gridf, derLeft, derRight);
}


particles::ParticleArraySph generatePosVelSamples(
    const math::IFunction& pot, const math::IFunction& df, const unsigned int numPoints)
{
    DFSphericalIntegrand fnc(pot, df);
    math::Matrix<double> result;  // sampled scaled coordinates/velocities
    double totalMass, errorMass;  // total normalization of the DF and its estimated error
    double xlower[6] = {0,0};     // boundaries of sampling region in scaled coordinates
    double xupper[6] = {1,1};
    math::sampleNdim(fnc, xlower, xupper, numPoints, result, NULL, &totalMass, &errorMass);
    const double pointMass = totalMass / result.rows();
    particles::ParticleArraySph points;
    points.data.reserve(result.rows());
    for(unsigned int i=0; i<result.rows(); i++) {
        double r, v, Phi,
            rtheta    = acos(math::random()*2-1),
            rphi      = 2*M_PI * math::random(),
            vcostheta = math::random()*2-1,
            vsintheta = sqrt(1-pow_2(vcostheta)),
            vphi      = 2*M_PI * math::random();
        fnc.unscalerv(result(i, 0), result(i, 1), r, v, Phi);
        points.add(coord::PosVelSph(r, rtheta, rphi,
            v*vsintheta*cos(vphi), v*vsintheta*sin(vphi), v*vcostheta), pointMass);
    }
    return points;
}


std::vector<double> densityFromCumulativeMass(
    const std::vector<double>& gridr, const std::vector<double>& gridm)
{
    unsigned int size = gridr.size();
    if(size<3 || gridm.size()!=size)
        throw std::invalid_argument("densityFromCumulativeMass: invalid array sizes");
    // check monotonicity and convert to log-scaled radial grid
    std::vector<double> gridlogr(size), gridlogm(size), gridrho(size);
    for(unsigned int i=0; i<size; i++) {
        if(!(gridr[i] > 0 && gridm[i] > 0))
            throw std::invalid_argument("densityFromCumulativeMass: negative input values");
        if(i>0 && (gridr[i] <= gridr[i-1] || gridm[i] <= gridm[i-1]))
            throw std::invalid_argument("densityFromCumulativeMass: arrays are not monotonic");
        gridlogr[i] = log(gridr[i]);
    }
    // determine if the cumulative mass approaches a finite limit at large radii,
    // that is, M = Minf - A * r^B  with A>0, B<0
    double B = math::findRoot(SlopeFinder(
        gridlogr[size-1], gridlogr[size-2], gridlogr[size-3],
        gridm   [size-1], gridm   [size-2], gridm   [size-3] ), -INFINITY, 0, EPSROOT);
    double invMinf = 0;  // 1/Minf, or remain 0 if no finite limit is detected
    if(B<0) {
        double A =  (gridm[size-1] - gridm[size-2]) / 
            (exp(B * gridlogr[size-2]) - exp(B * gridlogr[size-1]));
        if(A>0) {  // viable extrapolation
            invMinf = 1 / (gridm[size-1] + A * exp(B * gridlogr[size-1]));
            utils::msg(utils::VL_VERBOSE, "densityFromCumulativeMass",
                "Extrapolated total mass=" + utils::toString(1/invMinf) +
                ", rho(r)~r^" + utils::toString(B-3) + " at large radii" );
        }
    }
    // scaled mass to interpolate:  log[ M / (1 - M/Minf) ] as a function of log(r),
    // which has a linear asymptotic behaviour with slope -B as log(r) --> infinity;
    // if Minf = infinity, this additional term has no effect
    for(unsigned int i=0; i<size; i++)
        gridlogm[i] = log(gridm[i] / (1 - gridm[i]*invMinf));
    math::CubicSpline spl(gridlogr, gridlogm);
    if(!spl.isMonotonic())
        throw std::runtime_error("densityFromCumulativeMass: interpolated mass is not monotonic");
    // compute the density at each point of the input radial grid
    for(unsigned int i=0; i<size; i++) {
        double val, der;
        spl.evalDeriv(gridlogr[i], &val, &der);
        val = exp(val);
        gridrho[i] = der * val / (4*M_PI * pow_3(gridr[i]) * pow_2(1 + val * invMinf));
        if(gridrho[i] <= 0)   // shouldn't occur if the spline is (strictly) monotonic
            throw std::runtime_error("densityFromCumulativeMass: interpolated density is non-positive");
    }
    return gridrho;
}


std::vector<double> computeDensity(const math::IFunction& df, const potential::PhaseVolume& pv,
    const std::vector<double> &gridPhi)
{
    unsigned int gridsize = gridPhi.size();
    std::vector<double> result(gridsize);
    // we assume that the grid in Phi is monotonic and sufficiently dense!
    double glnodes[GLORDER], glweights[GLORDER];
    math::prepareIntegrationTableGL(0, 1, GLORDER, glnodes, glweights);
    for(unsigned int i=0; i<gridsize; i++) {
        double deltaPhi = (i<gridsize-1 ? gridPhi[i+1] : 0) - gridPhi[i];
        for(int k=0; k<GLORDER; k++) {
            // node of Gauss-Legendre quadrature within the current segment (Phi[i] .. Phi[i+1]);
            // the integration variable y ranges from 0 to 1, and Phi(y) is defined below
            double y   = glnodes[k];
            double Phi = gridPhi[i] + y*y * deltaPhi;
            // contribution of this point to each integral on the current segment, taking into account
            // the transformation of variable y -> Phi, multiplied by the value of f(h(Phi))
            double weight = glweights[k] * 2*y * deltaPhi * df(pv(Phi)) * (4*M_PI*M_SQRT2);
            // add a contribution to the integrals expressing rho(Phi[j]) for all Phi[j] < Phi
            for(unsigned int j=0; j<=i; j++) {
                double v   = sqrt(fmax(0, Phi - gridPhi[j]));
                result[j] += weight * v;
            }
        }
    }
    return result;
}


//---- Spherical model specified by a DF f(h) and phase volume h(E) ----//

SphericalModel::SphericalModel(const potential::PhaseVolume& _phasevol, const math::IFunction& df) :
    phasevol(_phasevol)
{
    // 1. determine the range of h that covers the region of interest
    // and construct the grid in log[h(Phi)]
    std::vector<double> gridLogH = math::createInterpolationGrid(math::LogLogScaledFnc(df), EPSDER2);

    // 2. store the values of f, g, h at grid nodes (ensure to consider only positive values of f)
    std::vector<double> gridF, gridG, gridH;
    for(unsigned int i=0; i<gridLogH.size();) {
        double h = exp(gridLogH[i]);
        double f = df(h);
        if(f > MIN_VALUE) {
            gridF.push_back(f);
            gridH.push_back(h);
            gridG.push_back(0);
            phasevol.E(h, &gridG.back());
            i++;
        } else {
            gridLogH.erase(gridLogH.begin()+i);
        }
    }
    const unsigned int npoints = gridLogH.size();
    std::vector<double> gridFint(npoints), gridFGint(npoints), gridFHint(npoints);

    // 3a. determine the asymptotic behaviour of f(h):
    // f(h) ~ h^outerFslope as h-->inf  or  h^innerFslope as h-->0
    double innerFslope = log(gridF[1] / gridF[0]) / (gridLogH[1] - gridLogH[0]);
    double outerFslope = log(gridF[npoints-1] / gridF[npoints-2]) /
        (gridLogH[npoints-1] - gridLogH[npoints-2]);
    if(!(innerFslope > -1))
        throw std::runtime_error("SphericalModel: f(h) rises too rapidly as h-->0");
    if(!(outerFslope < -1))
        throw std::runtime_error("SphericalModel: f(h) falls off too slowly as h-->infinity");

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
        throw std::runtime_error("SphericalModel: weird behaviour of potential");
    if(Phi0 != -INFINITY)   // determination of inner slope depends on whether the potential is finite
        innerE -= Phi0;
    double innerEslope = gridH.front() / gridG.front() / innerE;
    double outerEslope = gridH.back()  / gridG.back()  / outerE;
    double outerRatio  = outerFslope  / outerEslope;
    if(!(outerRatio > 0 && innerEslope + innerFslope > -1))
        throw std::runtime_error("SphericalModel: weird asymptotic behaviour of phase volume");

    // 4. compute integrals
    // \int f(E) dE        = \int f(h) / g(h) h d(log h),
    // \int f(E) g(E) dE   = \int f(h) h d(log h),
    // \int f(E) h(E) dE   = \int f(h) / g(h) h^2 d(log h),
    // \int f(E) g(E) E dE = \int f(h) E h d(log h),
    // the last one starts from an analytically calculated contribution on the segment (0..h[0])
    totalEnergy = gridF[0] * gridH[0] * (innerEslope >= 0 ?
        Phi0   / (1 + innerFslope) :
        innerE / (1 + innerFslope + innerEslope) );

    // 4a. integrate over all interior segments
    double glnodes[GLORDER], glweights[GLORDER];
    math::prepareIntegrationTableGL(0, 1, GLORDER, glnodes, glweights);
    for(unsigned int i=1; i<npoints; i++) {
        for(int k=0; k<GLORDER; k++) {
            // node of Gauss-Legendre quadrature within the current segment (logh[i-1] .. logh[i]);
            double logh = gridLogH[i-1] + (gridLogH[i]-gridLogH[i-1]) * glnodes[k];
            // GL weight -- contribution of this point to each integral on the current segment
            double weight = glweights[k] * (gridLogH[i]-gridLogH[i-1]);
            // compute E, f, g, h at the current point h (GL node)
            double h = exp(logh), g, E = phasevol.E(h, &g);
            // the original integrals are formulated in terms of  \int f(E) weight(E) dE,
            // where weight = 1, g, h for the three integrals,
            // and we replace  dE  by  d(log h) * [ dh / d(log h) ] / [ dh / dE ],
            // that's why there are extra factors h and 1/g below.
            double integrand = df(h) * h * weight;
            gridFint[i-1] += integrand / g;
            gridFGint[i]  += integrand;
            gridFHint[i]  += integrand / g * h;
            totalEnergy   += integrand * E;
        }
    }

    // 4b. integral of f(h) dE = f(h) / g(h) dh -- compute from outside in,
    // summing contributions from all intervals of h above its current value
    // the outermost segment from h_max to infinity is integrated analytically
    gridFint.back() = -gridF.back() * outerE / (1 + outerRatio);
    for(int i=npoints-1; i>=1; i--) {
        gridFint[i-1] += gridFint[i];
    }

    // 4c. integrands of f*g dE  and  f*h dE;  note that g = dh/dE.
    // compute from inside out, summing contributions from all previous intervals of h
    // integrals over the first segment (0..gridH[0]) are computed analytically
    gridFGint[0] = gridF[0] * gridH[0] / (1 + innerFslope);
    gridFHint[0] = gridF[0] * pow_2(gridH[0]) / gridG[0] / (1 + innerEslope + innerFslope);
    for(unsigned int i=1; i<npoints; i++) {
        gridFGint[i] += gridFGint[i-1];
        gridFHint[i] += gridFHint[i-1];
    }
    // add the contribution of integrals from the last grid point up to infinity (very small anyway)
    gridFGint.back() -= gridF.back() * gridH.back() / (1 + outerFslope);
    gridFHint.back() -= gridF.back() * pow_2(gridH.back()) / gridG.back() / (1 + outerEslope + outerFslope);
    totalMass = gridFGint.back();

    // 5. construct 1d interpolating splines for these integrals
    // 5a. log-scale the computed values and prepare derivatives for quintic spline
    std::vector<double> gridFder(npoints), gridFGder(npoints), gridFHder(npoints);
    for(unsigned int i=0; i<npoints; i++) {
        gridFder [i] = gridH[i] / -gridFint[i] * gridF[i] / gridG[i];
        gridFGder[i] = gridH[i] / gridFGint[i] * gridF[i];
        gridFHder[i] = gridH[i] / gridFHint[i] * gridF[i] * gridH[i] / gridG[i];
        gridFint [i] = log(gridFint[i]);
        gridFGint[i] = log(gridFGint[i]);
        gridFHint[i] = log(gridFHint[i]);
        if(!(gridFder[i]<=0 && gridFGder[i]>=0 && gridFHder[i]>=0 && 
            isFinite(gridFint[i] + gridFGint[i] + gridFHint[i])))
            throw std::runtime_error("SphericalModel: cannot construct valid interpolators");
    }
    // integrals of f*g and f*h have finite limit as h-->inf;
    // extrapolate them as constants beyond the last grid point
    gridFGder.back() = gridFHder.back() = 0;

    // 5b. initialize splines for log-scaled integrals
    intf  = math::QuinticSpline(gridLogH, gridFint,  gridFder);
    intfg = math::QuinticSpline(gridLogH, gridFGint, gridFGder);
    intfh = math::QuinticSpline(gridLogH, gridFHint, gridFHder);
}

double SphericalModel::value(const double h) const
{
    // intfg represents log-scaled  \int_0^h f(h) dh
    double logh = log(h), val, der, g;
    intfg.evalDeriv(logh, &val, &der);
    double mass = exp(val);
    // at large h, intfg reaches a limit (totalMass), thus its derivative may be inaccurate
    if(mass < totalMass * 0.9999 && der > 0)
        return der * mass / h;   // still ok
    // otherwise we compute it from a different spline which tends to zero at large h
    intf.evalDeriv(logh, &val, &der);
    phasevol.E(h, &g);
    return -der * exp(val) / h * g;
}

double SphericalModel::I0(const double logh) const
{
    return exp(intf(logh));
}

double SphericalModel::cumulMass(const double logh) const
{
    if(logh==INFINITY)
        return totalMass;
    return exp(intfg(logh));
}

double SphericalModel::cumulEkin(const double logh) const
{
    return 1.5 * exp(intfh(logh));
}


DiffusionCoefs::DiffusionCoefs(const potential::PhaseVolume& phasevol, const math::IFunction& df) :
    model(phasevol, df)
{
    // 1. determine the range of h that covers the region of interest
    // and construct the grid in X = log[h(Phi)] and Y = log[h(E)/h(Phi)]
    std::vector<double> gridLogH = math::createInterpolationGrid(math::LogLogScaledFnc(df), EPSDER2);
    const double logHmin         = gridLogH.front(),  logHmax = gridLogH.back();
    const unsigned int npoints   = gridLogH.size();
    const unsigned int npointsY  = 100;
    const double mindeltaY       = fmin(0.1, (logHmax-logHmin)/npointsY);
    std::vector<double> gridY    = math::createNonuniformGrid(npointsY, mindeltaY, logHmax-logHmin, true);

    // 3. determine the asymptotic behaviour of f(h) and g(h):
    // f(h) ~ h^outerFslope as h-->inf and  g(h) ~ h^(1-outerEslope)
    double outerH = exp(gridLogH.back()), outerG, outerE = phasevol.E(outerH, &outerG);
    double outerFslope = log(df(outerH) / df(exp(gridLogH[npoints-2]))) /
        (gridLogH[npoints-1] - gridLogH[npoints-2]);
    if(!(outerFslope < -1))
        throw std::runtime_error("DiffusionCoefs: f(h) falls off too slowly as h-->infinity");
    double outerEslope = outerH / outerG / outerE;
    double outerRatio  = outerFslope / outerEslope;
    if(!(outerRatio > 0))   // TODO: this may happen if f(h_out)=0 which is a valid case (?)
        throw std::runtime_error("DiffusionCoefs: weird asymptotic behaviour of phase volume");

    // 5. construct 2d interpolating splines for dv2par, dv2per as functions of Phi and E

    // 5a. asymptotic values for J1/J0 and J3/J0 as Phi --> 0 and (E/Phi) --> 0
    double outerJ1 = 0.5*M_SQRTPI * math::gamma(2 + outerRatio) / math::gamma(2.5 + outerRatio);
    double outerJ3 = outerJ1 * 1.5 / (2.5 + outerRatio);

    // 5b. compute the values of J1/J0 and J3/J0 at nodes of 2d grid in X=log(h(Phi)), Y=log(h(E)/h(Phi))
    math::Matrix<double> gridv2par(npoints, npointsY), gridv2per(npoints, npointsY);
    for(unsigned int i=0; i<npoints; i++)
    {
        // The first coordinate of the grid is X = log(h(Phi)), the second is Y = log(h(E)) - X.
        // For each pair of values of X and Y, we compute the following integrals:
        // J_n = \int_\Phi^E f(E') [(E'-\Phi) / (E-\Phi)]^{n/2}  dE';  n = 0, 1, 3.
        // Then the value of 2d interpolants are assigned as
        // \log[ J3 / J0 ], \log[ (3*J1-J3) / J0 ] .
        // In practice, we replace the integration over dE by integration over dy = d(log h),
        // and accumulate the values of modified integrals sequentially over each segment in Y.
        // Here the modified integrals are  J{n}acc = \int_X^Y f(y) (dE'/dy) (E'(y)-\Phi)^{n/2}  dy,
        // i.e., without the term [E(Y,X)-\Phi(X)]^{n/2} in the denominator,
        // which is invoked later when we assign the values to the 2d interpolants.
        double J0acc = 0, J1acc = 0, J3acc = 0;  // accumulators
        DFIntegrand<0> intJ0(df, phasevol, gridLogH[i]);
        DFIntegrand<1> intJ1(df, phasevol, gridLogH[i]);
        DFIntegrand<3> intJ3(df, phasevol, gridLogH[i]);
        gridv2par(i, 0) = log(2./5);  // analytic limiting values for Phi=E
        gridv2per(i, 0) = log(8./5);
        for(unsigned int j=1; j<npointsY; j++) {
            double logHprev = gridLogH[i] + gridY[j-1];
            double logHcurr = gridLogH[i] + gridY[j];
            if(j==1) {
                // integration over the first segment uses a more accurate quadrature rule
                // to accounting for a possible endpoint singularity at Phi=E
                J0acc = math::integrateGL(math::ScaledIntegrandEndpointSing(
                    intJ0, logHprev, logHcurr), 0, 1, GLORDER);
                J1acc = math::integrateGL(math::ScaledIntegrandEndpointSing(
                    intJ1, logHprev, logHcurr), 0, 1, GLORDER);
                J3acc = math::integrateGL(math::ScaledIntegrandEndpointSing(
                    intJ3, logHprev, logHcurr), 0, 1, GLORDER);
            } else {
                J0acc += math::integrateGL(intJ0, logHprev, logHcurr, GLORDER);
                J1acc += math::integrateGL(intJ1, logHprev, logHcurr, GLORDER);
                J3acc += math::integrateGL(intJ3, logHprev, logHcurr, GLORDER);
            }
            if(i==npoints-1) {
                // last row: analytic limiting values for Phi-->0 and any E/Phi
                double EoverPhi = exp(gridY[j] * outerEslope);  // strictly < 1
                double oneMinusJ0overI0 = pow(EoverPhi, 1+outerRatio);  // < 1
                double Fval1 = math::hypergeom2F1(-0.5, 1+outerRatio, 2+outerRatio, EoverPhi);
                double Fval3 = math::hypergeom2F1(-1.5, 1+outerRatio, 2+outerRatio, EoverPhi);
                double I0    = model.I0(gridLogH[i]);
                double sqPhi = sqrt(-outerE);
                if(isFinite(Fval1+Fval3)) {
                    J0acc = I0 * (1 - oneMinusJ0overI0);
                    J1acc = I0 * (outerJ1 - oneMinusJ0overI0 * Fval1) * sqPhi;
                    J3acc = I0 * (outerJ3 - oneMinusJ0overI0 * Fval3) * pow_3(sqPhi);
                } else {
                    // this procedure sometimes fails, since hypergeom2F1 is not very robust;
                    // in this case we simply keep the values computed by numerical integration
                    utils::msg(utils::VL_WARNING, "DiffusionCoefs", "Can't compute asymptotic value");
                }
            }
            double dv = sqrt(phasevol.deltaE(logHcurr, gridLogH[i]));
            double J1overJ0 = J1acc / J0acc / dv;
            double J3overJ0 = J3acc / J0acc / pow_3(dv);
            if(J1overJ0<=0 || J3overJ0<=0 || !isFinite(J1overJ0+J3overJ0)) {
                utils::msg(utils::VL_WARNING, "DiffusionCoefs", "Invalid value"
                    "  J0="+utils::toString(J0acc)+
                    ", J1="+utils::toString(J1acc)+
                    ", J3="+utils::toString(J3acc));
                J1overJ0 = 2./3;   // fail-safe values corresponding to E=Phi
                J3overJ0 = 2./5;
            }
            gridv2par(i, j) = log(J3overJ0);
            gridv2per(i, j) = log(3 * J1overJ0 - J3overJ0);
        }
    }

    // debugging output
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::ofstream strm("diffcoefs");
        for(unsigned int i=0; i<npoints; i++) {
            double Phi = phasevol.E(exp(gridLogH[i]));
            for(unsigned int j=0; j<npointsY; j++) {
                double E = phasevol.E(exp(gridLogH[i] + gridY[j]));
                strm << utils::pp(gridLogH[i],10) +' '+ utils::pp(gridY[j],10) +'\t'+
                utils::pp(Phi,14) +' '+ utils::pp(E,14) +'\t'+
                utils::pp(exp(gridv2par(i, j)),10) +' '+ utils::pp(exp(gridv2per(i, j)),10)+'\n';
            }
            strm << '\n';
        }
    }

    // 5c. construct the 2d splines
    intv2par = math::CubicSpline2d(gridLogH, gridY, gridv2par);
    intv2per = math::CubicSpline2d(gridLogH, gridY, gridv2per);
}

void DiffusionCoefs::evalOrbitAvg(double E, double &DeltaE, double &DeltaE2) const
{
    double h, g;
    model.phasevol.evalDeriv(E, &h, &g);
    double totalMass = model.cumulMass(),
    logh = log(h),
    IF   = model.I0(logh),
    IFG  = model.cumulMass(logh),
    IFH  = model.cumulEkin(logh) * (2./3);
    DeltaE  = 16*M_PI*M_PI * totalMass * (IF - IFG / g);
    DeltaE2 = 32*M_PI*M_PI * totalMass * (IF * h + IFH) / g;
}

void DiffusionCoefs::evalLocal(double Phi, double E, double &dvpar, double &dv2par, double &dv2per) const
{
    double loghPhi = log(model.phasevol(Phi));
    double loghE   = log(model.phasevol(E));
    if(!(Phi<0 && loghE >= loghPhi))
        throw std::invalid_argument("DiffusionCoefs: incompatible values of E and Phi");

    // compute the 1d interpolators for I0, J0
    double I0 = model.I0(loghE);
    double J0 = fmax(model.I0(loghPhi) - I0, 0);
    // restrict the arguments of 2d interpolators to the range covered by their grids
    double X = fmin(fmax(loghPhi, intv2par.xmin()), intv2par.xmax());
    double Y = fmin(fmax(loghE-loghPhi, intv2par.ymin()), intv2par.ymax());
    // compute the 2d interpolators for J1, J3
    double v2par = exp(intv2par.value(X, Y)) * J0;
    double v2per = exp(intv2per.value(X, Y)) * J0;
    if(E>=0) {  // in this case, the coefficients were computed for E=0, need to scale them to E>0
        double J1 = (v2par + v2per) / 3;
        double corr = 1 / sqrt(1 - E / Phi);  // correction factor <1
        J1    *= corr;
        v2par *= pow_3(corr);
        v2per  = 3 * J1 - v2par;
    }
    double mult = 32*M_PI*M_PI/3 * model.cumulMass();
    dvpar  = -mult * (v2par + v2per);
    dv2par =  mult * (v2par + I0);
    dv2per =  mult * (v2per + I0 * 2);
    /*if(loghPhi<X)
        utils::msg(utils::VL_WARNING, "DiffusionCoefs",
        "Extrapolating to small h: log(h(Phi))="+utils::toString(loghPhi)+
        ", log(h(E))="+utils::toString(loghE)+
        ", I0="+utils::toString(I0)+", J0="+utils::toString(J0));*/
}


// ------ Fokker-Planck solver ------ //

namespace {

// helper routine for solving the Poisson equation and constructing the spherical potential interpolator
static potential::Interpolator computePotential(
    const math::IFunction& modelDensity, const potential::PtrPotential& externalPotential,
    double rmin, double rmax, /*output*/ double& Phi0)
{
    potential::PtrPotential modelPotential =
        potential::Multipole::create(potential::FunctionToDensityWrapper(modelDensity),
        /*lmax*/ 0, /*mmax*/ 0, /*gridsize*/ 60, rmin, rmax);
    Phi0 = modelPotential->value(coord::PosCyl(0,0,0));
    if(externalPotential) {
        std::vector<potential::PtrPotential> components(2);
        components[0] = modelPotential;
        components[1] = externalPotential;
        return potential::Interpolator(potential::CompositeCyl(components));
    } else
        return potential::Interpolator(*modelPotential);
}

} // internal namespace

FokkerPlanckSolver::FokkerPlanckSolver(
    const math::IFunction& initDensity, const potential::PtrPotential& externalPotential,
    const std::vector<double>& inputgridh) :
    extPot(externalPotential),
    totalPot(computePotential(initDensity, externalPotential, 0, 0, /*diagnostic output*/ Phi0)),
    phasevol(totalPot),
    gridh(inputgridh)
{
    // construct the initial distribution function
    makeEddingtonDF(initDensity, totalPot, /*output*/ gridh, gridf);
    if(!inputgridh.empty()) {
        // a grid in phase space was provided and we will try to respect it,
        // even though the Eddington inversion routine may have eliminated some of its nodes:
        // we re-interpolate the values of DF onto the original grid nodes,
        // but retain only the nodes where DF is larger than the absolute minimum value
        // (otherwise it will cause problems in constructing the log-spline interpolator later)
        math::LogLogSpline spl(gridh, gridf);
        gridh.clear();
        gridf.clear();
        for(unsigned int i=0; i<inputgridh.size(); i++) {
            double dfval = spl(inputgridh[i]);
            if(dfval > MIN_VALUE) {
                gridh.push_back(inputgridh[i]);
                gridf.push_back(dfval);
            }
        }
    }
    // compute diffusion coefficients
    reinitDifCoefs();
}

void FokkerPlanckSolver::reinitPotential()
{
    // 1. construct the interpolated distribution function from the values of f on the grid
    math::LogLogSpline df(gridh, gridf);

    // 2. compute the density profile by integrating the DF over velocity at each point of the radial grid
    unsigned int gridsize = gridh.size()/2;
    std::vector<double> gridr(gridsize), gridPhi(gridsize);
    // 2a. assign the values of potential at grid nodes
    for(unsigned int i=0; i<gridsize; i++) {
        gridPhi[i] = phasevol.E(gridh[i*2]);
        gridr  [i] = totalPot.R_max(gridPhi[i]);
    }
    // 2b. compute the density at all grid nodes
    std::vector<double> gridrho = computeDensity(df, phasevol, gridPhi);
    
    // 2c. sanity check - eliminate possible zeros that can't be represented in log-scaled density
    for(unsigned int i=0; i<gridr.size(); i++)
        if(gridrho[i]==0) {
            gridr  .erase(gridr  .begin()+i);
            gridrho.erase(gridrho.begin()+i);
        }

    // 3. construct the density interpolator from the values computed on the radial grid
    math::LogLogSpline newDensity(gridr, gridrho);

    // 4. solve the Poisson equation, reinit the potential and the phase volume
    totalPot = computePotential(newDensity, extPot, gridr.front(), gridr.back(),
        /*diagnostic output*/ Phi0);
    phasevol = potential::PhaseVolume(totalPot);
}

void FokkerPlanckSolver::reinitDifCoefs()
{
    // 1. construct the interpolated distribution function from the values of f on the grid
    math::LogLogSpline df(gridh, gridf);

    // 2. construct the spherical model for this DF in the current potential, used to compute dif.coefs
    SphericalModel model(phasevol, df);
    double mult = 16*M_PI*M_PI * model.cumulMass();
    // 2a. store diagnostic quantities
    Mass = model.cumulMass();
    Etot = model.cumulEtotal();
    Ekin = model.cumulEkin();
    
    // 2b. allocate temporary arrays of various coefficients
    unsigned int gridsize = gridh.size();  // = M+1
    std::vector<double>     // notation follows Park&Petrosian 1996 for the Chang&Cooper scheme (eqs.27-34)
        xnode(gridsize),    // xnode[i]  = x_m = log(h_m), i=0..M
        xcenter(gridsize+1),// xcenter[i]= x_{m-1/2} = (x_m+x_{m-1})/2,  i=1..M
        A(gridh),           // A[i]      = A_m,  i=0..M
        Cdiv(gridsize+1),   // Cdiv[i]   = C_{m-1/2} / (x_m - x_{m-1}),  i=1..M
        Wplus(gridsize+1),  // Wplus[i]  = W_{m-1/2}^{+},  i=1..M
        Wminus(gridsize+1); // Wminus[i] = W_{m-1/2}^{-},  i=1..M
    for(unsigned int i=0; i<gridsize; i++)
        xnode[i] = log(gridh[i]);
    for(unsigned int i=1; i<gridsize; i++)
        xcenter[i] = (xnode[i]+xnode[i-1])/2;
    xcenter[0] = 2*xnode[0]-xcenter[1];
    xcenter[gridsize] = 2*xnode[gridsize-1]-xcenter[gridsize-1];
    
    // 3. compute the drift and diffusion coefficients
    // 3a. intermediate quantities W_{+,-} and C
    for(unsigned int i=1; i<gridsize; i++) {
        double
            intf  = model.I0(xcenter[i]),
            intfg = model.cumulMass(xcenter[i]),
            intfh = model.cumulEkin(xcenter[i]) * (2./3),
            h     = exp(xcenter[i]), g;
        phasevol.E(h, &g);
        double B  = mult * intfg;                   // drift coefficient D_h
        double C  = mult * g * (intf + intfh / h);  // diffusion coefficient D_hh / h
        // we use  D_hh / h  here because the derivative of f is taken w.r.t. ln h
        Cdiv  [i] = C / (xnode[i] - xnode[i-1]);
        double w  = B / Cdiv[i];
        Wminus[i] = fabs(w) > 0.02 ?    // use a more accurate asymptotic expression for small w
            w / (exp(w)-1) :
            1 - 0.5 * w * (1 - (1./6) * w * (1 - (1./60) * w * w));
        Wplus [i] = Wminus[i] + w;
    }
    
    // 3b. final coefficients of the tridiagonal system
    diag. resize(gridsize);
    above.resize(gridsize-1);
    below.resize(gridsize-1);
    
    for(unsigned int i=0; i<gridsize; i++) {
        double denom = 1. / (A[i] * (xcenter[i+1] - xcenter[i]));
        if(i>0)
            below[i-1] = -denom * Cdiv[i]   * Wminus[i];
        if(i<gridsize-1)
            above[i]   = -denom * Cdiv[i+1] * Wplus[i+1];
        diag[i] = denom * (Cdiv[i] * Wplus[i] + Cdiv[i+1] * Wminus[i+1]);
    }
}

double FokkerPlanckSolver::doStep(double dt)
{
    // prepare the actual tridiagonal system for the given delta t
    unsigned int gridsize = gridh.size();
    std::vector<double> c_diag(gridsize), c_above(gridsize-1), c_below(gridsize-1);
    for(unsigned int i=0; i<gridsize; i++) {
        if(i>0)
            c_below[i-1] = dt * below[i-1];
        if(i<gridsize-1)
            c_above[i]   = dt * above[i];
        c_diag[i] = 1 + dt * diag[i];
    }
    // solve the system and overwrite the values of DF at grid nodes with the new ones
    std::vector<double> newf = math::solveTridiag(c_diag, c_above, c_below, gridf);
    double maxdf = 0;
    for(unsigned int i=0; i<gridsize; i++)
        maxdf = fmax(maxdf, fabs(log(newf[i]/gridf[i])));
    gridf = newf;
    return maxdf;
}

}