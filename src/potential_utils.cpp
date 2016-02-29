#include "potential_utils.h"
#include "math_core.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace potential{

/// relative accuracy of locating the radius of circular orbit
static const double EPSREL_RCIRC = 1e-10;

// -------- Various routines for potential --------- //

double v_circ(const BasePotential& potential, double radius)
{
    if(!isZRotSymmetric(potential))
        throw std::invalid_argument("Potential is not axisymmetric, "
            "no meaningful definition of circular velocity is possible");
    coord::GradCyl deriv;
    potential.eval(coord::PosCyl(radius, 0, 0), NULL, &deriv);
    return sqrt(radius*deriv.dR);
}

/** helper class to find the root of  L_z^2 - R^3 d\Phi(R)/dR = 0
    (i.e. the radius R of a circular orbit with the given energy E).
*/
class RcircRootFinder: public math::IFunction {
public:
    RcircRootFinder(const BasePotential& _poten, double _E) :
        poten(_poten), E(_E) {};
    virtual void evalDeriv(const double R, double* val=0, double* deriv=0, double* deriv2=0) const {
        double Phi;
        coord::GradCyl grad;
        coord::HessCyl hess;
        poten.eval(coord::PosCyl(R,0,0), &Phi, &grad, &hess);
        if(val) {
            if(R==INFINITY && !math::isFinite(Phi))
                *val = -1-fabs(E);  // safely negative value
            else
                *val = 2*(E-Phi) - (R>0 && R!=INFINITY ? R*grad.dR : 0);
        }
        if(deriv)
            *deriv = -3*grad.dR - R*hess.dR2;
        if(deriv2)
            *deriv2 = NAN;
    }
    virtual unsigned int numDerivs() const { return 1; }
private:
    const BasePotential& poten;
    const double E;
};

/** helper class to find the root of  L_z^2 - R^3 d\Phi(R)/dR = 0
    (i.e. the radius R of a circular orbit with the given angular momentum L_z).
    For the reason of accuracy, we multiply the equation by  1/(R+1), 
    which ensures that the value stays finite as R -> infinity or R -> 0.
*/
class RfromLzRootFinder: public math::IFunction {
public:
    RfromLzRootFinder(const BasePotential& _poten, double _Lz) :
        poten(_poten), Lz2(_Lz*_Lz) {};
    virtual void evalDeriv(const double R, double* val=0, double* deriv=0, double* deriv2=0) const {
        coord::GradCyl grad;
        coord::HessCyl hess;
        if(R < math::UNREASONABLY_LARGE_VALUE) {
            poten.eval(coord::PosCyl(R,0,0), NULL, &grad, &hess);
            if(val)
                *val = ( Lz2 - (R>0 ? pow_3(R)*grad.dR : 0) ) / (R+1);
            if(deriv)
                *deriv = -(Lz2 + pow_2(R)*( (3+2*R)*grad.dR + R*(R+1)*hess.dR2) ) / pow_2(R+1);
        } else {   // at large R, Phi(R) ~ -M/R, we may use this asymptotic approximation even at infinity
            poten.eval(coord::PosCyl(math::UNREASONABLY_LARGE_VALUE,0,0), NULL, &grad);
            if(val)
                *val = Lz2/(R+1) - pow_2(math::UNREASONABLY_LARGE_VALUE) * grad.dR / (1+1/R);
            if(deriv)
                *deriv = NAN;
        } 
        if(deriv2)
            *deriv2 = NAN;
    }
    virtual unsigned int numDerivs() const { return 1; }
private:
    const BasePotential& poten;
    const double Lz2;
};

double R_circ(const BasePotential& potential, double energy) {
    if(!isZRotSymmetric(potential))
        throw std::invalid_argument("Potential is not axisymmetric, "
            "no meaningful definition of circular orbit is possible");
    return math::findRoot(RcircRootFinder(potential, energy), 0, INFINITY, EPSREL_RCIRC);
}

double L_circ(const BasePotential& potential, double energy) {
    double R = R_circ(potential, energy);
    return R * v_circ(potential, R);
}

double R_from_Lz(const BasePotential& potential, double Lz) {
    if(Lz==0)
        return 0;
    if(!isZRotSymmetric(potential))
        throw std::invalid_argument("Potential is not axisymmetric, "
            "no meaningful definition of circular orbit is possible");
    return math::findRoot(RfromLzRootFinder(potential, Lz), 0, INFINITY, EPSREL_RCIRC);
}

void epicycleFreqs(const BasePotential& potential, const double R,
    double& kappa, double& nu, double& Omega)
{
    if(!isZRotSymmetric(potential))
        throw std::invalid_argument("Potential is not axisymmetric, "
            "no meaningful definition of circular orbit is possible");
    coord::GradCyl grad;
    coord::HessCyl hess;
    potential.eval(coord::PosCyl(R, 0, 0), NULL, &grad, &hess);
    double gradR_over_R = (R==0 && grad.dR==0) ? hess.dR2 : grad.dR/R;
    //!!! no attempt to check if the expressions under sqrt are non-negative - 
    // they could well be for a physically plausible potential of a flat disk with an inner hole
    kappa = sqrt(hess.dR2 + 3*gradR_over_R);
    nu    = sqrt(hess.dz2);
    Omega = sqrt(gradR_over_R);
}

InterpEpicycleFreqs::InterpEpicycleFreqs(const potential::BasePotential& potential)
{
    if(!isZRotSymmetric(potential))
        throw std::invalid_argument("Potential is not axisymmetric, "
            "no meaningful definition of circular orbit is possible");
    const double dlogR = 0.2;  // grid spacing in log radius
    // start the scan in radius from a reasonable value (half-mass radius)
    double logRinit = log(getRadiusByMass(potential, potential.totalMass()*0.5));
    if(!math::isFinite(logRinit))
        logRinit = 0.;
    std::vector<double> Lc, fSum, fKappa, fNu;
    double kappa, nu, Omega;
    double logR = logRinit;
    int stage = 0;     // 0 means scan inward, 1 - outward, 2 - done
    while(stage<2) {   // first scan inward in radius, then outward, then stop
        epicycleFreqs(potential, exp(logR), kappa, nu, Omega);
        double sum = kappa + nu + Omega;
        Lc.    push_back(log(Omega) + 2*logR);  // log-scaled angular momentum of circular orbit
        fSum.  push_back(log(sum));             // log-scaled characteristic frequency (sum of three)
        fKappa.push_back(kappa / sum);          // relative contribution of kappa to the sum
        fNu.   push_back(nu / sum);             // same for nu
        if(!math::isFinite(fSum.back()+Lc.back()))
            throw std::runtime_error("Bad behaviour of potential in epicycle frequencies interpolator");
        // check if we have reached an asymptotic regime,
        // by examining the curvature (2nd derivative) of relation between log(Lcirc) and log(Freq),
        // and the 1st derivative of relative fractions of kappa and nu in the total sum of 3 freqs.
        // The idea is that in the asymptotic limit, both for large and small radii,
        // Lcirc(R) and frequencies(R) should have power-law dependence on radius, thus 
        // the relation between their logarithms has asymptotically constant slope;
        // similarly, the relative fractions of kappa and nu should tend to a constant limit.
        const double EPS = 1e-3;  // required tolerance on the derivative to declare the asymptotic limit
        unsigned int np = Lc.size();
        if(np>=3 && fabs(logR - logRinit)>=2.) {
            double dfSumdLc1 = (fSum[np-1] - fSum[np-2]) / (Lc[np-1] - Lc[np-2]);
            double dfSumdLc2 = (fSum[np-2] - fSum[np-3]) / (Lc[np-2] - Lc[np-3]);
            if( fabs(dfSumdLc1 - dfSumdLc2) < EPS &&   // reaching asymptotic constant slope in log-log
                fabs(fKappa[np-1] - fKappa[np-2]) < EPS &&  // relative fractions reach asymptotic limit
                fabs(fNu[np-1] - fNu[np-2]) < EPS)
            {
                if(stage==0) {   // we've been assembling the arrays inward, now need to reverse them
                    std::reverse(Lc.begin(), Lc.end());
                    std::reverse(fSum.begin(), fSum.end());
                    std::reverse(fKappa.begin(), fKappa.end());
                    std::reverse(fNu.begin(), fNu.end());
                }
                logR = logRinit;  // restart from the middle
                ++stage;          // switch direction in scanning, or finish
            }
            if(np>=1000)
                throw std::runtime_error("No convergence in epicyclic frequencies interpolator");
        }
        if(stage==0)
            logR -= dlogR;
        else
            logR += dlogR;
    }
    // check the behaviour of potential at origin
    epicycleFreqs(potential, 0, kappa, nu, Omega);
    // if the frequencies (namely, nu) are finite at origin,
    // ensure that the spline for freqSum has zero derivative as r->0 (extrapolate as a constant),
    // otherwise don't fix the derivative and let the spline be linearly extrapolated in log-log space
    freqSum   = math::CubicSpline(Lc, fSum, math::isFinite(nu) ? 0 : NAN);
    freqKappa = math::CubicSpline(Lc, fKappa, 0, 0);  // set zero derivatives at both ends
    freqNu    = math::CubicSpline(Lc, fNu, 0, 0);
}

void InterpEpicycleFreqs::eval(double Lz, double& kappa, double& nu, double& Omega) const
{
    double logL = log(fabs(Lz));
    double sum  = exp(freqSum(logL));
    double k = freqKappa(logL); // stays constant when Lz is outside the spline definition range
    double n = freqNu(logL);
    kappa = sum * k;
    nu    = sum * n;
    Omega = sum * (1 - k - n);
}

InterpLcirc::InterpLcirc(const BasePotential& potential)
{
    // find out characteristic energy values
    Ein = potential.value(coord::PosCar(0, 0, 0));
    if(!math::isFinite(Ein))  // limitation of the present implementation
        throw std::runtime_error("InterpLcirc: can only work with potentials "
            "that tend to a finite limit as r->0");
    Eout = potential.value(coord::PosCar(INFINITY, 0, 0));
    if(Eout==Eout && Eout!=0)
        throw std::runtime_error("InterpLcirc: can only work with potentials "
            "that tend to zero as r->infinity");
    else Eout = 0;
    const unsigned int gridSize = 100;
    std::vector<double> gridE(gridSize), gridL(gridSize), gridR(gridSize);
    for(unsigned int i=0; i<gridSize; i++) {
        double frac = i==0 ? 1e-3 : i==1 ? 2e-3 :   // refinement at the ends of interval
            i==gridSize-2 ? 1-2e-3 : i==gridSize-1 ? 1-1e-3 : (i-1)*1.0/(gridSize-3);
        gridE[i] = Ein + (Eout-Ein) * frac;
        double R = R_circ(potential, gridE[i]);
        double L = R * v_circ(potential, R);
        // scaling transformation
        gridE[i] = log(1/Ein-1/gridE[i]);
        gridR[i] = log(R);
        gridL[i] = log(L);
    }
    double derivIn = NAN;  // in principle could compute a reasonable value for extrapolation to r->0
    double derivOut= 0.5;  // extrapolation to large r assumes a Keplerian potential
    // construct 1d interpolators for Lcirc(E) and Rcirc(E)
    interpL = math::CubicSpline(gridE, gridL, derivIn, derivOut);
    interpR = math::CubicSpline(gridE, gridR);
}

void InterpLcirc::evalDeriv(const double E, double* val, double* der, double* der2) const
{
    if(E==Ein) {
        if(val)  *val=0;
        if(der)  *der=NAN;
        if(der2) *der2=NAN;
    }
    if(E<Ein || E>=Eout)
        throw std::invalid_argument("InterpLcirc: energy outside allowed range");
    double scaledE = log(1/Ein-1/E);
    double scaledEder = 1/E/(E/Ein-1);  // d(scaledE)/dE
    double splVal, splDer;
    interpL.evalDeriv(scaledE, &splVal, der!=0?&splDer:0);
    double Lcirc = exp(splVal);
    if(val)
        *val = Lcirc;
    if(der)
        *der = splDer * scaledEder * Lcirc;
    if(der2)
        *der2= NAN;
}

double InterpLcirc::Rcirc(double E) const
{
    if(E<Ein || E>=Eout)
        throw std::invalid_argument("InterpLcirc: energy outside allowed range");
    if(E==Ein)
        return 0;
    double scaledE = log(1/Ein-1/E);
    return exp(interpR.value(scaledE));
}

}  // namespace potential
