#include "actions_interfocal_distance_finder.h"
#include "math_core.h"
#include "math_fit.h"
#include "math_ode.h"
#include <cassert>
#include <stdexcept>
#include <cmath>

namespace actions{

/// number of sampling points for a shell orbit (equally spaced in time)
static const unsigned int NUM_STEPS_TRAJ = 16;
/// accuracy of root-finding for Rmin/Rmax
static const double ACCURACY_RMINMAX = 1e-10;
/// accuracy of root-finding for the radius of thin (shell) orbit
static const double ACCURACY_RTHIN = 1e-6;
/// accuracy of orbit integration for shell orbit
static const double ACCURACY_INTEGR = 1e-6;
/// upper limit on the number of timesteps in ODE solver (should be enough to track half of the orbit)
static const unsigned int MAX_NUM_STEPS_ODE = 100;


// estimate IFD for a series of points in R-z plane
template<typename PointT>
double estimateInterfocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<PointT>& traj)
{
    if(traj.size()==0)
        throw std::invalid_argument("Error in finding interfocal distance: empty array of points");
    std::vector<double> x(traj.size()), y(traj.size());
    double sumsq = 0;
    double minr2 = INFINITY;
    for(unsigned int i=0; i<traj.size(); i++) {
        const coord::PosCyl p = coord::toPosCyl(traj[i]);
        minr2 = fmin(minr2, p.R*p.R+p.z*p.z);
        coord::GradCyl grad;
        coord::HessCyl hess;
        potential.eval(p, NULL, &grad, &hess);
        x[i] = hess.dRdz;
        y[i] = 3*p.z * grad.dR - 3*p.R * grad.dz + p.R*p.z * (hess.dR2-hess.dz2)
             + (p.z*p.z - p.R*p.R) * hess.dRdz;
        sumsq += pow_2(x[i]);
    }
    double result = sumsq>0 ? math::linearFitZero(x, y, NULL) : 0;
    return sqrt( fmax( result, minr2*1e-4) );  // ensure that the computed value is positive
}

template double estimateInterfocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosCar>& traj);
template double estimateInterfocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosVelCar>& traj);
template double estimateInterfocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosCyl>& traj);
template double estimateInterfocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosVelCyl>& traj);
template double estimateInterfocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosSph>& traj);
template double estimateInterfocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<coord::PosVelSph>& traj);


/** find the best-fit value of interfocal distance for a shell orbit.
    \param[in] traj  contains the trajectory of this orbit in R-z plane,
    \return  the parameter of a prolate spheroidal coordinate system which minimizes
    the variation of `lambda` coordinate for this orbit.
    If the best-fit value is negative, it is replaced with a small positive quantity.
*/
static double fitInterfocalDistanceShellOrbit(const std::vector<coord::PosCyl>& traj)
{
    if(traj.size()==0)
        throw std::invalid_argument("Error in finding interfocal distance for a shell orbit: empty array");
    math::Matrix<double> coefs(traj.size(), 2);
    std::vector<double> rhs(traj.size());
    std::vector<double> result;  // regression parameters:  lambda/(lambda-delta), lambda
    double minr2 = INFINITY;
    for(unsigned int i=0; i<traj.size(); i++) {
        minr2 = fmin(minr2, pow_2(traj[i].R) + pow_2(traj[i].z));
        coefs(i, 0) = -pow_2(traj[i].R);
        coefs(i, 1) = 1.;
        rhs[i] = pow_2(traj[i].z);
    }
    math::linearMultiFit(coefs, rhs, NULL, result);
    return sqrt( fmax( result[1] * (1 - 1/result[0]), minr2*1e-4) );
}


/** Helper function for finding the roots of (effective) potential in R direction */
class OrbitSizeFunction: public math::IFunction {
public:
    const potential::BasePotential& potential;
    double E;
    double Lz2;
    enum { FIND_RMIN, FIND_RMAX, FIND_JR } mode;
    OrbitSizeFunction(const potential::BasePotential& p, double _E, double _Lz) :
        potential(p), E(_E), Lz2(_Lz*_Lz), mode(FIND_RMIN) {};
    virtual unsigned int numDerivs() const { return 2; }
    /** This function is used in the root-finder for Rmin/Rmax, and in computing the radial action.
        In the regime for locating the turnaround points of a planar orbit, it returns (1/2) v_R^2.
        Moreover, to compute the location of pericenter this is multiplied by R^2 to curb the sharp rise 
        of effective potential at zero, which is problematic for root-finder.
        In the regime for computing the radial action, it returns v_R.
    */
    virtual void evalDeriv(const double R, 
        double* val=0, double* deriv=0, double* deriv2=0) const
    {
        double Phi=0;
        coord::GradCyl grad;
        coord::HessCyl hess;
        if(math::isFinite(R)) {
            potential.eval(coord::PosCyl(R, 0, 0), &Phi, deriv? &grad : NULL, deriv2? &hess: NULL);
        } else {  // we're at infinity in root-finder
            if(deriv) 
                grad.dR = NAN;
            if(deriv2)
                hess.dR2 = NAN;
        }
        if(mode == FIND_RMIN) {    // f(R) = (1/2) v_R^2 * R^2
            if(val)
                *val = (E-Phi)*R*R - Lz2/2;
            if(deriv) 
                *deriv = 2*R*(E-Phi) - R*R*grad.dR;
            if(deriv2)
                *deriv2 = 2*(E-Phi) - 4*R*grad.dR - R*R*hess.dR2;
        } else if(mode == FIND_RMAX) {  // f(R) = (1/2) v_R^2 = E - Phi(R) - Lz^2/(2 R^2)
            if(val)
                *val = Lz2>0 && R<INFINITY ? 
                    // the bizarre expression should yield the same roundoff error as for FIND_RMIN
                    ((E-Phi)*R*R - Lz2/2) / (R*R) :
                    E-Phi;
            if(deriv)
                *deriv = -grad.dR + (Lz2>0 ? Lz2/(R*R*R) : 0);
            if(deriv2)
                *deriv2 = -hess.dR2 - (Lz2>0 ? 3*Lz2/(R*R*R*R) : 0);
        } else if(mode == FIND_JR) {  // f(R) = v_R
            *val = sqrt(fmax(0, 2*(E-Phi) - (Lz2>0 ? Lz2/(R*R) : 0) ) );
        } else
            assert("Invalid operation mode in OrbitSizeFunction"==0);
    }
};

/** Helper function for finding the roots of (effective) potential in R direction,
    for a power-law asymptotic form potential at small radii */
class OrbitSizeFunctionSmallE: public math::IFunction {
public:
    double EminusPhi0, twominusgamma, coefA, Lz2;  // parameters of power-law potential
    enum { FIND_RMINMAX, FIND_JR } mode;
    OrbitSizeFunctionSmallE(double _EminusPhi0, double _twominusgamma, double _coefA, double _Lz) :
        EminusPhi0(_EminusPhi0), twominusgamma(_twominusgamma), coefA(_coefA), Lz2(_Lz*_Lz), 
        mode(FIND_RMINMAX) {};
    virtual unsigned int numDerivs() const { return 1; }
    virtual void evalDeriv(const double R, double* val=0, double* deriv=0, double* =0) const
    {
        double AR2minusgamma = coefA*pow(R, twominusgamma);
        double EminusPhi = EminusPhi0 - AR2minusgamma;
        if(mode == FIND_RMINMAX) {    // f(R) = (1/2) v_R^2 * R^2
            if(val)
                *val = EminusPhi*R*R - Lz2/2;
            if(deriv) 
                *deriv = R*(2*EminusPhi0 - (2+twominusgamma)*AR2minusgamma);
        } else if(mode == FIND_JR) {  // f(R) = v_R
            *val = sqrt(fmax(0, 2*(EminusPhi) - (Lz2>0 ? Lz2/(R*R) : 0) ) );
        } else
            assert("Invalid operation mode in OrbitSizeFunctionSmallE"==0);
    }
};
    
/// accurate treatment of limiting case E --> Phi(0), assuming a power-law behaviour of potential at small r
static void findPlanarOrbitExtentSmallE(const potential::BasePotential& poten, double Phi0, double E, double Lz, 
    double& Rmin, double& Rmax, double* Jr)
{
    // determine asymptotic power-law behaviour of Phi(r) at small r: Phi = Phi(0) + A r^(2-gamma)
    // by computing the value of potential at three points: r0=0, r1, and r2=r1/2
    double Phi1 = Phi0*(1-1e-8);
    OrbitSizeFunction fnc1(poten, Phi1, 0);
    fnc1.mode  = OrbitSizeFunction::FIND_RMAX;
    double R1 = math::findRoot(fnc1, 0, INFINITY, ACCURACY_RMINMAX);
    // don't blindly trust the root-finder, because it may suffer from roundoff errors
    Phi1 = poten.value(coord::PosCyl(R1,0,0));
    double R2 = R1/2;
    double Phi2 = poten.value(coord::PosCyl(R2,0,0));
    double twominusgamma = log( (Phi1-Phi0)/(Phi2-Phi0) ) / log(2.);
    if(twominusgamma>2) twominusgamma=2;  // very unlikely
    if(twominusgamma<0) twominusgamma=0;  // shouldn't ever occur
    double coefA = (Phi1-Phi0) / pow(R1, twominusgamma);

    // now compute everything under the assumption of power-law Phi(r)
    double Rinit = pow(Lz*Lz/coefA/twominusgamma, 1/(2+twominusgamma));
    OrbitSizeFunctionSmallE fnc(E-Phi0, twominusgamma, coefA, Lz);
    if(fnc(Rinit)<0)
        throw std::runtime_error("Error in findPlanarOrbitExtentSmallE: E and Lz have incompatible values");
    double Rupper = pow((E-Phi0)/coefA, 1/twominusgamma);
    if(Lz==0) {
        Rmin=0;
        Rmax=Rupper;
    } else {
        Rmin = math::findRoot(fnc, 0., Rinit, ACCURACY_RMINMAX);
        Rmax = math::findRoot(fnc, Rinit, Rupper, ACCURACY_RMINMAX);
    }
    if(!math::isFinite(Rmin+Rmax))
        throw std::runtime_error("Error in locating Rmin/max in findPlanarOrbitExtentSmallE");
    if(Jr!=NULL) {  // compute radial action
        fnc.mode = OrbitSizeFunctionSmallE::FIND_JR;
        *Jr = math::integrateGL(fnc, Rmin, Rmax, 10) / M_PI;
    }
}

void findPlanarOrbitExtent(const potential::BasePotential& poten, double E, double Lz, 
    double& Rmin, double& Rmax, double* Jr)
{
    if(!isAxisymmetric(poten))
        throw std::invalid_argument("findPlanarOrbitExtent only works for axisymmetric potentials");
    double Phi0 = poten.value(coord::PosCyl(0,0,0));
    if(math::isFinite(Phi0) && E-Phi0 < fabs(Phi0)*1e-8) {   // accurate treatment of very low-energy values
        findPlanarOrbitExtentSmallE(poten, Phi0, E, Lz, Rmin, Rmax, Jr);
        return;
    }
    // the function to use in root-finder for locating the roots of v_R^2=0
    OrbitSizeFunction fnc(poten, E, Lz);
    // first guess for the radius that should lie between Rmin and Rmax
    double Rinit = R_from_Lz(poten, Lz);
    if(!(Rinit>=0))  // could be NaN or inf, although it's a really bad luck
        throw std::runtime_error("Error in findPlanarOrbitExtent: cannot determine R(Lz)");
    // we make sure that f(R)>=0, since otherwise we cannot initiate root-finding
    math::PointNeighborhood nh(fnc, Rinit);
    double dR_to_zero = nh.dxToNearestRoot();
    int nIter = 0;
    while(nh.f0<0 && nIter<4) {       // safety measure to avoid roundoff errors
        if(Rinit+dR_to_zero == Rinit)  // delta-step too small
            Rinit *= (1 + 1e-15*math::sign(dR_to_zero));
        else if(Rinit+dR_to_zero<0)    // delta-step negative and too large
            Rinit /= 2;
        else if(Rinit<dR_to_zero)      // delta-step positive and too large
            Rinit *= 2;
        else
            Rinit += dR_to_zero;
        nh = math::PointNeighborhood(fnc, Rinit);
        dR_to_zero = nh.dxToNearestRoot();
        nIter++;
    }
    if(nh.f0<0)
        throw std::runtime_error("Error in findPlanarOrbitExtent: E and Lz have incompatible values");
    Rmin = Rmax = Rinit;
    double maxPeri = Rinit, minApo = Rinit;    // endpoints of interval for locating peri/apocenter radii
    if(fabs(dR_to_zero) < Rinit*ACCURACY_RMINMAX) {    // we are already near peri- or apocenter radius
        if(nh.dxBetweenRoots() < Rinit*ACCURACY_RMINMAX) {  // the range between Rmin and Rmax is too small
            maxPeri = minApo = NAN;  // do not attempt to locate them
        } else if(nh.fder < 0) {
            minApo  = NAN;  // will skip the search for Rmax
            maxPeri = Rinit + nh.dxToPositive();
        } else {
            maxPeri = NAN;  // will skip the search for Rmin
            minApo  = Rinit + nh.dxToPositive();
        }
    }
    if(fnc.Lz2>0) {
        if(math::isFinite(maxPeri)) {
            fnc.mode = OrbitSizeFunction::FIND_RMIN;
            Rmin = math::findRoot(fnc, 0., maxPeri, ACCURACY_RMINMAX);
            if(!math::isFinite(Rmin))  // could be that our initial upper bound was wrong
                Rmin = math::findRoot(fnc, maxPeri, Rinit, ACCURACY_RMINMAX);
            // ensure that E-Phi(Rmin) >= 0
            // (due to finite accuracy in root-finding, a small adjustment may be needed)
            math::PointNeighborhood pn(fnc, Rmin);
            if(pn.f0<0) {   // ensure that E>=Phi(Rmin)
                double dx = pn.dxToPositive();
                if(Rmin+dx>=0 && Rmin+dx<=maxPeri)
                    Rmin += dx;
                else  // most likely due to some roundoff errors
                    Rmin = Rinit;  // safe value
            }
            if(!math::isFinite(Rmin))
                throw std::runtime_error("Error in locating Rmin in findPlanarOrbitExtent");
        }
    } else  // angular momentum is zero
        Rmin = 0; // !! this assumes a monotonic potential !! 
    if(math::isFinite(minApo)) {
        fnc.mode = OrbitSizeFunction::FIND_RMAX;
        Rmax = math::findRoot(fnc, minApo, INFINITY, ACCURACY_RMINMAX);
        if(!math::isFinite(Rmax))  // could be that our initial lower bound was wrong
            Rmax = math::findRoot(fnc, Rinit, minApo, ACCURACY_RMINMAX);
        math::PointNeighborhood pn(fnc, Rmax);
        if(pn.f0<0) {   // ensure that E>=Phi(Rmax)
            double dx = pn.dxToPositive();
            if(Rmax+dx>=Rmin)
                Rmax += dx;
            else  // most likely due to some roundoff errors
                Rmax = Rinit;  // safe value
        }
        if(!math::isFinite(Rmax)) {
            if(E>=0 && fnc(INFINITY)>0)   // could not find the upper limit because it simply does not exist!
                throw std::invalid_argument("findPlanarOrbitExtent: E>=0");
            else
                throw std::runtime_error("Error in locating Rmax in findPlanarOrbitExtent");
        }
    }   // else Rmax=Rinit
//!!!    assert(Rmin>=0 && Rmin<=Rinit && Rinit<=Rmax);
    if(Jr!=NULL) {  // compute radial action
        fnc.mode = OrbitSizeFunction::FIND_JR;
        *Jr = math::integrateGL(fnc, Rmin, Rmax, 10) / M_PI;
    }
}


/// function to use in ODE integrator
class OrbitIntegratorMeridionalPlane: public math::IOdeSystem {
public:
    OrbitIntegratorMeridionalPlane(const potential::BasePotential& p, double Lz) :
        poten(p), Lz2(Lz*Lz) {};

    /** apply the equations of motion in R,z plane without tracking the azimuthal motion */
    virtual void eval(const double /*t*/, const math::OdeStateType& y, math::OdeStateType& dydt) const
    {
        coord::GradCyl grad;
        poten.eval(coord::PosCyl(y[0], y[1], 0), NULL, &grad);
        dydt[0] = y[2];
        dydt[1] = y[3];
        dydt[2] = -grad.dR + (Lz2>0 ? Lz2/pow_3(y[0]) : 0);
        dydt[3] = -grad.dz;
    }
    
    /** return the size of ODE system: R, z, vR, vz */
    virtual unsigned int size() const { return 4;}
private:
    const potential::BasePotential& poten;
    const double Lz2;
};

/// function to use in locating the exact time of the x-y plane crossing
class FindCrossingPointZequal0: public math::IFunction {
public:
    FindCrossingPointZequal0(const math::BaseOdeSolver& _solver) :
        solver(_solver) {};
    /** used in root-finder to locate the root z(t)=0 */
    virtual void evalDeriv(const double time, 
        double* val=0, double* der=0, double* /*der2*/=0) const {
        if(val)
            *val = solver.value(time, 1);  // z
        if(der)
            *der = solver.value(time, 3);  // vz
    }
    virtual unsigned int numDerivs() const { return 1; }
private:
    const math::BaseOdeSolver& solver;
};

/** launch an orbit perpendicularly to x-y plane from radius R with vz>0,
    and record the radius at which it crosses this plane downward (vz<0).
    \param[out] timeCross stores the time required to complete the half-oscillation in z;
    \param[out] traj stores the trajectory recorded at equal intervals of time;
    \param[out] Jz stores the vertical action computed for this trajectory;
    \return  the crossing radius
*/
static double findCrossingPointR(
    const potential::BasePotential& poten, double E, double Lz, double R,
    double* timeCross, std::vector<coord::PosCyl>* traj, double* Jz)
{
    double vz = sqrt(fmax( 2 * (E-poten.value(coord::PosCyl(R, 0, 0))) - (Lz>0 ? pow_2(Lz/R) : 0), R*R*1e-16));
    OrbitIntegratorMeridionalPlane odeSystem(poten, Lz);
    math::OdeStateType vars(odeSystem.size());
    vars[0] = R;
    vars[1] = 0;
    vars[2] = 0;
    vars[3] = vz;
    math::OdeSolverDOP853 solver(odeSystem, 0, ACCURACY_INTEGR);
    solver.init(vars);
    bool finished = false;
    unsigned int numStepsODE = 0;
    double timePrev = 0;
    double timeCurr = 0;
    double timeTraj = 0;
    const double timeStepTraj = timeCross!=NULL ? *timeCross*0.5/(NUM_STEPS_TRAJ-1) : INFINITY;
    if(traj!=NULL)
        traj->clear();
    if(Jz!=NULL)
        *Jz = 0;
    while(!finished) {
        if(solver.step() <= 0 || numStepsODE >= MAX_NUM_STEPS_ODE)  // signal of error
            finished = true;
        else {
            numStepsODE++;
            timePrev = timeCurr;
            timeCurr = solver.getTime();
            if(timeStepTraj!=INFINITY && traj!=NULL)
            {   // store trajectory
                while(timeTraj <= timeCurr && traj->size() < NUM_STEPS_TRAJ) {
                    traj->push_back(coord::PosCyl(  // store R and z at equal intervals of time
                        fabs(solver.value(timeTraj, 0)), solver.value(timeTraj, 1), 0)); 
                    timeTraj += timeStepTraj;
                }
            }
            if(solver.value(timeCurr, 1) < 0) {  // z<0 - we're done
                finished = true;
                timeCurr = math::findRoot(FindCrossingPointZequal0(solver),
                    timePrev, timeCurr, ACCURACY_RTHIN);
            }
            if(Jz!=NULL)
            {   // compute vertical action  (very crude approximation! one integration point per timestep)
                *Jz +=
                 (  solver.value((timePrev+timeCurr)/2, 2) *   // vR at the mid-timestep
                   (solver.value(timeCurr, 0) - solver.value(timePrev, 0))  // delta R over timestep
                  + solver.value((timePrev+timeCurr)/2, 3) *   // vz at the mid-timestep
                   (solver.value(timeCurr, 1) - solver.value(timePrev, 1))  // delta z over timestep
                  ) / M_PI;
            }
        }
    }
    if(timeCross!=NULL)
        *timeCross = timeCurr;
    return fabs(solver.value(timeCurr, 0));   // value of R at the moment of crossing x-y plane
}

/// function to be used in root-finder for locating the thin orbit in R-z plane
class FindClosedOrbitRZplane: public math::IFunctionNoDeriv {
public:
    FindClosedOrbitRZplane(const potential::BasePotential& p, 
        double _E, double _Lz, double _Rmin, double _Rmax,
        double* _timeCross, std::vector<coord::PosCyl>* _traj, double* _Jz) :
        poten(p), E(_E), Lz(_Lz), Rmin(_Rmin), Rmax(_Rmax), 
        timeCross(_timeCross), traj(_traj), Jz(_Jz) {};
    /// report the difference in R between starting point (R, z=0, vz>0) and return point (R1, z=0, vz<0)
    virtual double value(const double R) const {
        // first two calls in root-finder are for the boundary points, we already know the answer
        if(R==Rmin)
            return R-Rmax;
        if(R==Rmax)
            return R-Rmin;
        double R1 = findCrossingPointR(poten, E, Lz, R, timeCross, traj, Jz);
        return R-R1;
    }
private:
    const potential::BasePotential& poten;
    const double E, Lz;               ///< parameters of motion in the R-z plane
    const double Rmin, Rmax;          ///< boundaries of interval in R (to skip the first two calls)
    double* timeCross;                ///< keep track of time required to complete orbit
    std::vector<coord::PosCyl>* traj; ///< store the trajectory
    double* Jz;                       ///< store the estimated value of vertical action
};


double estimateInterfocalDistanceShellOrbit(
    const potential::BasePotential& poten, double E, double Lz, 
    double* R, double* Jz)
{
    double Rmin, Rmax;
    findPlanarOrbitExtent(poten, E, Lz, Rmin, Rmax);
    double timeCross = INFINITY;
    std::vector<coord::PosCyl> traj;
    FindClosedOrbitRZplane fnc(poten, E, Lz, Rmin, Rmax, &timeCross, &traj, Jz);
    // locate the radius of thin orbit;
    // as a by-product, store the orbit in 'traj' and the vertical action in Jz (if necessary)
    double Rthin = math::findRoot(fnc, Rmin, Rmax, ACCURACY_RTHIN);
    if(R!=NULL)
        *R=Rthin;
    if(Rthin!=Rthin || traj.size()==0)
        return Rmin;  // anything
    // now find the best-fit value of delta for this orbit
    return fitInterfocalDistanceShellOrbit(traj);
}


// ----------- Interpolation of interfocal distance in E,Lz plane ------------ //
InterfocalDistanceFinder::InterfocalDistanceFinder(
    const potential::BasePotential& potential, const unsigned int gridSizeE) :
    interpLcirc(potential)
{
    if(!isAxisymmetric(potential))
        throw std::invalid_argument("Potential is not axisymmetric, "
            "interfocal distance estimator is not suitable for this case");
    
    if(gridSizeE<10 || gridSizeE>500)
        throw std::invalid_argument("InterfocalDistanceFinder: incorrect grid size");

    // find out characteristic energy values
    double Ein  = potential.value(coord::PosCar(0, 0, 0));
    double Eout = 0;  // default assumption for Phi(r=infinity)
    if(!math::isFinite(Ein) || Ein>=Eout)
        throw std::runtime_error("InterfocalDistanceFinder: weird behaviour of potential");

    // create a grid in energy
    Ein *= 1-0.5/gridSizeE;  // slightly offset from zero
    std::vector<double> gridE(gridSizeE);
    for(unsigned int i=0; i<gridSizeE; i++) 
        gridE[i] = Ein + i*(Eout-Ein)/gridSizeE;

    // create a uniform grid in Lz/Lcirc(E)
    const unsigned int gridSizeLzrel = gridSizeE<80 ? gridSizeE/4 : 20;
    std::vector<double> gridLzrel(gridSizeLzrel);
    for(unsigned int i=0; i<gridSizeLzrel; i++)
        gridLzrel[i] = (i+0.01) / (gridSizeLzrel-0.98);

    // fill a 2d grid in (E, Lz/Lcirc(E) )
    math::Matrix<double> grid2d(gridE.size(), gridLzrel.size());
    for(unsigned int iE=0; iE<gridE.size(); iE++) {
        const double Lc = L_circ(potential, gridE[iE]); //interpLcirc(gridE[iE]);
        for(unsigned int iL=0; iL<gridLzrel.size(); iL++) {
            double Lz = gridLzrel[iL] * Lc;
            grid2d(iE, iL) = estimateInterfocalDistanceShellOrbit(potential, gridE[iE], Lz);
        }
    }

    // create a 2d interpolator
    interp = math::LinearInterpolator2d(gridE, gridLzrel, grid2d);
}

double InterfocalDistanceFinder::value(double E, double Lz) const
{
    E = fmin(fmax(E, interp.xmin()), interp.xmax());
    double Lc = interpLcirc(E);
    double Lzrel = fmin(fmax(fabs(Lz)/Lc, interp.ymin()), interp.ymax());
    return interp.value(E, Lzrel);
}

}  // namespace actions
