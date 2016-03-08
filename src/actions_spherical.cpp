#include "actions_spherical.h"
#include "actions_interfocal_distance_finder.h"
#include "potential_utils.h"
#include "math_core.h"
#include <stdexcept>
#include <cassert>
#include <cmath>

namespace actions{

namespace {
/// required tolerance on the value of Jr(E)
const double ACCURACY_JR = 1e-6;

/// order of Gauss-Legendre quadrature
const unsigned int INTEGR_ORDER = 10;

/// helper class to find the energy corresponding to the given radial action
class HamiltonianFinderFnc: public math::IFunctionNoDeriv {
public:
    HamiltonianFinderFnc(const potential::BasePotential& p,
        double _Jr, double _L, double _Emin, double _Emax) :
        potential(p), Jr(_Jr), L(_L), Emin(_Emin), Emax(_Emax) {};
    /// report the difference between target Jr and the one computed at the given energy
    virtual double value(const double E) const {
        // first two calls in root-finder are for the boundary points, we already know the answer
        if(E==Emin)
            return -Jr;
        if(E==Emax)
            return Jr+1e-10;  // at r==infinity should return some positive value
        double JrE, Rmin, Rmax;
        findPlanarOrbitExtent(potential, E, L, Rmin, Rmax, &JrE);
        return JrE - Jr;
    }
private:
    /// the instance of potential
    const potential::BasePotential& potential;
    /// the values of actions
    const double Jr, L;
    /// boundaries of the energy interval (to use in the first two calls from the root-finder)
    const double Emin, Emax;
};


/// integrand for computing the angles in a spherical potential
template<bool denomr2>
class AngleIntegrand: public math::IFunctionNoDeriv {
public:
    AngleIntegrand(const potential::BasePotential& p, double _E, double _L) :
        potential(p), E(_E), L(_L) {};
    virtual double value(const double r) const {
        double Phi = potential.value(coord::PosSph(r, M_PI_2, 0));
        double vr2 = 2*(E-Phi) - (L!=0 ? pow_2(L/r) : 0);
        return vr2<0 ? 0 : (denomr2 ? L/(r*r) : 1) / sqrt(vr2);
    }
private:
    const potential::BasePotential& potential;
    double E, L;   ///< integrals of motion (energy and total angular momentum)
};
typedef AngleIntegrand<false> AngleIntegrandR;  ///< integrand for theta_r and Omega_r: 1/v_r(r,E,L)
typedef AngleIntegrand<true > AngleIntegrandZ;  ///< integrand for theta_z and Omega_z: L/(r^2*v_r(r,E,L))

/// helper function to find the upper limit of integral such that its value equals the target
class IntegralUpperLimitFinderFnc: public math::IFunction {
public:
    IntegralUpperLimitFinderFnc(const math::IFunction &_integrand, double _target) :
        integrand(_integrand), target(_target) {};
    virtual void evalDeriv(const double x, double *val, double *der, double*) const {
        if(val)
            *val = math::integrateGL(integrand, 0, x, INTEGR_ORDER) - target;
        if(der)
            *der = integrand(x);
    }
    virtual unsigned int numDerivs() const { return 1; }
private:
    const math::IFunction &integrand;
    const double target;  // target value of the integral
};

/// common routine for computing everything: actions, frequencies, angles (the latter two optionally)
template<bool needAngles>
static ActionAngles actionAnglesFrequencies(
    const potential::BasePotential& potential,
    const coord::PosVelCyl& point, Frequencies* freqout)
{
    if(!isSpherical(potential))
        throw std::invalid_argument("actionsAnglesSpherical can only deal with spherical potentials");
    ActionAngles aa;
    Frequencies freq;
    double E = totalEnergy(potential, point);
    double L = Ltotal(point);
    aa.Jphi  = Lz(point);
    aa.Jz    = point.z==0 && point.vz==0 ? 0 : L - fabs(aa.Jphi);  // avoid roundoff errors if Jz==0
    if(E>=0) {
        aa.Jr = aa.thetar = aa.thetaz = aa.thetaphi = NAN;
        return aa;
    }
    double Rmin, Rmax;
    findPlanarOrbitExtent(potential, E, L, Rmin, Rmax, &aa.Jr);
    if(!needAngles)  // in this case don't compute frequencies either
        return aa;
    double r = sqrt(pow_2(point.R)+pow_2(point.z));
    double vtheta = (point.vR * point.z - point.vz * point.R) / r;
    // make sure that the spherical radius is within the limits
    Rmin = fmin(Rmin, r);
    Rmax = fmax(Rmax, r);
    // two integrands used in computing both frequencies and angles, with a suitable transformation
    AngleIntegrandR integrand_r(potential, E, L);
    AngleIntegrandZ integrand_z(potential, E, L);
    math::ScaledIntegrandEndpointSing transf_r(integrand_r, Rmin, Rmax);
    math::ScaledIntegrandEndpointSing transf_z(integrand_z, Rmin, Rmax);
    // below may wish to add a special case of Jr==0 (output the epicyclic frequencies, but no angles?)
    freq.Omegar = M_PI / math::integrateGL(transf_r, 0, 1, INTEGR_ORDER);
    freq.Omegaz = freq.Omegar * math::integrateGL(transf_z, 0, 1, INTEGR_ORDER) / M_PI;
    freq.Omegaphi = freq.Omegaz * math::sign(aa.Jphi);
    if(freqout)  // freak out only if requested
        *freqout = freq;
    // aux angles:  sin(psi) = cos(theta) / sin(i),  sin(chi) = cot(i) cot(theta)
    double psi = atan2(point.z * L,  -point.R * vtheta * r);
    double chi = atan2(point.z * point.vphi, -vtheta * r);
    aa.thetar  = math::integrateGL(transf_r, 0, transf_r.y_from_x(r), INTEGR_ORDER) * freq.Omegar;
    double thr = aa.thetar;
    double thz = math::integrateGL(transf_z, 0, transf_z.y_from_x(r), INTEGR_ORDER);
    if(point.R * point.vR + point.z * point.vz < 0) {  // v_r<0 - we're on the second half of radial period
        aa.thetar = 2*M_PI - aa.thetar;
        thz       = -thz;
        thr       = aa.thetar - 2*M_PI;
    }
    aa.thetaz   = math::wrapAngle(psi + thr * freq.Omegaz / freq.Omegar - thz);
    aa.thetaphi = math::wrapAngle(point.phi - chi + math::sign(aa.Jphi) * aa.thetaz);
    if(aa.Jz == 0)  // in this case the value of theta_z is meaningless
        aa.thetaz = 0;
    return aa;
}

coord::PosVelCyl mapPointFromActions(
    const potential::BasePotential &potential,
    const ActionAngles &aa, Frequencies& freq)
{
    if(!isSpherical(potential))
        throw std::invalid_argument("mapSpherical: potential must be spherically symmetric");
    if(aa.Jr<0 || aa.Jz<0)
        throw std::invalid_argument("mapSpherical: input actions are negative");
    double L = aa.Jz + fabs(aa.Jphi);  // total angular momentum
    double E = computeHamiltonianSpherical(potential, aa);
    // find peri/apocenter radii
    double Rmin, Rmax;
    findPlanarOrbitExtent(potential, E, L, Rmin, Rmax);
    // two integrands used in computing both frequencies and angles
    AngleIntegrandR integrand_r(potential, E, L);
    AngleIntegrandZ integrand_z(potential, E, L);
    math::ScaledIntegrandEndpointSing transf_r(integrand_r, Rmin, Rmax);
    math::ScaledIntegrandEndpointSing transf_z(integrand_z, Rmin, Rmax);
    // compute the frequencies
    double intr = math::integrateGL(transf_r, 0, 1, INTEGR_ORDER);
    freq.Omegar = M_PI / intr;
    freq.Omegaz = freq.Omegar * math::integrateGL(transf_z, 0, 1, INTEGR_ORDER) / M_PI;
    freq.Omegaphi = freq.Omegaz * math::sign(aa.Jphi);
    // find r from theta_r:
    // radial phase ranging from 0 (peri) to 1 (apocenter) to 2 (back to pericenter)
    double phase_r = math::wrapAngle(aa.thetar) / M_PI;
    IntegralUpperLimitFinderFnc finder(transf_r, (phase_r<=1 ? phase_r : 2-phase_r) * intr);
    double rsc = math::findRoot(finder, 0, 1, ACCURACY_JR);
    double r   = transf_r.x_from_y(rsc);
    double vr  = (phase_r<=1 ? 1 : -1) *   // phase_r>1 means moving inwards from apo to pericenter
        sqrt(fmax(0, 2 * (E - potential.value(coord::PosSph(r, M_PI_2, 0))) - (L>0 ? pow_2(L/r) : 0) ));
    // find other auxiliary angles
    double thr = phase_r<=1 ? aa.thetar : aa.thetar - 2*M_PI;
    double thz = math::integrateGL(transf_z, 0, rsc, INTEGR_ORDER) * (phase_r<=1 ? 1 : -1);
    double psi = aa.thetaz + thz - thr * freq.Omegaz / freq.Omegar;
    double sinpsi   = sin(psi);
    double cospsi   = cos(psi);
    double chi      = aa.Jz != 0 ? atan2(fabs(aa.Jphi) * sinpsi, L * cospsi) : psi;
    double sini     = sqrt(1 - pow_2(aa.Jphi / L)); // inclination angle of the orbital plane
    double costheta = sini * sinpsi;                // z/r
    double sintheta = sqrt(1 - pow_2(costheta));    // R/r is always non-negative
    double vtheta   = L * sini * cospsi / (r * sintheta);
    // finally, output position/velocity
    coord::PosVelCyl point;
    point.R    = r * sintheta;
    point.z    = r * costheta;
    point.vR   = vr * sintheta - vtheta * costheta;
    point.vz   = vr * costheta + vtheta * sintheta;
    point.phi  = math::wrapAngle(aa.thetaphi + (chi-aa.thetaz) * math::sign(aa.Jphi));
    point.vphi = aa.Jphi!=0 ? aa.Jphi / point.R : 0;
    return point;
}


/// interpolated action finder
static const math::LinearInterpolator2d createInterpJr(
    const potential::BasePotential& potential,
    const unsigned int gridSizeE,
    const potential::InterpLcirc& interpLcirc)
{
    if(gridSizeE<10 || gridSizeE>500)
        throw std::invalid_argument("ActionFinderSpherical: incorrect grid size");

    // find out characteristic energy values
    double Ein  = potential.value(coord::PosCar(0, 0, 0));
    double Eout = 0;  // default assumption for Phi(r=infinity)
    if(!math::isFinite(Ein) || Ein>=Eout)
        throw std::runtime_error("ActionFinderSpherical: weird behaviour of potential");

    // create a grid in energy
    std::vector<double> gridE(gridSizeE);
    for(unsigned int i=0; i<gridSizeE; i++) 
        gridE[i] = Ein + (Eout-Ein)*i/(gridSizeE-1);

    // create a uniform grid in L/Lcirc(E)
    const unsigned int gridSizeLrel = gridSizeE<80 ? gridSizeE/4 : 20;
    std::vector<double> gridLrel(gridSizeLrel);
    for(unsigned int i=0; i<gridSizeLrel; i++)
        gridLrel[i] = 1.*i/(gridSizeLrel-1);

    // fill a 2d grid in (E, L/Lcirc(E) )
    math::Matrix<double> grid2d(gridSizeE, gridSizeLrel);
    for(unsigned int iE=0; iE<gridSizeE-1; iE++) {
        double E = iE==0 ? gridE[1]*0.1+gridE[0]*0.9 : gridE[iE]; // slightly offset from the boundary
        const double Lc = interpLcirc(E);
        for(unsigned int iL=0; iL<gridSizeLrel; iL++) {
            double L = (iL<gridSizeLrel-1 ? gridLrel[iL] : 1-1e-3) * Lc;
            double R1, R2, Jr;
            findPlanarOrbitExtent(potential, E, L, R1, R2, &Jr);
            grid2d(iE, iL) = Jr/(Lc-L);
            //std::cout << gridE[iE] << "\t"<<gridLrel[iL]<<"\t"<<grid2d(iE,iL)<<"\n";
        }
        //std::cout<<"\n";
    }
    // end point at E=0: all values set to the limiting value (1)
    for(unsigned int iL=0; iL<gridSizeLrel; iL++)
        grid2d(gridSizeE-1, iL) = 1;
    
    // create a 2d interpolator
    return math::LinearInterpolator2d(gridE, gridLrel, grid2d);
}

}  //internal namespace

double computeHamiltonianSpherical(const potential::BasePotential& potential, const Actions& acts)
{
    if(!isSpherical(potential))
        throw std::invalid_argument("computeHamiltonianSpherical: potential must be spherically symmetric");
    if(acts.Jr<0 || acts.Jz<0)
        throw std::invalid_argument("computeHamiltonianSpherical: input actions are negative");
    double L = acts.Jz + fabs(acts.Jphi);  // total angular momentum
    // radius of a circular orbit with this angular momentum
    double rcirc = R_from_Lz(potential, L);
    // initial guess (more precisely, lower bound) for Hamiltonian
    double Ecirc = potential.value(coord::PosSph(rcirc, M_PI_2, 0)) + (L>0 ? 0.5 * pow_2(L/rcirc) : 0);
    // upper bound for Hamiltonian
    double Einf  = potential.value(coord::PosSph(INFINITY, 0, 0));
    if(!math::isFinite(Einf) && Einf != INFINITY)  // some potentials may return NAN for r=infinity
        Einf = 0;  // assume the default value for potential at infinity
    // find E such that Jr(E, L) equals the target value
    HamiltonianFinderFnc fnc(potential, acts.Jr, L, Ecirc, Einf);
    return math::findRoot(fnc, Ecirc, Einf, ACCURACY_JR);
}

coord::PosVelCyl mapSpherical(
    const potential::BasePotential &potential,
    const ActionAngles &aa, Frequencies* freq)
{
    Frequencies tmp;  // temp.storage, ignored if not requested for output
    return mapPointFromActions(potential, aa, freq? *freq : tmp);
}


Actions actionsSpherical(
    const potential::BasePotential& potential,
    const coord::PosVelCyl& point)
{
    return actionAnglesFrequencies<false>(potential, point, NULL);
}

ActionAngles actionAnglesSpherical(
    const potential::BasePotential& potential,
    const coord::PosVelCyl& point,
    Frequencies* freq)
{
    return actionAnglesFrequencies<true>(potential, point, freq);
}

ActionFinderSpherical::ActionFinderSpherical(
    const potential::PtrPotential& _potential, const unsigned int gridSizeE) :
    potential(_potential), interpLcirc(*potential),
    interpJr(createInterpJr(*potential, gridSizeE, interpLcirc))
{
    if(!isSpherical(*potential))
        throw std::invalid_argument("ActionFinderSpherical: potential is not spherically-symmetric");
}

Actions ActionFinderSpherical::actions(const coord::PosVelCyl& point) const
{
    Actions acts;
    double L  = Ltotal(point);
    double E  = totalEnergy(*potential, point);
    double Lc = std::max<double>(L, interpLcirc(E));
    acts.Jphi = Lz(point);
    acts.Jz   = L - fabs(acts.Jphi);
    acts.Jr   = Lc>0 ? interpJr.value(E, L/Lc) * (Lc-L) : 0;  // NAN if out of range
    return acts;
}

ActionAngles ActionFinderSpherical::actionAngles(const coord::PosVelCyl& , Frequencies* ) const
{
    throw std::runtime_error("ActionFinderSpherical: angle determination not implemented");
}

}  // namespace actions
