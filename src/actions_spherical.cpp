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
        double R1, R2, JrE;
        findPlanarOrbitExtent(potential, E, L, R1, R2, &JrE);
        return JrE - Jr;
    }
private:
    const potential::BasePotential& potential;
    const double Jr, L;
    const double Emin, Emax;
};

}  // internal namespace

double computeHamiltonianSpherical(const potential::BasePotential& potential, const Actions& acts)
{
    if(!isSpherical(potential))
        throw std::invalid_argument("computeHamiltonianSpherical: potential must be spherically symmetric");
    if(acts.Jr<0 || acts.Jz<0)
        throw std::invalid_argument("computeHamiltonianSpherical: input actions are negative");
    // total angular momentum
    double L     = acts.Jz + fabs(acts.Jphi);
    // radius of a circular orbit with this angular momentum
    double rcirc = R_from_Lz(potential, L);
    // initial guess (more precisely, lower bound) for Hamiltonian
    double Ecirc = potential.value(coord::PosSph(rcirc, 0, 0)) + (L>0 ? 0.5 * pow_2(L/rcirc) : 0);
    // upper bound for Hamiltonian
    double Einf  = potential.value(coord::PosSph(INFINITY, 0, 0));
    if(!math::isFinite(Einf) && Einf != INFINITY)  // some potentials may return NAN for r=infinity
        Einf = 0;  // assume the default value for potential at infinity
    // find E such that Jr(E, L) equals the target value
    HamiltonianFinderFnc fnc(potential, acts.Jr, L, Ecirc, Einf);
    return math::findRoot(fnc, Ecirc, Einf, ACCURACY_JR);
}

Actions actionsSpherical(const potential::BasePotential& potential,
    const coord::PosVelCyl& point)
{
    if(!isSpherical(potential))
        throw std::invalid_argument("actionsSpherical only can deal with actions in a spherical potential");
    Actions acts;
    double Etot = totalEnergy(potential, point);
    double Ltot = Ltotal(point);
    acts.Jphi = Lz(point);
    acts.Jz = Ltot - fabs(acts.Jphi);
    if(Etot<0) {
        double R1, R2;
        findPlanarOrbitExtent(potential, Etot, Ltot, R1, R2, &acts.Jr);
    } else
        acts.Jr = NAN;
    return acts;
}

ActionAngles actionAnglesSpherical(const potential::BasePotential& potential,
    const coord::PosVelCyl& point, Frequencies* Freq)
{
    throw std::runtime_error("actionAnglesSpherical: angle determination not implemented");
}

namespace {

// interpolated action finder
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

ActionAngles ActionFinderSpherical::actionAngles(const coord::PosVelCyl& point, Frequencies* freq) const
{
    throw std::runtime_error("ActionFinderSpherical: angle determination not implemented");
}

}  // namespace actions
