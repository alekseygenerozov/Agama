#include "actions_spherical.h"
#include "actions_interfocal_distance_finder.h"
#include "math_core.h"
#include <stdexcept>
#include <cassert>
#include <cmath>

namespace actions{

Actions sphericalActions(const potential::BasePotential& potential,
    const coord::PosVelCyl& point)
{
    if(!isSpherical(potential))
        throw std::invalid_argument("This routine only can deal with actions in a spherical potential");
    Actions acts;
    double Ltot = Ltotal(point);
    acts.Jphi = Lz(point);
    acts.Jz = Ltot - fabs(acts.Jphi);
    double R1, R2;
    findPlanarOrbitExtent(potential, totalEnergy(potential, point), Ltot, R1, R2, &acts.Jr);
    return acts;
}

ActionAngles sphericalActionAngles(const potential::BasePotential& potential,
    const coord::PosVelCyl& point, Frequencies* Freq)
{
    throw std::runtime_error("Angle determination not implemented");
}

// interpolated action finder

ActionFinderSpherical::ActionFinderSpherical(
    const potential::BasePotential& _potential, const unsigned int gridSizeE) :
    potential(_potential), interpLcirc(_potential)
{
    if(!isSpherical(potential))
        throw std::invalid_argument("Potential is not spherically-symmetric");

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
    interpJr = math::LinearInterpolator2d(gridE, gridLrel, grid2d);
}

Actions ActionFinderSpherical::actions(const coord::PosVelCyl& point) const
{
    Actions acts;
    double L  = Ltotal(point);
    double E  = totalEnergy(potential, point);
    double Lc = std::max<double>(L, interpLcirc(E));
    acts.Jphi = Lz(point);
    acts.Jz   = L - fabs(acts.Jphi);
    acts.Jr   = Lc>0 ? interpJr.value(E, L/Lc) * (Lc-L) : 0;
    /*if(!math::isFinite(acts.Jr))
        throw std::runtime_error("ActionFinderSpherical: bad value encountered for Jr");*/
    return acts;
}

ActionAngles ActionFinderSpherical::actionAngles(const coord::PosVelCyl& point, Frequencies* freq) const
{
    throw std::runtime_error("Angle determination not implemented");
}

}  // namespace actions
