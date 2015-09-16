#include "actions_spherical.h"
#include "actions_interfocal_distance_finder.h"
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

}  // namespace actions
