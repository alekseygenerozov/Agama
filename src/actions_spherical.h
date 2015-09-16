/** \file    actions_spherical.h
    \brief   Action-angle finders for a generic spherical potential
    \author  Eugene Vasiliev
    \date    2015

*/
#pragma once
#include "actions_base.h"
#include "potential_base.h"

namespace actions {

/** Compute actions in a given spherical potential.
    \param[in]  potential is the arbitrary spherical potential;
    \param[in]  point     is the position/velocity point;
    \return     actions for the given point;
    \throw      std::invalid_argument exception if the potential is not spherical,
    or the energy is positive, or some other error occurs.
*/
Actions sphericalActions(
    const potential::BasePotential& potential,
    const coord::PosVelCyl& point);

ActionAngles sphericalActionAngles(
    const potential::BasePotential& potential,
    const coord::PosVelCyl& point,
    Frequencies* freq=0);

class ActionFinderSpherical: public BaseActionFinder {
public:
    ActionFinderSpherical(const potential::BasePotential& potential) :
        pot(potential) {};
    virtual ~ActionFinderSpherical() {};
    virtual Actions actions(const coord::PosVelCyl& point) const {
        return sphericalActions(pot, point); }
    virtual ActionAngles actionAngles(const coord::PosVelCyl& point, Frequencies* freq=0) const {
        return sphericalActionAngles(pot, point, freq); }
private:
    const potential::BasePotential& pot;    ///< the generic spherical potential in which actions are computed
};

}  // namespace actions
