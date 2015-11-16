/** \file    actions_spherical.h
    \brief   Action-angle finders for a generic spherical potential
    \author  Eugene Vasiliev
    \date    2015

*/
#pragma once
#include "actions_base.h"
#include "potential_utils.h"
#include "math_spline.h"

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

/** Fast computation of actions in any spherical potential by using
    2d interpolation of radial action as a function of E and L.
    The reference to the potential provided to the constructor is stored internally
    in this object, thus it must exist during its lifetime and will be used
    when the actions need to be computed.
*/
class ActionFinderSpherical: public BaseActionFinder {
public:
    /// Initialize the internal interpolation tables
    ActionFinderSpherical(const potential::BasePotential& potential, const unsigned int gridSize=50);
    virtual ~ActionFinderSpherical() {};
    virtual Actions actions(const coord::PosVelCyl& point) const;
    virtual ActionAngles actionAngles(const coord::PosVelCyl& point, Frequencies* freq=0) const;
private:
    const potential::BasePotential& potential;  ///< reference to the spherical potential
    const potential::InterpLcirc interpLcirc;   ///< interpolator for Lcirc(E)
    math::LinearInterpolator2d interpJr;        ///< 2d interpolator for Jr(E,L)
};

}  // namespace actions
