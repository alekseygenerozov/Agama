/** \file    actions_spherical.h
    \brief   Action-angle finders for a generic spherical potential
    \author  Eugene Vasiliev
    \date    2015-2016
*/
#pragma once
#include "actions_base.h"
#include "potential_utils.h"
#include "math_spline.h"
#include "smart.h"

namespace actions {

/** Compute actions in a given spherical potential.
    \param[in]  potential is the arbitrary spherical potential;
    \param[in]  point     is the position/velocity point;
    \return     actions for the given point, or Jr=NAN if the energy is positive;
    \throw      std::invalid_argument exception if the potential is not spherical
    or some other error occurs.
*/
Actions actionsSpherical(
    const potential::BasePotential& potential,
    const coord::PosVelCyl& point);

/** Compute actions, angles and frequencies in a spherical potential.
    \param[in]  potential is the arbitrary spherical potential;
    \param[in]  point     is the position/velocity point;
    \param[out] freq      if not NULL, output the frequencies in this variable;
    \return     actions and angles for the given point, or Jr=NAN if the energy is positive;
    \throw      std::invalid_argument exception if the potential is not spherical
    or some other error occurs.
*/ 
ActionAngles actionAnglesSpherical(
    const potential::BasePotential& potential,
    const coord::PosVelCyl& point,
    Frequencies* freq=0);


/** Compute the total energy for an orbit in a spherical potential from the given values of actions.
    \param[in]  potential  is the arbitrary spherical potential;
    \param[in]  acts       are the actions;
    \return     the value of Hamiltonian (total energy) corresponding to the given actions;
    \throw      std::invalid_argument exception if the potential is not spherical
    or Jr/Jz actions are negative.
*/
double computeHamiltonianSpherical(const potential::BasePotential& potential, const Actions& acts);


/** Compute position/velocity from actions/angles in an arbitrary spherical potential.
    \param[in]  potential   is the instance of a spherical potential;
    \param[in]  actAng  is the action/angle point
    \param[out] freq    if not NULL, store the frequencies for these actions.
    \return     position and velocity point
*/
coord::PosVelCyl mapSpherical(
    const potential::BasePotential &potential,
    const ActionAngles &actAng, Frequencies* freq=0);


/** Fast computation of actions in any spherical potential by using
    2d interpolation of radial action as a function of E and L.
*/
class ActionFinderSpherical: public BaseActionFinder {
public:
    /// Initialize the internal interpolation tables
    ActionFinderSpherical(const potential::PtrPotential& potential, const unsigned int gridSize=50);
    virtual ~ActionFinderSpherical() {};
    virtual Actions actions(const coord::PosVelCyl& point) const;
    /// actionAngles not implemented
    virtual ActionAngles actionAngles(const coord::PosVelCyl& point, Frequencies* freq=0) const;
private:
    const potential::PtrPotential potential;   ///< pointer to the spherical potential
    const potential::InterpLcirc interpLcirc;  ///< interpolator for Lcirc(E)
    const math::LinearInterpolator2d interpJr; ///< 2d interpolator for Jr(E,L)
};


/** Class for performing action/angle to coordinate/momentum transformation for 
    an arbitrary spherical potential, using modified spherical coordinates for output */
class ToyMapSpherical: public BaseToyMap<coord::SphMod>{
public:
    ToyMapSpherical(const potential::BasePotential& p) : potential(p) {};
    virtual coord::PosVelSphMod map(
        const ActionAngles& actAng,
        Frequencies* freq=0,
        DerivAct<coord::SphMod>* derivAct=0,
        DerivAng<coord::SphMod>* derivAng=0,
        coord::PosVelSphMod* derivParam=0) const;
private:
    const potential::BasePotential& potential;  ///< !!!!! need to use a smart pointer here
};

}  // namespace actions
