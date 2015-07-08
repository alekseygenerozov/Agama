#pragma once
#include "actions_base.h"
#include "potential_staeckel.h"

namespace actions{

/// Action finder for the Axisymmetric Stackel Perfect Ellipsoid Potential
class ActionFinderAxisymmetricStaeckel : public BaseActionFinder{
private:
    const potential::StaeckelOblatePerfectEllipsoid& poten;
public:
    ActionFinderAxisymmetricStaeckel(const potential::StaeckelOblatePerfectEllipsoid& _poten) :
        BaseActionFinder(), poten(_poten) {};
    virtual Actions actions(const coord::PosVelCar& point) const;
};

/// Action finder for arbitrary axisymmetric potential using Jason Sander's method
class ActionFinderAxisymmetricFudgeJS : public BaseActionFinder{
private:
    const potential::BasePotential& poten;
public:
    ActionFinderAxisymmetricFudgeJS(const potential::BasePotential& _poten) :
        BaseActionFinder(), poten(_poten) {};
    virtual Actions actions(const coord::PosVelCar& point) const;
};

}  // namespace actions