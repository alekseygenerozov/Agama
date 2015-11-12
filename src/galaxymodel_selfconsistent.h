/** \file    galaxymodel_selfconsistent.h
    \brief   Self-consistent model of a multi-component galaxy specified by distribution functions
    \date    2015
    \author  Eugene Vasiliev
*/
#pragma once
#include "potential_base.h"
#include "actions_base.h"
#include "df_base.h"

namespace galaxymodel{

/// Class for providing a progress report during self-consistent modelling
class ProgressReportCallback {
public:
    virtual ~ProgressReportCallback() {};
    
    virtual void generalMessage(const char* /*msg*/) {};

    virtual void reportDensityAtPoint(unsigned int /*componentIndex*/,
        const coord::PosCyl& /*point*/, double /*densityValue*/) {};

    virtual void reportDensityUpdate(unsigned int /*componentIndex*/,
        const potential::BaseDensity& /*density*/) {};

    virtual void reportTotalPotential(const potential::BasePotential& /*potential*/) {};
};

/// Description of a single density component of the total model
struct Component {
    /// action-based distribution function (stays constant)
    const df::BaseDistributionFunction* distrFunc;
    /// definition of grid for computing the density profile:
    double rmin, rmax;             ///< range of radii for the logarithmic grid
    unsigned int numCoefsRadial;   ///< number of grid points in radius
    unsigned int numCoefsAngular;  ///< maximum order of angular-harmonic expansion (l_max)
    /// spherical-harmonic expansion of density profile of this component
    const potential::BaseDensity* density;

    /// empty constructor needed for placing this structure into std::vector
    Component():
        distrFunc(NULL), rmin(0), rmax(0),
        numCoefsRadial(0), numCoefsAngular(0), density(NULL) {};

    /// create a component with the given distribution function and parameters of density profile
    Component(const df::BaseDistributionFunction* _df, double _rmin, double _rmax, 
        unsigned int _numCoefsRadial, unsigned int _numCoefsAngular,
        const potential::BaseDensity* initDens=NULL);
};

class SelfConsistentModel {
public:
    SelfConsistentModel(
        const std::vector<Component>& _components,
        ProgressReportCallback* _callback=NULL,
        double _relError=1e-3,
        unsigned int _maxNumEval=10000);
    
    /// recompute the densities of all components and update the total potential
    void doIteration();
    
    /// return the total potential
    const potential::BasePotential& getPotential() const {
        return *totalPotential; }
    
    /// return the density profile for the given component
    const potential::BaseDensity& getComponentDensity(unsigned int index) const {
        return *(components[index].density); }  // no range check performed!

private:
    /// array of density components and their DFs
    std::vector<Component> components;

    /// Total gravitational potential of all components
    const potential::BasePotential* totalPotential;
    
    /// Action finder associated with the total potential
    const actions::BaseActionFinder* actionFinder;
    
    /// Parameters controlling the accuracy of density computation:
    double relError;          ///< required relative error in density
    unsigned int maxNumEval;  ///< maximum number of DF evaluations during density computation at a single point
    
    /// External progress reporting interface (may be NULL)
    ProgressReportCallback* callback;


    /// recompute spherical-harmonic density expansion for the given component
    void updateComponentDensity(unsigned int index);

    /// recompute the total potential using the current density profiles for all components
    void updateTotalPotential();
};
    
}  // namespace
