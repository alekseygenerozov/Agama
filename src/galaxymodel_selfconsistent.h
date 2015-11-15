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

/** Description of a single density component of the total model.
    A component consists of a distribution function (specified in terms of actions)
    and an associated density profile.
    The density profile is calculated iteratively, as the integral of DF over velocities,
    performed in the overall potential.
    The overall potential, in turn, is determined by density profiles of all components,
    computed at the previous step of iterative procedure.
    Alternatively, a component may be a 'deadweight' density profile without associated DF,
    in which case it only contributes to the total potential, but the density profile stays fixed.
    This case is specified by providing NULL as the pointer to DF.
*/
struct Component {
    /// action-based distribution function (not owned by this object and stays constant),
    const df::BaseDistributionFunction* distrFunc;  ///< also may be NULL
    
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

    /** create a component with the given distribution function and parameters of density profile.
        Two variants are possible: 
        (1) DF is provided, and the density profile will be determined iteratively;
        in this case one may still provide the initial guess for it.
        (2) DF is not provided (set to NULL), and the density profile stays fixed throughout 
        the entire iterative procedure.

        \param[in]  df -- distribution function of this component (which remains constant
                    during the iterative procedure), or NULL if it is not provided.
        \param[in]  rmin,rmax -- determine the extent of (logarithmic) radial grid
                    used to compute the density profile of this component.
        \param[in]  numCoefsRadial -- number of grid points in this radial grid.
        \param[in]  numCoefsAngular -- max order of spherical-harmonic expansion (l_max).
        \param[in]  initDens -- the initial guess for density profile of this component;
                    if df is provided, it is not necessary to supply this parameter
                    (an initial guess will then be constructed automatically),
                    but if df is not provided, it is obligatory to specify the density profile;
                    a copy of the provided object will be created internally in any case.
    */
    Component(const df::BaseDistributionFunction* df, double rmin, double rmax, 
        unsigned int numCoefsRadial, unsigned int numCoefsAngular,
        const potential::BaseDensity* initDens=NULL);

    /** delete the internally created density model */
    ~Component() { delete density; }
    /** implement an appropriate copy constructor */
    Component(const Component& src);
    /** implement an appropriate assignment operator */
    Component& operator= (Component src);
};

/** The driver class for self-consistend modelling.
    The list of model components is provided at initialization;
    each component may be 'dead' (with a fixed density profile and unspecified DF) 
    or 'live' (with a specified DF, which stays constant throughout the iterative procedure).
    All components have associated density profiles (either fixed at the beginning for 'dead'
    components, or recomputed from DF after each iteration for 'live' components);
    the total potential is computed from the combined density profile of all components,
    and is in turn used in recomputing the live density profiles through integration of 
    their DFs over velocities. This two-stage process is performed at each iteration:
    first the density recomputation, then the potential update.
    The density profile of each component and the overall potential may be queried
    after each iteration, but these objects exist only until the subsequent iteration, 
    or until the `SelfConsistentModel` object is destroyed.
    One may create copies of them with the `clone()` method of `BaseDensity` or `BasePotential`,
    and then use to initialize a new SelfConsistentModel from the already computed approximations
    for the density, possibly with a different combination of dead/live components.
*/
class SelfConsistentModel {
public:
    /// Construct the model and initialize the first guesses for density profiles and the total potential
    SelfConsistentModel(
        const std::vector<Component>& _components,  ///< array of model components
        ProgressReportCallback* _callback=NULL,     ///< optional pointer to the progress reporting routine
        double _relError=1e-3,                      ///< relative accuracy of density computation
        unsigned int _maxNumEval=10000              ///< max # of DF evaluations per density computation
    );
    ~SelfConsistentModel();
    
    /// recompute the densities of all components with existing DFs and update the total potential
    void doIteration();
    
    /** return the pointer to the internally used total potential: 
        this pointer is valid until another iteration is performed, or the object itself is destroyed;
        if one needs to have a permanent copy then the `BasePotential::clone()` method should be used,
        like  const BasePotential* pot = model.getPotential()->clone();
    */
    const potential::BasePotential* getPotential() const {
        return totalPotential; }
    
    /** return the pointer to the internally used density profile for the given component
        (valid until next iteration or until the SelfConsistentModel object is destroyed)
    */
    const potential::BaseDensity* getComponentDensity(unsigned int index) const;

private:
    /// array of density components and their DFs
    std::vector<Component> components;

    /// Total gravitational potential of all components
    const potential::BasePotential* totalPotential;
    
    /// Action finder associated with the total potential
    const actions::BaseActionFinder* actionFinder;
    
    /// Parameters controlling the accuracy of density computation:
    /// required relative error in density
    double relError;
    
    /// maximum number of DF evaluations during density computation at a single point
    unsigned int maxNumEval;
    
    /// External progress reporting interface (may be NULL)
    ProgressReportCallback* callback;

    // methods //

    /// recompute spherical-harmonic density expansion for the given component
    void updateComponentDensity(unsigned int index);

    /// recompute the total potential using the current density profiles for all components
    void updateTotalPotential();
};
    
}  // namespace
