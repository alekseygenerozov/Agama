/** \file    galaxymodel_selfconsistent.h
    \brief   Self-consistent model of a multi-component galaxy specified by distribution functions
    \date    2015
    \author  Eugene Vasiliev

This module deals with models consisting of one or several mass components,
which together generate the (axisymmetric) total gravitational potential.
Each component contributes to it by either a density profile, a fixed potential, or both.
The density profile is computed from a distribution function (DF) by integrating
it over velocities; in doing so, the transformation between position/velocity and
action/angle variables (the latter are the arguments of DF) is provided by
the action finder associated with the total potential.
Not every component needs to be generated from a DF: it may present a static
density or potential profile.
In the present implementation, the overall potential is assembled from the contributions
of each component that provide a potential (not all of them need to do so), plus a single
multipole potential that is computed from the overall density profile of all components
that provide them.
For instance, a single static disk-like object (without a DF) could be represented by
two components: a DiskAnsatz potential and a DiskResidual density profile, which together
generate the required density distribution of a separable disk model - same as in GalPot.
Alternatively, a disk specified by DF can provide an appropriately tuned DiskAnsatz potential,
plus a residual density model that is obtained by subtracting the DiskAnsatz's density profile
from the density computed from a DF.
Finally, a halo-like DF component provides only a density computed from DF.
 
The workflow of self-consistent modelling is the following:
1) assemble the list of components and generate the first guess for the overall potential;
2) initialize the action finder for the total potential
3) recompute the density profiles of all components that have a specified DF (are not static);
4) recalculate the multipole expansion and put together all potential constituents to form
an updated potential;
5) repeat from step 2 until convergence.
*/
#pragma once
#include "potential_base.h"
#include "actions_base.h"
#include "df_base.h"
#include "potential_galpot.h"

namespace galaxymodel{

/** Description of a single component of the total model.
    It may provide the density profile, to be used in the multipole expansion,
    or a potential to be added directly to the total potential, or both.
    In the latter case, the overall density profile consists of two parts,
    one with a known potential (which may be strongly flattened),
    and the other with density that is not strongly flattened and can be efficiently
    represented with a spherical-harmonic expansion.
    This combination is primarily suited for disk-like components.
    In case that this component has a DF, its density and potential may be recomputed
    with the `update` method, using the given total potential and its action finder.
*/
class BaseComponent {
public:
    BaseComponent() {};
    virtual ~BaseComponent() {};

    /** recalculate the density profile (and possibly the additional potential)
        by integrating the DF over velocities in the given total potential.
        In case that the component does not have a DF, this method does nothing.
    */
    virtual void update(const potential::BasePotential& totalPotential,
        const actions::BaseActionFinder& actFinder) = 0;

    /** return the pointer to the internally used density profile for the given component
        (valid until the next call to `update` or until the object is destroyed);
        if necessary, one may make a copy of it with the `BaseDensity::clone()` method.
        This pointer to density object may be NULL, in which case it is not used
        in the spherical-harmonic expansion of the SelfConsistentModel.
    */    
    virtual const potential::BaseDensity*   getDensity()   const = 0;

    /** return the pointer to the additional potential component to be used as part of
        the total potential, or NULL if not applicable. This pointer remains valid
        until the next call to `update` or until the object is destroyed.
    */
    virtual const potential::BasePotential* getPotential() const = 0;

private:
    // do not allow to copy or assign objects of this type
    BaseComponent(const BaseComponent& src);
    BaseComponent& operator= (const BaseComponent& src);
};

/** A specialization for the component that provides static (unchanging) density and/or
    potential profiles. For instance, a disk-like structure may be described by
    a combination of DiskAnsatz potential and DiskResidual density profile. */
class ComponentStatic: public BaseComponent {
public:
    /** create copies of the input density and/or potential profiles */
    ComponentStatic(const potential::BaseDensity& dens) : density(dens.clone()), potential(0) {}
    ComponentStatic(const potential::BasePotential& pot) : density(0), potential(pot.clone()) {}
    ComponentStatic(const potential::BaseDensity& dens, const potential::BasePotential& pot) :
        density(dens.clone()), potential(pot.clone()) {}
    /** destroy the internally created copies of density/potentials */
    virtual ~ComponentStatic() { delete density; delete potential; }
    /** update does nothing */
    virtual void update(const potential::BasePotential&, const actions::BaseActionFinder&) {}
    virtual const potential::BaseDensity*   getDensity()   const { return density; }
    virtual const potential::BasePotential* getPotential() const { return potential; }
private:
    const potential::BaseDensity* density;
    const potential::BasePotential* potential;
};

/** A specialization for the component with the density profile computed from a DF,
    using a spherical-harmonic expansion */
class ComponentWithDF: public BaseComponent {
public:
    /** create a component with the given distribution function and parameters of density profile.
        \param[in]  df -- distribution function of this component (which is not owned
                    by this object and remains constant during the iterative procedure).
        \param[in]  rmin,rmax -- determine the extent of (logarithmic) radial grid
                    used to compute the density profile of this component.
        \param[in]  numCoefsRadial -- number of grid points in this radial grid.
        \param[in]  numCoefsAngular -- max order of spherical-harmonic expansion (l_max).
        \param[in]  initDensity -- the initial guess for the density profile of this component;
                    a copy of it is created internally.
        \param[in]  relError -- relative accuracy of density computation.
        \param[in]  maxNumEval -- max # of DF evaluations per single density computation.
    */
    ComponentWithDF(
        const df::BaseDistributionFunction& df,
        double rmin, double rmax,
        unsigned int numCoefsRadial, unsigned int numCoefsAngular,
        const potential::BaseDensity& initDensity,
        double relError=1e-3,
        unsigned int maxNumEval=1e5);

    /** same as the above, with the difference that the initial guess for the density profile
        is not provided by the user, and instead a plausible guess is constructed internally.
    */
    ComponentWithDF(
        const df::BaseDistributionFunction& df,
        double rmin, double rmax,
        unsigned int numCoefsRadial, unsigned int numCoefsAngular,
        double relError=1e-3,
        unsigned int maxNumEval=1e5);
    
    /** delete the internally created density model */
    virtual ~ComponentWithDF() { delete density; }

    /** reinitialize the density profile by recomputing the values of density at a set of 
        grid points in the meridional plane, and then constructing a spherical-harmonic
        density expansion from these values.
    */
    virtual void update(const potential::BasePotential& pot, const actions::BaseActionFinder& af);

    /** return the pointer to the internal density profile */
    virtual const potential::BaseDensity*   getDensity()   const { return density; }

    /** no additional potential component is provided */
    virtual const potential::BasePotential* getPotential() const { return 0; }

protected:
    /// action-based distribution function (not owned by this object and stays constant),
    const df::BaseDistributionFunction& distrFunc;
    
    /// spherical-harmonic expansion of density profile of this component
    const potential::BaseDensity* density;
    
    /// definition of grid for computing the density profile:
    const double rmin, rmax;             ///< range of radii for the logarithmic grid
    const unsigned int numCoefsRadial;   ///< number of grid points in radius
    const unsigned int numCoefsAngular;  ///< maximum order of angular-harmonic expansion (l_max)

    /// Parameters controlling the accuracy of density computation:
    /// required relative error in density
    const double relError;
    
    /// maximum number of DF evaluations during density computation at a single point
    const unsigned int maxNumEval;
};

/** A further modification of a component specified by a DF, suitable for disk-like profiles.
    It provides two contributions to the total model: first, an analytic potential of
    of a strongly flattened density distribution (DiskAnsatz); second, the difference
    between the true density generated by this DF and the Laplacian of the analytic potential.
    The latter is provided in terms of a spherical-harmonic expansion of this residual density
    profile, which is not strongly confined to the disk plane and should be efficiently
    approximated with a moderate number of terms in the angular expansion.
*/
class ComponentWithDisklikeDF: public ComponentWithDF {
public:
    /** create the component with the given DF, parameters of the grid for representing
        the spherical-harmonic expansion of the residual density, and parameters of the analytic
        potential. Most arguments are the same as for `ComponentWithDF`.
        \param[in] df -- the distribution function;
        \param[in] rmin,rmax -- the extent of the logarithmic grid in radius;
        \param[in] numCoefsRadial -- the size of radial grid;
        \param[in] numCoefsAngular -- the maximum order of spherical-harmonic expansion (l_max);
        \param[in] diskParams -- parameters of the analytic potential (DiskAnsatz);
        \param[in] relError -- relative accuracy in density computation;
        \param[in] maxNumEval -- maximum # of DF evaluations for a single density value.
    */
    ComponentWithDisklikeDF(
        const df::BaseDistributionFunction& df,
        double rmin, double rmax,
        unsigned int numCoefsRadial, unsigned int numCoefsAngular,
        const potential::DiskParam& diskParams,
        double relError=1e-3,
        unsigned int maxNumEval=1e5
        );

    virtual ~ComponentWithDisklikeDF() { delete diskAnsatzPotential; }

    /** recompute both the analytic disk potential and the spherical-harmonic expansion
        of the residual density profile.
    */
    virtual void update(const potential::BasePotential& pot, const actions::BaseActionFinder& af);

    /** return the pointer to the internal analytic potential model */
    virtual const potential::BasePotential* getPotential() const { return diskAnsatzPotential; }
private:
    const potential::BasePotential* diskAnsatzPotential;  ///< analytic potential
};

/// Class for providing a progress report during self-consistent modelling
class ProgressReportCallback {
public:
    virtual ~ProgressReportCallback() {};
    
    /** any message from the SelfConsistentModel engine not covered by specialized methods */
    virtual void generalMessage(const char* /*msg*/) {};

    /** called after a component with the given index has been updated */
    virtual void reportComponentUpdate(unsigned int /*componentIndex*/,
        const BaseComponent& /*component*/) {};

    /** called after the total potential has been recomputed */
    virtual void reportTotalPotential(const potential::BasePotential& /*potential*/) {};
};


/** The driver class for self-consistend modelling.
    The list of model components is provided at initialization;
    each component may be 'dead' (with a fixed density profile and unspecified DF) 
    or 'live' (with a specified DF, which stays constant throughout the iterative procedure).
    All components have associated density and/or potential profiles
    (either fixed at the beginning for 'dead' components, or recomputed from DF after 
    each iteration for 'live' components).
    The total potential consists of a multipole expansion, computed from the sum of density 
    profiles of all components that provide them, plus any additional potential models
    that may be provided by the components.
    Recall that a component specified by its DF provides its density profile 
    already in terms of a spherical-harmonic expansion. If this is the only density
    component, its density is directly used in the Multipole potential, while if 
    there are several density constituents, another spherical-harmonic expansion of
    the combined density is created. This double work has in fact a negligible overhead,
    because most of the computational effort is spent on the first stage (computing 
    the density profile by integration over DF, and taking its spherical-harmonic expansion).
    Moreover, a transformation between two spherical-harmonic expansions is exact
    if the order of the second one (used in the Multipole potential) is the same or greater
    than the first one (provided by components), and if their radial grids coincide;
    if the latter is not true, it introduces a generally small additional interpolation error.
    The overall potential is in turn used in recomputing the live density profiles through
    integration of their DFs over velocities. This two-stage process is performed 
    at each iteration: first the density recomputation, then the potential update.
    The overall potential may be queried at any time, but this object exists only until
    the next iteration, or until the `SelfConsistentModel` object is destroyed;
    same is true for the density profiles of each component.
    One may create copies of them with the `clone()` method of `BaseDensity` or `BasePotential`,
    and then use to initialize a new SelfConsistentModel from the already computed 
    approximations for the density, possibly with a different combination of dead/live components.
*/
class SelfConsistentModel {
public:
    /** Construct the model and initialize the first guess for the total potential.
        \param[in] components  is the array of pointers to the model components
        (they must have been constructed before, exist during the lifetime of 
        SelfConsistentModel object, and are not deleted when this object is destroyed).
        \param[in] rmin,rmax  determine the extent of radial grid used in multipole expansion;
        \param[in] numCoefsRadial  is the number of nodes in the (logarithmic) radial grid;
        \param[in] numCoefsAngular  is the order of spherical-harmonic expansion;
        in general, these parameters should encompass the range of analogous parameters 
        of all components that have a spherical-harmonic density representation.
        \param[in] callback  is the optional pointer to the progress reporting routine (may be NULL).
    */
    SelfConsistentModel(
        const std::vector<BaseComponent*>& components,
        double rmin, double rmax,
        unsigned int numCoefsRadial, unsigned int numCoefsAngular,
        ProgressReportCallback* callback=0
    );
    
    /** Destructor deletes the internally created total potential and action finder objects,
        but does not touch the model components passed to the constructor (nor the callback routine).
    */
    ~SelfConsistentModel();
    
    /** Main iteration step: recompute the densities of all components
        (if appropriate, i.e. if they have a specified DFs and can recompute 
        their density profile using the current total potential), 
        and then update the total potential and reinit action finder.
    */
    void doIteration();
    
    /** return the pointer to the internally used total potential: 
        this pointer is valid until another iteration is performed, or the object itself is destroyed;
        if one needs to have a permanent copy then the `BasePotential::clone()` method should be used,
        like  const BasePotential* pot = model.getPotential()->clone();
    */
    const potential::BasePotential* getPotential() const {
        return totalPotential; }
    
    /** return the pointer to the given component, or throw an exception if index is out of range */
    const BaseComponent* getComponent(unsigned int index) const;

private:
    /// array of model components
    std::vector<BaseComponent*> components;

    /// definition of grid for computing the multipole expansion of the total density profile:
    const double rmin, rmax;             ///< range of radii for the logarithmic grid
    const unsigned int numCoefsRadial;   ///< number of grid points in radius
    const unsigned int numCoefsAngular;  ///< maximum order of angular-harmonic expansion (l_max)
    
    /// Total gravitational potential of all components
    const potential::BasePotential* totalPotential;
    
    /// Action finder associated with the total potential
    const actions::BaseActionFinder* actionFinder;
    
    /// External progress reporting interface (may be NULL)
    ProgressReportCallback* callback;

    /// recompute the total potential using the current density profiles for all components
    void updateTotalPotential();
};
    
}  // namespace
