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
#include <vector>

// forward declaration
namespace potential{ struct DiskParam; }

namespace galaxymodel{

/** Description of a single component of the total model.
    It may provide the density profile, to be used in the multipole or CylSpline
    expansion, or a potential to be added directly to the total potential, or both.
    In case that this component has a DF, its density and potential may be recomputed
    with the `update` method, using the given total potential and its action finder.
*/
class BaseComponent {
public:
    BaseComponent(bool _isDensityDisklike) : isDensityDisklike(_isDensityDisklike)  {};
    virtual ~BaseComponent() {};

    /** recalculate the density profile (and possibly the additional potential)
        by integrating the DF over velocities in the given total potential.
        In case that the component does not have a DF, this method does nothing.
    */
    virtual void update(const potential::BasePotential& totalPotential,
        const actions::BaseActionFinder& actFinder) = 0;

    /** return the pointer to the internally used density profile for the given component;
        if it returns NULL, then this component does not participate in 
        the spherical-harmonic density expansion of the SelfConsistentModel.
    */    
    virtual potential::PtrDensity   getDensity()   const = 0;

    /** return the pointer to the additional potential component to be used as part of
        the total potential, or NULL if not applicable.
    */
    virtual potential::PtrPotential getPotential() const = 0;
    
    /** in case the component has an associated density profile, it may be used
        in construction of either multipole (spherical-harmonic) potential expansion,
        or a 'CylSpline' expansion of potential in the meridional plane.
        The former is more suitable for spheroidal and the latter for disk-like components.
    */
    const bool isDensityDisklike;
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
    ComponentStatic(const potential::PtrDensity& dens, bool _isDensityDisklike) :
        BaseComponent(_isDensityDisklike), density(dens), potential() {}
    ComponentStatic(const potential::PtrPotential& pot) :
        BaseComponent(false), density(), potential(pot) {}
    ComponentStatic(const potential::PtrDensity& dens, bool _isDensityDisklike,
        const potential::PtrPotential& pot) :
        BaseComponent(_isDensityDisklike), density(dens), potential(pot) {}
    /** update does nothing */
    virtual void update(const potential::BasePotential&, const actions::BaseActionFinder&) {}
    virtual potential::PtrDensity   getDensity()   const { return density; }
    virtual potential::PtrPotential getPotential() const { return potential; }
private:
    potential::PtrDensity density;     ///< shared pointer to the input density, if provided
    potential::PtrPotential potential; ///< shared pointer to the input potential, if exists
};


/** A (partial) specialization for the component with the density profile computed from a DF,
    using either a spherical-harmonic expansion or a 2d interpolation in meridional plane
    (detailed in two derived classes).
    Since the density computation from DF is very expensive, the density object provided by
    this component does not directly represent this interface. 
    Instead, during the update procedure, the DF-integrated density is computed at a moderate
    number of points (<~ 10^3) and used in creating an intermediate representation, that in turn
    provides the density everywhere in space by suitably interpolating from the computed values.
    The two derived classes differ in the way this intermediate representation is constructed:
    either as a spherical-harmonic expansion, or as 2d interpolation in R-z plane.
*/
class BaseComponentWithDF: public BaseComponent {
public:
    /** create a component with the given distribution function
        (the DF remains constant during the iterative procedure) */
    BaseComponentWithDF(const df::PtrDistributionFunction& df,
        const potential::PtrDensity& initDensity, bool _isDensityDisklike,
        double _relError, unsigned int _maxNumEval) :
    BaseComponent(_isDensityDisklike), distrFunc(df), density(initDensity),
    relError(_relError), maxNumEval(_maxNumEval) {}
    
    /** return the pointer to the internal density profile */
    virtual potential::PtrDensity   getDensity()   const { return density; }

    /** no additional potential component is provided, i.e. an empty pointer is returned */
    virtual potential::PtrPotential getPotential() const { return potential::PtrPotential(); }

protected:
    /// shared pointer to the action-based distribution function (remains unchanged)
    const df::PtrDistributionFunction distrFunc;
    
    /// spherical-harmonic expansion of density profile of this component
    potential::PtrDensity density;

    /// Parameters controlling the accuracy of density computation:
    /// required relative error in density
    const double relError;
    
    /// maximum number of DF evaluations during density computation at a single point
    const unsigned int maxNumEval;
};


/** Specialization of a component with DF and spheroidal density profile,
    which will be represented by a spherical-harmonic expansion */
class ComponentWithSpheroidalDF: public BaseComponentWithDF {
public:
    /** construct a component with given DF and parameters of spherical-harmonic expansion
        for representing its density profile.
        \param[in]  df -- shared pointer to the distribution function of this component.
        \param[in]  initDensity -- the initial guess for the density profile of this component;
                    if hard to guess, one may start e.g. with a simple Plummer sphere with
                    correct total mass and a reasonable scale radius, but it doesn't matter much.
        \param[in]  rmin,rmax -- determine the extent of (logarithmic) radial grid
                    used to compute the density profile of this component.
        \param[in]  numCoefsRadial -- number of grid points in this radial grid.
        \param[in]  numCoefsAngular -- max order of spherical-harmonic expansion (l_max).
        \param[in]  relError -- relative accuracy of density computation.
        \param[in]  maxNumEval -- max # of DF evaluations per single density computation.
    */
    ComponentWithSpheroidalDF(const df::PtrDistributionFunction& df,
        const potential::PtrDensity& initDensity,
        double rmin, double rmax, unsigned int numCoefsRadial, unsigned int numCoefsAngular,
        double relError=1e-3, unsigned int maxNumEval=1e5);

    /** reinitialize the density profile by recomputing the values of density at a set of 
        grid points in the meridional plane, and then constructing a spherical-harmonic
        density expansion from these values.
    */
    virtual void update(const potential::BasePotential& pot, const actions::BaseActionFinder& af);
    
private:
    /// definition of grid for computing the density profile:
    const double rmin, rmax;             ///< range of radii for the logarithmic grid
    const unsigned int numCoefsRadial;   ///< number of grid points in radius
    const unsigned int numCoefsAngular;  ///< maximum order of angular-harmonic expansion (l_max)
};


/** Specialization of a component with DF and flattened density profile,
    which will be represented using 2d interpolation in the meridional plane */
class ComponentWithDisklikeDF: public BaseComponentWithDF {
public:
    /** create the component with the given DF and parameters of the grid for representing
        the density in the meridional plane.
        \param[in]  df -- shared pointer to the distribution function of this component.
        \param[in]  initDensity -- the initial guess for the density profile of this component.
        \param[in]  gridR -- grid in cylindrical radius defining the R-coordinate of points
                    at which density is computed. 0th element must be at R=0, and the grid
                    should cover the range in which the density is presumed to be non-negligible.
                    A suitable grid may be constructed by `math::createNonuniformGrid`.
        \param[in]  gridz -- grid in vertical direction, with the same requirements as gridR.
        \param[in]  relError -- relative accuracy in density computation;
        \param[in]  maxNumEval -- maximum # of DF evaluations for a single density value.
    */
    ComponentWithDisklikeDF(const df::PtrDistributionFunction& df,
        const potential::PtrDensity& initDensity,
        const std::vector<double> _gridR, const std::vector<double> _gridz,
        double relError=1e-3, unsigned int maxNumEval=1e5);

    /** recompute both the analytic disk potential and the spherical-harmonic expansion
     of the residual density profile. */
    virtual void update(const potential::BasePotential& pot, const actions::BaseActionFinder& af);
private:
    /// coordinates of grid nodes for computing the density profile
    std::vector<double> gridR, gridz;
};


#if 0
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
        const df::PtrDistributionFunction& df,
        double rmin, double rmax,
        unsigned int numCoefsRadial, unsigned int numCoefsAngular,
        const potential::DiskParam& diskParams,
        double relError=1e-3,
        unsigned int maxNumEval=1e5);

    /** recompute both the analytic disk potential and the spherical-harmonic expansion
        of the residual density profile. */
    virtual void update(const potential::BasePotential& pot, const actions::BaseActionFinder& af);

    /** return the pointer to the internal analytic potential model */
    virtual potential::PtrPotential getPotential() const { return diskAnsatzPotential; }
private:
    math::PtrFunction radialFncBase, verticalFncBase;
    potential::PtrPotential diskAnsatzPotential;  ///< analytic potential
    std::vector<double> radialNodes, verticalNodes;
};
#endif

/// smart pointer to the model component
#ifdef HAVE_CXX11
typedef std::shared_ptr<BaseComponent> PtrComponent;
#else
typedef std::tr1::shared_ptr<BaseComponent> PtrComponent;
#endif

/// array of components
typedef std::vector<PtrComponent> ComponentArray;

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
    The overall potential may be queried at any time, returning a shared pointer, which,
    if assigned, continues to exist even after the `SelfConsistentModel` object is destroyed;
    same is true for the density profiles returned by each component.
    One may use them to initialize a new SelfConsistentModel from the already computed 
    approximations for the density, possibly with a different combination of dead/live components.
*/
struct SelfConsistentModel {
public:
    /** parameters of grid for computing the multipole expansion of the combined
        density profile of spheroidal components;
        in general, these parameters should encompass the range of analogous parameters 
        of all components that have a spherical-harmonic density representation. */
    double rminSph, rmaxSph;      ///< range of radii for the logarithmic grid
    unsigned int sizeRadialSph;   ///< number of grid points in radius
    unsigned int lmaxAngularSph;  ///< maximum order of angular-harmonic expansion (l_max)
    
    /** parameters of grid for computing CylSpline expansion of the combined
        density profile of flattened (disk-like) components;
        the radial and vertical extent should be somewhat larger than the region where
        the overall density is non-negligible, and the resolution should match that
        of the density profiles of components. */
    double RminCyl, RmaxCyl;      ///< innermost (non-zero) and outermost grid nodes in cylindrical radius
    double zminCyl, zmaxCyl;      ///< innermost and outermost grid nodes in vertical direction
    unsigned int sizeRadialCyl;   ///< number of grid nodes in cylindrical radius
    unsigned int sizeVerticalCyl; ///< number of grid nodes in vertical (z) direction

    /// total gravitational potential of all components (empty at the beginning)
    potential::PtrPotential totalPotential;

    /// action finder associated with the total potential (empty at the beginning)
    actions::PtrActionFinder actionFinder;

    /// array of model components
    ComponentArray components;
};

/// recompute the total potential using the current density profiles for all components,
/// and reinitialize the action finder;
/// throws a runtime_error exception if the total potential does not have any constituents
void updateTotalPotential(SelfConsistentModel& model);

/** Main iteration step: recompute the densities of all components
    (if appropriate, i.e. if they have a specified DFs and can recompute 
    their density profile using the current total potential), 
    and then update the total potential and reinit action finder.
*/
void doIteration(SelfConsistentModel& model);
    
}  // namespace
