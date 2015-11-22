#include "galaxymodel_selfconsistent.h"
#include "galaxymodel.h"
#include "actions_staeckel.h"
#include "potential_analytic.h"
#include "potential_sphharm.h"
#include "potential_composite.h"
#include "potential_galpot.h"
#include <cmath>
#include <stdexcept>
#include <cassert>

namespace galaxymodel{

using potential::PtrDensity;
using potential::PtrPotential;

/// Helper class for providing a BaseDensity interface to a density computed via integration over DF
class DensityFromDF: public potential::BaseDensity{
public:
    DensityFromDF(
        const potential::BasePotential& pot,
        const actions::BaseActionFinder& af,
        const df::BaseDistributionFunction& df,
        double _relError, unsigned int _maxNumEval,
        const potential::BaseDensity* _densityToSubtract=NULL) :
    model(pot, af, df), 
    relError(_relError), maxNumEval(_maxNumEval),
    densityToSubtract(_densityToSubtract) {};
    
    virtual potential::SymmetryType symmetry() const { return potential::ST_AXISYMMETRIC; }
    virtual const char* name() const { return myName(); };
    static const char* myName() { return "DensityFromDF"; };

private:
    const GalaxyModel model;  ///< aggregate of potential, action finder and DF
    double relError;          ///< requested relative error of density computation
    unsigned int maxNumEval;  ///< max # of DF evaluations per one density calculation
    /// pointer to an (optional) density profile that is subtracted from the computed one
    const potential::BaseDensity* densityToSubtract; ///< (may be NULL, not owned by this object)
    
    virtual double densityCar(const coord::PosCar &pos) const {
        return densityCyl(toPosCyl(pos)); }
    
    virtual double densitySph(const coord::PosSph &pos) const {
        return densityCyl(toPosCyl(pos)); }
    
    /// compute the density as the integral of DF over velocity at a given position
    virtual double densityCyl(const coord::PosCyl &point) const {
        double result;
        computeMoments(model, point, relError, maxNumEval, &result, NULL, NULL, NULL, NULL, NULL);
        if(densityToSubtract != NULL)
            result -= densityToSubtract->density(point);
        return result;
    }
};

template<typename T>
const T& ensureNotNull(const T& x) {
    if(x) return x;
    throw std::invalid_argument("NULL pointer in assignment");
}

ComponentWithDF::ComponentWithDF(
    const df::PtrDistributionFunction& df,
    double _rmin, double _rmax,
    unsigned int _numCoefsRadial, unsigned int _numCoefsAngular,
    const PtrDensity& initDensity,
    double _relError, unsigned int _maxNumEval) :
distrFunc(ensureNotNull(df)),
density(initDensity ? initDensity :  // if the first guess for density is provided, use it;
    // otherwise initialize it as a simple Plummer model with the correct total mass
    // and a (hopefully) reasonable scale radius
    PtrDensity(new potential::Plummer(df->totalMass(), sqrt(_rmin*_rmax)))),
rmin(_rmin), rmax(_rmax),
numCoefsRadial(_numCoefsRadial), numCoefsAngular(_numCoefsAngular),
relError(_relError), maxNumEval(_maxNumEval)
{}

void ComponentWithDF::update(
    const potential::BasePotential& totalPotential,
    const actions::BaseActionFinder& actionFinder)
{
    // temporary density wrapper object
    const DensityFromDF densityWrapper(
        totalPotential, actionFinder, *distrFunc, relError, maxNumEval);
    
    // recompute the spherical-harmonic expansion for the density
    density = PtrDensity(new potential::DensitySphericalHarmonic(
        numCoefsRadial, numCoefsAngular, densityWrapper, rmin, rmax));
}

ComponentWithDisklikeDF::ComponentWithDisklikeDF(
    const df::PtrDistributionFunction& df,
    double _rmin, double _rmax,
    unsigned int _numCoefsRadial, unsigned int _numCoefsAngular,
    const potential::DiskParam& _params,
    double _relError, unsigned int _maxNumEval) :
ComponentWithDF(df, _rmin, _rmax, _numCoefsRadial, _numCoefsAngular, 
    PtrDensity(new potential::DiskResidual(_params)), _relError, _maxNumEval),
diskAnsatzPotential(new potential::DiskAnsatz(_params))
{}

// in the present implementation, we do not (yet) reinitialize the analalytic disk potential
// on each iteration, even though it is possible in principle.
void ComponentWithDisklikeDF::update(
    const potential::BasePotential& totalPotential,
    const actions::BaseActionFinder& actionFinder)
{
    // temporary density wrapper object that subtracts the density of the analytic disk component
    // from the overall density computed from the DF
    const DensityFromDF densityWrapper(
        totalPotential, actionFinder, *distrFunc, relError, maxNumEval, diskAnsatzPotential.get());
    
    // recompute the spherical-harmonic expansion for the density
    density = PtrDensity(new potential::DensitySphericalHarmonic(
        numCoefsRadial, numCoefsAngular, densityWrapper, rmin, rmax));
}
    
//------------ Driver class for self-consistent modelling ------------//

SelfConsistentModel::SelfConsistentModel(
    const std::vector<PtrComponent>& _components,
    double _rmin, double _rmax,
    unsigned int _numCoefsRadial, unsigned int _numCoefsAngular,
    const PtrProgressReportCallback& _callback) :
components(_components),
rmin(_rmin), rmax(_rmax),
numCoefsRadial(_numCoefsRadial), numCoefsAngular(_numCoefsAngular),
totalPotential(), actionFinder(), callback(_callback)
{
    if(components.size()<=0)
        throw std::invalid_argument("SelfConsistentModel: no components provided");
    updateTotalPotential();  // init totalPotential and actionFinder
}

void SelfConsistentModel::doIteration()
{
    for(unsigned int index=0; index<components.size(); index++) {
        // update the density of each component
        // (this may be a no-op if the component is 'dead', i.e. provides only a fixed density or 
        // potential, but does not possess a DF) -- the implementation is at the discretion of 
        // each component individually.
        components[index]->update(*totalPotential, *actionFinder);

        // report progress if necessary
        if(callback)
            callback->reportComponentUpdate(index, *components[index]);
    }

    // now update the overall potential and reinit the action finder
    updateTotalPotential();
}

void SelfConsistentModel::updateTotalPotential()
{
    assert(components.size()>=1);
    // temporary array of density and potential objects from components
    std::vector<PtrDensity> compDens;
    std::vector<PtrPotential> compPot;
    
    // first retrieve non-zero density and potential objects from all components
    for(unsigned int i=0; i<components.size(); i++) {
        PtrDensity d = components[i]->getDensity();
        if(d)
            compDens.push_back(d);
        PtrPotential p = components[i]->getPotential();
        if(p)
            compPot.push_back(p);
    }
    
    // the total density to be used in multipole expansion
    PtrDensity totalDensity;
    if(compDens.size()>1)  // if more than one density component is present, 
        totalDensity =     // create a temporary composite density object;
#ifdef HAVE_CXX11
        std::make_shared<potential::CompositeDensity>(compDens);
#else
        PtrDensity(new potential::CompositeDensity(compDens));
#endif
    else if(compDens.size()>0)       // if only one component is present,  
        totalDensity = compDens[0];  // simply copy it;
    // otherwise don't use multipole expansion at all
    
    // construct potential expansion from the total density
    if(totalDensity != NULL) {
        if(callback)
            callback->generalMessage("Initializing Multipole");
        // add it as one of potential components (possibly the only one)
        compPot.push_back(
            PtrPotential(new potential::Multipole(*totalDensity,
            rmin, rmax, numCoefsRadial, numCoefsAngular)));
    }
    
    // now check if the total potential is elementary or composite
    if(compPot.size()==0)
        throw std::runtime_error("No potential is present in SelfConsistentModel");
    if(compPot.size()==1)
        totalPotential = compPot[0];
    else
#ifdef HAVE_CXX11
        totalPotential = std::make_shared<potential::CompositeCyl>(compPot);
#else
        totalPotential = PtrPotential(new potential::CompositeCyl(compPot));
#endif

    // update the action finder
    if(callback)
        callback->generalMessage("Initializing action finder");
    // replace the raw pointer with a new one and dispose of the old one
    actionFinder.reset(new actions::ActionFinderAxisymFudge(totalPotential));
    
    // report progress if necessary
    if(callback)
        callback->reportTotalPotential(*totalPotential);
}

}  // namespace
