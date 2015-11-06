#include "galaxymodel_selfconsistent.h"
#include "galaxymodel.h"
#include "actions_staeckel.h"
#include "potential_analytic.h"
#include "potential_sphharm.h"
#include "potential_composite.h"
#include "potential_galpot.h"
#include <cmath>
#include <stdexcept>

namespace galaxymodel{

/// Helper class for providing a BaseDensity interface to a density computed via integration over DF
class DensityFromDF: public potential::BaseDensity{
public:
    DensityFromDF(
        const potential::BasePotential& pot,
        const actions::BaseActionFinder& af,
        const df::BaseDistributionFunction& df,
        double _relError, unsigned int _maxNumEval,
        ProgressReportCallback* _callback, unsigned int _compIndex) :
    model(pot, af, df), 
    relError(_relError), maxNumEval(_maxNumEval),
    callback(_callback), compIndex(_compIndex) {};
    
    virtual potential::SymmetryType symmetry() const { return potential::ST_AXISYMMETRIC; }
    virtual const char* name() const { return myName(); };
    static const char* myName() { return "DensityFromDF"; };
    
private:
    const galaxymodel::GalaxyModel model;
    double relError;
    unsigned int maxNumEval;
    ProgressReportCallback* callback;
    unsigned int compIndex;
    
    virtual double densityCar(const coord::PosCar &pos) const {
        return densityCyl(toPosCyl(pos)); }
    
    virtual double densitySph(const coord::PosSph &pos) const {
        return densityCyl(toPosCyl(pos)); }
    
    /// compute the density as the integral of DF over velocity at a given position
    virtual double densityCyl(const coord::PosCyl &point) const {
        double result;
        computeMoments(model, point, relError, maxNumEval, &result, NULL, NULL, NULL, NULL, NULL);
        if(callback!=NULL)
            callback->reportDensityAtPoint(compIndex, point, result);
        return result;
    }
};

Component::Component(const df::BaseDistributionFunction* _distrFunc,
    double _rmin, double _rmax, 
    unsigned int _numCoefsRadial, unsigned int _numCoefsAngular):
    distrFunc(_distrFunc), rmin(_rmin), rmax(_rmax),
    numCoefsRadial(_numCoefsRadial), numCoefsAngular(_numCoefsAngular),
    // density is initialized as a simple Plummer model
    // with the correct total mass and a reasonable scale radius
    density(new potential::Plummer(distrFunc->totalMass(), sqrt(rmin*rmax)))
{}


SelfConsistentModel::SelfConsistentModel(
    const std::vector<Component>& _components,
    ProgressReportCallback* _callback,
    double _relError,
    unsigned int _maxNumEval) :
    components(_components), totalPotential(NULL), actionFinder(NULL),
    relError(_relError), maxNumEval(_maxNumEval), callback(_callback)
{
    if(components.size()<=0)
        throw std::invalid_argument("SelfConsistentModel: no components provided");
    updateTotalPotential();
}

void SelfConsistentModel::updateTotalPotential()
{
    // temporary array of density objects
    std::vector<const potential::BaseDensity*> comp_array(components.size());
    
    // determine the grid parameters for the overall potential expansion
    int numCoefsRadial  = 0;
    int numCoefsAngular = 0;
    double rmin = INFINITY, rmax = 0, deltalogr=INFINITY;
    for(std::vector<Component>::const_iterator comp=components.begin(); comp!=components.end(); ++comp) {
        rmin = fmin(rmin, comp->rmin);
        rmax = fmax(rmax, comp->rmax);
        numCoefsAngular = std::max<int>(numCoefsAngular, comp->numCoefsAngular);
        deltalogr = fmin(deltalogr, log(comp->rmax/comp->rmin) / comp->numCoefsRadial);
        comp_array.push_back(comp->density);
    }
    numCoefsRadial = log(rmax/rmin) / deltalogr;
    
    delete totalPotential;
    delete actionFinder;
    
    // if more than one component, create a temporary composite density object
    const potential::BaseDensity* totalDensity = components.size()>1 ?
        new potential::CompositeDensity(comp_array) : components[0].density;
    
    // construct potential expansion from the total density
    if(callback)
        callback->generalMessage("Initializing Multipole");
    totalPotential = new potential::Multipole(*totalDensity,
        rmin, rmax, numCoefsRadial, numCoefsAngular);
    
    // update the action finder
    if(callback)
        callback->generalMessage("Initializing action finder");
    actionFinder = new actions::ActionFinderAxisymFudge(*totalPotential);
    
    // delete temporary composite density if necessary
    if(components.size()>1)
        delete totalDensity;
    
    // report progress if necessary
    if(callback)
        callback->reportTotalPotential(*totalPotential);
}

void SelfConsistentModel::updateComponentDensity(unsigned int index)
{
    // temporary density wrapper object
    const DensityFromDF density(*totalPotential, *actionFinder, *components[index].distrFunc,
        relError, maxNumEval, callback, index);
    
    // recompute the spherical-harmonic expansion for the density
    delete components[index].density;
    components[index].density = new potential::DensitySphericalHarmonic(
        components[index].numCoefsRadial, components[index].numCoefsAngular,
        density, components[index].rmin, components[index].rmax);

    // report progress if necessary
    if(callback)
        callback->reportDensityUpdate(index, *components[index].density);
}

void SelfConsistentModel::doIteration()
{
    for(unsigned int index=0; index<components.size(); index++)
        updateComponentDensity(index);
    updateTotalPotential();
}

}  // namespace
