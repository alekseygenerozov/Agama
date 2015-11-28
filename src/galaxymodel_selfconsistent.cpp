#include "galaxymodel_selfconsistent.h"
#include "galaxymodel.h"
#include "actions_staeckel.h"
#include "potential_sphharm.h"
#include "potential_cylspline.h"
#include "potential_galpot.h"
#include "potential_composite.h"
#include <stdexcept>
#include <cassert>
#include <cmath>

namespace galaxymodel{

using potential::PtrDensity;
using potential::PtrPotential;

template<typename T>
const T& ensureNotNull(const T& x) {
    if(x) return x;
    throw std::invalid_argument("NULL pointer in assignment");
}

namespace{

/// Helper class for providing a BaseDensity interface to a density computed via integration over DF
class DensityFromDF: public potential::BaseDensity{
public:
    DensityFromDF(
        const potential::BasePotential& pot,
        const actions::BaseActionFinder& af,
        const df::BaseDistributionFunction& df,
        double _relError, unsigned int _maxNumEval) :
    model(pot, af, df), relError(_relError), maxNumEval(_maxNumEval) {};

    virtual potential::SymmetryType symmetry() const { return potential::ST_AXISYMMETRIC; }
    virtual const char* name() const { return myName(); };
    static const char* myName() { return "DensityFromDF"; };
    virtual double enclosedMass(const double) const {  // should never be used -- too slow
        throw std::runtime_error("DensityFromDF: enclosedMass not implemented"); }
private:
    const GalaxyModel model;  ///< aggregate of potential, action finder and DF
    double       relError;    ///< requested relative error of density computation
    unsigned int maxNumEval;  ///< max # of DF evaluations per one density calculation
    
    virtual double densityCar(const coord::PosCar &pos) const {
        return densityCyl(toPosCyl(pos)); }
    
    virtual double densitySph(const coord::PosSph &pos) const {
        return densityCyl(toPosCyl(pos)); }
    
    /// compute the density as the integral of DF over velocity at a given position
    virtual double densityCyl(const coord::PosCyl &point) const {
        double result;
        computeMoments(model, point, relError, maxNumEval, &result, NULL, NULL, NULL, NULL, NULL);
        return result;
    }
};
} // anonymous namespace

//--------- Components with DF ---------//

ComponentWithSpheroidalDF::ComponentWithSpheroidalDF(
    const df::PtrDistributionFunction& df,
    const potential::PtrDensity& initDensity,
    double _rmin, double _rmax,
    unsigned int _numCoefsRadial, unsigned int _numCoefsAngular,
    double _relError, unsigned int _maxNumEval) :
BaseComponentWithDF(ensureNotNull(df), ensureNotNull(initDensity), false, _relError, _maxNumEval),
rmin(_rmin), rmax(_rmax), numCoefsRadial(_numCoefsRadial), numCoefsAngular(_numCoefsAngular)
{
    if(rmin<=0 || rmax<=rmin || numCoefsRadial<2 || numCoefsAngular<0)
        throw std::invalid_argument("ComponentWithSpheroidalDF: Invalid grid parameters");
}

void ComponentWithSpheroidalDF::update(
    const potential::BasePotential& totalPotential,
    const actions::BaseActionFinder& actionFinder)
{
    // temporary density wrapper object
    const DensityFromDF densityWrapper(
        totalPotential, actionFinder, *distrFunc, relError, maxNumEval);

    // recompute the spherical-harmonic expansion for the density
    density.reset(new potential::DensitySphericalHarmonic(
        numCoefsRadial, numCoefsAngular, densityWrapper, rmin, rmax));
}

ComponentWithDisklikeDF::ComponentWithDisklikeDF(
    const df::PtrDistributionFunction& df,
    const potential::PtrDensity& initDensity,
    const std::vector<double> _gridR, const std::vector<double> _gridz,
    double _relError, unsigned int _maxNumEval) :
BaseComponentWithDF(ensureNotNull(df), ensureNotNull(initDensity), true, _relError, _maxNumEval),
gridR(_gridR), gridz(_gridz)
{
    if(gridR[0]!=0 || gridR.size()<2 || gridz[0]!=0 || gridz.size()<2)
        throw std::invalid_argument("ComponentWithDisklikeDF: Invalid grid parameters");
    gridR[0] = gridR[1]*1e-3;  ///!!! FIXME: apparently there is a problem for points exactly on z axis
    // in principle should also check if the grid is monotonic,
    // but this will be done by 2d interpolator anyway
}

void ComponentWithDisklikeDF::update(
    const potential::BasePotential& totalPotential,
    const actions::BaseActionFinder& actionFinder)
{
    // temporary density wrapper object
    const DensityFromDF densityWrapper(
        totalPotential, actionFinder, *distrFunc, relError, maxNumEval);

    // reinit the interpolator for density in meridional plane
    density.reset(new potential::DensityCylGrid(gridR, gridz, densityWrapper));
}


//------------ Driver routines for self-consistent modelling ------------//

void doIteration(SelfConsistentModel& model)
{
    // need to initialize the potential before the first iteration
    if(!model.totalPotential)
        updateTotalPotential(model);

    for(unsigned int index=0; index<model.components.size(); index++) {
        // update the density of each component (this may be a no-op if the component is 'dead',
        // i.e. provides only a fixed density or potential, but does not possess a DF) -- 
        // the implementation is at the discretion of each component individually.
        model.components[index]->update(*model.totalPotential, *model.actionFinder);
    }

    // now update the overall potential and reinit the action finder
    updateTotalPotential(model);
}

void updateTotalPotential(SelfConsistentModel& model)
{
    // temporary array of density and potential objects from components
    std::vector<PtrDensity> compDensSph;
    std::vector<PtrDensity> compDensDisk;
    std::vector<PtrPotential> compPot;

    // first retrieve non-zero density and potential objects from all components
    for(unsigned int i=0; i<model.components.size(); i++) {
        PtrDensity d = model.components[i]->getDensity();
        if(d) {
            if(model.components[i]->isDensityDisklike)
                compDensDisk.push_back(d);
            else
                compDensSph.push_back(d);
        }
        PtrPotential p = model.components[i]->getPotential();
        if(p)
            compPot.push_back(p);
    }

    // the total density to be used in multipole expansion for spheroidal components
    PtrDensity totalDensitySph;
    // if more than one density component is present, create a temporary composite density object;
    if(compDensSph.size()>1)
        totalDensitySph.reset(new potential::CompositeDensity(compDensSph));
    else
    // if only one component is present, simply copy it;
    if(compDensSph.size()>0)
        totalDensitySph = compDensSph[0];
    // otherwise don't use multipole expansion at all

    // construct potential expansion from the total density
    // and add it as one of potential components (possibly the only one)
    if(totalDensitySph != NULL)
        compPot.push_back(PtrPotential(
            new potential::Multipole(*totalDensitySph,
            model.rminSph, model.rmaxSph, model.sizeRadialSph, model.lmaxAngularSph)));

    // now the same for the total density to be used in CylSplineExp for the flattened components
    PtrDensity totalDensityDisk;
    if(compDensDisk.size()>1)
        totalDensityDisk.reset(new potential::CompositeDensity(compDensDisk));
    else if(compDensDisk.size()>0)
        totalDensityDisk = compDensDisk[0];

    if(totalDensityDisk != NULL)
        compPot.push_back(PtrPotential(
            new potential::CylSplineExp(model.sizeRadialCyl, model.sizeVerticalCyl, 0,
                *totalDensityDisk,
                model.RminCyl, model.RmaxCyl, model.zminCyl, model.zmaxCyl)));

    // now check if the total potential is elementary or composite
    if(compPot.size()==0)
        throw std::runtime_error("No potential is present in SelfConsistentModel");
    if(compPot.size()==1)
        model.totalPotential = compPot[0];
    else
        model.totalPotential.reset(new potential::CompositeCyl(compPot));

    // update the action finder
    model.actionFinder.reset(new actions::ActionFinderAxisymFudge(model.totalPotential));
}

}  // namespace
