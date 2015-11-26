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

#ifdef VERBOSE_REPORT
#include <iostream>
#endif

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

#if 0
// old approach, not used anymore //
//---------- Component with disk-like DF and its auxiliary routines ----------//

/** Disk radial and vertical functions are modified by multiplying by a spline function */
class FncTimesSpline: public math::IFunction {
public:
    FncTimesSpline(const math::PtrFunction& _fnc, const math::CubicSpline& _spl) :
        fnc(_fnc), spl(_spl) {};
private:
    const math::PtrFunction fnc;  /// shared pointer to another function
    const math::CubicSpline spl;  /// exclusively owned spline function
    /**  evaluate  f(x) and optionally its two derivatives, if these arguments are not NULL  */
    virtual void evalDeriv(double x, double* val=NULL, double* der=NULL, double* der2=NULL) const {
        double fval, fder, fder2, sval, sder, sder2;
        bool der1 = der2!=NULL || der!=NULL;
        fnc->evalDeriv(x, &fval, der1 ? &fder : NULL, der2 ? &fder2 : NULL);
        spl. evalDeriv(x, &sval, der1 ? &sder : NULL, der2 ? &sder2 : NULL);
        if(val)
            *val = fval * sval;
        if(der)
            *der = fval * sder + fder * sval;
        if(der2)
            *der2 = fder2 * sval + 2 * fder * sder + fval * sder2;
    }
    virtual unsigned int numDerivs() const { return 2; }
};

class SeparableDiskFittingFnc: public math::IFunctionNdim {
public:
    SeparableDiskFittingFnc(const potential::DensitySphericalHarmonic& densOrig,
        const math::PtrFunction& _radialFncBase, const math::PtrFunction& _verticalFncBase,
        const std::vector<double>& _radialNodes, const std::vector<double>& _verticalNodes) :
    radialFncBase(_radialFncBase), verticalFncBase(_verticalFncBase),
    radialNodes(_radialNodes), verticalNodes(_verticalNodes) {
        densOrig.getCoefs(radii, coefsOrig);
    }

    void assembleResidualCoefs(const PtrPotential& diskAnsatz,
        std::vector< std::vector<double> >& coefsResidual) const
    {
        // create another sph-harm expansion for the analytic disk component to be subtracted
        potential::DensitySphericalHarmonic densSubtract(
            radii.size(), coefsOrig.size()-1, *diskAnsatz, radii.front(), radii.back());
        std::vector<double> radii1;
        densSubtract.getCoefs(radii1, coefsResidual);
        assert(radii.size() == radii1.size());
        assert(coefsOrig.size() == coefsResidual.size());
        for(unsigned int l=0; l<coefsOrig.size(); l++) {
            assert(coefsOrig[l].size() == coefsResidual[l].size());
            for(unsigned int n=0; n<coefsOrig[l].size(); n++)
                coefsResidual[l][n] = coefsOrig[l][n] - coefsResidual[l][n];
        }
    }

    PtrPotential createDiskAnsatz(const double vars[]) const {
        std::vector<double> valRad(radialNodes.size()), valVer(verticalNodes.size());
        for(unsigned int i=0; i<radialNodes.size(); i++)
            valRad[i] = vars[i];
        for(unsigned int i=0; i<verticalNodes.size(); i++)
            valVer[i] = vars[i+radialNodes.size()];
        return PtrPotential(new potential::DiskAnsatz(
            math::PtrFunction(new FncTimesSpline(radialFncBase,
                math::CubicSpline(radialNodes,   valRad))),
            /*math::PtrFunction(new FncTimesSpline(verticalFncBase,
                math::CubicSpline(verticalNodes, valVer, 0, 0)))*/
            verticalFncBase
        ) );
    }

    PtrDensity createDiskResidual(const PtrPotential& diskAnsatz) const {
        std::vector< std::vector<double> > coefsResidual;
        assembleResidualCoefs(diskAnsatz, coefsResidual);
        return PtrDensity(new potential::DensitySphericalHarmonic(radii, coefsResidual));
    }

    double computeResidual(const std::vector< std::vector<double> >& coefsResidual) const {
        double sum = 0;
        potential::DensitySphericalHarmonic densRes(radii, coefsResidual);
        for(unsigned int k=0; k<radii.size(); k++)
            sum += pow_2(/*densInPlane[k] -*/ densRes.density(coord::PosCyl(radii[k],0,0))) *
                pow_2(radii[k]) * (radii[k] - (k>0 ? radii[k-1] : 0));
        return sum;
    }

    virtual void eval(const double vars[], double values[]) const {
        PtrPotential diskAnsatz = createDiskAnsatz(vars);
        std::vector< std::vector<double> > coefsResidual;
        assembleResidualCoefs(diskAnsatz, coefsResidual);
        values[0] = computeResidual(coefsResidual);
#ifdef VERBOSE_REPORT
        std::cout << ": ";
        for(unsigned int i=0; i<numVars(); i++)
            std::cout << vars[i] << ' ';
        std::cout << "= " << values[0] << '\n';
#endif
    }

    virtual unsigned int numVars() const { return radialNodes.size() + verticalNodes.size(); }
    virtual unsigned int numValues() const { return 1; }
private:
    math::PtrFunction radialFncBase;
    math::PtrFunction verticalFncBase;
    std::vector<double> radialNodes;
    std::vector<double> verticalNodes;
    std::vector<double> radii;
    std::vector< std::vector<double> > coefsOrig;
};

ComponentWithDisklikeDF::ComponentWithDisklikeDF(
    const df::PtrDistributionFunction& df,
    double _rmin, double _rmax,
    unsigned int _numCoefsRadial, unsigned int _numCoefsAngular,
    const potential::DiskParam& params,
    double _relError, unsigned int _maxNumEval) :
ComponentWithDF(df, _rmin, _rmax, _numCoefsRadial, _numCoefsAngular, 
    PtrDensity(new potential::DiskResidual(params)), _relError, _maxNumEval),
radialFncBase(createRadialDiskFnc(params)),
verticalFncBase(createVerticalDiskFnc(params)),
diskAnsatzPotential(new potential::DiskAnsatz(radialFncBase, verticalFncBase))
{
    radialNodes.assign(5,0);
    //verticalNodes.assign(5,0);
    for(int i=1; i<5; i++) {   // place nodes at 0, 0.5, 1, 2, 4
        radialNodes[i]   = params.scaleRadius * pow(2., i-2.);
        //verticalNodes[i] = fabs(params.scaleHeight) * pow(2., i-2.);
    }
}

void ComponentWithDisklikeDF::update(
    const potential::BasePotential& totalPotential,
    const actions::BaseActionFinder& actionFinder)
{
    // first compute the spherical-harmonic expansion of the entire density profile
    ComponentWithDF::update(totalPotential, actionFinder);
    const potential::DensitySphericalHarmonic& densOrig =
        static_cast<const potential::DensitySphericalHarmonic&>(*density);
#ifdef VERBOSE_REPORT
    writeDensityExpCoefs("dens_orig", densOrig);
#endif

    const unsigned int NNODES = radialNodes.size() + verticalNodes.size();
    std::vector<double> initVals(NNODES), initSteps(NNODES), optimalVals(NNODES);
    for(unsigned int i=0; i<NNODES; i++) {
        initVals[i]  = 1.0;
        initSteps[i] = 0.1;
    }
    SeparableDiskFittingFnc fnc(densOrig, radialFncBase, verticalFncBase, radialNodes, verticalNodes);
#ifdef VERBOSE_REPORT
    writeDensityExpCoefs("dens_resid_init",
        static_cast<const potential::DensitySphericalHarmonic&>(
        *fnc.createDiskResidual(fnc.createDiskAnsatz(&initVals.front()))));
#endif
    math::findMinNdim(fnc, &initVals.front(), &initSteps.front(), 1e-2, 1000, &optimalVals.front());

    diskAnsatzPotential = fnc.createDiskAnsatz(&optimalVals.front());
    density = fnc.createDiskResidual(diskAnsatzPotential);
#ifdef VERBOSE_REPORT
    writeDensityExpCoefs("dens_residual", 
        static_cast<const potential::DensitySphericalHarmonic&>(*density));
#endif
}
#endif

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
