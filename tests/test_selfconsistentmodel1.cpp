/** \file    test_selfconsistentmodel.cpp
    \author  Eugene Vasiliev
    \date    November 2015

    This example demonstrates the machinery for constructing multicomponent self-consistent models
    specified by distribution functions in terms of actions.
*/
#include "df_halo.h"
#include "galaxymodel_selfconsistent.h"
#include "galaxymodel.h"
#include "potential_dehnen.h"
#include "particles_io.h"
#include "actions_staeckel.h"
#include <iostream>

// interface for progress reporting
class ProgRep: public galaxymodel::ProgressReportCallback {
public:

    virtual void generalMessage(const char* msg)
    {
        std::cout << msg << '\n';
    }
    virtual void reportComponentUpdate(unsigned int componentIndex,
        const galaxymodel::BaseComponent& comp)
    {
        const potential::BaseDensity* density = comp.getDensity();
        if(density==NULL)
            density = comp.getPotential();
        std::cout << "Component #" << componentIndex;
        if(density==NULL)
            std::cout << " is DEAD!!!\n";
        else
            std::cout <<
            ": inner density slope=" << getInnerDensitySlope(*density) <<
            ", rho(R=1)=" << density->density(coord::PosCyl(1,0,0)) <<
            ", rho(z=1)=" << density->density(coord::PosCyl(0,1,0)) <<
            ", total mass=" << density->totalMass() << '\n';
    }

    virtual void reportTotalPotential(const potential::BasePotential& potential)
    {
        std::cout << "Potential is updated; "
            "inner density slope=" << getInnerDensitySlope(potential) <<
            ", Phi(0)="   << potential.value(coord::PosCyl(0,0,0)) <<
            ", rho(R=1)=" << potential.density(coord::PosCyl(1,0,0)) <<
            ", rho(z=1)=" << potential.density(coord::PosCyl(0,1,0)) <<
            ", total mass="   << potential.totalMass() << '\n';
    }
};

int main()
{
    ProgRep progressReporter;  // print out various information during the process
    
    // parameters of the inner component
    double norm  = 1.0;  // approximately equals the total mass
    double alpha = 1.6;  // determines the inner density slope
    double beta  = 5.5;  // same for the outer slope
    double j0    = 0.5;  // determines the break radius in density profile
    double jcore = 0.0;  // inner plateau in density (disabled here)
    double ar    = 1.5;  // determines the velocity anisotropy in the inner region
    double az    = 1.0;  // the ratio between these two
    double aphi  = 0.5;  // determines the flattening of the inner region
    double br    = 1.0;  // same for
    double bz    = 1.2;  // the outer
    double bphi  = 0.8;  // region
    const df::DoublePowerLawParam paramInner = {norm,j0,jcore,alpha,beta,ar,az,aphi,br,bz,bphi};
    // create the instance of distribution function
    const df::DoublePowerLaw dfInner(paramInner);

    // may also provide a first guess for the density profile of a component,
    // even though it's not necessary, but may speed up convergence
    double pot_mass        = 1.0;   // total mass
    double pot_scalerad    = 0.05;  // break radius in the density profile
    double pot_axis_y_to_x = 1.;    // this must be unity for an axisymmetric model
    double pot_axis_z_to_x = 0.8;   // flattening in z direction
    double pot_gamma       = 1.0;   // inner density slope
    potential::BaseDensity* guessForDensityInner = new potential::Dehnen(
        pot_mass, pot_scalerad, pot_axis_y_to_x, pot_axis_z_to_x, pot_gamma);

    // parameters of the outer component
    norm  = 10.0;
    alpha = 0.5;
    beta  = 4.0;
    j0    = 5.0;
    jcore = 0.01;
    ar    = 1.3;
    az    = 1.1;
    aphi  = 0.6;
    br    = 1.5;
    bz    = 0.8;
    bphi  = 0.7;
    const df::DoublePowerLawParam paramOuter = {norm,j0,jcore,alpha,beta,ar,az,aphi,br,bz,bphi};
    const df::DoublePowerLaw dfOuter(paramOuter);

    // we can compute the masses even though we don't know the density profile yet
    std::cout << "Two-component model: masses are " << 
        dfInner.totalMass() << " and " << dfOuter.totalMass() << "\n";

    // First example: we use a 'dead' (fixed) density profile for the inner component,
    // without specifying its DF, and a 'live' outer component specified by its DF
    std::vector<galaxymodel::BaseComponent*> components;
    components.push_back(new galaxymodel::ComponentWithStaticDensity(*guessForDensityInner));
    components.push_back(new galaxymodel::ComponentWithDF(dfOuter, 5e-3, 500, 50, 4));

    galaxymodel::SelfConsistentModel* model = 
        new galaxymodel::SelfConsistentModel(components, 1e-3, 1e3, 100, 6, &progressReporter);

    // do a few iterations
    for(int iter=0; iter<3; iter++) {
        std::cout << "Starting iteration #"<<iter<<'\n';
        model->doIteration();
    }
    // receive the density profile of the outer ('live') component, and make a copy of it
    const potential::BaseDensity* guessForDensityOuter =
        model->getComponent(1)->getDensity()->clone();

    // destroy all traces
    delete model;
    for(unsigned int i=0; i<components.size(); i++)
        delete components[i];
    
    // Second example: now use two live components (both specified by their DFs),
    // and provide the initial guess for the density profile of the outer component,
    // obtained at the previous stage
    components.clear();
    components.push_back(new galaxymodel::ComponentWithDF(dfInner, 1e-3, 40,  50, 6, *guessForDensityInner));
    components.push_back(new galaxymodel::ComponentWithDF(dfOuter, 5e-3, 500, 50, 4, *guessForDensityOuter));
    delete guessForDensityInner;
    delete guessForDensityOuter;  // not needed anymore

    model = new galaxymodel::SelfConsistentModel(components, 1e-3, 1e3, 100, 6, &progressReporter);
    for(int iter=0; iter<5; iter++) {
        std::cout << "Starting iteration #"<<iter<<'\n';
        model->doIteration();
    }
    
    // receive the overall potential from the model, and make a copy of it
    const potential::BasePotential* totalPotential = model->getPotential()->clone();

    // also get a copy of the updated density profile for the outer component
    guessForDensityOuter = model->getComponent(1)->getDensity()->clone();
    
    delete model;  // don't keep the model anymore
    for(unsigned int i=0; i<components.size(); i++)
        delete components[i];

    // export model to an N-body snapshot
    std::cout << "Creating an N-body representation of the model\n";

    // first create a representation of density profile without velocities
    // (just for demonstration), by drawing samples from the density distribution
    particles::PointMassArray<coord::PosCyl> points_dens;
    galaxymodel::generateDensitySamples(*guessForDensityOuter, 5e5, points_dens);
    particles::writeSnapshot("model_outer_dens.txt", units::ExternalUnits(), points_dens, "Text");
    delete guessForDensityOuter;

    // now create genuinely self-consistent models of both components,
    // by drawing positions and velocities from the DF in the given (self-consistent) potential
    const actions::ActionFinderAxisymFudge af(*totalPotential);  // need an action finder for this
    particles::PointMassArrayCar points;
    galaxymodel::generatePosVelSamples(
        galaxymodel::GalaxyModel(*totalPotential, af, dfInner), 1e5, points);
    particles::writeSnapshot("model_inner.txt", units::ExternalUnits(), points, "Text");

    // same for the other component: they are only self-consistent if used together
    galaxymodel::generatePosVelSamples(
        galaxymodel::GalaxyModel(*totalPotential, af, dfOuter), 5e5, points);
    particles::writeSnapshot("model_outer.txt", units::ExternalUnits(), points, "Text");
    
    delete totalPotential;
    return 0;
}
