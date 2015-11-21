/** \file    test_selfconsistentmodel.cpp
    \author  Eugene Vasiliev
    \date    November 2015

    This example demonstrates the machinery for constructing multicomponent self-consistent models
    specified by distribution functions in terms of actions.
    We create a two-component galaxy with disk and halo components, using a two-stage approach:
    first, we take a static potential/density profile for the disk, and find a self-consistent
    density profile of the halo component in the presence of the disk potential;
    second, we replace the static disk with a DF-based component and find the overall self-consistent
    model for both components. The rationale is that a reasonable guess for the total potential
    is already needed before constructing the DF for the disk component, since the latter relies
    upon plausible radially-varying epicyclic frequencies.
    Both stages require a few iterations to converge.
*/
#include "df_halo.h"
#include "df_disk.h"
#include "galaxymodel_selfconsistent.h"
#include "galaxymodel.h"
#include "potential_composite.h"
#include "potential_utils.h"
#include "particles_io.h"
#include "actions_staeckel.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <cmath>

/// print the radial and angular dependence of a density profile into a text file
void printoutDensity(const std::string& fileName, const potential::BaseDensity& dens)
{
    const int NT=6;  // number of points in angle theta, which are Gauss-Legendre nodes for cos(theta)
    //double theta[NT] = {1.5707963268, 1.1528929537, 0.7354466143, 0.3204050903};
    double theta[NT] = {1.5707963268, 1.2405739234, 0.9104740292, 0.5807869795, 0.2530224166, 0};
    std::ofstream strm(fileName.c_str());
    strm << "#r\theta:";
    for(int t=0; t<NT; t++)
        strm << '\t' << theta[t];
    strm << '\n';
    for(double r=1./64; r<=1e6; r*=2) {
        strm << r;
        for(int t=0; t<NT; t++)
            strm << '\t' << dens.density(coord::PosSph(r, theta[t], 0));
        strm << '\n';
    }
}

void printoutRotationCurve(const std::string& fileName,
    const std::vector<const potential::BasePotential*>& potentials)
{
    std::ofstream strm(fileName.c_str());
    strm << "#radius";
    for(unsigned int i=0; i<potentials.size(); i++)
        strm << "\tcomp"<<i;
    strm << "\ttotal\n";
    for(double r=1./8; r<=30; r<1 ? r*=2 : r<16 ? r+=0.5 : r+=2) {
        strm << r;
        double v2sum = 0;
        for(unsigned int i=0; i<potentials.size(); i++) {
            coord::GradCyl deriv;
            potentials[i]->eval(coord::PosCyl(r, 0, 0), NULL, &deriv);
            v2sum += r*deriv.dR;
            strm << '\t' << sqrt(r*deriv.dR);
        }
        strm << '\t' << sqrt(v2sum) << '\n';
    }
}

/// generate an N-body representation of a density profile (without velocities) and store in a file
void storeNbodyDensity(const std::string& fileName, const potential::BaseDensity& dens)
{
    particles::PointMassArray<coord::PosCyl> points;
    galaxymodel::generateDensitySamples(dens, 1e5, points);
    particles::writeSnapshot(fileName+".nemo", units::ExternalUnits(), points, "Nemo");
}

/// generate an N-body representation of the entire model specified by its DF, and store in a file
void storeNbodyModel(const std::string& fileName,
    const potential::BasePotential& pot, const df::BaseDistributionFunction& df)
{
    particles::PointMassArrayCar points;
    galaxymodel::generatePosVelSamples(
        galaxymodel::GalaxyModel(pot, actions::ActionFinderAxisymFudge(pot), df), 1e5, points);
    particles::writeSnapshot(fileName+".nemo", units::ExternalUnits(), points, "Nemo");
}

// interface for progress reporting
class ProgRep: public galaxymodel::ProgressReportCallback {
public:
    int iteration;
    std::vector<const potential::BasePotential*> components;
    ProgRep(): iteration(0) {};

    virtual void generalMessage(const char* msg)
    {
        std::cout << msg << '\n';
    }

    virtual void reportComponentUpdate(unsigned int componentIndex,
        const galaxymodel::BaseComponent& comp)
    {
        const potential::BaseDensity* density = comp.getDensity();
        const potential::BasePotential* potential = comp.getPotential();
        // need to combine possibly both density and potential objects to find out total density profile
        if(density!=NULL && potential!=NULL) {
            std::vector<const potential::BasePotential*> pcomp(2);
            pcomp[0] = potential;
            pcomp[1] = new potential::Multipole(*density, 1e-3, 1e3, 100, 20);
            components.push_back(new potential::CompositeCyl(pcomp));
            delete pcomp[1];
        } else if(density!=NULL) {
            components.push_back(new potential::Multipole(*density, 1e-3, 1e3, 100, 20));
        } else if(potential!=NULL) {
            components.push_back(potential);
        } else return;
        std::cout << "Component #" << componentIndex <<
            ": inner density slope=" << getInnerDensitySlope(*components.back()) <<
            ", rho(R=1)=" << components.back()->density(coord::PosCyl(1,0,0)) <<
            ", rho(z=1)=" << components.back()->density(coord::PosCyl(0,1,0)) <<
            ", total mass=" << components.back()->totalMass() << '\n';
            printoutDensity("dens_comp"+utils::convertToString(componentIndex)+
                "_iter"+utils::convertToString(iteration)+".txt", *components.back());
    }

    virtual void reportTotalPotential(const potential::BasePotential& potential)
    {
        std::cout << "Potential is updated; "
            "inner density slope=" << getInnerDensitySlope(potential) <<
            ", Phi(0)="   << potential.value(coord::PosCyl(0,0,0)) <<
            ", rho(R=1)=" << potential.density(coord::PosCyl(1,0,0)) <<
            ", rho(z=1)=" << potential.density(coord::PosCyl(0,1,0)) <<
            ", total mass="   << potential.totalMass() << '\n';
        printoutDensity("dens_total_iter"+utils::convertToString(iteration)+".txt", potential);
        printoutRotationCurve("rotcurve_iter"+utils::convertToString(iteration)+".txt", components);
        iteration++;
        for(unsigned int i=0; i<components.size(); i++)
            delete components[i];
        components.clear();
    }
};

int main()
{
    ProgRep progressReporter;  // print out various information during the process

    // parameters of the inner component (disk)
    double dnorm   = 1.0;
    double Rdisk   = 2.0;   // scale radius of the disk
    double Hdisk   = 0.2;   // thickness of the (isothermal) disk
    double L0      = 0.0;   // angular momentum of transition from isotropic to rotating disk
    double Sigma0  = dnorm / (2*M_PI * pow_2(Rdisk));  // surface density normalization
    double sigmaz0 = sqrt(2*M_PI * Sigma0 * Hdisk);
    double sigmar0 = sigmaz0;
    double sigmamin= sigmar0*0.1;
    const df::PseudoIsothermalParam paramInner = {dnorm,Rdisk,L0,Sigma0,sigmar0,sigmaz0,sigmamin};
    // parameters of disk density profile should be in rough agreement with the DF params
    const potential::DiskParam      paramPot(Sigma0, Rdisk, -Hdisk, 0, 0);

    // parameters of the outer component (halo)
    double hnorm = 10.;  // approximately equals the total mass
    double alpha = 1.4;   // determines the inner density slope
    double beta  = 5.0;   // same for the outer slope: rho ~ r^-(3+beta)/2
    double j0    = 20.;   // determines the break radius in density profile
    double jcore = 0.0;   // inner plateau in density (disabled here)
    double ar    = 1.3;   // determines the velocity anisotropy in the inner region
    double az    = 1.1;   // the ratio between these two
    double aphi  = 0.6;   // determines the flattening of the inner region
    double br    = 1.5;   // same for
    double bz    = 0.8;   // the outer
    double bphi  = 0.7;   // region
    const df::DoublePowerLawParam paramOuter = {hnorm,j0,jcore,alpha,beta,ar,az,aphi,br,bz,bphi};
    // create the instance of distribution function of the halo
    const df::DoublePowerLaw dfOuter(paramOuter);

    // First stage: we use a 'dead' (fixed) density profile for the inner (disk) component,
    // without specifying its DF, and a 'live' outer component specified by its DF.
    // The inner component itself is represented with a DiskAnsatz potential and
    // DiskResidual density profile, both of them 'deadweight'
    std::vector<galaxymodel::BaseComponent*> components;
    components.push_back(new galaxymodel::ComponentStatic(
        potential::DiskResidual(paramPot), potential::DiskAnsatz(paramPot)));
    components.push_back(new galaxymodel::ComponentWithDF(dfOuter, 0.1, 500, 30, 6));

    galaxymodel::SelfConsistentModel* model = 
        new galaxymodel::SelfConsistentModel(components, 1e-3, 1e3, 100, 20, &progressReporter);

    // do a few iterations
    for(int i=0; i<5; i++) {
        std::cout << "Starting iteration #" << progressReporter.iteration << '\n';
        model->doIteration();
    }
    // receive the density profile of the outer ('live') component, and make a copy of it
    const potential::BaseDensity* guessForDensityOuter =
        model->getComponent(1)->getDensity()->clone();
    storeNbodyDensity("dens_outer_iter"+utils::convertToString(progressReporter.iteration),
        *guessForDensityOuter);
    
    // now that we have a reasonable guess for the total potential,
    // we may initialize the DF of the disk component
    const df::PseudoIsothermal dfInner(paramInner,
        potential::InterpEpicycleFreqs(*model->getPotential()));

    // destroy all traces
    delete model;
    for(unsigned int i=0; i<components.size(); i++)
        delete components[i];
    
    // we can compute the masses even though we don't know the density profile yet
    std::cout << "**** STARTING TWO-COMPONENT MODELLING ****\n"
        "Masses are " << dfInner.totalMass() << " and " << dfOuter.totalMass() << "\n";

    // Second stage: now use two live components (both specified by their DFs),
    // and provide the initial guess for the density profile of the outer component,
    // obtained at the previous stage
    components.clear();
    components.push_back(
        new galaxymodel::ComponentWithDisklikeDF(dfInner, 1e-2, 40., 40, 10, paramPot));
    components.push_back(
        new galaxymodel::ComponentWithDF(dfOuter, 0.01, 500., 50, 8, *guessForDensityOuter));
    delete guessForDensityOuter;  // not needed anymore

    model = new galaxymodel::SelfConsistentModel(components, 1e-3, 1e3, 100, 10, &progressReporter);
    for(int i=0; i<5; i++) {
        std::cout << "Starting iteration #" << progressReporter.iteration << '\n';
        model->doIteration();
    }
    
    // receive the overall potential from the model, and make a copy of it
    const potential::BasePotential* totalPotential = model->getPotential()->clone();

    // also get a copy of the updated density profiles for both components;
    // the inner one is actually a composite density - 
    // the Laplacian of DiskAnsatz potential and the residual density profile
    std::vector<const potential::BaseDensity*> denscomp(2);
    denscomp[0] = model->getComponent(0)->getDensity();
    denscomp[1] = model->getComponent(0)->getPotential();
    const potential::BaseDensity* densityInner = new potential::CompositeDensity(denscomp);
    // the outer one is just a simple density profile
    const potential::BaseDensity* densityOuter = model->getComponent(1)->getDensity()->clone();
    
    delete model;  // don't keep the model anymore
    for(unsigned int i=0; i<components.size(); i++)
        delete components[i];

    // export model to an N-body snapshot
    std::cout << "Creating an N-body representation of the model\n";

    // first create a representation of density profiles without velocities
    // (just for demonstration), by drawing samples from the density distribution
    storeNbodyDensity("dens_inner_iter"+
        utils::convertToString(progressReporter.iteration), *densityInner);
    storeNbodyDensity("dens_outer_iter"+
        utils::convertToString(progressReporter.iteration), *densityOuter);
    delete densityInner;
    delete densityOuter;
    storeNbodyDensity("dens_total", *totalPotential);

    // now create genuinely self-consistent models of both components,
    // by drawing positions and velocities from the DF in the given (self-consistent) potential
    storeNbodyModel("model_outer_iter"+
        utils::convertToString(progressReporter.iteration), *totalPotential, dfOuter);
#if 1
    storeNbodyModel("model_inner_iter"+
        utils::convertToString(progressReporter.iteration), *totalPotential, dfInner);
#endif

    delete totalPotential;
    return 0;
}
