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
#include "potential_factory.h"
#include "potential_composite.h"
#include "potential_analytic.h"
#include "potential_galpot.h"
#include "potential_cylspline.h"
#include "potential_sphharm.h"
#include "potential_utils.h"
#include "particles_io.h"
#include "actions_staeckel.h"
#include "math_core.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <cmath>

using potential::PtrDensity;
using potential::PtrPotential;

// various auxiliary functions for printing out information are non-essential
// for the modelling itself; the essential workflow is contained in main().

/// print the radial and angular dependence of a density profile into a text file
void writeDensityProfile(const std::string& fileName, const potential::BaseDensity& density)
{
    if(density.name() == potential::DensityCylGrid::myName())
        writeDensityCoefs(fileName, 
        dynamic_cast<const potential::DensityCylGrid&>(density));
    if(density.name() == potential::DensitySphericalHarmonic::myName())
        writeDensityCoefs(fileName,
        dynamic_cast<const potential::DensitySphericalHarmonic&>(density));
}

/// print the rotation curve for a collection of potential components into a text file
void writeRotationCurve(const std::string& fileName, const PtrPotential& potential)
{
    PtrPotential comp = potential->name()==potential::CompositeCyl::myName() ? potential :
        PtrPotential(new potential::CompositeCyl(std::vector<PtrPotential>(1, potential)));
    const potential::CompositeCyl& pot = dynamic_cast<const potential::CompositeCyl&>(*comp);
    std::ofstream strm(fileName.c_str());
    strm << "#radius";
    for(unsigned int i=0; i<pot.size(); i++)
        strm << "\t"<<pot.component(i)->name();
    if(pot.size()>1)
        strm << "\ttotal\n";
    for(double r=1./8; r<=30; r<1 ? r*=2 : r<16 ? r+=0.5 : r+=2) {
        strm << r;
        double v2sum = 0;
        for(unsigned int i=0; i<pot.size(); i++) {
            coord::GradCyl deriv;
            pot.component(i)->eval(coord::PosCyl(r, 0, 0), NULL, &deriv);
            v2sum += r*deriv.dR;
            strm << '\t' << sqrt(r*deriv.dR);
        }
        if(pot.size()>1)
            strm << '\t' << sqrt(v2sum);
        strm << '\n';
    }
}

/// generate an N-body representation of a density profile (without velocities) and write to a file
void writeNbodyDensity(const std::string& fileName, const potential::BaseDensity& dens)
{
    particles::PointMassArray<coord::PosCyl> points;
    galaxymodel::generateDensitySamples(dens, 1e5, points);
    particles::writeSnapshot(fileName+".nemo", units::ExternalUnits(), points, "Nemo");
}

/// generate an N-body representation of the entire model specified by its DF, and write to a file
void writeNbodyModel(const std::string& fileName, const galaxymodel::GalaxyModel& model)
{
    particles::PointMassArrayCar points;
    generatePosVelSamples(model, 1e5, points);
    particles::writeSnapshot(fileName+".nemo", units::ExternalUnits(), points, "Nemo");
}

/// print profiles of surface density and z-velocity dispersion to a file
void writeSurfaceDensityProfile(const std::string& fileName, const galaxymodel::GalaxyModel& model)
{
    std::vector<double> radii;
    for(double r=1./8; r<=30; r<1 ? r*=2 : r<16 ? r+=0.5 : r+=2)
        radii.push_back(r);
    int n=radii.size();
    std::vector<double> surfDens(n), losvdisp(n), dens(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1)
#endif
    for(int i=0; i<n; i++) {
        computeProjectedMoments(model, radii[i], 1e-3, 1e6, surfDens[i], losvdisp[i]);
        computeMoments(model, coord::PosCyl(radii[i],0,0), 1e-3, 1e5, &dens[i],
            NULL, NULL, NULL, NULL, NULL);
    }

    std::ofstream strm((fileName+".surfdens").c_str());
    strm << "#Radius\tsurfaceDensity\tverticalVelocityDispersion\tin-plane-density\n";
    for(int i=0; i<n; i++)
        strm << radii[i] << '\t' << surfDens[i] << '\t' << losvdisp[i] << '\t' << dens[i] << '\n';
}

/// report progress after an iteration
void printoutInfo(const galaxymodel::SelfConsistentModel& model, const std::string& iterationStr)
{
    const potential::BaseDensity& comp0 = *model.components[0]->getDensity();
    const potential::BaseDensity& comp1 = *model.components[1]->getDensity();
    std::cout << 
        "Disk inner density slope=" << getInnerDensitySlope(comp0) <<
        ", total mass=" << comp0.totalMass() <<
        ", rho(R=1)=" << comp0.density(coord::PosCyl(1,0,0)) <<
        ", rho(z=1)=" << comp0.density(coord::PosCyl(0,1,0)) << "\n"
        "Halo inner density slope=" << getInnerDensitySlope(comp1) <<
        ", total mass=" << comp1.totalMass() <<
        ", rho(R=1)=" << comp1.density(coord::PosCyl(1,0,0)) <<
        ", rho(z=1)=" << comp1.density(coord::PosCyl(0,1,0)) << "\n"
        "Potential at origin="   << model.totalPotential->value(coord::PosCyl(0,0,0)) <<
        ", total mass=" << model.totalPotential->totalMass() << '\n';
    writeDensityProfile("dens_inner_iter"+iterationStr, comp0);
    writeDensityProfile("dens_outer_iter"+iterationStr, comp1);
    writeRotationCurve("rotcurve_iter"+iterationStr, model.totalPotential);
}

int main()
{
    // parameters of the inner component (disk)
    double dnorm   = 1.0;   // approximately equals the total mass
    double Rdisk   = 2.0;   // scale radius of the disk
    double Hdisk   = 0.2;   // thickness of the (isothermal) disk
    double L0      = 0.0;   // angular momentum of transition from isotropic to rotating disk
    double Sigma0  = dnorm / (2*M_PI * pow_2(Rdisk));  // surface density normalization
    double sigmaz0 = sqrt(2*M_PI * Sigma0 * Hdisk);    // vertical velocity dispersion scale
    double sigmar0 = sigmaz0;     // same for radial velocity
    double sigmamin= sigmar0*0.1; // lower bound on velocity dispersion
    double Jphimin = 0.5;         // lower limit on azimuthal action used for computing epicyclic freqs
    const df::PseudoIsothermalParam paramInner = {dnorm,Rdisk,L0,Sigma0,sigmar0,sigmaz0,sigmamin,Jphimin};
    // parameters of disk density profile should be in rough agreement with the DF params
    const potential::DiskParam      paramPot(Sigma0, Rdisk, -Hdisk, 0, 0);

    // parameters of the outer component (halo)
    double hnorm = 5.0;   // approximately equals the total mass
    double alpha = 1.4;   // determines the inner density slope
    double beta  = 6.0;   // same for the outer slope: rho ~ r^-(3+beta)/2
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
    df::PtrDistributionFunction dfOuter(new df::DoublePowerLaw(paramOuter));

    // set up parameters of the entire Self-Consistent Model
    galaxymodel::SelfConsistentModel model;
    model.rminSph = 0.1;
    model.rmaxSph = 500;        // range of radii for the logarithmic grid
    model.sizeRadialSph = 100;  // number of grid points in radius
    model.lmaxAngularSph = 6;   // maximum order of angular-harmonic expansion (l_max)
    model.RminCyl = 0.2;
    model.RmaxCyl = 30;         // range of grid nodes in cylindrical radius
    model.zminCyl = 0.04;
    model.zmaxCyl = 15;         // grid nodes in vertical direction
    model.sizeRadialCyl = 30;   // number of grid nodes in cylindrical radius
    model.sizeVerticalCyl = 30; // number of grid nodes in vertical (z) direction
    int iteration=0;

    // First stage: we use a 'dead' (fixed) density profile for the inner (disk) component,
    // without specifying its DF, and a 'live' outer component specified by its DF.
    // The inner component itself is represented with a DiskAnsatz potential and
    // DiskResidual density profile, both of them 'deadweight'
    model.components.push_back(galaxymodel::PtrComponent(
        new galaxymodel::ComponentStatic(
        PtrDensity(new potential::DiskDensity(paramPot)), true /*disk-like*/)));
    model.components.push_back(galaxymodel::PtrComponent(
        new galaxymodel::ComponentWithSpheroidalDF(
        dfOuter, PtrDensity(new potential::Plummer(dfOuter->totalMass(), 10.0)),
        0.2, 500., 40, model.lmaxAngularSph)));

    // do a few iterations
    for(int i=0; i<5; i++) {
        std::cout << "Starting iteration #" << ++iteration << '\n';
        doIteration(model);
        printoutInfo(model, utils::convertToString(iteration));
    }

    // now that we have a reasonable guess for the total potential,
    // we may initialize the DF of the disk component
    df::PtrDistributionFunction dfInner(new df::PseudoIsothermal(paramInner,
        potential::InterpEpicycleFreqs(*model.totalPotential)));

    // we can compute the masses even though we don't know the density profile yet
    std::cout << "**** STARTING TWO-COMPONENT MODELLING ****\n"
        "Masses are " << dfInner->totalMass() << " and " << dfOuter->totalMass() << "\n";

    // Second stage: replace the inner component with a `live' one (specified by the DFs)
    model.components[0] = galaxymodel::PtrComponent(
        new galaxymodel::ComponentWithDisklikeDF(
        dfInner, PtrDensity(new potential::DiskDensity(paramPot)),
        math::createNonuniformGrid(15, Rdisk*0.1, Rdisk*10, true),
        math::createNonuniformGrid(12, Hdisk*0.1, Hdisk*10, true) ));

    for(int i=0; i<5; i++) {
        std::cout << "Starting iteration #" << ++iteration << '\n';
        doIteration(model);
        printoutInfo(model, utils::convertToString(iteration));
    }

    // export model to an N-body snapshot
    std::cout << "Creating an N-body representation of the model\n";
    std::string iterationStr(utils::convertToString(iteration));

    // first create a representation of density profiles without velocities
    // (just for demonstration), by drawing samples from the density distribution
    writeNbodyDensity("dens_inner_iter"+iterationStr,
        *model.components[0]->getDensity());
    writeNbodyDensity("dens_outer_iter"+iterationStr,
        *model.components[1]->getDensity());
    writeNbodyDensity("dens_total", *model.totalPotential);
    writeSurfaceDensityProfile("model_inner_iter"+iterationStr,
        galaxymodel::GalaxyModel(*model.totalPotential, *model.actionFinder, *dfInner));

    // now create genuinely self-consistent models of both components,
    // by drawing positions and velocities from the DF in the given (self-consistent) potential
    writeNbodyModel("model_outer_iter"+iterationStr,
        galaxymodel::GalaxyModel(*model.totalPotential, *model.actionFinder, *dfOuter));
    writeNbodyModel("model_inner_iter"+iterationStr,
        galaxymodel::GalaxyModel(*model.totalPotential, *model.actionFinder, *dfInner));

    return 0;
}
