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
#include "galaxymodel_selfconsistent.h"
#include "galaxymodel.h"
#include "df_factory.h"
#include "potential_factory.h"
#include "potential_composite.h"
#include "particles_io.h"
#include "math_core.h"
#include "math_spline.h"
#include "units.h"
#include "utils.h"
#include "utils_config.h"
#include <iostream>
#include <fstream>
#include <cmath>

using potential::PtrDensity;
using potential::PtrPotential;

// define internal unit system - arbitrary numbers here!
const units::InternalUnits intUnits(6.666*units::Kpc, 42*units::Myr);

// various auxiliary functions for printing out information are non-essential
// for the modelling itself; the essential workflow is contained in main()

/// print the rotation curve for a collection of potential components into a text file
void writeRotationCurve(const std::string& fileName, const PtrPotential& potential)
{
    PtrPotential comp = potential->name()==potential::CompositeCyl::myName() ? potential :
        PtrPotential(new potential::CompositeCyl(std::vector<PtrPotential>(1, potential)));
    const potential::CompositeCyl& pot = dynamic_cast<const potential::CompositeCyl&>(*comp);
    std::ofstream strm(fileName.c_str());
    strm << "#radius";
    for(unsigned int i=0; i<pot.size(); i++) {
        writePotential(fileName+"_"+pot.component(i)->name(), *pot.component(i));
        strm << "\t"<<pot.component(i)->name();
    }
    if(pot.size()>1)
        strm << "\ttotal\n";
    // print values at certain radii, expressed in units of Kpc
    for(double r=1./8; r<=100; r<1 ? r*=2 : r<16 ? r+=0.5 : r<30 ? r+=2 : r+=5) {
        strm << r;         // output radius in kpc
        double v2sum = 0;  // accumulate squared velocity in internal units
        double r_int = r * intUnits.from_Kpc;  // radius in internal units
        for(unsigned int i=0; i<pot.size(); i++) {
            coord::GradCyl deriv;  // potential derivatives in internal units
            pot.component(i)->eval(coord::PosCyl(r_int, 0, 0), NULL, &deriv);
            double v2comp = r_int*deriv.dR;
            v2sum += v2comp;
            strm << '\t' << (sqrt(v2comp) * intUnits.to_kms);  // output in km/s
        }
        if(pot.size()>1)
            strm << '\t' << (sqrt(v2sum) * intUnits.to_kms);
        strm << '\n';
    }
}

/// generate an N-body representation of a density profile (without velocities) and write to a file
void writeNbodyDensity(const std::string& fileName, const potential::BaseDensity& dens)
{
    particles::PointMassArray<coord::PosCyl> points;
    galaxymodel::generateDensitySamples(dens, 1e5, points);
    units::ExternalUnits extUnits(intUnits, 1.*units::Kpc, 977.8*units::kms, 2.223e+11*units::Msun);
    writeSnapshot(fileName+".nemo", extUnits, points, "Nemo");
}

/// generate an N-body representation of the entire model specified by its DF, and write to a file
void writeNbodyModel(const std::string& fileName, const galaxymodel::GalaxyModel& model)
{
    particles::PointMassArrayCar points;
    galaxymodel::generatePosVelSamples(model, 1e5, points);
    units::ExternalUnits extUnits(intUnits, 1.*units::Kpc, 977.8*units::kms, 2.223e+11*units::Msun);
    writeSnapshot(fileName+".nemo", units::ExternalUnits(), points, "Nemo");
}

/// print profiles of surface density and z-velocity dispersion to a file
void writeSurfaceDensityProfile(const std::string& fileName, const galaxymodel::GalaxyModel& model)
{
    std::vector<double> radii, heights;
    // convert radii to internal units
    for(double r=1./8; r<=30; r<1 ? r*=2 : r<16 ? r+=0.5 : r+=2)
        radii.push_back(r * intUnits.from_Kpc);
    for(double h=0; h<=5; h<1? h+=0.25 : h+=1)
        heights.push_back(h * intUnits.from_Kpc);
    int nr = radii.size(), nh = heights.size();
    std::vector<double> surfDens(nr), losvdisp(nr), dens(nr*nh);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int ir=0; ir<nr; ir++) {
        computeProjectedMoments(model, radii[ir], 1e-3, 1e6, surfDens[ir], losvdisp[ir]);
        for(int ih=0; ih<nh; ih++)
            computeMoments(model, coord::PosCyl(radii[ir],heights[ih],0), 1e-3, 1e5, &dens[ir*nh+ih],
                NULL, NULL, NULL, NULL, NULL);
    }

    std::ofstream strm((fileName+".surfdens").c_str());
    strm << "#Radius\tsurfaceDensity\tverticalVelocityDispersion\tdensity:";
    for(int ih=0; ih<nh; ih++)
        strm << (heights[ih] * intUnits.to_Kpc) << '\t';
    strm << '\n';
    for(int ir=0; ir<nr; ir++) {
        strm << (radii[ir] * intUnits.to_Kpc) << '\t' << 
        (surfDens[ir] * intUnits.to_Msun_per_pc2) << '\t' << 
        (losvdisp[ir] * intUnits.to_kms);
        for(int ih=0; ih<nh; ih++)
            strm << '\t' << (dens[ir*nh+ih] * intUnits.to_Msun_per_pc3);
        strm << '\n';
    }
}

/// report progress after an iteration
void printoutInfo(const galaxymodel::SelfConsistentModel& model, const std::string& iterationStr)
{
    const potential::BaseDensity& compHalo = *model.components[0]->getDensity();
    const potential::BaseDensity& compDisc = *model.components[1]->getDensity();
    coord::PosCyl pt0(8.3 * intUnits.from_Kpc, 0, 0);
    coord::PosCyl pt1(8.3 * intUnits.from_Kpc, 1 * intUnits.from_Kpc, 0);
    std::cout << 
        "Disk total mass="      << (compDisc.totalMass()  * intUnits.to_Msun) << " Msun"
        ", rho(Rsolar,z=0)="    << (compDisc.density(pt0) * intUnits.to_Msun_per_pc3) <<
        ", rho(Rsolar,z=1kpc)=" << (compDisc.density(pt1) * intUnits.to_Msun_per_pc3) << " Msun/pc^3\n"
        "Halo inner density slope=" << getInnerDensitySlope(compHalo) <<
        ", total mass="         << (compHalo.totalMass()  * intUnits.to_Msun) << " Msun"
        ", rho(Rsolar,z=0)="    << (compHalo.density(pt0) * intUnits.to_Msun_per_pc3) <<
        ", rho(Rsolar,z=1kpc)=" << (compHalo.density(pt1) * intUnits.to_Msun_per_pc3) << " Msun/pc^3\n"
        "Potential at origin=-("<< 
        (sqrt(-model.totalPotential->value(coord::PosCyl(0,0,0))) * intUnits.to_kms) << " km/s)^2"
        ", total mass=" << (model.totalPotential->totalMass() * intUnits.to_Msun) << " Msun\n";
    writeDensity("dens_disc_iter"+iterationStr, compDisc);
    writeDensity("dens_halo_iter"+iterationStr, compHalo);
    writeRotationCurve("rotcurve_iter"+iterationStr, model.totalPotential);
}

int main()
{
    // read parameters from the INI file
    const std::string iniFileName = "../data/SCM.ini";
    utils::ConfigFile ini(iniFileName);
    utils::KeyValueMap 
        iniPotenThinDisc = ini.findSection("Potential thin disc"),
        iniPotenThickDisc= ini.findSection("Potential thick disc"),
        iniPotenGasDisc  = ini.findSection("Potential gas disc"),
        iniPotenBulge    = ini.findSection("Potential bulge"),
        iniPotenDarkHalo = ini.findSection("Potential dark halo"),
        iniDFThinDisc    = ini.findSection("DF thin disc"),
        iniDFThickDisc   = ini.findSection("DF thick disc"),
        iniDFStellarHalo = ini.findSection("DF stellar halo"),
        iniDFDarkHalo    = ini.findSection("DF dark halo"),
        iniSCMHalo       = ini.findSection("SelfConsistentModel halo"),
        iniSCMDisc       = ini.findSection("SelfConsistentModel disc"),
        iniSCM           = ini.findSection("SelfConsistentModel");
    if(!iniSCM.contains("rminSphGrid")) {  // most likely file doesn't exist
        std::cout << "Invalid INI file " << iniFileName << "\n";
        return -1;
    }
    // define external unit system describing the data (including the parameters in INI file)
    const units::ExternalUnits extUnits(intUnits, 1.*units::Kpc, 1.*units::kms, 1.*units::Msun);

    // set up parameters of the entire Self-Consistent Model
    galaxymodel::SelfConsistentModel model;
    model.rminSph         = iniSCM.getDouble("rminSphGrid") * extUnits.lengthUnit;
    model.rmaxSph         = iniSCM.getDouble("rmaxSphGrid") * extUnits.lengthUnit;
    model.sizeRadialSph   = iniSCM.getInt("sizeRadialSph");
    model.lmaxAngularSph  = iniSCM.getInt("lmaxAngularSph");
    model.RminCyl         = iniSCM.getDouble("RminCylGrid") * extUnits.lengthUnit;
    model.RmaxCyl         = iniSCM.getDouble("RmaxCylGrid") * extUnits.lengthUnit;
    model.zminCyl         = iniSCM.getDouble("zminCylGrid") * extUnits.lengthUnit;
    model.zmaxCyl         = iniSCM.getDouble("zmaxCylGrid") * extUnits.lengthUnit;
    model.sizeRadialCyl   = iniSCM.getInt("sizeRadialCyl");
    model.sizeVerticalCyl = iniSCM.getInt("sizeVerticalCyl");

    // initialize density profiles of various components
    std::vector<PtrDensity> densityStellarDisc(2);
    PtrDensity densityBulge    = potential::createDensity(iniPotenBulge,    extUnits);
    PtrDensity densityDarkHalo = potential::createDensity(iniPotenDarkHalo, extUnits);
    densityStellarDisc[0]      = potential::createDensity(iniPotenThinDisc, extUnits);
    densityStellarDisc[1]      = potential::createDensity(iniPotenThickDisc,extUnits);
    PtrDensity densityGasDisc  = potential::createDensity(iniPotenGasDisc,  extUnits);

    // add components to SCM - at first, all of them are static density profiles
    model.components.push_back(galaxymodel::PtrComponent(
        new galaxymodel::ComponentStatic(densityDarkHalo, false)));
    model.components.push_back(galaxymodel::PtrComponent(
        new galaxymodel::ComponentStatic(PtrDensity(
        new potential::CompositeDensity(densityStellarDisc)), true)));
    model.components.push_back(galaxymodel::PtrComponent(
        new galaxymodel::ComponentStatic(densityBulge, false)));
    model.components.push_back(galaxymodel::PtrComponent(
        new galaxymodel::ComponentStatic(densityGasDisc, true)));

    // initialize total potential of the model (first guess)
    updateTotalPotential(model);
    writeRotationCurve("rotcurve_init", model.totalPotential);

    std::cout << "**** STARTING ONE-COMPONENT MODELLING ****\nMasses are: "
        "Mbulge=" << (densityBulge->totalMass() * intUnits.to_Msun) << " Msun, "
        "Mgas="   << (densityGasDisc->totalMass() * intUnits.to_Msun) << " Msun, "
        "Mdisc="  << (model.components[1]->getDensity()->totalMass() * intUnits.to_Msun) << " Msun, "
        "Mhalo="  << (densityDarkHalo->totalMass() * intUnits.to_Msun) << " Msun\n";

    // create the dark halo DF from the parameters in INI file;
    // here the initial potential is only used to create epicyclic frequency interpolation table
    df::PtrDistributionFunction dfHalo = df::createDistributionFunction(
        iniDFDarkHalo, model.totalPotential.get(), extUnits);

    // replace the halo SCM component with the DF-based one
    model.components[0] = galaxymodel::PtrComponent(
        new galaxymodel::ComponentWithSpheroidalDF(dfHalo, densityDarkHalo,
        iniSCMHalo.getDouble("rminSphGrid") * extUnits.lengthUnit,
        iniSCMHalo.getDouble("rmaxSphGrid") * extUnits.lengthUnit,
        iniSCMHalo.getInt("sizeRadialSph"),
        iniSCMHalo.getInt("lmaxAngularSph") ));

    // do a few iterations to determine the self-consistent density profile of the halo
    int iteration=0;
    for(int i=0; i<5; i++) {
        std::cout << "Starting iteration #" << ++iteration << '\n';
        doIteration(model);
        printoutInfo(model, utils::convertToString(iteration));
    }

    // now that we have a reasonable guess for the total potential,
    // we may initialize the DF of the stellar components
    std::vector<df::PtrDistributionFunction> dfStellarArray;
    dfStellarArray.push_back(df::createDistributionFunction(
        iniDFThinDisc, model.totalPotential.get(), extUnits));
    dfStellarArray.push_back(df::createDistributionFunction(
        iniDFThickDisc, model.totalPotential.get(), extUnits));
    dfStellarArray.push_back(df::createDistributionFunction(
        iniDFStellarHalo, model.totalPotential.get(), extUnits));
    // composite DF of all stellar components except the bulge
    df::PtrDistributionFunction dfStellar(new df::CompositeDF(dfStellarArray));

    // we can compute the masses even though we don't know the density profile yet
    std::cout << "**** STARTING TWO-COMPONENT MODELLING ****\n"
        "Masses are: Mdisc=" << (dfStellar->totalMass() * intUnits.to_Msun) <<
        " Msun; Mhalo="      << (dfHalo->totalMass() * intUnits.to_Msun) << " Msun\n";

    // prepare parameters for the density grid of the stellar component
    std::vector<double> gridRadialCyl = math::createNonuniformGrid(
        iniSCMDisc.getInt("sizeRadialCyl"),
        iniSCMDisc.getDouble("RminCylGrid") * extUnits.lengthUnit,
        iniSCMDisc.getDouble("RmaxCylGrid") * extUnits.lengthUnit, true);
    std::vector<double> gridVerticalCyl = math::createNonuniformGrid(
        iniSCMDisc.getInt("sizeVerticalCyl"),
        iniSCMDisc.getDouble("zminCylGrid") * extUnits.lengthUnit,
        iniSCMDisc.getDouble("zmaxCylGrid") * extUnits.lengthUnit, true);

    // replace the static disc density component of SCM with a DF-based disc component
    model.components[1] = galaxymodel::PtrComponent(
        new galaxymodel::ComponentWithDisklikeDF(
        dfStellar, PtrDensity(), gridRadialCyl, gridVerticalCyl));

    // do a few more iterations to obtain the self-consistent density profile for both discs
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
    writeNbodyDensity("dens_halo_iter"+iterationStr,
        *model.components.front()->getDensity());
    writeNbodyDensity("dens_disc_iter"+iterationStr,
        *model.components.back()->getDensity());

    writeSurfaceDensityProfile("model_disc_iter"+iterationStr,
        galaxymodel::GalaxyModel(*model.totalPotential, *model.actionFinder, *dfStellar));

    // now create genuinely self-consistent models of both components,
    // by drawing positions and velocities from the DF in the given (self-consistent) potential
    writeNbodyModel("model_halo_iter"+iterationStr,
        galaxymodel::GalaxyModel(*model.totalPotential, *model.actionFinder, *dfHalo));
    writeNbodyModel("model_disc_iter"+iterationStr,
        galaxymodel::GalaxyModel(*model.totalPotential, *model.actionFinder, *dfStellar));

    return 0;
}
