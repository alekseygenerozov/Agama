/** \file    test_selfconsistentmodel.cpp
    \author  Eugene Vasiliev
    \date    November 2015

*/
#include <iostream>
#include "df_halo.h"
#include "galaxymodel_selfconsistent.h"
#include "galaxymodel.h"
#include "particles_io.h"
#include "actions_staeckel.h"

//const double reqRelError = 1e-4;
//const int maxNumEval = 1e5;
const char* errmsg = "\033[1;31m **\033[0m";

class ProgRep: public galaxymodel::ProgressReportCallback {
public:
    
    virtual void generalMessage(const char* msg)
    {
        std::cout << msg << '\n';
    }
    
    virtual void reportDensityAtPoint(unsigned int /*componentIndex*/,
        const coord::PosCyl& point, double densityValue)
    {
        std::cout << point.R <<'\t'<< point.z <<'\t'<< densityValue <<'\n';
    }
    
    virtual void reportDensityUpdate(unsigned int /*componentIndex*/,
                                     const potential::BaseDensity& density)
    {
        std::cout << "Inner density slope is " << getInnerDensitySlope(density) << '\n';
    }
    
    virtual void reportTotalPotential(const potential::BasePotential& potential)
    {
        std::cout << "Potential is updated; "
        "inner density slope=" << getInnerDensitySlope(potential) <<
        ", Phi(0)="   << potential.value(coord::PosCyl(0,0,0)) <<
        ", rho(R=1)=" << potential.density(coord::PosCyl(1,0,0)) <<
        ", rho(z=1)=" << potential.density(coord::PosCyl(0,1,0)) <<
        ", Mtotal="   << potential.totalMass() << '\n';
    }
};

int main()
{
    double norm  = 0.956;
    double alpha = 1.407;
    double beta  = 5.628;
    double j0    = 1.745;
    double jcore = 0.;
    double ar    = 1.614;
    double az    = (3-ar)*0.6;
    double aphi  = (3-ar)*0.4;
    double br    = 1.0;
    double bz    = 1.2;
    double bphi  = 0.8;
    const df::DoublePowerLawParam paramDPL = {norm,j0,jcore,alpha,beta,ar,az,aphi,br,bz,bphi};
    const df::DoublePowerLaw distrFunc(paramDPL);
    galaxymodel::Component comp(&distrFunc, 0.01, 100, 49, 6);
    ProgRep progressReporter;
    galaxymodel::SelfConsistentModel model(std::vector<galaxymodel::Component>(1, comp), &progressReporter);
    for(int iter=0; iter<10; iter++) {
        std::cout << "Starting iteration #"<<iter<<'\n';
        model.doIteration();
    }
    if(1) {
        // output model as an N-body snapshot
        const actions::ActionFinderAxisymFudge af(model.getPotential());
        galaxymodel::GalaxyModel galmod(model.getPotential(), af, distrFunc);
        particles::PointMassArrayCar points;
        galaxymodel::generatePosVelSamples(galmod, 1e5, points);
        particles::BaseIOSnapshot* snap = particles::createIOSnapshotWrite(
            "Text", "sampled_model.txt", units::ExternalUnits());
        snap->writeSnapshot(points);
        delete snap;
    }
    return 0;
}
