#include "potential_galpot.h"
#include "actions_staeckel.h"
#include "df_disk.h"
#include "galaxymodel.h"
#include "particles_io.h"
#include "math_specfunc.h"
#include "debug_utils.h"
#include <iostream>
#include <fstream>

const double reqRelError = 1e-4;
const int maxNumEval = 1e6;

void testTotalMass(const galaxymodel::GalaxyModel& galmod, double massExact)
{
    // Calculate the total mass
    double err;
    int numEval;
    double mass = galmod.distrFunc.totalMass(reqRelError, maxNumEval, &err, &numEval);
    if(err > mass*reqRelError) {
        std::cout << "Mass=" << mass << " +- " << err << " in " << numEval << " DF evaluations\n";
        mass = galmod.distrFunc.totalMass(reqRelError, maxNumEval*10, &err, &numEval);
    }
    std::cout <<
        "Mass=" << mass << " +- " << err << " in " << numEval << " DF evaluations"
        " (analytic value=" << massExact <<")\n";
}

math::Matrix<double> data;
std::vector<coord::PosCyl> points;

void getDFmoments(const galaxymodel::GalaxyModel& galmod, unsigned int index,
    double densExact, double velThin, double velExact, double sigma2Exact)
{
    // compute density and velocity moments
    double density, densityErr;
    coord::VelCyl velocityFirstMoment, velocityFirstMomentErr;
    coord::Vel2Cyl velocitySecondMoment, velocitySecondMomentErr;
    computeMoments(galmod, points[index], reqRelError, maxNumEval,
        &density, &velocityFirstMoment, &velocitySecondMoment,
        &densityErr, &velocityFirstMomentErr, &velocitySecondMomentErr);
    data(index, 0) = density;
    data(index, 1) = densityErr;
    data(index, 2) = densExact;
    data(index, 3) = velocityFirstMoment.vphi;
    data(index, 4) = velocityFirstMomentErr.vphi;
    data(index, 5) = velExact;
    data(index, 6) = velThin;
    data(index, 7) = velocitySecondMoment.vR2;
    data(index, 8) = velocitySecondMomentErr.vR2;
    data(index, 9) = velocitySecondMoment.vz2;
    data(index,10) = velocitySecondMomentErr.vz2;
    data(index,11) = velocitySecondMoment.vphi2 - pow_2(velocityFirstMoment.vphi);
    data(index,12) = velocitySecondMomentErr.vphi2;
    data(index,13) = sigma2Exact;
    std::cout << points[index].R << ',' << points[index].z << '\t' << std::flush;
}

double vcircExpDisk(double Mdisk, double Rdisk, double R)
{
    double x = R/(2*Rdisk);
    return x==0 ? 0 : sqrt( Mdisk/Rdisk * 2*x*x *
        (math::besselI(0, x) * math::besselK(0, x) - math::besselI(1, x) * math::besselK(1, x)) );
}

int main(){
    // test pseudo-isothermal distribution function in an exponential-disk potential
    double norm    = 1.0;
    double Rdisk   = 2.5;   // scale radius of the disk
    double Hdisk   = 0.25;  // thickness of the (isothermal) disk
    double L0      = 0.0;   // angular momentum of transition from isotropic to rotating disk
    double Sigma0  = norm / (2*M_PI * pow_2(Rdisk));  // surface density normalization
    double sigmaz0 = sqrt(2*M_PI * Sigma0 * Hdisk);
    double sigmar0 = sigmaz0;
    const df::PseudoIsothermalParam param = {norm,Rdisk,L0,Sigma0,sigmar0,sigmaz0,sigmar0*0.1,0};
    const potential::DiskParam      paramPot(Sigma0, Rdisk, -Hdisk, 0, 0);
    const potential::PtrPotential pot    = potential::createGalaxyPotential(
        std::vector<potential::DiskParam>(1, paramPot),
        std::vector<potential::SphrParam>() );
    const actions::ActionFinderAxisymFudge act(pot);
    const df::PseudoIsothermal df(param, potential::InterpEpicycleFreqs(*pot));
    const galaxymodel::GalaxyModel galmod(*pot, act, df);

    double massExact = Sigma0 * 2*M_PI * pow_2(Rdisk);
    testTotalMass(galmod, massExact);
#if 0
    for(double R=0; R<=20; R<10 ? R+=0.5 : R+=1)
        for(double z=0; z<=1.0; z<0.5 ? z+=0.125 : z+=0.25)
            points.push_back(coord::PosCyl(R, z, 0));
    int np = points.size();
    data.resize(np, 14);
#pragma omp parallel for schedule(dynamic)
    for(int i=0; i<np; i++) {
        double R = points[i].R;
        double densExact  = pot->density(points[i]);           // analytical value of density
        double velThin    = vcircExpDisk(massExact, Rdisk, R); // circular speed for razor-thin disk
        double velExact   = v_circ(*pot, R);                   // circular speed for the actual disk
        double sigmaExact = sigmaz0 * exp(-0.5*R/Rdisk);       // vertical velocity dispersion
        getDFmoments(galmod, i, densExact, velThin, velExact, pow_2(sigmaExact));
    }
    delete pot;
    std::ofstream strm("disk_profile");
    for(int i=0; i<np; i++) {
        strm << points[i].R << '\t' << points[i].z;
        for(int k=0; k<14; k++)
            strm << '\t' << data(i, k);
        strm << '\n';
    }
    strm.close();
#endif

#if 1
    // test sampling of DF in 3d action space
    particles::PointMassArrayCar points_car;
    //galaxymodel::generateActionSamples(galmod, 1e5, points_car);
    //particles::writeSnapshot("sampled_actions.txt", units::ExternalUnits(), points_car);

    // test sampling of DF in 6d phase space
    galaxymodel::generatePosVelSamples(galmod, 1e5, points_car);
    particles::writeSnapshot("sampled_posvel.txt", units::ExternalUnits(), points_car);

    std::vector<actions::Actions> actsamples;
    double val, err;
    df::sampleActions(df, 1e5, actsamples, &val, &err);
    std::cout << "Sampled mass: "<<val<<" +- "<<err<<"\n";
    std::ofstream strm1("actions_disk.txt");
    for(unsigned int i=0; i<actsamples.size(); i++) 
        strm1<<actsamples[i].Jr<<"\t"<<actsamples[i].Jz<<"\t"<<actsamples[i].Jphi<<"\n";

    // test sampling of 3d density
    particles::PointMassArray<coord::PosCyl> points_cyl;
    galaxymodel::generateDensitySamples(galmod.potential, 1e5, points_cyl);
    particles::writeSnapshot("sampled_density.txt", units::ExternalUnits(), points_cyl);
#endif
    return 0;
}
