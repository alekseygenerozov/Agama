/** \file    test_distributionfunction.cpp
    \author  Eugene Vasiliev
    \date    August 2015

    This test demonstrates that the action-based double-power-law distribution function
    corresponds rather well to the ergodic distribution function obtained by
    the Eddington inversion formula for the known spherically-symmetric isotropic model.
    We create an instance of DF and compute several quantities (such as density and 
    velocity dispersion), comparing them to the standard analytical expressions.
*/
#include <iostream>
#include <fstream>
#include "potential_dehnen.h"
#include "actions_staeckel.h"
#include "df_halo.h"
#include "galaxymodel.h"
#include "particles_io.h"
#include "math_specfunc.h"
#include "debug_utils.h"

const double reqRelError = 1e-4;
const int maxNumEval = 1e5;
const char* errmsg = "\033[1;31m **\033[0m";

bool testTotalMass(const galaxymodel::GalaxyModel& galmod, double massExact)
{
    std::cout << "\033[1;33mTesting " << galmod.potential.name() << "\033[0m\n";
    // Calculate the total mass
    double err;
    int numEval;
    double mass = galmod.distrFunc.totalMass(reqRelError, maxNumEval, &err, &numEval);
    if(err > mass*0.01) {
        std::cout << "Mass=" << mass << " +- " << err << " in " << numEval << " DF evaluations\n";
        mass = galmod.distrFunc.totalMass(reqRelError, maxNumEval*10, &err, &numEval);
    }
    bool ok = math::fcmp(mass, massExact, 0.05) == 0; // 5% relative error allowed because ar!=az
    std::cout <<
        "Mass=" << mass << " +- " << err << " in " << numEval << " DF evaluations"
        " (analytic value=" << massExact << (ok?"":errmsg) <<")\n";
    return ok;
}

// return a random number or 0 or 1, to test all possibilities including the boundary cases
inline double rnd(int i) { return (i<=1) ? i*1. : math::random(); }
bool testActionSpaceScaling(const df::BaseActionSpaceScaling& s)
{
    for(int i=0; i<1000; i++) {
        double v[3] = { rnd(i%10), rnd(i/10%10), rnd(i/100) };
        actions::Actions J = s.toActions(v);
        if(J.Jr!=J.Jr || J.Jz!=J.Jz || J.Jphi!=J.Jphi)
            return false;
        double w[3];
        s.toScaled(J, w);
        if(!(w[0]>=0 && w[0]<=1 && w[1]>=0 && w[1]<=1 && w[2]>=0 && w[2]<=1))
            return false;
        actions::Actions J1 = s.toActions(w);
        if( J1.Jr!=J1.Jr || J1.Jz!=J1.Jz || J1.Jphi!=J1.Jphi ||
            math::isFinite(J1.Jr+J1.Jz+fabs(J1.Jphi)) && (math::fcmp(J.Jr, J1.Jr, 1e-10)!=0 ||
            math::fcmp(J.Jz, J1.Jz, 1e-10)!=0 || math::fcmp(J.Jphi, J1.Jphi, 1e-10)!=0) )
            return false;
    }
    return true;
}

bool testDFmoments(const galaxymodel::GalaxyModel& galmod, const coord::PosVelCyl& point,
    double dfExact, double densExact, double sigmaExact)
{
    std::cout << "\033[1mAt point " << point << "\033[0m we have\n";
    // compare the action-based distribution function f(J) with the analytic one
    const actions::Actions J = galmod.actFinder.actions(point);
    double dfValue = galmod.distrFunc.value(J);
    double energy = totalEnergy(galmod.potential, point);
    bool dfok = math::fcmp(dfValue, dfExact, 0.05) == 0;
    std::cout <<
        "f(J)=" << dfValue << " for actions " << J << "\n"
        "f(E)=" << dfExact << " for energy E=" << energy << (dfok?"":errmsg) <<"\n";

    // compute density and velocity moments
    double density, densityErr;
    coord::VelCyl velocityFirstMoment, velocityFirstMomentErr;
    coord::Vel2Cyl velocitySecondMoment, velocitySecondMomentErr;
    computeMoments(galmod, point, reqRelError, maxNumEval,
        &density, &velocityFirstMoment, &velocitySecondMoment,
        &densityErr, &velocityFirstMomentErr, &velocitySecondMomentErr);
    bool densok = math::fcmp(density, densExact, 0.05) == 0;
    bool sigmaok =
        math::fcmp(velocitySecondMoment.vR2,   sigmaExact, 0.05) == 0 &&
        math::fcmp(velocitySecondMoment.vz2,   sigmaExact, 0.05) == 0 &&
        math::fcmp(velocitySecondMoment.vphi2, sigmaExact, 0.05) == 0;

    std::cout << 
        "density=" << density << " +- " << densityErr << 
        "  compared to analytic value " << densExact << (densok?"":errmsg) <<"\n"
        "velocity"
        "  vR=" << velocityFirstMoment.vR << " +- " << velocityFirstMomentErr.vR <<
        ", vz=" << velocityFirstMoment.vz << " +- " << velocityFirstMomentErr.vz <<
        ", vphi=" << velocityFirstMoment.vphi << " +- " << velocityFirstMomentErr.vphi << "\n"
        "2nd moment of velocity"
        "  vR2="    << velocitySecondMoment.vR2    << " +- " << velocitySecondMomentErr.vR2 <<
        ", vz2="    << velocitySecondMoment.vz2    << " +- " << velocitySecondMomentErr.vz2 <<
        ", vphi2="  << velocitySecondMoment.vphi2  << " +- " << velocitySecondMomentErr.vphi2 <<
        ", vRvz="   << velocitySecondMoment.vRvz   << " +- " << velocitySecondMomentErr.vRvz <<
        ", vRvphi=" << velocitySecondMoment.vRvphi << " +- " << velocitySecondMomentErr.vRvphi <<
        ", vzvphi=" << velocitySecondMoment.vzvphi << " +- " << velocitySecondMomentErr.vzvphi <<
        "   compared to analytic value " << sigmaExact << (sigmaok?"":errmsg) <<"\n";
    return dfok && densok && sigmaok;
}

/// analytic expression for the ergodic distribution function f(E)
/// in a Hernquist model with mass m, scale radius a, at energy E.
double dfHernquist(double m, double a, double E)
{
    double q = sqrt(-E*a/m);
    return m / (4 * pow(2 * m * a * M_PI*M_PI, 1.5) ) * pow(1-q*q, -2.5) *
        (3*asin(q) + q * sqrt(1-q*q) * (1-2*q*q) * (8*q*q*q*q - 8*q*q - 3) );
}

/// analytic expression for isotropic 1d velocity dispersion sigma^2
/// in a Hernquist model with mass m, scale radius a, at a radius r
double sigmaHernquist(double m, double a, double r)
{
    double x = r/a;
    return m/a * ( x * pow_3(x + 1) * log(1 + 1/x) -
        x/(x+1) * ( 25./12 + 13./3*x + 7./2*x*x + x*x*x) );
}

const int NUM_POINTS_H = 3;
const double testPointsH[NUM_POINTS_H][6] = {
    {0,   1, 0, 0, 0, 0.5},
    {0.2, 0, 0, 0, 0, 0.2},
    {5.0, 2, 9, 0, 0, 0.3} };

int main(){
    bool ok = true;
    ok &= testActionSpaceScaling(df::ActionSpaceScalingTriangLog());
    ok &= testActionSpaceScaling(df::ActionSpaceScalingRect(0.765432,0.234567));

    // test double-power-law distribution function in a spherical Hernquist potential
    // NB: parameters obtained by fitting (test_df_fit.cpp)
    df::DoublePowerLawParam paramDPL;
    paramDPL.alpha = 1.407;
    paramDPL.beta  = 5.628;
    paramDPL.j0    = 1.745;
    paramDPL.jcore = 0.;
    paramDPL.ar    = 1.614;
    paramDPL.az    = (3-paramDPL.ar)/2;
    paramDPL.aphi  = paramDPL.az;
    paramDPL.br    = 1.0;
    paramDPL.bz    = 1.0;
    paramDPL.bphi  = 1.0;
    paramDPL.norm  = 0.956 * math::gamma(paramDPL.beta-paramDPL.alpha)
        / math::gamma(3-paramDPL.alpha) / math::gamma(paramDPL.beta-3);
    potential::PtrPotential potH(new potential::Dehnen(1., 1., 1., 1., 1.));  // potential
    const actions::ActionFinderAxisymFudge actH(potH);        // action finder
    const df::DoublePowerLaw dfH(paramDPL);                   // distribution function
    const galaxymodel::GalaxyModel galmodH(*potH, actH, dfH); // all together - the mighty triad

    ok &= testTotalMass(galmodH, 1.);

    for(int i=0; i<NUM_POINTS_H; i++) {
        const coord::PosVelCyl point(testPointsH[i]);
        double dfExact    = dfHernquist(1, 1, totalEnergy(*potH, point));    // f(E) for the Hernquist model
        double densExact  = potH->density(point);                            // analytical value of density
        double sigmaExact = sigmaHernquist(1, 1, coord::toPosSph(point).r);  // analytical value of sigma^2
        ok &= testDFmoments(galmodH, point, dfExact, densExact, sigmaExact);
    }

    // create an N-body model by sampling from DF
    particles::PointMassArrayCar points;
    galaxymodel::generatePosVelSamples(galmodH, 1e5, points);
    particles::writeSnapshot("sampled_model.nemo", units::ExternalUnits(), points, "Nemo");

    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    return 0;
}
