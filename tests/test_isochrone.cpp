/** \file    test_isochrone.cpp
    \author  Eugene Vasiliev
    \date    February 2016

    This test checks the correctness of (exact) action-angle determination for Isochrone potential.
*/
#include "potential_analytic.h"
#include "actions_isochrone.h"
#include "actions_spherical.h"
#include "actions_staeckel.h"
#include "orbit.h"
#include "debug_utils.h"
#include <iostream>
#include <fstream>
#include <cmath>

bool test_isochrone(const coord::PosVelCyl& initial_conditions)
{
    const bool output = false; // whether to write a text file
    const double eps  = 2e-4;  // accuracy of comparison for actions found with different methods
    const double epsd = 1e-7;  // accuracy of action conservation along the orbit for each method
    const double M = 2.5;      // mass and
    const double b = 0.5;      // scale radius of Isochrone potential
    const double total_time=50;// integration time
    const double timestep=1./8;// sampling rate of trajectory
    std::vector<coord::PosVelCyl > traj;
    potential::Isochrone pot(M, b);
    orbit::integrate(pot, initial_conditions, total_time, timestep, traj, 1e-10);
    actions::ActionStat statI, statS, statF;
    actions::ActionAngles aaI, aaF;
    actions::Actions acS;
    actions::Frequencies frI, frF;
    actions::Angles aoldF(0,0,0), aoldI(0,0,0);
    bool anglesMonotonic = true;
    std::ofstream strm;
    if(output)
        strm.open("test_isochrone.dat");
    double ifd = 1e-5;
    for(size_t i=0; i<traj.size(); i++) {
        aaI = actions::actionAnglesIsochrone(M, b,  traj[i], &frI);
        aaF = actions::actionAnglesAxisymFudge(pot, traj[i], ifd, &frF);
        acS = actions::actionsSpherical(pot, traj[i]);
        statI.add(aaI);
        statF.add(aaF);
        statS.add(acS);
        actions::Angles anewF, anewI;
        anewF.thetar   = math::unwrapAngle(aaF.thetar,   aoldF.thetar);
        anewF.thetaz   = math::unwrapAngle(aaF.thetaz,   aoldF.thetaz);
        anewF.thetaphi = math::unwrapAngle(aaF.thetaphi, aoldF.thetaphi);
        anewI.thetar   = math::unwrapAngle(aaI.thetar,   aoldI.thetar);
        anewI.thetaz   = math::unwrapAngle(aaI.thetaz,   aoldI.thetaz);
        anewI.thetaphi = math::unwrapAngle(aaI.thetaphi, aoldI.thetaphi);
        anglesMonotonic &= i==0 || (
            anewI.thetar   >= aoldI.thetar && ( anewF.thetar   >= aoldF.thetar || aaF.Jr<1e-10 ) &&
            anewI.thetaz   >= aoldI.thetaz   && anewF.thetaz   >= aoldF.thetaz &&
            anewI.thetaphi >= aoldI.thetaphi && anewF.thetaphi >= aoldF.thetaphi);
        aoldF = anewF;
        aoldI = anewI;
        if(output) {
            strm << i*timestep<<"   "<<traj[i].R<<" "<<traj[i].z<<"   "<<
                aaI.thetar<<" "<<aaI.thetaz<<" "<<aaI.thetaphi<<"  "<<
                aaF.thetar<<" "<<aaF.thetaz<<" "<<aaF.thetaphi<<"  "<<
            "\n";
        }
    }
    statI.finish();
    statF.finish();
    statS.finish();
    bool dispI_ok = statI.disp.Jr<epsd && statI.disp.Jz<epsd && statI.disp.Jphi<epsd;
    bool dispS_ok = statS.disp.Jr<epsd && statS.disp.Jz<epsd && statS.disp.Jphi<epsd;
    bool dispF_ok = statF.disp.Jr<epsd && statF.disp.Jz<epsd && statF.disp.Jphi<epsd;
    bool compareIF =
             fabs(statI.avg.Jr-statF.avg.Jr)<eps
          && fabs(statI.avg.Jz-statF.avg.Jz)<eps
          && fabs(statI.avg.Jphi-statF.avg.Jphi)<eps;
    std::cout << "Isochrone"
    ":  Jr="  <<statI.avg.Jr  <<" +- "<<statI.disp.Jr<<
    ",  Jz="  <<statI.avg.Jz  <<" +- "<<statI.disp.Jz<<
    ",  Jphi="<<statI.avg.Jphi<<" +- "<<statI.disp.Jphi<< (dispI_ok?"":" \033[1;31m**\033[0m")<<
    "\nSpherical"
    ":  Jr="  <<statS.avg.Jr  <<" +- "<<statS.disp.Jr<<
    ",  Jz="  <<statS.avg.Jz  <<" +- "<<statS.disp.Jz<<
    ",  Jphi="<<statS.avg.Jphi<<" +- "<<statS.disp.Jphi<< (dispS_ok?"":" \033[1;31m**\033[0m")<<
    "\nAxi.Fudge"
    ":  Jr="  <<statF.avg.Jr  <<" +- "<<statF.disp.Jr<<
    ",  Jz="  <<statF.avg.Jz  <<" +- "<<statF.disp.Jz<<
    ",  Jphi="<<statF.avg.Jphi<<" +- "<<statF.disp.Jphi<< (dispF_ok?"":" \033[1;31m**\033[0m")<<
    (compareIF?"":" \033[1;31mNOT EQUAL\033[0m")<<
    (anglesMonotonic?"":" \033[1;31mANGLES NON-MONOTONIC\033[0m")<<'\n';
    return dispI_ok && dispS_ok && dispF_ok && compareIF && anglesMonotonic;
}

int main()
{
    bool ok=true;
    ok &= test_isochrone(coord::PosVelCyl(1.0, 0.3, 1.1, 0.1, 0.4, 0.1));  // ordinary case
    ok &= test_isochrone(coord::PosVelCyl(1.0, 0.0, 2.0, 1.0, 0.0, 0.5));  // Jz==0
    ok &= test_isochrone(coord::PosVelCyl(1.0, 0.0, 3.0, 0.0, 0.21,0.9));  // Jr small
    //ok &= test_isochrone(coord::PosVelCyl(1.0, 0.0, 2.0, 5.0, 1.0, 1e-4));  // Jphi small
    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    return 0;
}
