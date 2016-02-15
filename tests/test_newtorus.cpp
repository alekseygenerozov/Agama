/** \file    test_newtorus.cpp
    \author  Eugene Vasiliev
    \date    February 2016

*/
#include "potential_perfect_ellipsoid.h"
#include "actions_staeckel.h"
#include "actions_newtorus.h"
#include "actions_torus.h"
#include "orbit.h"
#include "math_core.h"
#include "debug_utils.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>

const double axis_a=1.6, axis_c=1.0; // axes of perfect ellipsoid
const double EPSACT  = 1e-2;
const double EPSANG  = 1e-2;
const double EPSFREQ = 1e-2;

bool test_torus(const potential::OblatePerfectEllipsoid& pot, const coord::PosVelCyl& point)
{
    // obtain exact actions and frequencies corresponding to the given IC
    actions::Frequencies frOrig;
    actions::ActionAngles aaOrig = actions::actionAnglesAxisymStaeckel(pot, point, &frOrig);

    // numerically compute the orbit
    double totalTime = 50*M_PI/fmax(frOrig.Omegar, frOrig.Omegaz);
    double timeStep  = totalTime * 0.001;
    std::vector<coord::PosVelCyl> trajOrig;
    clock_t tbegin = std::clock();
    orbit::integrate(pot, point, totalTime, timeStep, trajOrig);
    double torbit = (std::clock()-tbegin)*1.0/CLOCKS_PER_SEC;

    // construct the torus
    tbegin = std::clock();
    //actions::ActionMapperTorus tor(pot, aaOrig);
    actions::ActionMapperNewTorus tor(pot, aaOrig);
    double ttorus = (std::clock()-tbegin)*1.0/CLOCKS_PER_SEC;
    actions::Frequencies frTorus;

    // create an image of the orbit using torus mapping
    const unsigned int NPOINTS = trajOrig.size();
    std::vector<coord::PosVelCyl> trajTorus(NPOINTS);
    std::vector<actions::ActionAngles> aaTorus(NPOINTS);
    tbegin = std::clock();
    for(unsigned int i=0; i<NPOINTS; i++) {
        // assign angles from the condition that they linearly vary with time along the orbit
        aaTorus[i] = actions::ActionAngles(aaOrig, actions::Angles(
            math::wrapAngle(i*timeStep * frOrig.Omegar   + aaOrig.thetar), 
            math::wrapAngle(i*timeStep * frOrig.Omegaz   + aaOrig.thetaz),
            math::wrapAngle(i*timeStep * frOrig.Omegaphi + aaOrig.thetaphi)));
        // map act/ang to pos/vel using the torus
        trajTorus[i] = tor.map(aaTorus[i], &frTorus);
    }
    double tmap = (std::clock()-tbegin)*1.0/CLOCKS_PER_SEC;

    // recover the exact action/angles from the orbit mapped by torus
    std::vector<actions::ActionAngles> aaStk(NPOINTS);
    actions::ActionStat statAct;
    math::Averager statOmegar, statOmegaz, statOmegaphi, statH;
    double statAng=0;
    tbegin = std::clock();
    for(unsigned int i=0; i<NPOINTS; i++) {
        // compute a/a from p/v using exact expressions for Staeckel potential
        actions::Frequencies frS;
        aaStk[i] = actions::actionAnglesAxisymStaeckel(pot, trajTorus[i], &frS);
        if(!math::isFinite(aaStk[i].thetar) && math::isFinite(aaStk[i].Jr))
            continue;  // subtle error in action/angle finder, not relevant to Torus
        // collect statistics
        statH.add(totalEnergy(pot, trajTorus[i]));
        statAct.add(aaStk[i]);
        statOmegar.  add(frS.Omegar);
        statOmegaz.  add(frS.Omegaz);
        statOmegaphi.add(frS.Omegaphi);
        // align angles first
        if( aaStk[i].thetar - aaTorus[i].thetar >  M_PI)
            aaStk[i].thetar -= 2*M_PI;
        if( aaStk[i].thetar - aaTorus[i].thetar < -M_PI)
            aaStk[i].thetar += 2*M_PI;
        if( aaStk[i].thetaz - aaTorus[i].thetaz >  M_PI)
            aaStk[i].thetaz -= 2*M_PI;
        if( aaStk[i].thetaz - aaTorus[i].thetaz < -M_PI)
            aaStk[i].thetaz += 2*M_PI;
        if( aaStk[i].thetaphi - aaTorus[i].thetaphi >  M_PI)
            aaStk[i].thetaphi -= 2*M_PI;
        if( aaStk[i].thetaphi - aaTorus[i].thetaphi < -M_PI)
            aaStk[i].thetaphi += 2*M_PI;
        statAng += pow_2(aaStk[i].thetar-aaTorus[i].thetar) + 
            pow_2(aaStk[i].thetaz-aaTorus[i].thetaz) +
            pow_2(aaStk[i].thetaphi-aaTorus[i].thetaphi);
    }
    double tstk = (std::clock()-tbegin)*1.0/CLOCKS_PER_SEC;

    // output
    std::ofstream strm("torus.dat");
    strm << "#t\tR z phi\ttorus_R z phi\ttorus_thetar thetaz thetaphi\t"
        "exact_thetar thetaz thetaphi\tJr Jz E\n";
    for(unsigned int i=0; i<NPOINTS; i++) {
        strm << (i*timeStep) << '\t' <<
        trajOrig [i].R << ' ' << trajOrig [i].z << ' ' << math::wrapAngle(trajOrig [i].phi) << '\t' <<
        trajTorus[i].R << ' ' << trajTorus[i].z << ' ' << trajTorus[i].phi << '\t' <<
        aaTorus[i].thetar << ' ' << aaTorus[i].thetaz << ' ' << aaTorus[i].thetaphi << '\t' <<
        aaStk  [i].thetar << ' ' << aaStk  [i].thetaz << ' ' << aaStk  [i].thetaphi << '\t' <<
        aaStk[i].Jr << ' ' << aaStk[i].Jz << ' ' << totalEnergy(pot, trajTorus[i]) << '\n';
    }
    strm.close();
    
    // summarize
    statAct.finish();
    statAng = sqrt(statAng/(NPOINTS*3));
    bool ok_act = math::fcmp(aaOrig.Jr,   statAct.avg.Jr,   EPSACT) == 0
               && math::fcmp(aaOrig.Jz,   statAct.avg.Jz,   EPSACT) == 0
               && math::fcmp(aaOrig.Jphi, statAct.avg.Jphi, EPSACT) == 0
               && statAct.rms.Jr < aaOrig.Jr * EPSACT && statAct.rms.Jz < aaOrig.Jz * EPSACT;
    bool ok_ang = statAng < EPSANG;
    bool ok_frq = math::fcmp(frTorus.Omegar,   statOmegar.mean(),   EPSFREQ) == 0
               && math::fcmp(frTorus.Omegaz,   statOmegaz.mean(),   EPSFREQ) == 0
               && math::fcmp(frTorus.Omegaphi, statOmegaphi.mean(), EPSFREQ) == 0
               && sqrt(statOmegar.disp())   < EPSFREQ * statOmegar.mean()
               && sqrt(statOmegaz.disp())   < EPSFREQ * statOmegaz.mean()
               && sqrt(statOmegaphi.disp()) < EPSFREQ * statOmegaphi.mean();
    
    std::cout << 
        "  Jr: "  <<aaOrig.Jr  <<" = "<<statAct.avg.Jr  <<" +- "<<statAct.rms.Jr<<
        ", Jz: "  <<aaOrig.Jz  <<" = "<<statAct.avg.Jz  <<" +- "<<statAct.rms.Jz<<
        ", Jphi: "<<aaOrig.Jphi<<" = "<<statAct.avg.Jphi<<" +- "<<statAct.rms.Jphi<<
        (ok_act ? "" : " \033[1;31m**\033[0m ")<<
        "; Omegar="  <<frTorus.Omegar  <<" = "<<statOmegar.mean()  <<" +- "<<sqrt(statOmegar.disp())<<
        ", Omegaz="  <<frTorus.Omegaz  <<" = "<<statOmegaz.mean()  <<" +- "<<sqrt(statOmegaz.disp())<<
        ", Omegaphi="<<frTorus.Omegaphi<<" = "<<statOmegaphi.mean()<<" +- "<<sqrt(statOmegaphi.disp())<<
        (ok_frq ? "" : " \033[1;31m**\033[0m ")<<
        "; deltatheta="<<statAng << (ok_ang ? "" : " \033[1;31m**\033[0m ") <<
        "; deltaHrel=" << (-sqrt(statH.disp())/statH.mean()) << "\n";
    std::cout << torbit << " s to integrate orbit, " << ttorus << " s to create torus, " <<
        tmap << " s to map " << NPOINTS << " points, " << tstk << " s to compute actions\n";
    return ok_act && ok_ang && ok_frq;
}

int main()
{
    const potential::OblatePerfectEllipsoid potential(1.0, axis_a, axis_c);
    bool allok=true;
    allok &= test_torus(potential, coord::PosVelCyl(2.0000000, 0, 0, 0, 0.24, 0.5));
    allok &= test_torus(potential, coord::PosVelCyl(4.4444444, 0, 0, 0, 0.40, 0.225));
    allok &= test_torus(potential, coord::PosVelCyl(1.4142136, 0, 0, 0, 0.30, 0.7071068));
    allok &= test_torus(potential, coord::PosVelCyl(1.2000000, 0, 0, 0, 0.80, 0.083333333));
    allok &= test_torus(potential, coord::PosVelCyl(3.2000000, 0, 0, 0, 0.54, 0.3125));
    if(allok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    return 0;
}