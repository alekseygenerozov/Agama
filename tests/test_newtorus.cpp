/** \file    test_newtorus.cpp
    \author  Eugene Vasiliev
    \date    February 2016

*/
#include "potential_perfect_ellipsoid.h"
#include "potential_analytic.h"
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

const double AXIS_A  = 1.6, AXIS_C = 1.0; // axes of perfect ellipsoid
const double CS_DELTA= sqrt(pow_2(AXIS_A)-pow_2(AXIS_C));
const double EPSACT  = 3e-2;
const double EPSANG  = 5e-2;
const double EPSFREQ = 1e-2;
const double EPSENER = 3e-3;
std::ofstream strm;

bool test_torus(const potential::BasePotential& pot, const coord::PosVelCyl& point)
{
    try{
    // obtain exact actions and frequencies corresponding to the given IC
    actions::Frequencies frOrig;
    actions::ActionAngles aaOrig = actions::actionAnglesAxisymFudge(pot, point, CS_DELTA, &frOrig);
    std::cout << "\033[1;39m" << (actions::Actions(aaOrig)) << "\033[0m\n";

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
        aaStk[i] = actions::actionAnglesAxisymFudge(pot, trajTorus[i], CS_DELTA, &frS);
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
    std::ofstream strm("torus.orb");
    strm << "#t\tR z phi\ttorus_R z phi\ttorus_thetar thetaz thetaphi\t"
        "exact_thetar thetaz thetaphi\tJr Jz E\n";
    for(unsigned int i=0; i<NPOINTS; i++) {
        coord::ProlMod cm(CS_DELTA);
        coord::PosVelProlMod pm(coord::toPosVel<coord::Cyl,coord::ProlMod>(trajOrig[i], cm));
        strm << (i*timeStep) << '\t' <<
        trajOrig [i].R << ' ' << trajOrig [i].z << ' ' << math::wrapAngle(trajOrig [i].phi) << '\t' <<
        trajTorus[i].R << ' ' << trajTorus[i].z << ' ' << trajTorus[i].phi << '\t' <<
        aaTorus[i].thetar << ' ' << aaTorus[i].thetaz << ' ' << aaTorus[i].thetaphi << '\t' <<
        aaStk  [i].thetar << ' ' << aaStk  [i].thetaz << ' ' << aaStk  [i].thetaphi << '\t' <<
        aaStk[i].Jr << ' ' << aaStk[i].Jz << ' ' << totalEnergy(pot, trajTorus[i]) << '\t' <<
        pot.value(trajOrig[i]) << ' ' << 
        0.5*(pow_2(trajOrig[i].vR)+pow_2(trajOrig[i].vz)+pow_2(trajOrig[i].vphi)) << '\t'<<
        pm.rho << ' ' << pm.tau << ' ' << pm.prho << ' ' << pm.ptau << '\n';
    }
    strm.close();
    
    // summarize
    statAct.finish();
    statAng = sqrt(statAng/(NPOINTS*3));
    double statEner = (-sqrt(statH.disp())/statH.mean());
    bool ok_ang = statAng < EPSANG;
    bool ok_ener= statEner< EPSENER;
    bool ok_act = math::fcmp(aaOrig.Jr,   statAct.avg.Jr,   EPSACT) == 0
               && math::fcmp(aaOrig.Jz,   statAct.avg.Jz,   EPSACT) == 0
               && math::fcmp(aaOrig.Jphi, statAct.avg.Jphi, EPSACT) == 0
               && statAct.rms.Jr < aaOrig.Jr * EPSACT && statAct.rms.Jz < aaOrig.Jz * EPSACT;
    bool ok_frq = math::fcmp(frTorus.Omegar,   statOmegar.mean(),   EPSFREQ) == 0
               && math::fcmp(frTorus.Omegaz,   statOmegaz.mean(),   EPSFREQ) == 0
               && math::fcmp(frTorus.Omegaphi, statOmegaphi.mean(), EPSFREQ) == 0
               && sqrt(statOmegar.disp())   < EPSFREQ * statOmegar.mean()
               && sqrt(statOmegaz.disp())   < EPSFREQ * statOmegaz.mean()
               && sqrt(statOmegaphi.disp()) < EPSFREQ * fabs(statOmegaphi.mean());
    
    strm << aaOrig.Jr<<' '<<aaOrig.Jz<<' '<<aaOrig.Jphi<<'\t'<<
    ((statAct.rms.Jr+statAct.rms.Jz)/(statAct.avg.Jr+statAct.avg.Jz))<<'\t'<<
    (sqrt(statOmegar.disp()+statOmegaz.disp()+statOmegaphi.disp())/
    (statOmegar.mean()+statOmegaz.mean()+statOmegaphi.mean()))<<'\t'<<
    statAng<<'\t'<<statEner<<std::endl;
    
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
        "; deltaHrel=" <<statEner<< (ok_ener? "" : " \033[1;31m**\033[0m ") << "\n";
    std::cout << torbit << " s to integrate orbit, " << ttorus << " s to create torus, " <<
        tmap << " s to map " << NPOINTS << " points, " << tstk << " s to compute actions\n";
    return ok_act && ok_ang && ok_frq && ok_ener;
    }
    catch(std::exception &e) {
        std::cout << "\033[1;31mException:\033[0m" << e.what() << '\n';
        return false;
    }
}

int main()
{
    strm.open("torus.stat");
    const potential::OblatePerfectEllipsoid potential(1.0, AXIS_A, AXIS_C);
    bool allok=true;
    allok &= test_torus(potential, coord::PosVelCyl(1.0000000, 0, 0, 0, 0.45/*0.446*/,0.1));
    //allok &= test_torus(potential, coord::PosVelCyl(2.0000000, 0, 0, 0, 0.01, 0.5));
    allok &= test_torus(potential, coord::PosVelCyl(2.0000000, 0, 0, 0, 0.24, 0.5));
    allok &= test_torus(potential, coord::PosVelCyl(4.4444444, 0, 0, 0, 0.40, 0.225));
    allok &= test_torus(potential, coord::PosVelCyl(1.4142136, 0, 0, 0, 0.30, 0.7071068));
    allok &= test_torus(potential, coord::PosVelCyl(1.2000000, 0, 0, 0, 0.80, 0.083333333));
    allok &= test_torus(potential, coord::PosVelCyl(3.2000000, 0, 0, 0, 0.54, 0.3125));
    for(int p=0; p<100; p++) {
        double m = math::random();
        double r = pow(1/sqrt(m)-1, -2./3);
        double costheta = math::random()*2 - 1;
        double sintheta = sqrt(1-pow_2(costheta));
        double phi = math::random()*2*M_PI;
        coord::PosVelCyl point(r*sintheta, r*costheta, phi, 0, 0, 0);
        double v = math::random() * sqrt(-2*totalEnergy(potential, point));   // uniform in [0..v_escape]
        costheta = math::random()*2 - 1;
        sintheta = sqrt(1-pow_2(costheta));
        phi = math::random()*2*M_PI;
        point.vR = v*sintheta*cos(phi);
        point.vz = v*sintheta*sin(phi);
        point.vphi = v*costheta;
        allok &= test_torus(potential, point);
    }
    if(allok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    return 0;
}