/** \file    test_newtorus.cpp
    \author  Eugene Vasiliev
    \date    February 2016

*/
#include "potential_perfect_ellipsoid.h"
#include "actions_staeckel.h"
#include "actions_newtorus.h"
#include "math_core.h"
#include "debug_utils.h"
#include <iostream>
#include <fstream>
#include <cmath>

const double axis_a=1.6, axis_c=1.0; // axes of perfect ellipsoid
const unsigned int NPOINTS = 100;
const double EPSACT  = 1e-3;
const double DISPACT = 1e-2;
const double EPSANG  = 1e-2;
const double EPSFREQ = 1e-3;
const double DISPFREQ= 1e-8;

bool test_torus(const potential::OblatePerfectEllipsoid& pot, const actions::Actions acts)
{
    actions::ActionMapperNewTorus tor(pot, acts);
    actions::ActionStat statAct;
    math::Averager statOmegar, statOmegaz, statOmegaphi;
    double statAng=0;
    actions::Frequencies frT, frS;
    for(unsigned int i=0; i<NPOINTS; i++) {
        // randomly sample angles
        actions::Angles angs(math::random()*2*M_PI, math::random()*2*M_PI, math::random()*2*M_PI);
        // map act/ang to pos/vel using the torus
        actions::ActionAngles aaT(acts, angs);
        coord::PosVelCyl pp = tor.map(aaT, &frT);
        // compute a/a from p/v using exact expressions for Staeckel potential
        actions::ActionAngles aaS = actions::actionAnglesAxisymStaeckel(pot, pp, &frS);
        // collect statistics
        statAct.add(aaS);
        statOmegar.  add(frS.Omegar);
        statOmegaz.  add(frS.Omegaz);
        statOmegaphi.add(frS.Omegaphi);
        // align angles first
        if(aaS.thetar - aaT.thetar >  M_PI)
            aaS.thetar -= 2*M_PI;
        if(aaS.thetar - aaT.thetar < -M_PI)
            aaS.thetar += 2*M_PI;
        if(aaS.thetaz - aaT.thetaz >  M_PI)
            aaS.thetaz -= 2*M_PI;
        if(aaS.thetaz - aaT.thetaz < -M_PI)
            aaS.thetaz += 2*M_PI;
        if(aaS.thetaphi - aaT.thetaphi >  M_PI)
            aaS.thetaphi -= 2*M_PI;
        if(aaS.thetaphi - aaT.thetaphi < -M_PI)
            aaS.thetaphi += 2*M_PI;
        statAng += pow_2(aaS.thetar-aaT.thetar) + 
            pow_2(aaS.thetaz-aaT.thetaz) + pow_2(aaS.thetaphi-aaT.thetaphi);
    }
    statAct.finish();
    statAng = sqrt(statAng/(NPOINTS*3));
    bool ok_act = math::fcmp(acts.Jr,   statAct.avg.Jr,   EPSACT) == 0
               && math::fcmp(acts.Jz,   statAct.avg.Jz,   EPSACT) == 0
               && math::fcmp(acts.Jphi, statAct.avg.Jphi, EPSACT) == 0
               && statAct.disp.Jr < DISPACT && statAct.disp.Jz < DISPACT;
    bool ok_ang = statAng < EPSANG;
    bool ok_frq = math::fcmp(frT.Omegar,   statOmegar.mean(),   EPSFREQ) == 0
               && math::fcmp(frT.Omegaz,   statOmegaz.mean(),   EPSFREQ) == 0
               && math::fcmp(frT.Omegaphi, statOmegaphi.mean(), EPSFREQ) == 0
               && statOmegar.disp() < DISPFREQ && statOmegaz.disp() < DISPFREQ
               && statOmegaphi.disp() < DISPFREQ;
    
    std::cout << 
        "  Jr: "  <<acts.Jr  <<" = "<<statAct.avg.Jr  <<" +- "<<statAct.disp.Jr<<
        ", Jz: "  <<acts.Jz  <<" = "<<statAct.avg.Jz  <<" +- "<<statAct.disp.Jz<<
        ", Jphi: "<<acts.Jphi<<" = "<<statAct.avg.Jphi<<" +- "<<statAct.disp.Jphi<<
        (ok_act ? "" : " \033[1;31m**\033[0m ")<<
        "; Omegar="  <<frT.Omegar  <<" = "<<statOmegar.mean()  <<" +- "<<statOmegar.disp()<<
        ", Omegaz="  <<frT.Omegaz  <<" = "<<statOmegaz.mean()  <<" +- "<<statOmegaz.disp()<<
        ", Omegaphi="<<frT.Omegaphi<<" = "<<statOmegaphi.mean()<<" +- "<<statOmegaphi.disp()<<
        (ok_frq ? "" : " \033[1;31m**\033[0m ")<<
        "; delta_theta="<<statAng << (ok_ang ? "" : " \033[1;31m**\033[0m ") << "\n";
    return ok_act && ok_ang && ok_frq;
}

int main()
{
    const potential::OblatePerfectEllipsoid potential(1.0, axis_a, axis_c);
    bool allok=true;
    allok &= test_torus(potential, actions::Actions(0.1, 1.0, 1.0));
    //allok &= test_torus(potential, actions::Actions(1.0, 0.1, 1.0));
    //allok &= test_torus(potential, actions::Actions(1.0, 1.0, 0.1));
    allok &= test_torus(potential, actions::Actions(1.0, 1.0, 1.0));
    if(allok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    return 0;
}