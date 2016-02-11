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
#include <iomanip>
#include <cmath>

//#define TEST_OLD_TORUS
#ifdef TEST_OLD_TORUS
#include "torus/Toy_Isochrone.h"
#endif

bool test_isochrone(const coord::PosVelCyl& initial_conditions, const char* title)
{
    const bool output = false; // whether to write a text file
    const double epsr = 5e-4;  // accuracy of comparison for radial action found with different methods
    const double epsd = 1e-7;  // accuracy of action conservation along the orbit for each method
    const double epst = 1e-9;  // accuracy of reverse transformation (pv=>aa=>pv)
    const double M = 2.7;      // mass and
    const double b = 0.6;      // scale radius of Isochrone potential
    const double total_time=50;// integration time
    const double timestep=1./8;// sampling rate of trajectory
    std::cout << "\033[1;32m"<<title<<"\033[0m\n";
    std::vector<coord::PosVelCyl > traj;
    potential::Isochrone pot(M, b);
    orbit::integrate(pot, initial_conditions, total_time, timestep, traj, 1e-10);
    actions::ActionStat statI, statS, statF;
    actions::ActionAngles aaI, aaF;
    actions::Actions acS;
    actions::Frequencies frI, frF, frIinv;
    math::Averager statfrIr, statfrIz;
    actions::Angles aoldF(0,0,0), aoldI(0,0,0);
    bool anglesMonotonic = true;  // check that angle determination is reasonable
    bool reversible = true;       // check that forward-reverse transform gives the original point
    bool deriv_ok = true;         // check that finite-difference derivs agree with analytic ones
    std::ofstream strm;
    if(output) {
        strm.open("test_isochrone.dat");
        strm << std::setprecision(12);
    }
    double ifd = 1e-4;
    int numWarnings = 0;
#ifdef TEST_OLD_TORUS
    Torus::IsoPar toypar;
    toypar[0] = sqrt(M);
    toypar[1] = sqrt(b);
    toypar[2] = Lz(initial_conditions);
    Torus::ToyIsochrone toy(toypar);
#endif
    for(size_t i=0; i<traj.size(); i++) {
        traj[i].phi = math::wrapAngle(traj[i].phi);
        aaI = actions::actionAnglesIsochrone(M, b,  traj[i], &frI);
        aaF = actions::actionAnglesAxisymFudge(pot, traj[i], ifd, &frF);
        acS = actions::actionsSpherical(pot, traj[i]);        
        statI.add(aaI);
        statF.add(aaF);
        statS.add(acS);
        statfrIr.add(frI.Omegar);
        statfrIz.add(frI.Omegaz);
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
            math::sign(aaI.Jphi) * anewI.thetaphi >= math::sign(aaI.Jphi) * aoldI.thetaphi &&
            math::sign(aaF.Jphi) * anewF.thetaphi >= math::sign(aaF.Jphi) * aoldF.thetaphi);
        aoldF = anewF;
        aoldI = anewI;
#ifdef TEST_OLD_TORUS
        coord::PosVelSph ps(toPosVelSph(traj[i]));
        Torus::PSPT pvs;
        pvs[0]=ps.r; pvs[1]=M_PI/2-ps.theta; pvs[2]=ps.phi;
        pvs[3]=ps.vr; pvs[4]=-ps.vtheta*ps.r; pvs[5]=ps.vphi*traj[i].R;
        Torus::PSPT aaT = toy.Backward3D(pvs); // (J_r,J_z,J_phi,theta_r,theta_z,theta_phi)
        Torus::PSPT pvi = toy.Forward3D(aaT);
        reversible &= 
            math::fcmp(pvs[0], pvi[0], eps) == 0 && math::fcmp(pvs[1], pvi[1], eps) == 0 &&
            math::fcmp(pvs[2], pvi[2], eps) == 0 && math::fcmp(pvs[3]+1, pvi[3]+1, eps) == 0 &&
            math::fcmp(pvs[4]+1, pvi[4]+1, eps) == 0 && math::fcmp(pvs[5]+1, pvi[5]+1, eps) == 0;
        reversible &=
            math::fcmp(aaT[0]+1, aaI.Jr+1, eps) == 0 &&  // ok when Jr<<1
            math::fcmp(aaT[1], aaI.Jz, eps) == 0 &&
            math::fcmp(aaT[2], aaI.Jphi, eps) == 0 &&
            (aaI.Jr<epsd || math::fcmp(aaT[3], aaI.thetar, eps) == 0) &&
            (aaI.Jz==0 || math::fcmp(math::wrapAngle(aaT[4]+1), math::wrapAngle(aaI.thetaz+1), eps) == 0) &&
            math::fcmp(aaT[5], aaI.thetaphi, eps) == 0;
        if(!reversible)
            std::cout << aaT <<'\t' <<aaI << '\n';
#endif
        // inverse transformation with derivs
        coord::PosVelCyl pd[2];
        actions::DerivAct ac;
        actions::DerivAng an;
        coord::PosVelCyl pp = actions::ToyMapIsochrone(M, b).mapDeriv(aaI, &frIinv, &ac, &an, pd);
        reversible &= equalPosVel(pp, traj[i], epst) && 
            math::fcmp(frI.Omegar, frIinv.Omegar, epst) == 0 &&
            math::fcmp(frI.Omegaz, frIinv.Omegaz, epst) == 0 &&
            math::fcmp(frI.Omegaphi, frIinv.Omegaphi, epst) == 0;
        // check derivs w.r.t. potential params
        coord::PosVelCyl pM = actions::ToyMapIsochrone(M*(1+epsd), b).map(aaI);
        coord::PosVelCyl pb = actions::ToyMapIsochrone(M, b*(1+epsd)).map(aaI);
        pM.R    = (pM.R  - pp.R)   / (M*epsd);
        pM.z    = (pM.z  - pp.z)   / (M*epsd);
        pM.phi  = (pM.phi- pp.phi) / (M*epsd);
        pM.vR   = (pM.vR - pp.vR)  / (M*epsd);
        pM.vz   = (pM.vz - pp.vz)  / (M*epsd);
        pM.vphi = (pM.vphi-pp.vphi)/ (M*epsd);
        pb.R    = (pb.R  - pp.R)   / (b*epsd);
        pb.z    = (pb.z  - pp.z)   / (b*epsd);
        pb.phi  = (pb.phi- pp.phi) / (b*epsd);
        pb.vR   = (pb.vR - pp.vR)  / (b*epsd);
        pb.vz   = (pb.vz - pp.vz)  / (b*epsd);
        pb.vphi = (pb.vphi-pp.vphi)/ (b*epsd);
        if(!equalPosVel(pM, pd[0], 1e-5) && ++numWarnings<10) {
            deriv_ok = false;
            std::cout << "d/dM: " << pM << pd[0] << '\n';
        }
        if(!equalPosVel(pb, pd[1], 1e-5) && ++numWarnings<10) {
            deriv_ok = false;
            std::cout << "d/db: " << pb << pd[1] << '\n';
        }
        // check derivs w.r.t. actions
        actions::ActionAngles aaT = aaI; aaT.Jr += epsd;
        coord::PosVelCyl pJr = mapIsochrone(M, b, aaT);
        pJr.R   = (pJr.R   - pp.R)   / epsd;
        pJr.z   = (pJr.z   - pp.z)   / epsd;
        pJr.phi = (pJr.phi - pp.phi) / epsd;
        pJr.vR  = (pJr.vR  - pp.vR)  / epsd;
        pJr.vz  = (pJr.vz  - pp.vz)  / epsd;
        pJr.vphi= (pJr.vphi- pp.vphi)/ epsd;
        aaT = aaI; aaT.Jz += epsd;
        coord::PosVelCyl pJz = mapIsochrone(M, b, aaT);
        pJz.R   = (pJz.R   - pp.R)   / epsd;
        pJz.z   = (pJz.z  - pp.z)  / epsd;
        pJz.phi = (pJz.phi - pp.phi) / epsd;
        pJz.vR  = (pJz.vR  - pp.vR)  / epsd;
        pJz.vz  = (pJz.vz - pp.vz) / epsd;
        pJz.vphi= (pJz.vphi- pp.vphi)/ epsd;
        if(aaI.Jz==0) {
            deriv_ok &= !math::isFinite(ac.dbyJz.z+ac.dbyJz.vz);  // should be infinite
            pJz.z=pJz.vz=ac.dbyJz.z=ac.dbyJz.vz=0;  // exclude from comparison
        }
        aaT = aaI; aaT.Jphi += epsd;
        coord::PosVelCyl pJp = mapIsochrone(M, b, aaT);
        pJp.R   = (pJp.R   - pp.R)   / epsd;
        pJp.z   = (pJp.z   - pp.z)   / epsd;
        pJp.phi = (pJp.phi - pp.phi) / epsd;
        pJp.vR  = (pJp.vR  - pp.vR)  / epsd;
        pJp.vz  = (pJp.vz  - pp.vz)  / epsd;
        pJp.vphi= (pJp.vphi- pp.vphi)/ epsd;
        if(!equalPosVel(pJr, ac.dbyJr, 1e-4) && ++numWarnings<10) {
            deriv_ok = false;
            std::cout << "d/dJr: " << pJr << ac.dbyJr << '\n';
        }
        if(!equalPosVel(pJz, ac.dbyJz, 1e-4) && ++numWarnings<10) {
            deriv_ok = false;
            std::cout << "d/dJz: " << pJz << ac.dbyJz << '\n';
        }
        if(!equalPosVel(pJp, ac.dbyJphi, 1e-4) && ++numWarnings<10) {
            deriv_ok = false;
            std::cout << "d/dJphi: " << pJp << ac.dbyJphi << '\n';
        }
        if(output) {
            strm << i*timestep<<"   "<<traj[i].R<<" "<<traj[i].z<<" "<<traj[i].phi<<"  "<<
                pp.R<<" "<<pp.z<<" "<<pp.phi<<"   "<<
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
             fabs(statI.avg.Jr-statF.avg.Jr)<epsr
          && fabs(statI.avg.Jz-statF.avg.Jz)<epsr
          && fabs(statI.avg.Jphi-statF.avg.Jphi)<epsd;
    bool freq_ok = statfrIr.disp() < epsd && statfrIz.disp() < epsd;
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
    (compareIF?"":" \033[1;31mNOT EQUAL\033[0m ")<<
    (reversible?"":" \033[1;31mNOT INVERTIBLE\033[0m ")<<
    (freq_ok?"":" \033[1;31mFREQS NOT CONST\033[0m ")<<
    (deriv_ok?"":" \033[1;31mDERIVS INCONSISTENT\033[0m ")<<
    (anglesMonotonic?"":" \033[1;31mANGLES NON-MONOTONIC\033[0m ")<<'\n';
    return dispI_ok && dispS_ok && dispF_ok && compareIF
        && freq_ok && reversible && deriv_ok && anglesMonotonic;
}

int main()
{
    bool ok=true;
    ok &= test_isochrone(coord::PosVelCyl(1.0, 0.3, 1.1, 0.1, 0.4,  0.1), "ordinary case");
    ok &= test_isochrone(coord::PosVelCyl(1.0, 0.0, 2.2, 1.0, 0.0,  0.5), "Jz==0");
    ok &= test_isochrone(coord::PosVelCyl(1.0, 0.0, 3.3, 0.0, 0.21, 0.9), "Jr small");
    ok &= test_isochrone(coord::PosVelCyl(1.0, 0.0, 4.4, 0.6, 1.0, 1e-4), "Jphi small");
    ok &= test_isochrone(coord::PosVelCyl(1.0, 0.5, 5.5, 0.5, 0.7, -0.5), "Jphi negative");
    ok &= test_isochrone(coord::PosVelCyl(1.0, 0.0,M_PI, 0.0, 0.0, -0.5), "Jz==0, Jphi<0");
    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    return 0;
}
