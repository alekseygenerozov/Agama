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
    const double epsf = 1e-6;  // accuracy of frequency determination
    const double M = 2.7;      // mass and
    const double b = 0.6;      // scale radius of Isochrone potential
    const double total_time=50;// integration time
    const double timestep=1./8;// sampling rate of trajectory
    std::cout << "\033[1;39m"<<title<<"\033[0m\n";
    std::vector<coord::PosVelCyl > traj;
    potential::Isochrone pot(M, b);
    orbit::integrate(pot, initial_conditions, total_time, timestep, traj, 1e-10);
    actions::ActionStat statI, statS, statF;
    actions::ActionAngles aaI, aaF;
    actions::Actions acS;
    actions::Frequencies frI, frF, frIinv;
    math::Averager statfrIr, statfrIz, statH;
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
        statH.add(actions::computeHamiltonianSpherical(pot, aaI));  // find H(J)
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
        coord::PosVelSphMod pd[2];
        actions::DerivAct<coord::SphMod> ac;
        coord::PosVelSphMod pp = actions::ToyMapIsochrone(M, b).map(aaI, &frIinv, &ac, NULL, pd);
        reversible &= equalPosVel(toPosVelCyl(pp), traj[i], epst) && 
            math::fcmp(frI.Omegar, frIinv.Omegar, epst) == 0 &&
            math::fcmp(frI.Omegaz, frIinv.Omegaz, epst) == 0 &&
            math::fcmp(frI.Omegaphi, frIinv.Omegaphi, epst) == 0;
        // check derivs w.r.t. potential params
        coord::PosVelSphMod pM = actions::ToyMapIsochrone(M*(1+epsd), b).map(aaI);
        coord::PosVelSphMod pb = actions::ToyMapIsochrone(M, b*(1+epsd)).map(aaI);
        pM.r   = (pM.r   - pp.r)   / (M*epsd);
        pM.tau = (pM.tau - pp.tau) / (M*epsd);
        pM.phi = (pM.phi - pp.phi) / (M*epsd);
        pM.pr  = (pM.pr  - pp.pr)  / (M*epsd);
        pM.ptau= (pM.ptau- pp.ptau)/ (M*epsd);
        pM.pphi= (pM.pphi- pp.pphi)/ (M*epsd);
        pb.r   = (pb.r   - pp.r)   / (b*epsd);
        pb.tau = (pb.tau - pp.tau) / (b*epsd);
        pb.phi = (pb.phi - pp.phi) / (b*epsd);
        pb.pr  = (pb.pr  - pp.pr)  / (b*epsd);
        pb.ptau= (pb.ptau- pp.ptau)/ (b*epsd);
        pb.pphi= (pb.pphi- pp.pphi)/ (b*epsd);
        if(!equalPosVel(pM, pd[0], 1e-4) && ++numWarnings<10) {
            deriv_ok = false;
            std::cout << "d/dM: " << pM << pd[0] << '\n';
        }
        if(!equalPosVel(pb, pd[1], 1e-4) && ++numWarnings<10) {
            deriv_ok = false;
            std::cout << "d/db: " << pb << pd[1] << '\n';
        }
        // check derivs w.r.t. actions
        actions::ActionAngles aaT = aaI; aaT.Jr += epsd;
        coord::PosVelSphMod pJr = actions::ToyMapIsochrone(M, b).map(aaT);
        pJr.r   = (pJr.r   - pp.r)   / epsd;
        pJr.tau = (pJr.tau - pp.tau) / epsd;
        pJr.phi = (pJr.phi - pp.phi) / epsd;
        pJr.pr  = (pJr.pr  - pp.pr)  / epsd;
        pJr.ptau= (pJr.ptau- pp.ptau)/ epsd;
        pJr.pphi= (pJr.pphi- pp.pphi)/ epsd;
        aaT = aaI; aaT.Jz += epsd;
        coord::PosVelSphMod pJz = actions::ToyMapIsochrone(M, b).map(aaT);
        pJz.r   = (pJz.r   - pp.r)   / epsd;
        pJz.tau = (pJz.tau - pp.tau) / epsd;
        pJz.phi = (pJz.phi - pp.phi) / epsd;
        pJz.pr  = (pJz.pr  - pp.pr)  / epsd;
        pJz.ptau= (pJz.ptau- pp.ptau)/ epsd;
        pJz.pphi= (pJz.pphi- pp.pphi)/ epsd;
        if(aaI.Jz==0) {
            deriv_ok &= !math::isFinite(ac.dbyJz.tau+ac.dbyJz.ptau);  // should be infinite
            pJz.tau=pJz.ptau=ac.dbyJz.tau=ac.dbyJz.ptau=0;  // exclude from comparison
        }
        aaT = aaI; aaT.Jphi += epsd;
        coord::PosVelSphMod pJp = actions::ToyMapIsochrone(M, b).map(aaT);
        pJp.r   = (pJp.r   - pp.r)   / epsd;
        pJp.tau = (pJp.tau - pp.tau) / epsd;
        pJp.phi = (pJp.phi - pp.phi) / epsd;
        pJp.pr  = (pJp.pr  - pp.pr)  / epsd;
        pJp.ptau= (pJp.ptau- pp.ptau)/ epsd;
        pJp.pphi= (pJp.pphi- pp.pphi)/ epsd;
        if(!equalPosVel(pJr, ac.dbyJr, 1e-4) && ++numWarnings<10) {
            deriv_ok = false;
            std::cout << "d/dJr: " << pJr << ac.dbyJr << '\n';
        }
        if(!equalPosVel(pJz, ac.dbyJz, 1e-4) && ++numWarnings<10) {
            deriv_ok = false;
            std::cout << "d/dJz: " << pJz << ac.dbyJz << '\n';
        }
        if(!equalPosVel(pJp, ac.dbyJphi, 1e-3) && ++numWarnings<10) {
            deriv_ok = false;
            std::cout << "d/dJphi: " << pJp << ac.dbyJphi << '\n';
        }
        if(output) {
            strm << i*timestep<<"   "<<traj[i].R<<" "<<traj[i].z<<" "<<traj[i].phi<<"  "<<
                toPosVelCyl(pp).R<<" "<<toPosVelCyl(pp).z<<" "<<pp.phi<<"   "<<
                aaI.thetar<<" "<<aaI.thetaz<<" "<<aaI.thetaphi<<"  "<<
                aaF.thetar<<" "<<aaF.thetaz<<" "<<aaF.thetaphi<<"  "<<
            "\n";
        }
    }
    statI.finish();
    statF.finish();
    statS.finish();
    bool dispI_ok = statI.rms.Jr<epsd && statI.rms.Jz<epsd && statI.rms.Jphi<epsd;
    bool dispS_ok = statS.rms.Jr<epsd && statS.rms.Jz<epsd && statS.rms.Jphi<epsd;
    bool dispF_ok = statF.rms.Jr<epsd && statF.rms.Jz<epsd && statF.rms.Jphi<epsd;
    bool compareIF =
             fabs(statI.avg.Jr-statF.avg.Jr)<epsr
          && fabs(statI.avg.Jz-statF.avg.Jz)<epsr
          && fabs(statI.avg.Jphi-statF.avg.Jphi)<epsd;
    bool freq_ok = statfrIr.disp() < epsf*epsf && statfrIz.disp() < epsf*epsf;
    bool HofJ_ok = statH.disp() < pow_2(epsf*statH.mean());
    std::cout << "Isochrone"
    ":  Jr="  <<statI.avg.Jr  <<" +- "<<statI.rms.Jr<<
    ",  Jz="  <<statI.avg.Jz  <<" +- "<<statI.rms.Jz<<
    ",  Jphi="<<statI.avg.Jphi<<" +- "<<statI.rms.Jphi<< (dispI_ok?"":" \033[1;31m**\033[0m")<<
    "\nSpherical"
    ":  Jr="  <<statS.avg.Jr  <<" +- "<<statS.rms.Jr<<
    ",  Jz="  <<statS.avg.Jz  <<" +- "<<statS.rms.Jz<<
    ",  Jphi="<<statS.avg.Jphi<<" +- "<<statS.rms.Jphi<< (dispS_ok?"":" \033[1;31m**\033[0m")<<
    "\nAxi.Fudge"
    ":  Jr="  <<statF.avg.Jr  <<" +- "<<statF.rms.Jr<<
    ",  Jz="  <<statF.avg.Jz  <<" +- "<<statF.rms.Jz<<
    ",  Jphi="<<statF.avg.Jphi<<" +- "<<statF.rms.Jphi<< (dispF_ok?"":" \033[1;31m**\033[0m")<<
    (compareIF?"":" \033[1;31mNOT EQUAL\033[0m ")<<
    (reversible?"":" \033[1;31mNOT INVERTIBLE\033[0m ")<<
    (freq_ok?"":" \033[1;31mFREQS NOT CONST\033[0m ")<<
    (deriv_ok?"":" \033[1;31mDERIVS INCONSISTENT\033[0m ")<<
    (anglesMonotonic?"":" \033[1;31mANGLES NON-MONOTONIC\033[0m ")<<
    "\nHamiltonian H(J)="<<statH.mean()<<" +- "<<sqrt(statH.disp())<<
    (HofJ_ok?"":" \033[1;31m**\033[0m") <<'\n';
    return dispI_ok && dispS_ok && dispF_ok && compareIF
        && freq_ok && reversible && deriv_ok && anglesMonotonic && HofJ_ok;
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
