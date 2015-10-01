/** \file    test_torus_new.cpp
    \author  Brian Jia Jiunn Khor
    \date    07/08/2015

    This test compares two methods for angle mapping in Torus machinery -
    the original one from McMillan&Binney 2008, and the one suggested in
    Laakso&Kaasalainen 2013 (a faster one).
*/
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include "torus/Torus.h"
#include "potential_factory.h"
#include "math_core.h"
#include "orbit.h"

/// Auxiliary class for using any of BasePotential-derived potentials with Torus code
class TorusPotentialWrapper: public Torus::Potential{
public:
    TorusPotentialWrapper(const potential::BasePotential& _poten) : poten(_poten) {};
    virtual ~TorusPotentialWrapper() {};
    virtual double operator()(const double R, const double z) const {
        return poten.value(coord::PosCyl(R, z, 0));
    }
    virtual double operator()(const double R, const double z, double& dPhidR, double& dPhidz) const {
        double val;
        coord::GradCyl grad;
        poten.eval(coord::PosCyl(R, z, 0), &val, &grad);
        dPhidR = grad.dR;
        dPhidz = grad.dz;
        return val;
    }
    virtual double RfromLc(double Lz, double* =0) const {
        return R_from_Lz(poten, Lz);
    }
    virtual double LfromRc(double R, double* ) const {
        return v_circ(poten, R) * R;
    }
    virtual Torus::Frequencies KapNuOm(double R) const {
        Torus::Frequencies freq;
        epicycleFreqs(poten, R, freq[0], freq[1], freq[2]);
        return freq;
    }
private:
    const potential::BasePotential& poten;
};

void test(bool useNewAngleMapping, const potential::BasePotential& poten, const Torus::Actions &J)
{
    TorusPotentialWrapper Phi(poten);
    Torus::Torus T(useNewAngleMapping);
    T.AutoFit(J,&Phi,0.001,700,300,15,5,24,200,24,0);

    // orbit integration part
    Torus::Angles theta;
    theta[0] = theta[1] = theta[2] = 0;
    Torus::PSPT P = T.Map3D(theta);
    double totaltime = 100 * 2*M_PI / T.omega(0);
    int numsteps     = 5000;
    double timestep  = totaltime / numsteps;
    std::vector<coord::PosVelCyl> traj;
    orbit::integrate(poten, coord::PosVelCyl(P[0], P[1], P[2], P[3], P[4], P[5]), totaltime, timestep, traj, 1e-8);

    // torus part
    std::ofstream strm(useNewAngleMapping ? "traj_new.txt" : "traj_old.txt");
    math::Averager avg;
    double phi = 0;
    for(unsigned int s=0; s<traj.size(); s++) {
        double t = s*timestep;
        for(int d=0; d<3; d++)
            theta[d] = T.omega(d) * t;
        P = T.Map3D(theta);
        phi = math::unwrapAngle(P[2], phi);
        strm<< t << '\t' << traj[s].R << '\t' << traj[s].z << '\t' << traj[s].phi
            << '\t' << P[0] << '\t' << P[1] << '\t' << phi << '\n';
        avg.add(totalEnergy(poten, coord::PosVelCyl(P[0], P[1], P[2], P[3], P[4], P[5])));
    }
    std::cout << std::setprecision(9) << "E = " << avg.mean() << "+-" << sqrt(avg.disp()) <<
        "; freqs = " << T.omega(0) << ", " << T.omega(1) << ", " << T.omega(2) << std::endl;
}

int main(int argc, char** argv) {
    const potential::BasePotential* poten = 
        potential::readGalaxyPotential("../temp/GalPot.pot", units::galactic_Myr);
    Torus::Actions J;
    J[0] = 1; // actions in whatever units
    J[1] = 1;
    J[2] = 2;
    if(argc>=3) {
        J[0] = atof(argv[1]); J[1] = atof(argv[2]); J[2] = atof(argv[3]);
    }
    test(false, *poten, J);
    test(true,  *poten, J);
    //std::cout << "ALL TESTS PASSED\n";
    return 0;
}
