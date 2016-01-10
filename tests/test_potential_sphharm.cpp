#include "potential_analytic.h"
#include "potential_dehnen.h"
#include "potential_cylspline.h"
#include "potential_sphharm.h"
#include "potential_factory.h"
#include "potential_galpot.h"
#include "particles_io.h"
#include "debug_utils.h"
#include "utils.h"
#include "utils_config.h"
#include "math_sphharm.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cstdio>

/// define test suite in terms of points for various coord systems
const int numtestpoints=8;
const double pos_sph[numtestpoints][3] = {   // order: R, theta, phi
    {1  , 2  , 3},   // ordinary point
    {2  , 1  , 0},   // point in x-z plane
    {1  , 0  , 0},   // point along z axis
    {2  , 3.14159,-1},   // point along z axis, z<0
    {0.5, 1.5707963, 1.5707963},  // point along y axis
    {111, 0.7, 2},   // point at a large radius
    {.01, 2.5,-1},   // point at a small radius origin
    {0  , 0  , 0} }; // point at origin

const bool output = false;

/// write potential coefs into file, load them back and create a new potential from these coefs
potential::PtrPotential write_read(const potential::BasePotential& pot)
{
    std::string coefFile("test_potential_sphharm");
    coefFile += getCoefFileExtension(pot);
    writePotential(coefFile, pot);
    potential::PtrPotential newpot = potential::readPotential(coefFile);
    std::remove(coefFile.c_str());
    return newpot;
}

/// create a triaxial Hernquist model (could use galaxymodel::sampleNbody in a general case)
particles::PointMassArrayCar make_hernquist(int nbody, double q, double p)
{
    particles::PointMassArrayCar pts;
    for(int i=0; i<nbody; i++) {
        double m = math::random();
        double r = 1/(1/sqrt(m)-1);
        double costheta = math::random()*2 - 1;
        double sintheta = sqrt(1-pow_2(costheta));
        double phi = math::random()*2*M_PI;
        pts.add(coord::PosVelCar(
            r*sintheta*cos(phi), r*sintheta*sin(phi)*q, r*costheta*p, 0, 0, 0), 1./nbody);
    }
    return pts;
}

potential::PtrPotential create_from_file(
    const particles::PointMassArrayCar& points, const std::string& potType)
{
    const std::string fileName = "test.txt";
    particles::writeSnapshot(fileName, units::ExternalUnits(), points, "Text");
    potential::PtrPotential newpot;

    // illustrates two possible ways of creating a potential from points
    if(potType == "BasisSetExp") {
        particles::PointMassArrayCar pts;
        particles::readSnapshot(fileName, units::ExternalUnits(), pts);
        newpot = potential::PtrPotential(new potential::BasisSetExp(
            1.0, /*alpha*/
            20,  /*numCoefsRadial*/
            4,   /*numCoefsAngular*/
            pts, /*points*/
            potential::ST_TRIAXIAL));  /*symmetry (default value)*/
    } else {
        // a rather lengthy way of setting parameters, used only for illustration:
        // normally these would be read from an INI file or from command line;
        // to create an instance of potential expansion of a known type, 
        // use directly its constructor as shown above
        utils::KeyValueMap params;
        params.set("file", fileName);
        params.set("type", potType);
        params.set("numCoefsRadial", 20);
        params.set("numCoefsAngular", 4);
        params.set("numCoefsVertical", 20);
        params.set("Density", "Nbody");
        newpot = potential::createPotential(params);
    }
    std::remove(fileName.c_str());
    std::remove((fileName+potential::getCoefFileExtension(potType)).c_str());
    return newpot;
}

void test_average_error(const potential::BasePotential& p1, const potential::BasePotential& p2)
{
    double gamma = getInnerDensitySlope(p2);
    std::string fileName = std::string("testerr_") + p1.name() + "_" + p2.name() + 
        "_gamma" + utils::convertToString(gamma);
    std::ofstream strm(fileName.c_str());
    const double dlogR=0.25;
    const int nptbin=100;
    for(double logR=-4; logR<4; logR+=dlogR) {
        math::Averager difPhi, difForce, difDens;
        for(int n=0; n<nptbin; n++) {
            coord::PosSph point( pow(10., logR+dlogR*n/nptbin),
                acos(math::random()*2-1), math::random()*2*M_PI);
            coord::GradCar g1, g2;
            double v1, v2, f1, f2, d1, d2;
            p1.eval(coord::toPosCar(point), &v1, &g1);
            p2.eval(coord::toPosCar(point), &v2, &g2);
            d1 = p1.density(point);
            d2 = p2.density(point);
            f1 = sqrt(pow_2(g1.dx)+pow_2(g1.dy)+pow_2(g1.dz));
            f2 = sqrt(pow_2(g2.dx)+pow_2(g2.dy)+pow_2(g2.dz));
            difPhi  .add((v1-v2)/(v1+v2)*2);
            difDens .add((d1-d2)/(fabs(d1)+fabs(d2))*2);
            difForce.add((f1-f2)/(f1+f2)*2);
        }
        strm << pow(10., logR+0.5*dlogR) << '\t' <<
            sqrt(pow_2(difPhi  .mean())+difPhi  .disp()) << '\t' <<
            sqrt(pow_2(difForce.mean())+difForce.disp()) << '\t' <<
            sqrt(pow_2(difDens .mean())+difDens .disp()) << '\n';
    }
}


/// compare potential and its derivatives between the original model and its spherical-harmonic approximation
bool test_suite(const potential::BasePotential& p, const potential::BasePotential& orig, double eps_pot)
{
    bool ok=true;
    potential::PtrPotential newpot = write_read(p);
    double gamma = getInnerDensitySlope(orig);
    std::cout << "\033[1;32m---- testing "<<p.name()<<" with "<<
        (isSpherical(orig) ? "spherical " : isAxisymmetric(orig) ? "axisymmetric" : "triaxial ") <<orig.name()<<
        " (gamma="<<gamma<<") ----\033[0m\n";
    const char* err = "\033[1;31m **\033[0m";
    std::string fileName = std::string("test_") + p.name() + "_" + orig.name() + 
        "_gamma" + utils::convertToString(gamma);
    if(output)
        writePotential(fileName + getCoefFileExtension(p), p);
    for(int ic=0; ic<numtestpoints; ic++) {
        double pot, pot_orig;
        coord::GradCyl der,  der_orig;
        coord::HessCyl der2, der2_orig;
        coord::PosSph point(pos_sph[ic][0], pos_sph[ic][1], pos_sph[ic][2]);
        newpot->eval(toPosCyl(point), &pot, &der, &der2);
        orig.eval(toPosCyl(point), &pot_orig, &der_orig, &der2_orig);
        double eps_der = eps_pot*100/point.r;
        double eps_der2= eps_der*10;
        bool pot_ok = (pot==pot) && fabs(pot-pot_orig)<eps_pot;
        bool der_ok = point.r==0 || equalGrad(der, der_orig, eps_der);
        bool der2_ok= point.r==0 || equalHess(der2, der2_orig, eps_der2);
        ok &= pot_ok && der_ok && der2_ok;
        std::cout << "Point:  " << point << 
            "Phi: " << pot << " (orig:" << pot_orig << (pot_ok?"":err) << ")\n"
            "Force sphharm: " << der  << "\nForce origin.: " << der_orig  << (der_ok ?"":err) << "\n"
            "Deriv sphharm: " << der2 << "\nDeriv origin.: " << der2_orig << (der2_ok?"":err) << "\n";
    }
    if(output)
        test_average_error(*newpot, orig);
    return ok;
}

void test_axi_dens()
{
    const potential::Dehnen deh_axi(1., 1., 1., 0.5, 1.2);
    const potential::DensitySphericalHarmonic deh_6(50, 6, deh_axi, 1e-2, 100);
    const potential::DensitySphericalHarmonic deh_10(50, 10, deh_axi, 1e-2, 100);
    const potential::DensitySphericalHarmonic deh_20(100, 20, deh_axi, 1e-2, 100);
    // creating a sph-harm expansion from another s-h expansion - should produce identical results
    // if the location of radial grid points is the same, and l_max is at least as large as the original one.
    const potential::DensitySphericalHarmonic deh_6a(99, 10, deh_6, 1e-2, 100);
    for(double r=0.125; r<=8; r*=4) {
        for(double theta=0; theta<M_PI/2; theta+=0.314) {
            coord::PosSph p(r, theta, 0);
            std::cout << r<< ' '<<theta<<" : " <<deh_axi.density(p) << " = " << 
                deh_6.density(p)<<", "<<deh_10.density(p)<<", "<<deh_20.density(p)<<", "<<deh_6a.density(p);
            if(theta==0) std::cout <<"  "<< deh_6.rho_l(r, 0)<<", "<<deh_10.rho_l(r, 0)<<", "<<
                deh_20.rho_l(r, 0)<<", "<< deh_6a.rho_l(r, 0);
            std::cout<<"\n";
        }
    }
}

// definition of a single spherical-harmonic term with indices (l,m)
template<int l, int m>
double myfnc(double theta, double phi);
// first few spherical harmonics (with arbitrary normalization)
template<> double myfnc<0, 0>(double      , double    ) { return 1; }
template<> double myfnc<1,-1>(double theta, double phi) { return sin(theta)*sin(phi); }
template<> double myfnc<1, 0>(double theta, double    ) { return cos(theta); }
template<> double myfnc<1, 1>(double theta, double phi) { return sin(theta)*cos(phi); }
template<> double myfnc<2,-2>(double theta, double phi) { return (1-cos(2*theta))*sin(2*phi); }
template<> double myfnc<2,-1>(double theta, double phi) { return sin(theta)*cos(theta)*sin(phi); }
template<> double myfnc<2, 0>(double theta, double    ) { return 3*cos(2*theta)+1; }
template<> double myfnc<2, 1>(double theta, double phi) { return sin(theta)*cos(theta)*cos(phi); }
template<> double myfnc<2, 2>(double theta, double phi) { return (1-cos(2*theta))*cos(2*phi); }

template<int l, int m>
bool checkSH(const math::SphHarmIndices& ind)
{
    math::SphHarmTransformForward tr(ind);
    // array of original function values
    std::vector<double> d(tr.size());
    for(int j=0; j<=ind.lmax; j+=ind.lstep)
        for(int k=ind.mmin; k<=ind.mmax; k++)
            d.at(tr.index(j, k)) = myfnc<l,m>(acos(tr.costheta(j)), tr.phi(k));
    // array of SH coefficients
    std::vector<double> c(ind.size());
    tr.transform(&d.front(), &c.front());
    // check that only one of them is non-zero
    unsigned int t0 = ind.index(l, m);  // index of the only non-zero coef
    for(unsigned int t=0; t<c.size(); t++)
        if((t==t0) ^ (c[t]!=0))  // xor operation
            return false;
    // array of function values after inverse transform
    std::vector<double> b(tr.size());
    for(int j=0; j<=ind.lmax; j+=ind.lstep)
        for(int k=ind.mmin; k<=ind.mmax; k++) {
            unsigned int index = tr.index(j, k);
            b.at(index) = math::sphHarmTransformInverse(ind, &c.front(),
                acos(tr.costheta(j)), tr.phi(k));
            if(fabs(d[index]-b[index])>1e-14)
                return false;
        }
    return true;
}

int main() {
    bool ok=true;

    // perform several tests, some of them are expected to fail - because... (see comments below)
    ok &= checkSH<0, 0>(math::SphHarmIndices(4, 2, 0, 2, 2));
    ok &= checkSH<1,-1>(math::SphHarmIndices(4, 1,-4, 4, 1));
    ok &= checkSH<1, 0>(math::SphHarmIndices(2, 1, 0, 0, 0));
    ok &= checkSH<1, 1>(math::SphHarmIndices(6, 1, 0, 3, 1));
    ok &= checkSH<1, 1>(math::SphHarmIndices(3, 1,-2, 2, 1));
    ok &=!checkSH<1, 1>(math::SphHarmIndices(6, 2, 0, 2, 1)); // lstep==2 but we have odd-l term
    ok &=!checkSH<1, 1>(math::SphHarmIndices(6, 1, 0, 2, 2)); // mstep==2 but we have odd-m term
    ok &= checkSH<1, 1>(math::SphHarmIndices(6, 1, 0, 2, 1));
    ok &=!checkSH<2,-2>(math::SphHarmIndices(6, 2, 0, 4, 1)); // mmin==0 but we have sine term
    ok &= checkSH<2,-2>(math::SphHarmIndices(6, 2,-4, 4, 1));
    ok &=!checkSH<2,-1>(math::SphHarmIndices(2, 2,-2, 2, 1)); // lstep==2 but we have odd-m term
    ok &= checkSH<2,-1>(math::SphHarmIndices(2, 1,-2, 2, 1));
    ok &=!checkSH<2,-1>(math::SphHarmIndices(6, 1,-4, 4, 2)); // mstep==2 but we have odd-m term
    ok &= checkSH<2,-1>(math::SphHarmIndices(6, 1,-4, 4, 1));
    ok &= checkSH<2, 0>(math::SphHarmIndices(2, 2, 0, 1, 1));
    ok &= checkSH<2, 1>(math::SphHarmIndices(4, 1,-2, 2, 1));
    ok &= checkSH<2, 2>(math::SphHarmIndices(3, 1, 0, 2, 2));

    // spherical, cored
    const potential::Plummer plum(10., 5.);
    const potential::BasisSetExp bs1(0., 30, 2, plum);
    const potential::SplineExp sp1(20, 2, plum);
    // this forces potential to be computed via integration of density over volume
    const potential::CylSplineExp cy1(20, 20, 0, static_cast<const potential::BaseDensity&>(plum));
    ok &= test_suite(bs1, plum, 1e-5);
    ok &= test_suite(sp1, plum, 1e-5);
    ok &= test_suite(cy1, plum, 1e-4);

    // mildly triaxial, cuspy
    const potential::Dehnen deh15(3., 1.2, 0.8, 0.6, 1.5);
    const potential::BasisSetExp bs2(2., 20, 6, deh15);
    const potential::SplineExp sp2(20, 6, deh15);
    ok &= test_suite(bs2, deh15, 2e-4);
    ok &= test_suite(sp2, deh15, 2e-4);

    // mildly triaxial, cored
    const potential::Dehnen deh0(1., 1., 0.8, 0.6, 0.);
    const potential::BasisSetExp bs3(1., 20, 6, deh0);
    const potential::SplineExp sp3(20, 6, deh0);
    const potential::CylSplineExp cy3(20, 20, 6, static_cast<const potential::BaseDensity&>(deh0));
    ok &= test_suite(bs3, deh0, 5e-5);
    ok &= test_suite(sp3, deh0, 5e-5);
    ok &= test_suite(cy3, deh0, 1e-4);

    // mildly triaxial, created from N-body samples
    const potential::Dehnen hernq(1., 1., 0.8, 0.6, 1.0);
    particles::PointMassArrayCar points = make_hernquist(100000, 0.8, 0.6);
    ok &= test_suite(*create_from_file(points, potential::BasisSetExp::myName()), hernq, 2e-2);
    // could also use create_from_file(points, "SplineExp");  below
    potential::SplineExp sp4(20, 4, points, potential::ST_TRIAXIAL);
    ok &= test_suite(sp4, hernq, 2e-2);
    ok &= test_suite(*create_from_file(points, potential::CylSplineExp::myName()), hernq, 2e-2);

#if 1
    //test_axi_dens();

    // axisymmetric multipole
    const potential::Dehnen deh1(1., 1., 1., .5, 1.2);
    const potential::Multipole deh1m(deh1, 1e-3, 1e3, 100, 16);
    test_average_error(deh1m, deh1);
    const potential::SplineExp deh1s(50, 8, deh1, 1e-3, 1e3);
    test_average_error(deh1s, deh1);
    std::vector<double> radii;
    std::vector<std::vector<double> > Phi, dPhi;
    potential::PtrPotential deh1m_clone = write_read(deh1m);
    test_average_error(deh1m, *deh1m_clone);
#endif

    if(ok)
        std::cout << "ALL TESTS PASSED\n";
    return 0;
}