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
    writePotentialCoefs(coefFile, pot);
    potential::PtrPotential newpot = potential::readPotentialCoefs(coefFile);
    std::remove(coefFile.c_str());
    return newpot;
}

/// create a triaxial Hernquist model
void make_hernquist(int nbody, double q, double p, particles::PointMassArrayCar& out)
{
    out.data.clear();
    for(int i=0; i<nbody; i++) {
        double m = math::random()*1.0/RAND_MAX;
        double r = 1/(1/sqrt(m)-1);
        double costheta = math::random()*2.0/RAND_MAX - 1;
        double sintheta = sqrt(1-pow_2(costheta));
        double phi = math::random()*2*M_PI/RAND_MAX;
        out.add(coord::PosVelCar(r*sintheta*cos(phi), r*sintheta*sin(phi)*q, r*costheta*p, 0, 0, 0), 1./nbody);
    }
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

particles::PointMassArrayCar points;  // sampling points

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
            difForce.add((f1-f2)/(f1+f2)*2);
            difDens .add((d1-d2)/(fabs(d1)+d2)*2);
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
        writePotentialCoefs(fileName + getCoefFileExtension(p), p);
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
    if(output) {
#if 0
        std::ofstream strmSample((fileName+".samples").c_str());
        for(unsigned int ic=0; ic<points.size(); ic++) {
            double pot, pot_orig;
            coord::GradCyl der,  der_orig;
            coord::HessCyl der2, der2_orig;
            newpot->eval(toPosCyl(points.point(ic)), &pot, &der, &der2);
            orig.eval(toPosCyl(points.point(ic)), &pot_orig, &der_orig, &der2_orig);
            strmSample << toPosSph(points.point(ic)).r << "\t" <<
            fabs((pot-pot_orig)/pot_orig) << "\t" <<
            fabs((der.dR-der_orig.dR)/der_orig.dR) << "\t" <<
            fabs((der.dz-der_orig.dz)/der_orig.dz) << "\t" <<
            fabs(1-newpot->density(points.point(ic))/orig.density(points.point(ic))) << "\n";
        }
#else
        test_average_error(*newpot, orig);
#endif
    }
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

int main() {
    bool ok=true;

#if __GNUC__ and (__GNUC__ <= 4) and (__GNUC_MINOR__ <= 2)
    std::cout << "Cannot compile this test:"
        " the current version of GNU compiler "<<__GNUC__<<'.'<<__GNUC_MINOR__<<
        " does not allow temporary objects without copy constructors to be passed as references\n";
#else
    make_hernquist(100000, 0.8, 0.6, points);

    //test_axi_dens();
    // spherical, cored
    const potential::Plummer plum(10., 5.);
    ok &= test_suite(potential::BasisSetExp(0., 30, 2, plum), plum, 1e-5);
    ok &= test_suite(potential::SplineExp(20, 2, plum), plum, 1e-5);
    // this forces potential to be computed via integration of density over volume
    ok &= test_suite(potential::CylSplineExp(20, 20, 0, static_cast<const potential::BaseDensity&>(plum)), plum, 1e-4);

    // mildly triaxial, cuspy
    const potential::Dehnen deh15(3., 1.2, 0.8, 0.6, 1.5);
    ok &= test_suite(potential::BasisSetExp(2., 20, 6, deh15), deh15, 2e-4);
    ok &= test_suite(potential::SplineExp(20, 6, deh15), deh15, 2e-4);

    // mildly triaxial, cored
    const potential::Dehnen deh0(1., 1., 0.8, 0.6, 0.);
    ok &= test_suite(potential::BasisSetExp(1., 20, 6, deh0), deh0, 5e-5);
    ok &= test_suite(potential::SplineExp(20, 6, deh0), deh0, 5e-5);
    ok &= test_suite(potential::CylSplineExp(20, 20, 6, static_cast<const potential::BaseDensity&>(deh0)), deh0, 1e-4);

    // mildly triaxial, created from N-body samples
    const potential::Dehnen hernq(1., 1., 0.8, 0.6, 1.0);
    ok &= test_suite(*create_from_file(points, potential::BasisSetExp::myName()), hernq, 2e-2);
    // could also use create_from_file(points, "SplineExp");  below
    ok &= test_suite(potential::SplineExp(20, 4, points, potential::ST_TRIAXIAL), hernq, 2e-2);
    ok &= test_suite(*create_from_file(points, potential::CylSplineExp::myName()), hernq, 2e-2);

    // axisymmetric multipole
    const potential::Dehnen deh1(1., 1., 1., .5, 1.2);
    test_average_error(potential::Multipole(deh1, 1e-3, 1e3, 100, 16), deh1);
    test_average_error(potential::SplineExp(50, 8, deh1, 1e-3, 1e3), deh1);
#endif

    if(ok)
        std::cout << "ALL TESTS PASSED\n";
    return 0;
}