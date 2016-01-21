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
#include <ctime>
#include "math_specfunc.h"

/// define test suite in terms of points in spherical coordinates
const int numtestpoints=8;
const double pos_sph[numtestpoints][3] = {   // order: r, theta, phi
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
        double r = 1/(1/sqrt(m)-1);  // known inversion of M(r)
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
            coord::ST_TRIAXIAL));  /*symmetry (default value)*/
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

// test the accuracy of potential, force and density approximation at different radii
void test_average_error(const potential::BasePotential& p1, const potential::BasePotential& p2)
{
    double gamma = getInnerDensitySlope(p2);
    std::string fileName = std::string("testerr_") + p1.name() + "_" + p2.name() + 
        "_gamma" + utils::convertToString(gamma);
    std::ofstream strm(fileName.c_str());
    const double dlogR=0.1;
    const int nptbin=500;
    for(double logR=-4; logR<4; logR+=dlogR) {
        math::Averager difPhi, difForce, difDens;
        for(int n=0; n<nptbin; n++) {
            coord::PosSph point( pow(10., logR+dlogR*n/nptbin),
                acos(math::random()*2-1), math::random()*2*M_PI);
            coord::GradCar g1, g2;
            double v1, v2, df, f2, d1, d2;
            p1.eval(coord::toPosCar(point), &v1, &g1);
            p2.eval(coord::toPosCar(point), &v2, &g2);
            d1 = p1.density(point);
            d2 = p2.density(point);
            df = sqrt(pow_2(g1.dx-g2.dx)+pow_2(g1.dy-g2.dy)+pow_2(g1.dz-g2.dz));
            f2 = sqrt(pow_2(g2.dx)+pow_2(g2.dy)+pow_2(g2.dz));
            difPhi  .add((v1-v2)/(v1+v2));
            difDens .add((d1-d2)/(fabs(d1)+fabs(d2)));
            difForce.add(df/f2);
        }
        strm << pow(10., logR+0.5*dlogR) << '\t' <<
            sqrt(pow_2(difPhi  .mean())+difPhi  .disp()) << '\t' <<
            sqrt(pow_2(difForce.mean())+difForce.disp()) << '\t' <<
            sqrt(pow_2(difDens .mean())+difDens .disp()) << '\n';
    }
}

// test the accuracy of density approximation at different radii
void test_average_error(const potential::BaseDensity& p1, const potential::BaseDensity& p2)
{
    double gamma = getInnerDensitySlope(p2);
    std::string fileName = std::string("testerr_") + p1.name() + "_" + p2.name() + 
        "_gamma" + utils::convertToString(gamma);
    std::ofstream strm(fileName.c_str());
    const double dlogR=0.1;
    const int nptbin=500;
    for(double logR=-4; logR<4; logR+=dlogR) {
        math::Averager avg;
        for(int n=0; n<nptbin; n++) {
            coord::PosSph point( pow(10., logR+dlogR*n/nptbin),
                acos(math::random()*2-1), math::random()*2*M_PI);
            double d1 = p1.density(point);
            double d2 = p2.density(point);
            avg.add((d1-d2)/(fabs(d1)+fabs(d2)));
        }
        strm << pow(10., logR+0.5*dlogR) << '\t' <<
            sqrt(pow_2(avg.mean())+avg.disp()) << '\n';
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

bool testDensSH()
{
    const potential::Dehnen dens(1., 1., 0.8, 0.5, 1.2);
    std::vector<double> radii1 = math::createExpGrid(51, 0.01, 100);
    // twice denser grid: every other node coincides with that of the first grid
    std::vector<double> radii2 = math::createExpGrid(101,0.01, 100);
    std::vector<std::vector<double> > coefs1, coefs2;
    computeDensityCoefsSph(dens, math::SphHarmIndices(8, 6, dens.symmetry()), radii1, coefs1);
    potential::DensitySphericalHarmonic dens1(radii1, coefs1);
    // creating a sph-harm expansion from another s-h expansion:
    // should produce identical results if the location of radial grid points 
    // (partly) coincides with the first one, and the order of expansions lmax,mmax
    // are at least as large as the original ones (the extra coefs will be zero).
    computeDensityCoefsSph(dens1, math::SphHarmIndices(10, 8, dens1.symmetry()), radii2, coefs2);
    potential::DensitySphericalHarmonic dens2(radii2, coefs2);
    bool ok = true;
    // check that the two sets of coefs are identical at equal radii
    for(unsigned int c=0; c<coefs2[0].size(); c++)
        for(unsigned int k=0; k<radii1.size(); k++)
            if( (c<coefs1[0].size() && fabs(coefs1[k][c] - coefs2[k*2][c]) > 1e-12) ||
                (c>=coefs1[0].size() && coefs2[k*2][c] != 0) )
                ok = false;

    // test the accuracy at grid radii
    for(unsigned int k=0; k<radii1.size(); k++)
        for(double theta=0; theta<M_PI/2; theta+=0.31) 
            for(double phi=0; phi<M_PI; phi+=0.31) {
                coord::PosSph point(radii1[k], theta, phi);
                double d1 = dens1.density(point);
                double d2 = dens2.density(point);
                if(fabs(d1-d2) / fabs(d1) > 1e-12)  // two SH expansions should give the same result
                    ok = false;
            }
    test_average_error(dens1, dens);
    test_average_error(dens1, dens2);
    return ok;
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

// test spherical-harmonic transformation with the given set of indices and a given SH term (l,m)
template<int l, int m>
bool checkSH(const math::SphHarmIndices& ind)
{
    math::SphHarmTransformForward tr(ind);
    // array of original function values
    std::vector<double> d(tr.size());
    for(unsigned int i=0; i<d.size(); i++)
        d[i] = myfnc<l,m>(tr.theta(i), tr.phi(i));
    // array of SH coefficients
    std::vector<double> c(ind.size());
    tr.transform(&d.front(), &c.front());
    math::eliminateNearZeros(c);
    // check that only one of them is non-zero
    unsigned int t0 = ind.index(l, m);  // index of the only non-zero coef
    for(unsigned int t=0; t<c.size(); t++)
        if((t==t0) ^ (c[t]!=0))  // xor operation
            return false;
    double tmp[100];
    // array of function values after inverse transform
    std::vector<double> b(tr.size());
    for(unsigned int i=0; i<d.size(); i++) {
        b[i] = math::sphHarmTransformInverse(ind, &c.front(), tr.theta(i), tr.phi(i), tmp);
        if(fabs(d[i]-b[i]) > 1e-14)
            return false;
    }
    return true;
}

int main() {
#if 0
    for(unsigned int m=0; m<=12; m++) {
    double dd=math::powInt(0.5,14);
    double sum=0, sumd, max=0, maxd=0;
    std::ofstream ff(("appr"+utils::convertToString(m)).c_str());
    ff << std::setprecision(15);
    for(double x=dd; x<1; x+=dd) {
        double der1, der2, val2;
        double val1 = math::legendreQ(m-0.5, 1/sqrt(x), &der1);
        try{ val2 = math::legendreQ(m-0.500000000000001, 1/sqrt(x), &der2); }
        catch(...) { val2=val1; der2=der1; }
        if(fabs(der2)>1e8) { val2=val1; der2=der1; } // overflow
        ff << x << ", "<<val1 << ", " << (val2-val1) << "; " << der1 << ", " << (der2-der1) << "\n";
        sum += pow_2(val2-val1)/pow_2(val1);
        sumd+= pow_2(der2-der1)/pow_2(der1);
        max = fmax(max, fabs((val2-val1)/val1));
        maxd= fmax(maxd,fabs((der2-der1)/der1));
    }
    ff.close();
    std::cout << "m="<<m<<": rel.error in fnc="<<sqrt(sum*dd)<<" (max="<<max<<
    "), in deriv="<<sqrt(sumd*dd)<<" (max="<<maxd<<")\n";
    }
#endif
    bool ok=true;

    // perform several tests, some of them are expected to fail - because... (see comments below)
    ok &= checkSH<0, 0>(math::SphHarmIndices(4, 2, coord::ST_TRIAXIAL));
    ok &= checkSH<1,-1>(math::SphHarmIndices(4, 4, coord::ST_XREFLECTION));
        // axisymmetric implies z-reflection symmetry, but we have l=1 term
    ok &=!checkSH<1, 0>(math::SphHarmIndices(2, 0, coord::ST_AXISYMMETRIC));
        // while in this case it's only z-rotation symmetric, so a correct variant is below
    ok &= checkSH<1, 0>(math::SphHarmIndices(2, 0, coord::ST_ZROTATION));
    ok &= checkSH<1, 1>(math::SphHarmIndices(6, 3, coord::ST_YREFLECTION));
    ok &= checkSH<1, 1>(math::SphHarmIndices(3, 2, coord::ST_ZREFLECTION));
        // odd-m cosine term is not possible with x-reflection symmetry
    ok &=!checkSH<1, 1>(math::SphHarmIndices(6, 1, coord::ST_XREFLECTION));
        // odd-l term is not possible with mirror symmetry
    ok &=!checkSH<1, 1>(math::SphHarmIndices(6, 2, coord::ST_REFLECTION));
        // y-reflection implies no sine terms (mmin==0), but we have them
    ok &=!checkSH<2,-2>(math::SphHarmIndices(6, 4, coord::ST_YREFLECTION));
        // x-reflection excludes even-m sine terms
    ok &=!checkSH<2,-2>(math::SphHarmIndices(5, 2, coord::ST_XREFLECTION));
    ok &= checkSH<2,-2>(math::SphHarmIndices(6, 4, coord::ST_ZREFLECTION));
        // terms with odd l+m are excluded under z-reflection 
    ok &=!checkSH<2,-1>(math::SphHarmIndices(2, 2, coord::ST_ZREFLECTION));
    ok &= checkSH<2,-1>(math::SphHarmIndices(2, 1,
        static_cast<coord::SymmetryType>(coord::ST_REFLECTION | coord::ST_XREFLECTION)));
    ok &= checkSH<2, 0>(math::SphHarmIndices(2, 2, coord::ST_AXISYMMETRIC));
    ok &= checkSH<2, 1>(math::SphHarmIndices(3, 1,
        static_cast<coord::SymmetryType>(coord::ST_REFLECTION | coord::ST_YREFLECTION)));
    ok &= checkSH<2, 2>(math::SphHarmIndices(5, 3, coord::ST_TRIAXIAL));

    ok &= testDensSH();
    const potential::MiyamotoNagai mn(1, 3, 0.5);
    potential::PtrDensity mna = potential::DensityAzimuthalHarmonic::create(
        mn, 0, 30, 1e-2, 1e2, 30, 1e-2, 1e2);
    test_average_error(*mna, mn);
#if 0

    // axisymmetric multipole
    const potential::Dehnen deh1(1., 1., 0.8, .5, 1.2);
    potential::PtrPotential deh1m = potential::Multipole::create(
        static_cast<const potential::BaseDensity&>(deh1), 10, 10, 50, 1e-3, 1e3);
    std::cout << "Created Multipole\n";
    clock_t clockbegin = std::clock();
    test_average_error(*deh1m, deh1);
    std::cout << (std::clock()-clockbegin)*1.0/CLOCKS_PER_SEC << " seconds to test Multipole\n";
    const potential::SplineExp deh1s(50, 10, deh1, 1e-3, 1e3);
    std::cout << "Created Spline\n";
    clockbegin = std::clock();
    test_average_error(deh1s, deh1);
    std::cout << (std::clock()-clockbegin)*1.0/CLOCKS_PER_SEC << " seconds to test Spline\n";
    potential::PtrPotential deh1m_clone = write_read(*deh1m);
    test_average_error(*deh1m_clone, *deh1m);

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
#endif
    const potential::Dehnen deh0(1., 1., 0.8, 0.6, 0.);
    //const potential::MiyamotoNagai deh0(1, 2, 0.5);
    potential::PtrPotential deh0m = potential::Multipole::create(
        deh0, 12, 8, 30, 1e-3, 1e3);
    clock_t clockbegin = std::clock();
    //test_average_error(*deh0m, deh0);
    std::cout << (std::clock()-clockbegin)*1.0/CLOCKS_PER_SEC << " seconds for Multipole\n";
    //const potential::BasisSetExp bs3(1., 20, 6, deh0);
    //const potential::SplineExp sp3(20, 6, deh0);
    //const potential::CylSplineExp cy3(20, 20, 6, static_cast<const potential::BaseDensity&>(deh0));
    //std::cout << (std::clock()-clockbegin)*1.0/CLOCKS_PER_SEC << " seconds to create old CylSpline\n";
    const potential::CylSplineExpOld cy3(30, 30, 8, static_cast<const potential::BaseDensity&>(deh0), 1e-3, 1e3, 1e-3, 1e3);
    clockbegin = std::clock();
    test_average_error(cy3, deh0);
    std::cout << (std::clock()-clockbegin)*1.0/CLOCKS_PER_SEC << " seconds for old spline\n";
    //ok &= test_suite(bs3, deh0, 5e-5);
    //ok &= test_suite(sp3, deh0, 5e-5);
    //ok &= test_suite(cy3, deh0, 1e-4);
    writePotential("CylSplineOld", cy3);
    //clockbegin = std::clock();
    std::vector<double> gridR = math::createNonuniformGrid(30, 1e-3, 1e3, true);
    std::vector<double> gridz = math::createNonuniformGrid(30, 1e-3, 1e3, true);
    std::vector<math::Matrix<double> > Phi, dPhidR, dPhidz;
    computePotentialCoefsCyl(static_cast<const potential::BaseDensity&>(deh0), 8, gridR, gridz, Phi, dPhidR, dPhidz);
    //computePotentialCoefsCyl(deh0, 8, gridR, gridz, Phi, dPhidR, dPhidz);
    //std::cout << (std::clock()-clockbegin)*1.0/CLOCKS_PER_SEC << " seconds to create new CylSpline\n";
    const potential::CylSplineExp cy5(gridR, gridz, Phi, dPhidR, dPhidz);
    clockbegin = std::clock();
    test_average_error(cy5, deh0);
    std::cout << (std::clock()-clockbegin)*1.0/CLOCKS_PER_SEC << " seconds for new spline\n";
    //ok &= test_suite(cy4, deh0, 1e-4);
    writePotential("CylSplineNew", cy5);
    potential::PtrPotential cy5_clone = write_read(cy5);
    //test_average_error(*cy5_clone, cy5);

    // mildly triaxial, created from N-body samples
    const potential::Dehnen hernq(1., 1., 0.8, 0.6, 1.0);
    particles::PointMassArrayCar points = make_hernquist(100000, 0.8, 0.6);
    ok &= test_suite(*create_from_file(points, potential::BasisSetExp::myName()), hernq, 2e-2);
    // could also use create_from_file(points, "SplineExp");  below
    potential::SplineExp sp4(20, 4, points, coord::ST_TRIAXIAL);
    ok &= test_suite(sp4, hernq, 2e-2);
    ok &= test_suite(*create_from_file(points, potential::CylSplineExp::myName()), hernq, 2e-2);

    if(ok)
        std::cout << "ALL TESTS PASSED\n";
    return 0;
}