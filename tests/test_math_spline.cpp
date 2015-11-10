//#define COMPARE_WD_PSPLINE
//#define STRESS_TEST

#include "math_spline.h"
#include "math_core.h"
#include "math_specfunc.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>
#ifdef COMPARE_WD_PSPLINE
#include "torus/WD_Pspline.h"
#endif

const int NNODES  = 20;
const int NSUBINT = 16;
const int NPOINTS = 10000;
const double XMIN = 0.2;
const double XMAX = 12.;
const double DISP = 0.5;  // y-dispersion
const bool OUTPUT = false;

// provides the integral of sin(x)*x^n
class testfnc: public math::IFunctionIntegral {
    virtual double integrate(double x1, double x2, int n=0) const {
        return antideriv(x2,n)-antideriv(x1,n);
    }
    double antideriv(double x, int n) const {
        switch(n) {
            case 0: return -cos(x);
            case 1: return -cos(x)*x+sin(x);
            case 2: return  cos(x)*(2-x*x) + 2*x*sin(x);
            case 3: return  cos(x)*x*(6-x*x) + 3*sin(x)*(x*x-2);
            default: return NAN;
        }
    }
};
// provides the integrand for numerical integration of sin(x)*f(x)
class testfncint: public math::IFunctionNoDeriv {
public:
    testfncint(const math::IFunction& _f): f(_f) {};
    virtual double value(const double x) const {
        return sin(x) * f(x);
    }
private:
    const math::IFunction& f;
};
// provides the integrand for numerical integration of f(x)^2
class squaredfnc: public math::IFunctionNoDeriv {
public:
    squaredfnc(const math::IFunction& _f): f(_f) {};
    virtual double value(const double x) const {
        return pow_2(f(x));
    }
private:
    const math::IFunction& f;
};

bool test_integral(const math::CubicSpline& f, double x1, double x2)
{
    double result_int = f.integrate(x1, x2);
    double result_ext = math::integrateAdaptive(f, x1, x2, 1e-10);
    std::cout << "Ordinary intergral on ["<<x1<<":"<<x2<<
        "]: internal routine = "<<result_int<<", adaptive integration = "<<result_ext<<"\n";
    if(fabs(result_int-result_ext)>1e-10) return false;
    result_int = f.integrate(x1, x2, testfnc());
    result_ext = math::integrateAdaptive(testfncint(f), x1, x2, 1e-10);
    std::cout << "Weighted intergral on ["<<x1<<":"<<x2<<
        "]: internal routine = "<<result_int<<", adaptive integration = "<<result_ext<<"\n";
    if(fabs(result_int-result_ext)>1e-10) return false;
    result_int = f.integrate(x1, x2, f);
    result_ext = math::integrateAdaptive(squaredfnc(f), x1, x2, 1e-10);
    std::cout << "Integral of f(x)^2 on ["<<x1<<":"<<x2<<
        "]: internal routine = "<<result_int<<", adaptive integration = "<<result_ext<<"\n";
    if(fabs(result_int-result_ext)>1e-10) return false;
    return true;
}

int main()
{
    //----------- test penalized smoothing spline fit to noisy data -------------//

    std::cout << std::setprecision(12);
    bool ok=true;
    std::vector<double> xnodes(NNODES);
    math::createNonuniformGrid(NNODES, XMIN, XMAX, true, xnodes);
    std::vector<double> xvalues(NPOINTS), yvalues1(NPOINTS), yvalues2(NPOINTS);
    for(int i=0; i<NPOINTS; i++) {
        xvalues [i] = rand()*XMAX/RAND_MAX;
        yvalues1[i] = sin(4*sqrt(xvalues[i])) + DISP*(rand()*1./RAND_MAX-0.5);
        yvalues2[i] = cos(4*sqrt(xvalues[i])) + DISP*(rand()*1./RAND_MAX-0.5)*4;
    }
    math::SplineApprox appr(xvalues, xnodes);
    if(appr.isSingular())
        std::cout << "Warning, matrix is singular\n";

    std::vector<double> ynodes1, ynodes2;
    double deriv_left, deriv_right, rms, edf, lambda;

    appr.fitDataOptimal(yvalues1, ynodes1, deriv_left, deriv_right, &rms, &edf, &lambda);
    std::cout << "case A: RMS="<<rms<<", EDF="<<edf<<", lambda="<<lambda<<"\n";
    math::CubicSpline fit1(xnodes, ynodes1, deriv_left, deriv_right);
    ok &= rms<0.1 && edf>=2 && edf<NNODES+2 && lambda>0;

    appr.fitDataOversmooth(yvalues2, .5, ynodes2, deriv_left, deriv_right, &rms, &edf, &lambda);
    std::cout << "case B: RMS="<<rms<<", EDF="<<edf<<", lambda="<<lambda<<"\n";
    math::CubicSpline fit2(xnodes, ynodes2, deriv_left, deriv_right);
    ok &= rms<1.0 && edf>=2 && edf<NNODES+2 && lambda>0;

    if(OUTPUT) {
        std::ofstream strm("test_math_spline_fit.dat");
        for(size_t i=0; i<xnodes.size(); i++)
            strm << xnodes[i] << "\t" << ynodes1[i] << "\t" << ynodes2[i] << "\n";
        strm << "\n";
        for(size_t i=0; i<xvalues.size(); i++)
            strm << xvalues[i] << "\t" << yvalues1[i] << "\t" << yvalues2[i] << "\t" <<
                fit1(xvalues[i]) << "\t" << fit2(xvalues[i]) << "\n";
    }

    //-------- test cubic and quintic splines ---------//
    // accuracy of approximation of an oscillating fnc //

    math::createNonuniformGrid(NNODES, XMIN, XMAX, false, xnodes);
    xnodes[1]=(xnodes[1]+xnodes[2])/2;  // slightly squeeze grid spacing to allow
    xnodes[0]*=2;                       // a better interpolation of a strongly varying function
    std::vector<double> yvalues(NNODES), yderivs(NNODES);
    for(int i=0; i<NNODES; i++) {
        yvalues[i] = sin(4*sqrt(xnodes[i]));
        yderivs[i] = cos(4*sqrt(xnodes[i])) * 2 / sqrt(xnodes[i]);
    }
    math::CubicSpline   fcubna(xnodes, yvalues);  // cubic spline with natural boundary conditions
    math::CubicSpline   fcubcl(xnodes, yvalues, 
        yderivs.front(), yderivs.back());         // cubic, clamped -- specify derivs at the boundaries
    math::QuinticSpline fquint(xnodes, yvalues, yderivs);  // quintic spline -- specify derivs at all nodes
    std::ofstream strm;
    if(OUTPUT)
        strm.open("test_math_spline1d.dat");
    double sumerr3n = 0, sumerr3 = 0, sumerr5 = 0;
    for(int i=0; i<=(NNODES-1)*NSUBINT; i++) {
        double xa = xnodes[i/NSUBINT];
        double xb = i<(NNODES-1)*NSUBINT ? xnodes[i/NSUBINT+1] : xa;
        double x  = xa*(1 - (i%NSUBINT)*1.0/NSUBINT) + xb*(i%NSUBINT)/NSUBINT;
        double y0 = sin(4*sqrt(x));
        double y0p  = cos(4*sqrt(x)) * 2 / sqrt(x);
        double y0pp = -(4*y0 + 0.5*y0p) / x;
        double y0ppp = (6*y0 + (0.75-4*x)*y0p) / pow_2(x);
        double y3, y3p, y3pp, y5, y5p, y5pp, y5ppp=fquint.deriv3(x);
        double y3n  = fcubna(x);
        fcubcl.evalDeriv(x, &y3, &y3p, &y3pp);
        fquint.evalDeriv(x, &y5, &y5p, &y5pp);
        sumerr3n += pow_2(y0-y3n);
        sumerr3  += pow_2(y0-y3);
        sumerr5  += pow_2(y0-y5);
        if(OUTPUT)
            strm << x << '\t' << y0 << '\t' << y3n << '\t' << y3 << '\t' << y5 << '\t' <<
            y0p << '\t' << y3p << '\t' << y5p << '\t' << 
            y0pp << '\t' << y3pp << '\t' << y5pp << '\t' << y0ppp << '\t' << y5ppp << "\n";
    }
    sumerr3n = (sqrt(sumerr3n / ((NNODES-1)*NSUBINT)));
    sumerr3  = (sqrt(sumerr3  / ((NNODES-1)*NSUBINT)));
    sumerr5  = (sqrt(sumerr5  / ((NNODES-1)*NSUBINT)));
    std::cout << "RMS error in cubic spline: " << sumerr3n <<
        ", in clamped cubic spline: " << sumerr3 <<
        ", in quintic spline: " << sumerr5 << "\n";
    ok &= sumerr3n<5e-3 && sumerr3<3e-4 && sumerr5 < 4e-5;
    if(OUTPUT)
        strm.close();

    // test the integration function //
    ok &= test_integral(fcubcl, (xnodes[0]+xnodes[1])/2, xnodes.back());
    ok &= test_integral(fcubna, -1.234567, xnodes.back()+1.);

    //-------- another test for cubic and quintic splines ---------//
    //      accuracy of approximation of Legendre polynomials      //

    for(int lmax=4; lmax<=32; lmax*=2) {   // vary the order of polynomials
        for(int np=lmax*2-2; np<=lmax*4-2; np+=lmax) {   // vary the number of grid points on the interval -1<=x<=1
            std::vector<double> legPl(lmax+1), legdPl(lmax+1);
            std::vector<double> x(np+1);
            std::vector<std::vector<double> > Pl(lmax+1), dPl(lmax+1);
            for(int l=0; l<=lmax; l++) {
                Pl [l].resize(x.size());
                dPl[l].resize(x.size());
            }
            // assign x nodes and store values of Pl(x_i) and dPl/dx
            for(int p=0; p<=np; p++) {
                x[p] = p==0 ? -1 : p==np ? 1 : p==np/2 ? 0 : -cos(M_PI*(1.*p/np));
                math::legendrePolyArray(lmax, 0, x[p], &legPl.front(), &legdPl.front());
                for(int l=0; l<=lmax; l++) {
                    Pl [l][p] = legPl[l];
                    dPl[l][p] = legdPl[l];
                }
            }
            // init cubic and quintic splines
            std::vector<math::CubicSpline>   spl3(lmax+1);
            std::vector<math::QuinticSpline> spl5(lmax+1);
            for(int l=0; l<=lmax; l++) {
                spl3[l] = math::CubicSpline  (x, Pl[l], dPl[l].front(), dPl[l].back());
                spl5[l] = math::QuinticSpline(x, Pl[l], dPl[l]);
            }
            // compute the rms error for each l
            std::vector<double> err3(lmax+1), err5(lmax+1), err3der(lmax+1), err5der(lmax+1);
            const int NPTCHECK=1000;
            for(int p=0; p<NPTCHECK; p++) {
                double xp = (2*p+0.5)/NPTCHECK - 1.;
                math::legendrePolyArray(lmax, 0, xp, &legPl.front(), &legdPl.front());
                for(int l=0; l<=lmax; l++) {
                    double v, d;
                    spl3[l].evalDeriv(xp, &v, &d);
                    err3[l] += pow_2(v - legPl[l]);
                    err3der[l] += pow_2(d - legdPl[l]);
                    spl5[l].evalDeriv(xp, &v, &d);
                    err5[l] += pow_2(v - legPl[l]);
                    err5der[l] += pow_2(d - legdPl[l]);
                }
            }
            double maxerr3=0, maxerr5=0, maxerr3der=0, maxerr5der=0;
            for(int l=0; l<=lmax; l++) {
                maxerr3 = fmax(maxerr3, sqrt(err3[l]/NPTCHECK));
                maxerr5 = fmax(maxerr5, sqrt(err5[l]/NPTCHECK));
                maxerr3der = fmax(maxerr3der, sqrt(err3der[l]/NPTCHECK));
                maxerr5der = fmax(maxerr5der, sqrt(err5der[l]/NPTCHECK));
            }
            std::cout << "Lmax="<<lmax<<", Npoints="<<(np+1)<<
                ": rms(3)= "<<maxerr3<<", rms(5)= "<<maxerr5<<
                ": rms(3')= "<<maxerr3der<<", rms(5')= "<<maxerr5der<<"\n";
        }
    }

    //----------- test 2d cubic and quintic spline ------------//

    const int NNODESX=8;
    const int NNODESY=4;
    const int NN=99;    // number of intermediate points for checking the values
    std::vector<double> xval(NNODESX,0);
    std::vector<double> yval(NNODESY,0);
    math::Matrix<double> zval(NNODESX,NNODESY);
    for(int i=1; i<NNODESX; i++)
        xval[i] = xval[i-1] + rand()*1.0/RAND_MAX + 0.5;
    for(int j=1; j<NNODESY; j++)
        yval[j] = yval[j-1] + rand()*1.0/RAND_MAX + 0.5;
    for(int i=0; i<NNODESX; i++) {
        for(int j=0; j<NNODESY; j++)
            zval(i, j) = rand()*1.0/RAND_MAX;
    }
    // create a 2d cubic spline with prescribed derivatives at three out of four edges
    math::CubicSpline2d spl2d(xval, yval, zval, 0., NAN, 1., -1.);

    // obtain the matrices of derivatives from the existing cubic spline
    math::Matrix<double> zderx(NNODESX,NNODESY), zdery(NNODESX,NNODESY);
    for(int i=0; i<NNODESX; i++)
        for(int j=0; j<NNODESY; j++)
            spl2d.evalDeriv(xval[i], yval[j], NULL, &zderx(i, j), &zdery(i, j));

    // create a 2d quintic spline with prescribed derivatives at all nodes
    math::QuinticSpline2d spl2d5(xval, yval, zval, zderx, zdery);

#ifdef COMPARE_WD_PSPLINE
    double *WD_X[2], **WD_Y[3], **WD_Z[4];
    WD_X[0] = new double[NNODESX];
    WD_X[1] = new double[NNODESY];
    int WD_K[2] = {NNODESX, NNODESY};
    WD::Alloc2D(WD_Y[0],WD_K);
    WD::Alloc2D(WD_Y[1],WD_K);
    WD::Alloc2D(WD_Y[2],WD_K);
    WD::Alloc2D(WD_Z[0],WD_K);
    WD::Alloc2D(WD_Z[1],WD_K);
    WD::Alloc2D(WD_Z[2],WD_K);
    WD::Alloc2D(WD_Z[3],WD_K);
    for(int i=0; i<NNODESX; i++)
        WD_X[0][i] = xval[i];
    for(int j=0; j<NNODESY; j++)
        WD_X[1][j] = yval[j];
    for(int i=0; i<NNODESX; i++)
        for(int j=0; j<NNODESY; j++) {
            WD_Y[0][i][j] = zval(i, j);
            WD_Y[1][i][j] = zderx(i, j);
            WD_Y[2][i][j] = zdery(i, j);
        }
    WD::Pspline2D(WD_X, WD_Y, WD_K, WD_Z);
#endif

    // check values and derivatives at grid nodes on all four grid edges
    for(int i=0; i<NNODESX; i++) {
        double z, dy;
        spl2d.evalDeriv(xval[i], yval.front(), &z, NULL, &dy);
        ok &= math::fcmp(dy, 1., 1e-13)==0 && math::fcmp(z, zval(i, 0), 1e-13)==0;
        spl2d.evalDeriv(xval[i], yval.back(), &z, NULL, &dy);
        ok &= math::fcmp(dy, -1., 1e-13)==0 && math::fcmp(z, zval(i, NNODESY-1), 1e-13)==0;

        spl2d5.evalDeriv(xval[i], yval.front(), &z, NULL, &dy);
        ok &= math::fcmp(dy, 1., 1e-13)==0 && math::fcmp(z, zval(i, 0), 1e-13)==0;
        spl2d5.evalDeriv(xval[i], yval.back(), &z, NULL, &dy);
        ok &= math::fcmp(dy, -1., 1e-13)==0 && math::fcmp(z, zval(i, NNODESY-1), 1e-13)==0;
    }
    for(int j=0; j<NNODESY; j++) {
        double z, dx;
        spl2d.evalDeriv(xval.front(), yval[j], &z, &dx);
        ok &= math::fcmp(dx, 0.)==0 && math::fcmp(z, zval(0, j), 1e-13)==0;
        spl2d.evalDeriv(xval.back(), yval[j], &z, &dx);
        ok &= fabs(dx)<10 && math::fcmp(z, zval(NNODESX-1, j), 1e-13)==0;

        spl2d5.evalDeriv(xval.front(), yval[j], &z, &dx);
        ok &= math::fcmp(dx, 0.)==0 && math::fcmp(z, zval(0, j), 1e-13)==0;
        spl2d5.evalDeriv(xval.back(), yval[j], &z, &dx);
        ok &= fabs(dx)<10 && math::fcmp(z, zval(NNODESX-1, j), 1e-13)==0;
    }

    // check derivatives on the entire edge at the three edges that had a prescribed value of derivative
    // (this is only valid for the cubic spline, not for the quintic one)
    for(int i=0; i<=NN; i++) {
        double x = i*xval.back()/NN;
        double dy;
        spl2d.evalDeriv(x, yval.front(), NULL, NULL, &dy);
        ok &= math::fcmp(dy, 1., 1e-13)==0;
        spl2d.evalDeriv(x, yval.back(), NULL, NULL, &dy);
        ok &= math::fcmp(dy, -1., 1e-13)==0;
        double y = i*yval.back()/NN;
        double dx;
        spl2d.evalDeriv(xval.front(), y, NULL, &dx);
        ok &= math::fcmp(dx, 0.)==0;
    }

    // check that the derivatives at all nodes agree between cubic and quintic splines
    for(int i=0; i<NNODESX; i++)
        for(int j=0; j<NNODESY; j++) {
            double c, cx, cy, q, qx, qy;
            spl2d .evalDeriv(xval[i], yval[j], &c, &cx, &cy);
            spl2d5.evalDeriv(xval[i], yval[j], &q, &qx, &qy);
            ok &= math::fcmp(c, q, 1e-13)==0 && 
                math::fcmp(cx, qx, 1e-13)==0 && math::fcmp(cy, qy, 1e-13)==0;
        }

    if(OUTPUT)  // output for Gnuplot splot routine
        strm.open("test_math_spline2d.dat");
    for(int i=0; i<=NN; i++) {
        double x = i*xval.back()/NN;
        for(int j=0; j<=NN; j++) {
            double y = j*yval.back()/NN;
            double z, dx, dy, dxx, dxy, dyy, z5, d5x, d5y, d5xx, d5xy, d5yy;
            spl2d.evalDeriv(x, y, &z, &dx, &dy, &dxx, &dxy, &dyy);
            ok &= z>=-1 && z<=2;
            spl2d5.evalDeriv(x, y, &z5, &d5x, &d5y, &d5xx, &d5xy, &d5yy);
            ok &= z5>=-1 && z5<=2 && fabs(z-z5)<0.1;
#ifdef COMPARE_WD_PSPLINE
            double wx[2] = {x, y};
            double wder[2];
            double wder2x[2], wder2y[2]; 
            double* wder2[] = {wder2x, wder2y};
            double wval = WD::Psplev2D(WD_X, WD_Y, WD_Z, WD_K, wx, wder, wder2);
            ok &=      (math::fcmp(z5, wval, 1e-13)==0 || fabs(z5  -wval   )  <1e-13) && 
                (math::fcmp(d5x,  wder[0],   1e-13)==0 || fabs(d5x -wder[0])  <1e-13) && 
                (math::fcmp(d5y,  wder[1],   1e-13)==0 || fabs(d5y -wder[1])  <1e-13) && 
                (math::fcmp(d5xx, wder2x[0], 1e-13)==0 || fabs(d5xx-wder2x[0])<1e-13) &&
                (math::fcmp(d5xy, wder2x[1], 1e-13)==0 || fabs(d5xy-wder2x[1])<1e-13) && 
                (math::fcmp(d5yy, wder2y[1], 1e-13)==0 || fabs(d5yy-wder2y[1])<1e-13);
#endif
            if(OUTPUT)
                strm << x << ' ' << y << ' ' << 
                    z << ' ' << dx << ' ' << dy << ' ' << dxx << ' ' << dxy << ' ' << dyy << '\t' <<
                    z5<< ' ' << d5x<< ' ' << d5y<< ' ' << d5xx<< ' ' << d5xy<< ' ' << d5yy << "\n";
        }
        if(OUTPUT)
        strm << "\n";
    }

#ifdef STRESS_TEST
    //----------- test the performance of 2d spline calculation -------------//
    double z, dx, dy, dxx, dxy, dyy;
    double wder[2];
    double wder2x[2], wder2y[2]; 
    double* wder2[] = {wder2x, wder2y};
    int NUM_ITER = 1000;
    clock_t clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NN; i++) {
            double x = i*xval.back()/NN;
            for(int j=0; j<=NN; j++) {
                double y = j*yval.back()/NN;
                spl2d.evalDeriv(x, y, &z, &dx, &dy, &dxx, &dxy, &dyy);
            }
        }
    }
    std::cout << "Cubic spline with 2nd deriv: " << (std::clock()-clk)*1.0/CLOCKS_PER_SEC;
    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NN; i++) {
            double x = i*xval.back()/NN;
            for(int j=0; j<=NN; j++) {
                double y = j*yval.back()/NN;
                spl2d.evalDeriv(x, y, &z, &dx, &dy);
            }
        }
    }
    std::cout << ", 1st deriv: " << (std::clock()-clk)*1.0/CLOCKS_PER_SEC;
    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NN; i++) {
            double x = i*xval.back()/NN;
            for(int j=0; j<=NN; j++) {
                double y = j*yval.back()/NN;
                spl2d.evalDeriv(x, y, &z);
            }
        }
    }
    std::cout << ", no deriv: " << (std::clock()-clk)*1.0/CLOCKS_PER_SEC << " seconds\n";

    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NN; i++) {
            double x = i*xval.back()/NN;
            for(int j=0; j<=NN; j++) {
                double y = j*yval.back()/NN;
                spl2d5.evalDeriv(x, y, &z, &dx, &dy, &dxx, &dxy, &dyy);
            }
        }
    }
    std::cout << "Quintic spline with 2nd deriv: " << (std::clock()-clk)*1.0/CLOCKS_PER_SEC;
    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NN; i++) {
            double x = i*xval.back()/NN;
            for(int j=0; j<=NN; j++) {
                double y = j*yval.back()/NN;
                spl2d5.evalDeriv(x, y, &z, &dx, &dy);
            }
        }
    }
    std::cout << ", 1st deriv: " << (std::clock()-clk)*1.0/CLOCKS_PER_SEC;
    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NN; i++) {
            double x = i*xval.back()/NN;
            for(int j=0; j<=NN; j++) {
                double y = j*yval.back()/NN;
                spl2d5.evalDeriv(x, y, &z);
            }
        }
    }
    std::cout << ", no deriv: " << (std::clock()-clk)*1.0/CLOCKS_PER_SEC << " seconds\n";
#ifdef COMPARE_WD_PSPLINE
    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NN; i++) {
            double x = i*xval.back()/NN;
            for(int j=0; j<=NN; j++) {
                double y = j*yval.back()/NN;
                double wx[2] = {x, y};
                WD::Psplev2D(WD_X, WD_Y, WD_Z, WD_K, wx, wder, wder2);
            }
        }
    }
    std::cout << "WD's Pspline with 2nd deriv: " << (std::clock()-clk)*1.0/CLOCKS_PER_SEC;
    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NN; i++) {
            double x = i*xval.back()/NN;
            for(int j=0; j<=NN; j++) {
                double y = j*yval.back()/NN;
                double wx[2] = {x, y};
                WD::Psplev2D(WD_X, WD_Y, WD_Z, WD_K, wx, wder);
            }
        }
    }
    std::cout << ", 1st deriv: " << (std::clock()-clk)*1.0/CLOCKS_PER_SEC;
    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NN; i++) {
            double x = i*xval.back()/NN;
            for(int j=0; j<=NN; j++) {
                double y = j*yval.back()/NN;
                double wx[2] = {x, y};
                WD::Psplev2D(WD_X, WD_Y, WD_Z, WD_K, wx);
            }
        }
    }
    std::cout << ", no deriv: " << (std::clock()-clk)*1.0/CLOCKS_PER_SEC << " seconds\n";
#endif
#endif

    if(ok)
        std::cout << "ALL TESTS PASSED\n";
    return 0;
}
