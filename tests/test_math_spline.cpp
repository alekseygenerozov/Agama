//#define COMPARE_WD_PSPLINE
//#define STRESS_TEST

#include "math_spline.h"
#include "math_core.h"
#include "math_sphharm.h"
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
const bool OUTPUT = true;

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

// test the integration of a spline function
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

// provides a function of 3 variables to interpolate
class testfnc3d: public math::IFunctionNdim {
public:
    virtual void eval(const double vars[], double values[]) const {
        values[0] = sin(vars[0]+0.5*vars[1]+0.25*vars[2]) * cos(-0.2*vars[0]*vars[1]+vars[2]) /
            sqrt(1 + 0.1*(pow_2(vars[0]) + pow_2(vars[1]) + pow_2(vars[2])));
    }
    virtual unsigned int numVars() const { return 3; }
    virtual unsigned int numValues() const { return 1; }
};

int main()
{
    //----------- test penalized smoothing spline fit to noisy data -------------//

    std::cout << std::setprecision(12);
    bool ok=true;
    std::vector<double> xnodes = math::createNonuniformGrid(NNODES, XMIN, XMAX, true);
    std::vector<double> xvalues(NPOINTS), yvalues1(NPOINTS), yvalues2(NPOINTS);
    for(int i=0; i<NPOINTS; i++) {
        xvalues [i] = math::random()*XMAX;
        yvalues1[i] = sin(4*sqrt(xvalues[i])) + DISP*(math::random()-0.5);
        yvalues2[i] = cos(4*sqrt(xvalues[i])) + DISP*(math::random()-0.5)*4;
    }
    math::SplineApprox appr(xvalues, xnodes);
    if(appr.isSingular())
        std::cout << "Warning, matrix is singular\n";

    std::vector<double> ynodes1, ynodes2;
    double deriv_left, deriv_right, rms, edf, lambda;

    appr.fitDataOptimal(yvalues1, ynodes1, deriv_left, deriv_right, &rms, &edf, &lambda);
    std::cout << "case A: RMS="<<rms<<", EDF="<<edf<<", lambda="<<lambda<<"\n";
    math::CubicSpline fit1(xnodes, ynodes1, deriv_left, deriv_right);
    ok &= rms<0.2 && edf>=2 && edf<NNODES+2 && lambda>0;

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

    std::vector<double> yvalues(NNODES), yderivs(NNODES);
    xnodes = math::createNonuniformGrid(NNODES, XMIN, XMAX, false);
    xnodes[1]=(xnodes[1]+xnodes[2])/2;  // slightly squeeze grid spacing to allow
    xnodes[0]*=2;                       // a better interpolation of a strongly varying function
    for(int i=0; i<NNODES; i++) {
        yvalues[i] = sin(4*sqrt(xnodes[i]));
        yderivs[i] = cos(4*sqrt(xnodes[i])) * 2 / sqrt(xnodes[i]);
    }
    math::CubicSpline   fcubna(xnodes, yvalues);  // cubic spline with natural boundary conditions
    math::CubicSpline   fcubcl(xnodes, yvalues,
        yderivs.front(), yderivs.back());         // cubic, clamped -- specify derivs at the boundaries
    math::HermiteSpline fhermi(xnodes, yvalues, yderivs);  // hermite cubic spline -- specify derivs at all nodes
    math::QuinticSpline fquint(xnodes, yvalues, yderivs);  // quintic spline -- specify derivs at all nodes
    std::ofstream strm;
    if(OUTPUT)
        strm.open("test_math_spline1d.dat");
    double sumerr3n = 0, sumerr3 = 0, sumerr5 = 0, sumerrh = 0;
    for(int i=0; i<=(NNODES-1)*NSUBINT; i++) {
        double xa = xnodes[i/NSUBINT];
        double xb = i<(NNODES-1)*NSUBINT ? xnodes[i/NSUBINT+1] : xa;
        double x  = xa*(1 - (i%NSUBINT)*1.0/NSUBINT) + xb*(i%NSUBINT)/NSUBINT;
        double y0 = sin(4*sqrt(x));
        double y0p  = cos(4*sqrt(x)) * 2 / sqrt(x);
        double y0pp = -(4*y0 + 0.5*y0p) / x;
        double y0ppp = (6*y0 + (0.75-4*x)*y0p) / pow_2(x);
        double y3n = fcubna(x);
        double y3, y3p, y3pp;
        fcubcl.evalDeriv(x, &y3, &y3p, &y3pp);
        double yh, yhp, yhpp;
        fhermi.evalDeriv(x, &yh, &yhp, &yhpp);
        double y5, y5p, y5pp, y5ppp=fquint.deriv3(x);
        fquint.evalDeriv(x, &y5, &y5p, &y5pp);
        sumerr3n += pow_2(y0-y3n);
        sumerr3  += pow_2(y0-y3);
        sumerrh  += pow_2(y0-yh);
        sumerr5  += pow_2(y0-y5);
        if(OUTPUT)
            strm << x << '\t' << y0 << '\t' << 
            y3n << '\t' << y3  << '\t' << yh  << '\t' << y5  << '\t' <<
            y0p << '\t' << y3p << '\t' << yhp << '\t' << y5p << '\t' <<
            y0pp<< '\t' << y3pp<< '\t' << yhpp<< '\t' << y5pp<< '\t' <<
            y0ppp << '\t' << y5ppp << "\n";
    }
    sumerr3n = (sqrt(sumerr3n / ((NNODES-1)*NSUBINT)));
    sumerr3  = (sqrt(sumerr3  / ((NNODES-1)*NSUBINT)));
    sumerrh  = (sqrt(sumerrh  / ((NNODES-1)*NSUBINT)));
    sumerr5  = (sqrt(sumerr5  / ((NNODES-1)*NSUBINT)));
    std::cout << "RMS error in cubic spline: " << sumerr3n <<
        ", in clamped cubic spline: " << sumerr3 <<
        ", in hermite cubic spline: " << sumerrh <<
        ", in quintic spline: " << sumerr5 << "\n";
    ok &= sumerr3n<5e-3 && sumerr3<3e-4 && sumerr5 < 4e-5;
    if(OUTPUT)
        strm.close();

    // test the integration function //
    ok &= test_integral(fcubcl, (xnodes[0]+xnodes[1])/2, xnodes.back());
    ok &= test_integral(fcubna, -1.234567, xnodes.back()+1.);

    //----------- test 2d cubic and quintic spline ------------//

    const int NNODESX=8;
    const int NNODESY=4;
    const int NN=99;    // number of intermediate points for checking the values
    std::vector<double> xval(NNODESX,0);
    std::vector<double> yval(NNODESY,0);
    math::Matrix<double> fval(NNODESX,NNODESY);
    for(int i=1; i<NNODESX; i++)
        xval[i] = xval[i-1] + math::random() + 0.5;
    for(int j=1; j<NNODESY; j++)
        yval[j] = yval[j-1] + math::random() + 0.5;
    for(int i=0; i<NNODESX; i++) {
        for(int j=0; j<NNODESY; j++)
            fval(i, j) = math::random();
    }
    // create a 2d cubic spline with prescribed derivatives at three out of four edges
    math::CubicSpline2d spl2d(xval, yval, fval, 0., NAN, 1., -1.);

    // obtain the matrices of derivatives from the existing cubic spline
    math::Matrix<double> fderx(NNODESX,NNODESY), fdery(NNODESX,NNODESY);
    for(int i=0; i<NNODESX; i++)
        for(int j=0; j<NNODESY; j++)
            spl2d.evalDeriv(xval[i], yval[j], NULL, &fderx(i, j), &fdery(i, j));

    // create a 2d quintic spline with prescribed derivatives at all nodes
    math::QuinticSpline2d spl2d5(xval, yval, fval, fderx, fdery);

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
            WD_Y[0][i][j] = fval(i, j);
            WD_Y[1][i][j] = fderx(i, j);
            WD_Y[2][i][j] = fdery(i, j);
        }
    WD::Pspline2D(WD_X, WD_Y, WD_K, WD_Z);
#endif

    // check values and derivatives at grid nodes on all four grid edges
    for(int i=0; i<NNODESX; i++) {
        double f, dy;
        spl2d.evalDeriv(xval[i], yval.front(), &f, NULL, &dy);
        ok &= math::fcmp(dy, 1., 1e-13)==0 && math::fcmp(f, fval(i, 0), 1e-13)==0;
        spl2d.evalDeriv(xval[i], yval.back(), &f, NULL, &dy);
        ok &= math::fcmp(dy, -1., 1e-13)==0 && math::fcmp(f, fval(i, NNODESY-1), 1e-13)==0;

        spl2d5.evalDeriv(xval[i], yval.front(), &f, NULL, &dy);
        ok &= math::fcmp(dy, 1., 1e-13)==0 && math::fcmp(f, fval(i, 0), 1e-13)==0;
        spl2d5.evalDeriv(xval[i], yval.back(), &f, NULL, &dy);
        ok &= math::fcmp(dy, -1., 1e-13)==0 && math::fcmp(f, fval(i, NNODESY-1), 1e-13)==0;
    }
    for(int j=0; j<NNODESY; j++) {
        double f, dx;
        spl2d.evalDeriv(xval.front(), yval[j], &f, &dx);
        ok &= math::fcmp(dx, 0.)==0 && math::fcmp(f, fval(0, j), 1e-13)==0;
        spl2d.evalDeriv(xval.back(), yval[j], &f, &dx);
        ok &= fabs(dx)<10 && math::fcmp(f, fval(NNODESX-1, j), 1e-13)==0;

        spl2d5.evalDeriv(xval.front(), yval[j], &f, &dx);
        ok &= math::fcmp(dx, 0.)==0 && math::fcmp(f, fval(0, j), 1e-13)==0;
        spl2d5.evalDeriv(xval.back(), yval[j], &f, &dx);
        ok &= fabs(dx)<10 && math::fcmp(f, fval(NNODESX-1, j), 1e-13)==0;
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
    if(OUTPUT)
        strm.close();

    //----------- test 3d interpolation ------------//

    testfnc3d fnc3d;
    const int NNODESZ=6;
    std::vector<double> zval(NNODESZ,0);
    for(int i=1; i<NNODESZ; i++)
        zval[i] = zval[i-1] + math::random() + 0.5;
    std::vector<double> lval3d(math::createInterpolator3dArray<1>(fnc3d, xval, yval, zval));
    std::vector<double> cval3d(math::createInterpolator3dArray<3>(fnc3d, xval, yval, zval));
    math::LinearInterpolator3d lin3d(xval, yval, zval);
    math::CubicInterpolator3d  cub3d(xval, yval, zval);
    double point[3];
    // test the values of interpolated function at grid nodes
    for(int i=0; i<NNODESX; i++) {
        point[0] = xval[i];
        for(int j=0; j<NNODESY; j++) {
            point[1] = yval[j];
            for(int k=0; k<NNODESZ; k++) {
                point[2] = zval[k];
                double v;
                fnc3d.eval(point, &v);
                double l = lin3d.interpolate(point, lval3d);
                double c = cub3d.interpolate(point, cval3d);
                ok &= math::fcmp(v, l, 1e-14)==0 && math::isFinite(c);
                // cubic interpolation coincides with the original values only at grid corners
                if((i==0 || i==NNODESX-1) && (j==0 || j==NNODESY-1) && (k==0 || k==NNODESZ-1))
                    ok &= math::fcmp(v, c, 1e-14)==0;
            }
        }
    }
    // test accuracy of approximation
    double sumsqerr_l=0, sumsqerr_c=0;
    const int NNN=24;    // number of intermediate points for checking the values
    if(OUTPUT)
        strm.open("test_math_spline3d.dat");
    for(int i=0; i<=NNN; i++) {
        point[0] = i*xval.back()/NNN;
        for(int j=0; j<=NNN; j++) {
            point[1] = j*yval.back()/NNN;
            for(int k=0; k<=NNN; k++) {
                point[2] = k*zval.back()/NNN;
                double v;
                fnc3d.eval(point, &v);
                double l = lin3d.interpolate(point, lval3d);
                double c = cub3d.interpolate(point, cval3d);
                sumsqerr_l += pow_2(l-v);
                sumsqerr_c += pow_2(c-v);
                if(OUTPUT)
                    strm << point[0] << ' ' << point[1] << ' ' << point[2] << '\t' <<
                    v << ' ' << l << ' ' << c << "\n";
            }
            if(OUTPUT)
                strm << "\n";
        }
    }
    if(OUTPUT)
        strm.close();
    sumsqerr_l = sqrt(sumsqerr_l / pow_3(NNN+1));
    sumsqerr_c = sqrt(sumsqerr_c / pow_3(NNN+1));
    std::cout << "RMS error in linear 3d interpolator: " << sumsqerr_l << 
        ", cubic 3d interpolator:" << sumsqerr_c << "\n";
    ok &= sumsqerr_l<0.1 && sumsqerr_c<0.2;

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
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    return 0;
}
