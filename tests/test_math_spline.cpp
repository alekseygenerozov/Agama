#include "math_spline.h"
#include "math_core.h"
#include "math_sphharm.h"
#include "math_sample.h"
#include "math_specfunc.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cassert>

#include "debug_utils.h"
const bool OUTPUT = utils::verbosityLevel >= utils::VL_VERBOSE;

bool testCond(bool condition, const char* errorMessage)
{
    if(!condition)
        std::cout << errorMessage << '\n';
    return condition;
}

// provides the integral of sin(x)*x^n
class testfncsin: public math::IFunctionIntegral {
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
class testfncintsin: public math::IFunctionNoDeriv {
public:
    testfncintsin(const math::IFunction& _f): f(_f) {};
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

//#define TESTFNC1D_SMOOTH

// provides a function of 1 variable to interpolate
class testfnc1d: public math::IFunction {
public:
    void evalDeriv(const double x, double* val, double* der, double* der2, double* der3) const
    {
#ifdef TESTFNC1D_SMOOTH
        // a very special case of a function with 3 zero derivatives at x=0 and x=pi
        double y = x / M_PI;
        double q = M_PI*y*y*(3-2*y);
        double s = sin(q);
        double c = cos(q);
        double t = 6*y*(1-y);
        if(val)
            *val = s*s;
        if(der)
            *der = 2*t*s*c;
        if(der2)
            *der2 = 2*t*t*(c*c-s*s) + 12/M_PI*(1-2*y)*s*c;
        if(der3)
            *der3 = -8*(t*t*t+3/M_PI/M_PI)*c*s + 216/M_PI*y*(1-y)*(1-2*y)*(c*c-s*s);
#else
        // a more general case of an oscillating function
        double sqx = sqrt(x);
        double y0 = sin(4*sqx);
        double y0p  = cos(4*sqx) * 2 / sqx;
        if(val)
            *val = y0;
        if(der)
            *der = y0p;
        if(der2)
            *der2 = -(4*y0 + 0.5*y0p) / x;
        if(der3)
            *der3 = (6*y0 + (0.75-4*x)*y0p) / pow_2(x);
#endif
    }
    virtual void evalDeriv(const double x, double* val=NULL, double* der=NULL, double* der2=NULL) const {
        evalDeriv(x, val, der, der2, NULL);
    }
    virtual unsigned int numDerivs() const { return 3; }
};

// provides a function of 2 variables to interpolate
class testfnc2d: public math::IFunctionNdimDeriv {
public:
    virtual void evalDeriv(const double vars[], double values[], double *derivs=NULL) const
    {
        double val = exp(-pow_2(vars[0])-pow_2(vars[1]));
        if(values)
            *values = val;
        if(derivs) {
            derivs[0] = -2*vars[0]*val;
            derivs[1] = -2*vars[1]*val;
            // output second and higher derivs as well
            derivs[2] = (4*pow_2(vars[0])-2)*val; // f_xx
            derivs[3] =  4* vars[0]*vars[1] *val; // f_xy
            derivs[4] = (4*pow_2(vars[1])-2)*val; // f_yy
            derivs[5] = 4*vars[1]*(1-2*pow_2(vars[0]))*val; // f_xxy
            derivs[6] = 4*vars[0]*(1-2*pow_2(vars[1]))*val; // f_xyy
            derivs[7] = 4*(1-pow_2(vars[0]))*(1-2*pow_2(vars[1]))*val; // f_xxyy
        }
    }
    virtual unsigned int numVars() const { return 2; }
    virtual unsigned int numValues() const { return 1; }
};

// provides a function of 3 variables to interpolate
class testfnc3d: public math::IFunctionNdim {
public:
    virtual void eval(const double vars[], double values[]) const {
        values[0] = (sin(vars[0]+0.5*vars[1]+0.25*vars[2]) * cos(-0.2*vars[0]*vars[1]+vars[2]) + 1) /
            sqrt(1 + 0.1*(pow_2(vars[0]) + pow_2(vars[1]) + pow_2(vars[2])));
    }
    virtual unsigned int numVars() const { return 3; }
    virtual unsigned int numValues() const { return 1; }
};

//----------- test penalized smoothing spline fit to noisy data -------------//
bool testPenalizedSplineFit()
{
    std::cout << "\033[1;33m1d penalized fitting\033[0m\n";
    bool ok=true;
    const int NNODES  = 20;
    const int NPOINTS = 10000;
    const double XMIN = 0.2;
    const double XMAX = 12.;
    const double DISP = 0.5;  // y-dispersion
    std::vector<double> xnodes = math::createNonuniformGrid(NNODES, XMIN, XMAX, false);
    xnodes.pop_back();
    std::vector<double> xvalues(NPOINTS), yvalues1(NPOINTS), yvalues2(NPOINTS);

    for(int i=0; i<NPOINTS; i++) {
        xvalues [i] = math::random()*XMAX;
        yvalues1[i] = sin(4*sqrt(xvalues[i])) + DISP*(math::random()-0.5);
        yvalues2[i] = cos(4*sqrt(xvalues[i])) + DISP*(math::random()-0.5)*4;
    }
    math::SplineApprox appr(xnodes, xvalues);
    double rms, edf;

    math::CubicSpline fit1(xnodes, appr.fitOptimal(yvalues1, &rms, &edf));
    std::cout << "case A: RMS=" << rms<<", EDF="<<edf<<"\n";
    ok &= rms<0.2 && edf>=2 && edf<NNODES+2;

    math::CubicSpline fit2(xnodes, appr.fitOversmooth(yvalues2, .5, &rms, &edf));
    std::cout << "case B: RMS="<<rms<<", EDF="<<edf<<"\n";
    ok &= rms<1.0 && edf>=2 && edf<NNODES+2;

    if(OUTPUT) {
        std::ofstream strm("test_math_spline_fit.dat");
        for(size_t i=0; i<xvalues.size(); i++)
            strm << xvalues[i] << "\t" << yvalues1[i] << "\t" << yvalues2[i] << "\t" <<
                fit1(xvalues[i]) << "\t" << fit2(xvalues[i]) << "\n";
    }
    return ok;
}

//-------- test penalized spline log-density estimation ---------//

// density distribution described by a sum of two Gaussians
class Density1: public math::IFunctionNoDeriv {
public:
    double mean1, disp1, mean2, disp2;  // parameters of the density
    double norm;  // overall normalization
    int d;        // multiply P(x) by ln(P(x)^d
    Density1(double _mean1, double _disp1, double _mean2, double _disp2, double _norm) :
        mean1(_mean1), disp1(_disp1), mean2(_mean2), disp2(_disp2), norm(_norm), d(0) {}
    // evaluate the density at the given point
    virtual double value(double x) const {
        double P = norm * (
            0.5 * exp( -0.5 * pow_2((x-mean1)/disp1) ) / sqrt(2*M_PI) / disp1 +
            0.5 * exp( -0.5 * pow_2((x-mean2)/disp2) ) / sqrt(2*M_PI) / disp2 );
        return d==0 ? P : P * math::powInt(log(P), d);
    }
    // sample a point from this density
    double sample() const {
        double x1, x2;
        math::getNormalRandomNumbers(x1, x2);
        if(x2>=0)  // attribute the point to either of the two Gaussians with 50% probability
            return mean1 + disp1 * x1;
        else
            return mean2 + disp2 * x1;
    }
};

// density described by the beta distribution
class Density2: public math::IFunctionNoDeriv {
public:
    double a, b;  // parameters of the density (must be >=0)
    double norm;  // normalization factor
    double cap;   // upper bound on the value of density (needed for rejection sampling)
    int d;        // multiply P(x) by ln(P(x)^d
    Density2(double _a, double _b, double _norm) :
        a(_a), b(_b),
        norm(_norm * math::gamma(a+b+2) / math::gamma(a+1) / math::gamma(b+1)),
        cap(norm * pow(a, a) * pow(b, b) / pow(a+b, a+b)),
        d(0) {}
    // evaluate the density at the given point
    virtual double value(double x) const {
        double P = x>=0 && x<=1 ? norm * pow(x, a) * pow(1-x, b) : 0;
        return d==0 ? P : P * math::powInt(log(P), d);
    }
    // sample a point from this density using the rejection algorithm
    double sample() const {
        while(1) {
            double x = math::random(), p = value(x), y = math::random() * cap;
            assert(p<=cap);
            if(y<p)
                return x;
        }
    }
};

// Gaussian kernel density estimate for an array of points
double kernelDensity(double x, const std::vector<double>& xvalues,
    const std::vector<double>& weights)
{
    const double width = 0.05;
    double sum=0;
    for(unsigned int i=0; i<xvalues.size(); i++)
        sum += weights[i] * exp( -0.5*pow_2((x-xvalues[i])/width) );
    return sum / sqrt(2*M_PI) / width;
}

bool testPenalizedSplineDensity()
{
    std::cout << "\033[1;33m1d density estimate\033[0m\n";
    bool ok=true;
#if 1
    const double MEAN1= 2.0, DISP1=0.1, MEAN2=3.0, DISP2=1.0;// parameters of density function
    const double NORM = 1.;   // normalization (sum of weights of all samples)
    const double XMIN = 0;//fmin(MEAN1-3*DISP1, MEAN2-3*DISP2);  // limits of the interval for
    const double XMAX = 6;//fmax(MEAN1+3*DISP1, MEAN2+3*DISP2);  // constructing the estimators
    // whether to assume that the domain is infinite
    const math::FitOptions OPTIONS = math::FitOptions(math::FO_INFINITE_LEFT | math::FO_INFINITE_RIGHT);
    // density function from which the samples are drawn
    Density1 dens(MEAN1, DISP1, MEAN2, DISP2, NORM);
#else
    const double A    = 0., B = 0.5;  // parameters of density fnc
    const double NORM = 10.;
    const double XMIN = 0., XMAX = 1.;
    const math::FitOptions OPTIONS = math::FitOptions();
    Density2 dens(A, B, NORM);
#endif
    const int NPOINTS = 1000; // # of points to sample
    const int NNODES  = 49;    // nodes in the estimated density function
    const int NCHECK  = 321;   // points to measure the estimated density
    const double SMOOTHING=.5; // amount of smoothing applied to penalized spline estimate
    const int NTRIALS = 120;   // number of different realizations of samples
    const double XCUT = 3.;    // unequal-mass sampling: for x>XCUT, retain only a subset of
    const int MASSMULT= 1;     // samples with proportionally higher weight each
    std::vector<double> xvalues, weights;  // array of sample points

    // first perform Monte Carlo experiment to estimate the average log-likelihood of
    // a finite array of samples drawn from the density function, and its dispersion.
    math::Averager avgL;
    for(int t=0; t<NTRIALS; t++) {
        double logL = 0;
        xvalues.clear();
        weights.clear();
        for(int i=0; i<NPOINTS; i++) {
            double x = dens.sample();
            if(x<XCUT || math::random()<1./MASSMULT) {
                xvalues.push_back(x);
                weights.push_back(NORM/NPOINTS * (x<XCUT ? 1 : MASSMULT) /* (x>XMIN&&x<XMAX)*/);
                logL += log(dens(x)) * weights.back();
            }
        }
        avgL.add(logL);
    }
    std::cout << "Finite-sample log L = " << avgL.mean() << " +- " << sqrt(avgL.disp());
    // compare with theoretical expectation
    dens.d=1;   // integrate P(x) times ln(P(x)^d
    double E = math::integrateAdaptive(dens, XMIN-5, XMAX+5, 1e-6);
    dens.d=2;
    double Q = math::integrateAdaptive(dens, XMIN-5, XMAX+5, 1e-6)*NORM;
    double D = sqrt((Q-E*E)/xvalues.size());  // estimated rms scatter in log-likelihood
    dens.d=0;   // restore the original function
    std::cout << "  Expected log L = " << E << " +- " << D << "\n";

    if(OUTPUT) {
        std::ofstream strm("test_math_spline_logdens_points.dat");
        for(size_t i=0; i<xvalues.size(); i++)
            strm << xvalues[i] << "\t" << weights[i] << "\n";
    }

    // grid defining the logdensity functions
    std::vector<double> grid = math::createUniformGrid(NNODES, XMIN, XMAX);

    // grid of points to check the results
    std::vector<double> testgrid = math::createUniformGrid(NCHECK, grid.front()-1, grid.back()+1);
    std::vector<double> truedens(NCHECK);
    for(int j=0; j<NCHECK; j++)
        truedens[j] = log(dens(testgrid[j]));
    // spline approximation for the true density - to test how well it is described
    // by a cubic spline defined by a small number of nodes
    math::CubicSpline spltrue(grid, math::SplineApprox(grid, testgrid).fit(truedens));
    // estimators of various degree constructed from a finite array of samples
    math::LinearInterpolator spl1(grid,
        math::splineLogDensity<1>(grid, xvalues, weights, OPTIONS));  // linear fit
    math::CubicSpline spl3o(grid,
        math::splineLogDensity<3>(grid, xvalues, weights, OPTIONS));  // optimally smoothed cubic
    math::CubicSpline spl3p(grid,
        math::splineLogDensity<3>(grid, xvalues, weights, OPTIONS, SMOOTHING));  // penalized cubic
    double logLtrue=0, logL1=0, logL3o=0, logL3p=0, logL3s=0;
    for(unsigned int i=0; i<xvalues.size(); i++) {
        // evaluate the likelihood of the sampled points against the true underlying density
        // and against all approximations
        logLtrue += weights[i] * log(dens(xvalues[i]));
        logL1    += weights[i] * spl1(xvalues[i]);
        logL3o   += weights[i] * spl3o(xvalues[i]);
        logL3p   += weights[i] * spl3p(xvalues[i]);
        logL3s   += weights[i] * spltrue(xvalues[i]);
    }
    ok &= fabs(logLtrue-logL1) < 3*D && fabs(logLtrue-logL3o) < 3*D &&
         fabs(logLtrue-logL3p) < 3*D && fabs(logL3o-logL3p-SMOOTHING*D) < 0.5*D;
    std::cout << "Log-likelihood: true density = " << logLtrue <<
        ", its cubic spline approximation = " << logL3s <<
        ", linear B-spline estimate = " << logL1 <<
        ", optimally smoothed cubic B-spline estimate = " << logL3o <<
        ", more heavily smoothed cubic = " << logL3p << '\n';
    if(OUTPUT) {
        std::ofstream strm("test_math_spline_logdens.dat");
        for(int j=0; j<NCHECK; j++) {
            double x = testgrid[j];
            double kernval = log(kernelDensity(x, xvalues, weights));
            strm << x << '\t' << spl1(x) << '\t' << spl3o(x) << '\t' << spl3p(x) << '\t' <<
                kernval << '\t' << truedens[j] << '\t' << spltrue(x) << '\n';
        }
    }
    return ok;
}

//-------- test cubic and quintic splines ---------//

// test the integration of a spline function
bool test_integral(const math::CubicSpline& f, double x1, double x2)
{
    double result_int = f.integrate(x1, x2);
    double result_ext = math::integrateAdaptive(f, x1, x2, 1e-13);
    double error1 = fabs((result_int-result_ext) / result_ext);
    std::cout << "Ordinary intergral on [" + utils::pp(x1,10) +':'+ utils::pp(x2,10) +
        "]: result=" + utils::pp(result_int,8) + ", error=" + utils::pp(error1,8) + '\n';
    result_int = f.integrate(x1, x2, testfncsin());
    result_ext = math::integrateAdaptive(testfncintsin(f), x1, x2, 1e-13);
    double error2 = fabs((result_int-result_ext) / result_ext);
    std::cout << "Weighted intergral on [" + utils::pp(x1,10) +':'+ utils::pp(x2,10) +
        "]: result=" + utils::pp(result_int,8) + ", error=" + utils::pp(error2,8) + '\n';
    result_int = f.integrate(x1, x2, f);
    result_ext = math::integrateAdaptive(squaredfnc(f), x1, x2, 1e-13);
    double error3 = fabs((result_int-result_ext) / result_ext);
    std::cout << "Integral of f(x)^2 on [" + utils::pp(x1,10) +':'+ utils::pp(x2,10) +
        "]: result=" + utils::pp(result_int,8) + ", error=" + utils::pp(error3,8) + '\n';
    return error1 < 1e-13 && error2 < 1e-13 && error3 < 3e-12;
    // the large error in the last case is apparently due to roundoff errors
}

bool testLogScaledSplines()
{
    // function to approximate: f(x) = cos(x-1) + x, x=1..Pi+1
    const int NNODES=8;
    std::vector<double> xnodes = math::createUniformGrid(NNODES, 1, 1+M_PI);
    std::vector<double> yvalues(NNODES);
    for(int i=0; i<NNODES; i++)
        yvalues[i] = cos(xnodes[i]-1) + xnodes[i];
    double derLeft = 1., derRight = 1.;
    math::CubicSpline  fSpl(xnodes, yvalues, derLeft, derRight);
    math::LogSpline    fLog(xnodes, yvalues, derLeft, derRight);
    math::LogLogSpline fDbl(xnodes, yvalues, derLeft, derRight);
    double errValS = 0, errValL = 0, errValD = 0, errDerS = 0, errDerL = 0, errDerD = 0,
        errDer2S = 0, errDer2L = 0, errDer2D = 0;
    for(double x=1.; x<1.+M_PI; x+=0.01) {
        double fval = cos(x-1) + x, fder = 1 - sin(x-1), fder2 = -cos(x-1);
        double sval, sder, sder2, lval, lder, lder2, dval, dder, dder2;
        fSpl.evalDeriv(x, &sval, &sder, &sder2);
        fLog.evalDeriv(x, &lval, &lder, &lder2);
        fDbl.evalDeriv(x, &dval, &dder, &dder2);
        errValS = fmax(errValS, fabs(sval - fval));
        errValL = fmax(errValL, fabs(lval - fval));
        errValD = fmax(errValD, fabs(dval - fval));
        errDerS = fmax(errDerS, fabs(sder - fder));
        errDerL = fmax(errDerL, fabs(lder - fder));
        errDerD = fmax(errDerD, fabs(dder - fder));
        errDer2S= fmax(errDer2S,fabs(sder2- fder2));
        errDer2L= fmax(errDer2L,fabs(lder2- fder2));
        errDer2D= fmax(errDer2D,fabs(dder2- fder2));
    }
    return
        errValS < 2e-4 && errValL < 4e-4 && errValD < 3e-4 &&
        errDerS < 1e-3 && errDerL < 3e-3 && errDerD < 3e-3 &&
        errDer2S< 2e-2 && errDer2L< 5e-2 && errDer2D< 4e-2;
}

#ifdef  TESTFNC1D_SMOOTH
const double XMIN1D = 0;
const double XMAX1D = M_PI;
#else
const double XMIN1D = 0.4;
const double XMAX1D = 6.2832;
#endif

// simple wrapper that binds together a 1d B-spline interpolator and the array of its amplitudes
class BsplineWrapper: public math::IFunctionNoDeriv {
    const math::BsplineInterpolator1d<3>& spl;
    const std::vector<double>& ampl;
public:
    BsplineWrapper(const math::BsplineInterpolator1d<3>& _spl, const std::vector<double>& _ampl) :
        spl(_spl), ampl(_ampl) {}
    virtual double value(double x) const {
        return spl.interpolate(x, ampl);
    }
};

// performance test - repeat spline evaluation many times
template<int numDeriv>
std::string evalSpline(const math::IFunction& fnc)
{
    const int NUM_ITER = 100000, NPOINTS = 100;
    double RATE = 1.0 * NUM_ITER * NPOINTS * CLOCKS_PER_SEC;
    clock_t clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NPOINTS; i++) {
            double x = i * (XMAX1D-XMIN1D) / NPOINTS + XMIN1D, v, d, s;
            fnc.evalDeriv(x, &v, numDeriv>=1 ? &d : NULL, numDeriv>=2 ? &s : NULL);
        }
    }
    return utils::pp(RATE/(std::clock()-clk), 6);
}

bool test1dSpline()
{
    std::cout << "\033[1;33m1d interpolation\033[0m\n";
    const int NNODES  = 20;
    const int NSUBINT = 50;
    std::vector<double> yvalues(NNODES), yderivs(NNODES);
    std::vector<double> xnodes = math::createUniformGrid(NNODES, XMIN1D, XMAX1D);
    testfnc1d fnc;   // the original function that we are approximating
    for(int i=0; i<NNODES; i++)
        fnc.evalDeriv(xnodes[i], &yvalues[i], &yderivs[i]);

    // a collection of approximating splines of various types:
    // 1. linear interpolator
    math::LinearInterpolator fLinear(xnodes, yvalues);

    // 2. cubic spline with natural boundary conditions
    math::CubicSpline   fNatural(xnodes, yvalues);

    // 3. cubic, clamped -- specify derivs at the boundaries
    math::CubicSpline   fClamped(xnodes, yvalues, yderivs.front(), yderivs.back());

    // 4. hermite cubic spline -- specify derivs at all nodes
    math::HermiteSpline fHermite(xnodes, yvalues, yderivs);

    // 5. hermite spline constructed from an ordinary cubic spline,
    // by collecting the first derivatives at all grid nodes - should be equivalent
    std::vector<double> yderivsNatural(NNODES);
    for(int i=0; i<NNODES; i++)
        fNatural.evalDeriv(xnodes[i], NULL, &yderivsNatural[i]);
    math::HermiteSpline fHermiteEquivNatural(xnodes, yvalues, yderivsNatural);

    // 6. quintic spline constructed using the derivatives computed by an ordinary cubic spline -
    // doesn't give much improvement over the cubic spline, but provides a smoother interpolant
    math::QuinticSpline fQuiCube(xnodes, yvalues, yderivsNatural);

    // 7. quintic spline constructed using exact derivatives at all nodes
    math::QuinticSpline fQuintic(xnodes, yvalues, yderivs);

    // 8-9: 1d cubic B-spline represented as a sum over basis functions;
    // this class only stores the x-grid and presents a method for computing the interpolant
    // for any array of amplitudes passed as a parameter -- we will use two different ones:
    math::BsplineInterpolator1d<3> fBspline(xnodes);

    // 8. compute the amplitudes from another cubic spline, which results in an equivalent
    // representation in terms of B-splines
    std::vector<double> amplBsplineEquivClamped =
        math::createBsplineInterpolator1dArray<3>(fClamped, xnodes);

    // 9. compute the amplitudes from the original function in a different way:
    // instead of using just from the function values and possibly the derivatives at the grid
    // nodes, as in all previous cases, it collects the function values at several points
    // inside each grid segment, thus achieving a somewhat better overall approximation
    std::vector<double> amplBsplineOrig = math::createBsplineInterpolator1dArray<3>(fnc, xnodes);

    // 10. 1d clamped cubic spline equivalent to a B-spline (this demonstrates the equivalence
    // in both directions, as the amplitudes of this B-spline were computed from another cubic spline).
    // The array of amplitudes can be simply passed to the constructor of CubicSpline instead of
    // an array of function values -- the mode of operation is determined from the size of this array;
    // if it is equal to the number of x-nodes, it refers to the values of the function at these nodes;
    // if it is longer by two elements, it refers to the B-spline amplitudes.
    math::CubicSpline fClampedEquivBspline(xnodes, amplBsplineEquivClamped);

    // output file
    std::ofstream strm;

    // accumulators for computing rms errors in values, derivatives and second derivatives
    // of these approximations
    double errLinearVal =0,  errNaturalVal=0,  errBsplineVal=0,
           errClampedVal=0,  errHermiteVal=0,  errQuiCubeVal=0,  errQuinticVal=0,
           errClampedDer=0,  errHermiteDer=0,  errQuiCubeDer=0,  errQuinticDer=0,
           errClampedDer2=0, errHermiteDer2=0, errQuiCubeDer2=0, errQuinticDer2=0,
           errBsplineEquivClamped=0, errClampedEquivBspline=0, errHermiteEquivNatural=0;
    bool oknat = true, okcla = true, okher = true, okqui = true, okhcl = true;

    // loop through the range of x covering the input grid,
    // using points both at grid nodes and inside grid segments
    const int NPOINTS = (NNODES-1)*NSUBINT;
    for(int i=0; i<=NPOINTS; i++) {
        double xa = xnodes[i/NSUBINT];
        double xb = i<NPOINTS ? xnodes[i/NSUBINT+1] : xa;
        double x  = xa*(1 - (i%NSUBINT)*1.0/NSUBINT) + xb*(i%NSUBINT)/NSUBINT;

        // 0. the original function value and derivs
        double origVal, origDer, origDer2, origDer3;
        fnc.evalDeriv(x, &origVal, &origDer, &origDer2, &origDer3);

        // 1. linear interpolator
        double fLinearVal = fLinear.value(x);

        // 2. natural cubic spline
        double fNaturalVal, fNaturalDer, fNaturalDer2;
        fNatural.evalDeriv(x, &fNaturalVal, &fNaturalDer, &fNaturalDer2);

        // 3. clamped cubic spline
        double fClampedVal, fClampedDer, fClampedDer2;
        fClamped.evalDeriv(x, &fClampedVal, &fClampedDer, &fClampedDer2);

        // 4. hermite spline constructed from the original function
        double fHermiteVal, fHermiteDer, fHermiteDer2;
        fHermite.evalDeriv(x, &fHermiteVal, &fHermiteDer, &fHermiteDer2);

        // 5. hermite spline equivalent to the natural cubic spline
        double fHermiteEquivNaturalVal, fHermiteEquivNaturalDer, fHermiteEquivNaturalDer2;
        fHermiteEquivNatural.evalDeriv(x, &fHermiteEquivNaturalVal,
            &fHermiteEquivNaturalDer, &fHermiteEquivNaturalDer2);

        // 6. quintic spline from approximate derivatives
        double fQuiCubeVal, fQuiCubeDer, fQuiCubeDer2;
        fQuiCube.evalDeriv(x, &fQuiCubeVal,
            &fQuiCubeDer, &fQuiCubeDer2);

        // 7. quintic spline from exact derivatives
        double fQuinticVal, fQuinticDer, fQuinticDer2;
        fQuintic.evalDeriv(x, &fQuinticVal, &fQuinticDer, &fQuinticDer2);

        // 8. b-spline equivalent to the clamped cubic spline
        double fBsplineEquivClampedVal = fBspline.interpolate(x, amplBsplineEquivClamped);

        // 9. b-spline constructed from the original function
        double fBsplineVal = fBspline.interpolate(x, amplBsplineOrig);

        // 10. clamped cubic spline equivalent to the b-spline
        double fClampedEquivBsplineVal = fClampedEquivBspline(x);

        // accumulate errors of various approximations
        errLinearVal  += pow_2(origVal - fLinearVal );
        errBsplineVal += pow_2(origVal - fBsplineVal);
        errNaturalVal += pow_2(origVal - fNaturalVal);
        errClampedVal += pow_2(origVal - fClampedVal);
        errHermiteVal += pow_2(origVal - fHermiteVal);
        errQuiCubeVal += pow_2(origVal - fQuiCubeVal);
        errQuinticVal += pow_2(origVal - fQuinticVal);
        errClampedDer += pow_2(origDer - fClampedDer);
        errHermiteDer += pow_2(origDer - fHermiteDer);
        errQuiCubeDer += pow_2(origDer - fQuiCubeDer);
        errQuinticDer += pow_2(origDer - fQuinticDer);
        errClampedDer2+= pow_2(origDer2- fClampedDer2);
        errHermiteDer2+= pow_2(origDer2- fHermiteDer2);
        errQuiCubeDer2+= pow_2(origDer2- fQuiCubeDer2);
        errQuinticDer2+= pow_2(origDer2- fQuinticDer2);

        // keep track of error in equivalent representations of the same interpolant
        errBsplineEquivClamped = fmax(errBsplineEquivClamped,
            fabs(fBsplineEquivClampedVal - fClampedVal));
        errClampedEquivBspline = fmax(errClampedEquivBspline,
            fabs(fBsplineEquivClampedVal - fClampedEquivBsplineVal));
        errHermiteEquivNatural = fmax(errHermiteEquivNatural,
            fabs(fHermiteEquivNaturalVal - fNaturalVal) +
            fabs(fHermiteEquivNaturalDer - fNaturalDer) +
            fabs(fHermiteEquivNaturalDer2- fNaturalDer2));

        // if x coincides with a grid node, the values of interpolants should be
        // exact to machine precision by construction
        if(i%NSUBINT == 0) {
            int k = i/NSUBINT;
            oknat &= fNaturalVal == yvalues[k];
            okcla &= fClampedVal == yvalues[k];
            okher &= fHermiteVal == yvalues[k] && fHermiteDer == yderivs[k];
            okqui &= fQuinticVal == yvalues[k] && fQuinticDer == yderivs[k];
            okhcl &= fHermiteEquivNaturalVal == fNaturalVal;
        }

        // dump the values to a file if requested
        if(OUTPUT) {
            if(i==0) {
                strm.open("test_math_spline1d.dat");
                strm << "x\torig:f f' f''\tbspline:f\tnatural:f f' f''\t"
                    "clamped:f f' f''\thermite:f f' f''\t"
                    "quintic-from-cubic:f f' f''\tquintic:f f' f''\n";
            }
            strm << utils::pp(x,   9) +'\t'+
            utils::pp(origVal,    10) +' ' +
            utils::pp(origDer,     9) +' ' +
            utils::pp(origDer2,    9) +'\t'+
            utils::pp(fBsplineVal,10) +'\t'+
            utils::pp(fNaturalVal,10) +' ' +
            utils::pp(fNaturalDer, 9) +' ' +
            utils::pp(fNaturalDer2,9) +'\t'+
            utils::pp(fClampedVal,10) +' ' +
            utils::pp(fClampedDer, 9) +' ' +
            utils::pp(fClampedDer2,9) +'\t'+
            utils::pp(fHermiteVal,10) +' ' +
            utils::pp(fHermiteDer, 9) +' ' +
            utils::pp(fHermiteDer2,9) +'\t'+
            utils::pp(fQuiCubeVal,10) +' ' +
            utils::pp(fQuiCubeDer, 9) +' ' +
            utils::pp(fQuiCubeDer2,9) +'\t'+
            utils::pp(fQuinticVal,10) +' ' +
            utils::pp(fQuinticDer, 9) +' ' +
            utils::pp(fQuinticDer2,9) +'\n';
            if(i==NPOINTS)
                strm.close();
        }
    }
    errLinearVal   = sqrt(errLinearVal  / NPOINTS);
    errNaturalVal  = sqrt(errNaturalVal / NPOINTS);
    errBsplineVal  = sqrt(errBsplineVal / NPOINTS);
    errClampedVal  = sqrt(errClampedVal / NPOINTS);
    errHermiteVal  = sqrt(errHermiteVal / NPOINTS);
    errQuinticVal  = sqrt(errQuinticVal / NPOINTS);
    errClampedDer  = sqrt(errClampedDer / NPOINTS);
    errHermiteDer  = sqrt(errHermiteDer / NPOINTS);
    errQuinticDer  = sqrt(errQuinticDer / NPOINTS);
    errClampedDer2 = sqrt(errClampedDer2/ NPOINTS);
    errHermiteDer2 = sqrt(errHermiteDer2/ NPOINTS);
    errQuinticDer2 = sqrt(errQuinticDer2/ NPOINTS);
    errQuiCubeVal  = sqrt(errQuiCubeVal / NPOINTS);

    std::cout << "RMS error in linear interpolator: " + utils::pp(errLinearVal, 8) +
        ", in ordinary cubic spline: " + utils::pp(errNaturalVal, 8) +
        ", in clamped cubic spline: "  + utils::pp(errClampedVal, 8) +
        ", in hermite cubic spline: "  + utils::pp(errHermiteVal, 8) +
        ", in cubic B-spline: "        + utils::pp(errBsplineVal, 8) +
        ", in quintic-from-cubic spline: " + utils::pp(errQuiCubeVal, 8) +
        ", in quintic spline: " + utils::pp(errQuinticVal, 8) +
        ", max|cubic-hermite|=" + utils::pp(errHermiteEquivNatural, 8) +
        ", max|cubic-bspline|=" + utils::pp(errBsplineEquivClamped, 8) +
        " and " + utils::pp(errClampedEquivBspline, 8) + "\n";

    // test the integration functions //
    const double X1 = (xnodes[0]+xnodes[1])/2, X2 = xnodes[xnodes.size()-2]-0.1;
    bool okintcla = test_integral(fClamped, X1, X2);
    bool okintnat = test_integral(fNatural, -1.234567, xnodes.back()+1.);
    double intClamped = fClamped.integrate(X1, X2);
    double intBspline = fBspline.integrate(X1, X2, amplBsplineEquivClamped);
    double intNumeric = math::integrateAdaptive(fClamped, X1, X2, 1e-14);
    bool okintnum = fabs(intClamped-intNumeric) < 1e-13 && fabs(intClamped-intBspline) < 1e-13;

    // test log-scaled splines
    bool oklogspl = testLogScaledSplines();

    bool ok = 
    testCond(oknat, "natural cubic spline values at grid nodes are inexact") &&
    testCond(okcla, "clamped cubic spline values at grid nodes are inexact") &&
    testCond(oknat, "hermite cubic spline values or derivatives at grid nodes are inexact") &&
    testCond(oknat, "quintic spline values or derivatives at grid nodes are inexact") &&
    testCond(oknat, "hermite and clamped cubic spline values are not equal") &&
    testCond(errHermiteEquivNatural < 1e-13, "hermite<>natural") &&
    testCond(errBsplineEquivClamped < 1e-14, "bspline<>clamped") &&
    testCond(errClampedEquivBspline < 1e-14, "clamped<>bspline") &&
    testCond(errNaturalVal < 1.0e-3, "error in natural cubic spline is too large") &&
    testCond(errBsplineVal < 1.0e-4, "error in cubic b-spline is too large") &&
    testCond(errClampedVal < 4.0e-4, "error in clamped cubic spline is too large") &&
    testCond(errHermiteVal < 3.1e-4, "error in hermite cubic spline is too large") &&
    testCond(errQuinticVal < 1.2e-4, "error in quintic spline is too large") &&
    testCond(errClampedDer < 4.5e-3, "error in clamped cubic spline derivative is too large") &&
    testCond(errHermiteDer < 3.5e-3, "error in hermite cubic spline derivative is too large") &&
    testCond(errQuinticDer < 1.4e-3, "error in quintic spline derivative is too large") &&
    testCond(errClampedDer2< 0.085,  "error in clamped cubic spline 2nd derivative is too large")   &&
    testCond(errHermiteDer2< 0.075,  "error in hermite cubic spline 2nd derivative is too large")   &&
    testCond(errQuinticDer2< 0.035,  "error in quintic spline 2nd derivative is too large") &&
    testCond(okintcla, "integral of clamped spline is incorrect") &&
    testCond(okintnat, "integral of natural spline is incorrect") &&
    testCond(okintnum, "integral of B-spline is incorrect") &&
    testCond(oklogspl, "log-scaled splines failed");

    //----------- test the performance of 1d spline calculation -------------//
    std::cout << "Cubic   spline w/o deriv: " + evalSpline<0>(fClamped) + ", 1st deriv: " +
    evalSpline<1>(fClamped) + ", 2nd deriv: " + evalSpline<2>(fClamped) + " eval/s\n";
    std::cout << "Hermite spline w/o deriv: " + evalSpline<0>(fHermite) + ", 1st deriv: " +
    evalSpline<1>(fHermite) + ", 2nd deriv: " + evalSpline<2>(fHermite) + " eval/s\n";
    std::cout << "Quintic spline w/o deriv: " + evalSpline<0>(fQuintic) + ", 1st deriv: " +
    evalSpline<1>(fQuintic) + ", 2nd deriv: " + evalSpline<2>(fQuintic) + " eval/s\n";
    std::cout << "B-spline of degree N=3:   " +
    evalSpline<0>(BsplineWrapper(fBspline, amplBsplineOrig)) + " eval/s\n";
    return ok;
}

//----------- test 2d cubic and quintic spline ------------//

// performance test - repeat spline evaluation many times
template<int numDeriv>
std::string evalSpline2d(const math::BaseInterpolator2d& fnc)
{
    const int NUM_ITER = 500, NPOINTS = 100;
    const double XMIN = -1.9, XMAX = 2.2, YMIN = -2.1, YMAX = 1.7;
    double RATE = 1.0 * NUM_ITER * pow_2(NPOINTS+1) * CLOCKS_PER_SEC;
    clock_t clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NPOINTS; i++) {
            double x = (XMAX-XMIN)*(i*1./NPOINTS)+XMIN;
            for(int j=0; j<=NPOINTS; j++) {
                double y = (YMAX-YMIN)*(j*1./NPOINTS)+YMIN;
                double z, dx, dy, dxx, dxy, dyy;
                fnc.evalDeriv(x, y, &z, numDeriv>=1 ? &dx : NULL, numDeriv>=1 ? &dy : NULL,
                    numDeriv>=2 ? &dxx : NULL, numDeriv>=2 ? &dxy : NULL, numDeriv>=2 ? &dyy : NULL);
            }
        }
    }
    return utils::pp(RATE/(std::clock()-clk), 6);
}

bool test2dSpline()
{
    std::cout << "\033[1;33m2d interpolation\033[0m\n";
    bool ok=true;
    const int NNODESX=8;
    const int NNODESY=7;
    const int NN=100;    // number of intermediate points for checking the values
    const double XMIN = -1.9, XMAX = 2.2, YMIN = -2.1, YMAX = 1.7;
    /*const int NNODESX=3;
    const int NNODESY=3;
    const int NN=100;    // number of intermediate points for checking the values
    const double XMIN = -1.9, XMAX = -1.5, YMIN = -2.1, YMAX = -1.7;*/
    
    std::vector<double> xval = math::createUniformGrid(NNODESX, XMIN, XMAX);
    std::vector<double> yval = math::createUniformGrid(NNODESY, YMIN, YMAX);
    math::Matrix<double> fval(NNODESX, NNODESY), fderx(NNODESX, NNODESY), fdery(NNODESX, NNODESY);
    testfnc2d fnc;
    for(int i=0; i<NNODESX; i++)
        for(int j=0; j<NNODESY; j++) {
            double xy[2] = {xval[i], yval[j]};
            double der[8];
            fnc.evalDeriv(xy, &fval(i, j), der);
            fderx(i, j) = der[0];
            fdery(i, j) = der[1];
        }
    // 2d bilinear interpolator
    math::LinearInterpolator2d lin2d(xval, yval, fval);
    // 2d cubic spline with natural boundary conditions at three edges
    // and a prescribed derivative at the fourth edge
    math::CubicSpline2d cub2d(xval, yval, fval);
    // 2d quintic spline with prescribed derivatives at all nodes
    math::QuinticSpline2d qui2d(xval, yval, fval, fderx, fdery);

    // check the values (for both cubic and quintic splines) and derivatives (for quintic spline only)
    // at all nodes of 2d grid -- should exactly equal the input values (with machine precision)
    for(int i=0; i<NNODESX; i++) {
        for(int j=0; j<NNODESY; j++) {
            double xy[2] = {xval[i], yval[j]};
            double f, fder[8], l, c, q, qder[8], cxy, qxy, qxx, qyy;
            fnc.evalDeriv(xy, &f, fder);
            l = lin2d .value(xy[0], xy[1]);
            cub2d.evalDeriv(xy[0], xy[1], &c, NULL, NULL, NULL, &cxy);
            qui2d.evalDeriv(xy[0], xy[1], &q, &qder[0], &qder[1], &qxx, &qxy, &qyy);
            if(l != f || c != f || q != f || fder[0] != qder[0] || fder[1] != qder[1])
                ok = false;
        }
    }

    std::ofstream strm;
    if(OUTPUT) { // output for Gnuplot splot routine
        strm.open("test_math_spline2d.dat");
        strm << "x y\tfunc fx fy fxx fxy fyy\tcubic cx cy cxx cxy cyy\tquintic qx qy qxx qxy qyy\n";
    }
    double sumerrl = 0, sumerrc = 0, sumerrq = 0,
        sumerrcder = 0, sumerrqder = 0, sumerrcder2 = 0, sumerrqder2 = 0;
    for(int i=0; i<=NN; i++) {
        double x = (XMAX-XMIN)*(i*1./NN)+XMIN;
        for(int j=0; j<=NN; j++) {
            double y = (YMAX-YMIN)*(j*1./NN)+YMIN;
            double xy[2] = {x, y}, f, d[8];
            fnc.evalDeriv(xy, &f, d);
            double l, c, cx, cy, cxx, cxy, cyy, q, qx, qy, qxx, qxy, qyy;
            l =  lin2d.value(x, y);
            cub2d.evalDeriv(x, y, &c, &cx, &cy, &cxx, &cxy, &cyy);
            qui2d.evalDeriv(x, y, &q, &qx, &qy, &qxx, &qxy, &qyy);

            sumerrl     += pow_2(l-f);
            sumerrc     += pow_2(c-f);
            sumerrq     += pow_2(q-f);
            sumerrcder  += pow_2(cx-d[0]) + pow_2(cy-d[1]);
            sumerrqder  += pow_2(qx-d[0]) + pow_2(qy-d[1]);
            sumerrcder2 += pow_2(cxx-d[2]) + pow_2(cxy-d[3]) + pow_2(cyy-d[4]);
            sumerrqder2 += pow_2(qxx-d[2]) + pow_2(qxy-d[3]) + pow_2(qyy-d[4]);

            if(OUTPUT)
                strm << utils::pp(x, 7) + ' ' + utils::pp(y, 7) + '\t' +
                utils::pp(f, 10) + ' ' + utils::pp(d[0],8)+ ' ' + utils::pp(d[1],8)+ ' ' +
                utils::pp(d[2],7)+ ' ' + utils::pp(d[3],7)+ ' ' + utils::pp(d[4],7)+ '\t'+
                utils::pp(c, 10) + ' ' + utils::pp(cx, 8) + ' ' + utils::pp(cy, 8) + ' ' +
                utils::pp(cxx,7) + ' ' + utils::pp(cxy,7) + ' ' + utils::pp(cyy,7) + '\t'+
                utils::pp(q, 10) + ' ' + utils::pp(qx, 8) + ' ' + utils::pp(qy, 8) + ' ' +
                utils::pp(qxx,7) + ' ' + utils::pp(qxy,7) + ' ' + utils::pp(qyy,7) + '\n';
        }
        if(OUTPUT)
            strm << "\n";
    }
    if(OUTPUT)
        strm.close();
    sumerrl     = sqrt(sumerrl)       / (NN+1);
    sumerrc     = sqrt(sumerrc)       / (NN+1);
    sumerrq     = sqrt(sumerrq)       / (NN+1);
    sumerrcder  = sqrt(sumerrcder /2) / (NN+1);
    sumerrqder  = sqrt(sumerrqder /2) / (NN+1);
    sumerrcder2 = sqrt(sumerrcder2/3) / (NN+1);
    sumerrqder2 = sqrt(sumerrqder2/3) / (NN+1);
    std::cout << "RMS error in linear 2d interpolator: " + utils::pp(sumerrl, 8) +
        "\ncubic   2d spline value:  " + utils::pp(sumerrc, 8) +
        ", deriv: " + utils::pp(sumerrcder, 8) + ", 2nd deriv: " + utils::pp(sumerrcder2, 8) +
        "\nquintic 2d spline value:  " + utils::pp(sumerrq, 8) +
        ", deriv: " + utils::pp(sumerrqder, 8) + ", 2nd deriv: " + utils::pp(sumerrqder2, 8) +"\n";
    ok &= sumerrc < 0.003 && sumerrcder < 0.012 && sumerrcder2 < 0.075 &&
          sumerrq < 1.e-4 && sumerrqder < 7.e-4 && sumerrqder2 < 0.006;

    //----------- test the performance of 2d spline calculation -------------//
    std::cout <<"Linear interpolator:      " + evalSpline2d<0>(lin2d) + " eval/s\n";
    std::cout <<"Cubic   spline w/o deriv: " + evalSpline2d<0>(cub2d) + ", 1st deriv: " +
    evalSpline2d<1>(cub2d) + ", 2nd deriv: " + evalSpline2d<2>(cub2d) + " eval/s\n";
    std::cout <<"Quintic spline w/o deriv: " + evalSpline2d<0>(qui2d) + ", 1st deriv: " +
    evalSpline2d<1>(qui2d) + ", 2nd deriv: " + evalSpline2d<2>(qui2d) + " eval/s\n";
    return ok;
}

//----------- test 3d interpolation ------------//
bool test3dSpline()
{
    std::cout << "\033[1;33m3d interpolation\033[0m\n";
    bool ok=true;
    testfnc3d fnc3d;
    const int NNODESX=10, NNODESY=8, NNODESZ=7;
    std::vector<double>
    xval=math::createUniformGrid(NNODESX, 0, 6.283185),
    yval=math::createUniformGrid(NNODESY, 0, 2.718282),
    zval=math::createUniformGrid(NNODESZ, 0, 5.012345);
    // B-spline interpolators of degree 1 and 3 are constructed by computing the amplitudes;
    // for the N=1 B-spline these are just the function values at grid points,
    // but for the N=3 one we must solve a (sparse) linear system
    std::vector<double> lval3d(math::createBsplineInterpolator3dArray<1>(fnc3d, xval, yval, zval));
    std::vector<double> cval3d(math::createBsplineInterpolator3dArray<3>(fnc3d, xval, yval, zval));
    math::BsplineInterpolator3d<1> blin3d(xval, yval, zval);
    math::BsplineInterpolator3d<3> bcub3d(xval, yval, zval);
    // Linear interpolator constructed from the values of function at grid points
    math::LinearInterpolator3d lin3d(xval, yval, zval, lval3d);
    // Cubic spline in 3d constructed from the values of function at grid points
    math::CubicSpline3d        spl3d(xval, yval, zval, lval3d);
    // 3d cubic spline constructed from the amplitudes of a 3d B-spline of degree N=3
    math::CubicSpline3d        spc3d(xval, yval, zval, cval3d);
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
                double ll = lin3d.value(point[0], point[1], point[2]);
                double sl = spl3d.value(point[0], point[1], point[2]);
                double sc = spc3d.value(point[0], point[1], point[2]);
                double bl = blin3d.interpolate(point, lval3d);
                double bc = bcub3d.interpolate(point, cval3d);
                ok &= v == ll && v == sl && v == bl;  // this should be exact
                ok &= math::fcmp(v, sc, 1e-13)==0;
                ok &= math::fcmp(v, bc, 1e-13)==0;
            }
        }
    }
    // test accuracy of approximation
    double sumsqerr_l=0, sumsqerr_c=0, sumsqerr_s=0;
    const int NNN=24;    // number of intermediate points for checking the values
    std::ofstream strm;
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
                double ll = lin3d.value(point[0], point[1], point[2]);
                double sl = spl3d.value(point[0], point[1], point[2]);
                double sc = spc3d.value(point[0], point[1], point[2]);
                double bl = blin3d.interpolate(point, lval3d);
                double bc = bcub3d.interpolate(point, cval3d);
                sumsqerr_l += pow_2(v-bl);
                sumsqerr_c += pow_2(v-bc);
                sumsqerr_s += pow_2(sl-bc) + pow_2(sc-bc) + pow_2(ll-bl);
                if(OUTPUT)
                    strm << point[0] << ' ' << point[1] << ' ' << point[2] << '\t' <<
                    v << ' ' << bl << ' ' << bc << "\n";
            }
            if(OUTPUT)
                strm << "\n";
        }
    }
    if(OUTPUT)
        strm.close();
    sumsqerr_l = sqrt(sumsqerr_l / pow_3(NNN+1));
    sumsqerr_c = sqrt(sumsqerr_c / pow_3(NNN+1));
    sumsqerr_s = sqrt(sumsqerr_s / pow_3(NNN+1));
    std::cout << "RMS error in linear 3d interpolator: " << utils::pp(sumsqerr_l, 8) << 
        ", cubic 3d interpolator: " << utils::pp(sumsqerr_c, 8) <<
        ", difference between cubic spline and B-spline: " << utils::pp(sumsqerr_s, 8) << "\n";
    ok &= sumsqerr_l<0.1 && sumsqerr_c<0.05 && sumsqerr_s<1e-15;

    // test performance of various interpolators
    int NUM_ITER = 200;
    double RATE = 1.0 * NUM_ITER * pow_3(NNN+1) * CLOCKS_PER_SEC;
    clock_t clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NNN; i++) {
            point[0] = i*xval.back()/NNN;
            for(int j=0; j<=NNN; j++) {
                point[1] = j*yval.back()/NNN;
                for(int k=0; k<=NNN; k++) {
                    point[2] = k*zval.back()/NNN;
                    double s;
                    lin3d.eval(point, &s);
                }
            }
        }
    }
    std::cout << "Linear interpolator:    " + utils::pp(RATE/(std::clock()-clk), 6) + " eval/s\n";

    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NNN; i++) {
            point[0] = i*xval.back()/NNN;
            for(int j=0; j<=NNN; j++) {
                point[1] = j*yval.back()/NNN;
                for(int k=0; k<=NNN; k++) {
                    point[2] = k*zval.back()/NNN;
                    double s;
                    spl3d.eval(point, &s);
                }
            }
        }
    }
    std::cout << "Cubic spline:           " + utils::pp(RATE/(std::clock()-clk), 6) + " eval/s\n";

    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NNN; i++) {
            point[0] = i*xval.back()/NNN;
            for(int j=0; j<=NNN; j++) {
                point[1] = j*yval.back()/NNN;
                for(int k=0; k<=NNN; k++) {
                    point[2] = k*zval.back()/NNN;
                    blin3d.interpolate(point, lval3d);
                }
            }
        }
    }
    std::cout << "B-spline of degree N=1: " + utils::pp(RATE/(std::clock()-clk), 6) + " eval/s\n";

    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NNN; i++) {
            point[0] = i*xval.back()/NNN;
            for(int j=0; j<=NNN; j++) {
                point[1] = j*yval.back()/NNN;
                for(int k=0; k<=NNN; k++) {
                    point[2] = k*zval.back()/NNN;
                    bcub3d.interpolate(point, cval3d);
                }
            }
        }
    }
    std::cout << "B-spline of degree N=3: " + utils::pp(RATE/(std::clock()-clk), 6) + " eval/s\n";

    return ok;
}

bool printFail(const char* msg)
{
    std::cout << "\033[1;31m " << msg << " failed\033[0m\n";
    return (msg==0);  // false
}

int main()
{
    bool ok=true;
    ok &= testPenalizedSplineFit() || printFail("Penalized spline fit");
    ok &= testPenalizedSplineDensity() || printFail("Penalized spline density estimator");
    ok &= test1dSpline() || printFail("1d spline");
    ok &= test2dSpline() || printFail("2d spline");
    ok &= test3dSpline() || printFail("3d spline");
    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
