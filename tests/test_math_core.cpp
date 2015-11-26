#include "math_core.h"
#include "math_fit.h"
#include "math_sample.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
int numEval=0;

class test1: public math::IFunctionNoDeriv{
    virtual double value(double x) const{
        return 1/sqrt(1-x*x);
    }
};

class test2: public math::IFunctionNoDeriv{
    virtual double value(double x) const{
        return pow(1-x*x*x*x,-2./3);
    }
};

class test3: public math::IFunction{
public:
    int nd;
    test3(int nder) : nd(nder) {};
    virtual void evalDeriv(double x, double* val, double* der=0, double* =0) const{
        numEval++;
        *val = math::sign(x-0.3)*pow(fabs(x-0.3), 1./5);
        if(der) *der = *val/5/(x-0.3);
    }
    virtual unsigned int numDerivs() const { return nd; }
};

class test4: public math::IFunction{
public:
    int nd;
    test4(int nder) : nd(nder) {};
    virtual void evalDeriv(double x, double* val, double* der=0, double* =0) const{
        numEval++;
        *val = x-1+1e-3/sqrt(x);
        if(der) *der = 1-0.5e-3/pow(x,1.5);
    }
    virtual unsigned int numDerivs() const { return nd; }
};

class test4Ndim: public math::IFunctionNdim{
public:
    virtual void eval(const double x[], double val[]) const{
        val[0] = test4(0)(x[0]);
    }
    virtual unsigned int numVars() const { return 1; }
    virtual unsigned int numValues() const { return 1; }
};

class test5: public math::IFunctionNoDeriv{
    virtual double value(double x) const{
        numEval++;
        return exp(1.0001-x)*(x<INFINITY ? (x-1)-1e5*(x-1.0001)*(x-1.0001)-1e-4 : 1) - 1e-12*(1+1/x);
    }
};

class test6: public math::IFunctionNoDeriv{
    virtual double value(double x) const{
        numEval++;;
        return sin(1e4*x);
    }
};

class test_powerlaw: public math::IFunctionNoDeriv{
public:
    test_powerlaw(double _p): p(_p) {};
    virtual double value(double x) const{
        return pow(x, p);
    }
    double exactValue(double xmin, double xmax) const{
        return p==-1 ? log(xmax/xmin) : (pow(xmax, p+1) - pow(xmin, p+1)) / (p+1);
    }
    double p;
};

static const double  // rotation
    A00 = 0.8786288646, A01 = -0.439043856, A02 = 0.1877546558,
    A10 = 0.4474142786, A11 = 0.8943234085, A12 = -0.002470791,
    A20 = -0.166828598, A21 = 0.0861750222, A22 = 0.9822128505,
    c0 = -0.5, c1 = -1., c2 = 2,    // center
    s0 = 2.0,  s1 = 0.5, s2 = 0.1;  // scale
class test7Ndim: public math::IFunctionNdim{
public:
    // 3-dimensional paraboloid centered at c[], scaled with s[] and rotated with orthogonal matrix A[][]
    virtual void eval(const double x[], double val[]) const{
        double x0 = (x[0]-c0)*s0, x1 = (x[1]-c1)*s1, x2 = (x[2]-c2)*s2;
        double v0 = x0*A00+x1*A01+x2*A02;
        double v1 = x0*A10+x1*A11+x2*A12;
        double v2 = x0*A20+x1*A21+x2*A22;
        double v  = x0*v0 +x1*v1 +x2*v2;
        val[0] = 1-1./(1+v*v);
        numEval++;
    }
    virtual unsigned int numVars() const { return 3; }
    virtual unsigned int numValues() const { return 1; }
};

static const double Rout = 3, Rin = 1;  // outer and inner radii of the torus
class test8Ndim: public math::IFunctionNdim{
public:
    // 3-dimensional torus rotated with orthogonal matrix A[][]
    virtual void eval(const double x[], double val[]) const{
        double x0 = x[0]*A00+x[1]*A01+x[2]*A02;
        double x1 = x[0]*A10+x[1]*A11+x[2]*A12;
        double x2 = x[0]*A20+x[1]*A21+x[2]*A22;
        val[0] = pow_2(sqrt(x0*x0+x1*x1)-Rout)+x2*x2 <= Rin*Rin ? 1.0+x0*0.2 : 0.0;
#ifdef _OPENMP
#pragma omp atomic
#endif
        ++numEval;
    }
    virtual unsigned int numVars() const { return 3; }
    virtual unsigned int numValues() const { return 1; }
};

int main()
{
    std::cout << std::setprecision(10);
    bool ok=true;

    // integration routines
    const double toler = 1e-6;
    double exact = (M_PI*2/3), error=0, result;
    result = math::integrate(test1(), -1, 1./2, toler, &error, &numEval);
    std::cout << "Int1: naive="<<result<<" +- "<<error<<" (delta="<<(result-exact)<<", neval="<<numEval;
    ok &= fabs(1-result/exact)<2e-3 && fabs(result-exact)<error;
    result = math::integrateAdaptive(test1(), -1, 1./2, toler, &error, &numEval);
    std::cout << "), adaptive="<<result<<" +- "<<error<<" (delta="<<(result-exact)<<", neval="<<numEval;
    ok &= fabs(1-result/exact)<toler && fabs(result-exact)<error;
    test1 t1;
    math::ScaledIntegrandEndpointSing test1s(t1, -1, 1);
    result = math::integrate(test1s, test1s.y_from_x(-1), test1s.y_from_x(1./2), toler, &error, &numEval);
    std::cout<<"), scaled="<<result<<" +- "<<error<<" (delta="<<(result-exact)<<", neval="<<numEval<<")\n";
    ok &= fabs(1-result/exact)<1e-8 && fabs(result-exact)<error;

    exact = 2.274454287;
    result = math::integrate(test2(), -1, 2./3, toler, &error, &numEval);
    std::cout << "Int1: naive="<<result<<" +- "<<error<<" (delta="<<(result-exact)<<", neval="<<numEval;
    ok &= fabs(1-result/exact)<2e-2 && fabs(result-exact)<error;
    result = math::integrateAdaptive(test2(), -1, 2./3, toler*15, &error, &numEval);
    std::cout << "), adaptive="<<result<<" +- "<<error<<" (delta="<<(result-exact)<<", neval="<<numEval;
    ok &= fabs(1-result/exact)<toler*15 && fabs(result-exact)<error;
    test2 t2;
    math::ScaledIntegrandEndpointSing test2s(t2, -1, 1);
    result = math::integrate(test2s, test2s.y_from_x(-1), test2s.y_from_x(2./3), toler, &error, &numEval);
    std::cout<<"), scaled="<<result<<" +- "<<error<<" (delta="<<(result-exact)<<", neval="<<numEval<<")\n";
    ok &= fabs(1-result/exact)<2e-4 && fabs(result-exact)<error;

    // root-finding
    exact=0.3;
    numEval=0;
    result = math::findRoot(test3(0), 0, 0.8, toler);
    std::cout << "Root3="<<result<<" (delta="<<(result-exact)<<"; neval="<<numEval<<")\n";
    ok &= fabs(1-result/exact)<toler;
    numEval=0;
    result = math::findRoot(test3(1), 0, 0.8, toler);
    std::cout << "with derivative: Root3="<<result<<" (delta="<<(result-exact)<<"; neval="<<numEval<<")\n";
    ok &= fabs(1-result/exact)<toler;

    exact=1.000002e-6;
    numEval=0;
    result = math::findRoot(test4(0), 1e-15, 0.8, 1e-8);
    std::cout << "Root4="<<result<<" (delta="<<(result-exact)<<"; neval="<<numEval<<")\n";
    ok &= fabs(result-exact)<1e-8*0.8;
    numEval=0;
    result = math::findRoot(test4(1), 1e-15, 0.8, 1e-8);
    std::cout << "with derivative: Root4="<<result<<" (delta="<<(result-exact)<<"; neval="<<numEval<<")\n";
    ok &= fabs(result-exact)<1e-8*0.8;

    double x0 = exact*2;
    double x1 = x0 + math::PointNeighborhood(test4(0), x0).dxToPositive();
    result = test4(0)(x1);
    std::cout << "positive value at x="<<x1<<", value="<<result<<"\n";
    ok &= math::isFinite(x1+result) && x1>0 && x1<exact && result>0;
    x0 = exact*0.9;
    x1 = x0 + math::PointNeighborhood(test4(0), x0).dxToNegative();
    result = test4(0)(x1);
    std::cout << "negative value at x="<<x1<<", value="<<result<<"\n";
    ok &= math::isFinite(x1+result) && result<0;
    x1 = x0 + math::PointNeighborhood(test4(1), x0).dxToNegative();
    result = test4(0)(x1);
    std::cout << "(with deriv) negative value at x="<<x1<<", value="<<result<<"\n";
    ok &= math::isFinite(x1+result) && result<0;

    x0 = 1.00009;
    exact = 1.000100000002;
    x1 = x0 + math::PointNeighborhood(test5(), x0).dxToPositive();
    result = test5()(x1);
    std::cout << "f5: positive value at x="<<exact<<"+"<<(x1-exact)<<", value="<<result<<"\n";
    ok &= math::isFinite(x1+result) && result>0;
    numEval=0;
    result = math::findRoot(test5(), 1, x1, toler);
    std::cout << "Root5="<<result<<" (delta="<<(result-exact)<<"; neval="<<numEval<<")\n";
    ok &= fabs(result-exact)<toler*exact;

    exact=1.000109999998;
    numEval=0;
    result = math::findRoot(test5(), x1, INFINITY, toler);
    std::cout << "Another root="<<result<<" (delta="<<(result-exact)<<"; neval="<<numEval<<")\n";
    ok &= fabs(result-exact)<toler*exact;

    // minimization
    numEval=0;
    exact=0.006299605249;
    result = math::findMin(test4(0), 1e-15, 1, NAN, toler);
    std::cout << "Minimum of f4(x) at x="<<result<<" is "<<test4(0)(result)<<
        " (delta="<<(result-exact)<<"; neval="<<numEval<<")\n";
    ok &= fabs(result-exact)<toler*exact;
    numEval=0;
    double xinit[] = {0.5};
    double xstep[] = {0.1};
    double xresult[1];
    int numIter = findMinNdim(test4Ndim(), xinit, xstep, toler, 100, xresult);
    std::cout << "N-dimensional minimization (N=1) of the same function: minimum at x="<<xresult[0]<<
        " is "<<test4(0)(xresult[0])<<" (delta="<<(xresult[0]-exact)<<"; neval="<<numEval<<", nIter="<<numIter<<")\n";
    ok &= fabs(result-exact)<toler;

    numEval=0;
    double yinit[] = {5.0,-4.,2.5};
    double ystep[] = {0.1,0.1,0.1};
    double yresult[3];
    numIter = findMinNdim(test7Ndim(), yinit, ystep, 1e-10, 1000, yresult);
    test7Ndim().eval(yresult, &result);
    std::cout << "N-dimensional minimization (N=3): minimum at x=("<<
        yresult[0]<<","<<yresult[1]<<","<<yresult[2]<<")"
        " is "<<result<<" (neval="<<numEval<<", nIter="<<numIter<<")\n";
    ok &= fabs(yresult[0]-c0) * fabs(yresult[1]-c1) * fabs(yresult[2]-c2) < 1e-10;

    numEval=0;
    double ymin[] = {-4,-4,-2};
    double ymax[] = {+4,+4,+2};
    test8Ndim fnc8;
    exact = 2*pow_2(M_PI*Rin)*Rout;  // volume of a torus
    integrateNdim(fnc8, ymin, ymax, toler, 1000000, &result, &error);
    std::cout << "Volume of a 3d torus = "<<result<<" +- "<<error<<" (delta="<<(result-exact)<<"; neval="<<numEval<<")\n";
    ok &= fabs(result-exact)<error;

    numEval=0;
    math::Matrix<double> points;
    sampleNdim(fnc8, ymin, ymax, 100000, points, NULL, &result, &error);
    std::cout << "Monte Carlo Volume of a 3d torus = "<<result<<" +- "<<error<<" (delta="<<(result-exact)<<"; neval="<<numEval<<")\n";
    ok &= fabs(result-exact)<error*2;  // loose tolerance on MC error estimate
    if(0) {
        std::ofstream fout("torus.dat");
        for(unsigned int i=0; i<points.numRows(); i++)
            fout << points(i,0) << "\t" << points(i,1) << "\t" << points(i,2) << "\n";
    }

#if 0
    for(double p=-40; p<=40; p+=1.77) {
        test_powerlaw tpl(p);
        for(int n=8; n<=32; n*=2) {
            double xmin=1., xmax=1.5;
            result = math::integrateGL(tpl, xmin, xmax, n);
            exact  = tpl.exactValue(xmin, xmax);
            std::cout << "p="<<p<<", N="<<n<<": error("<<xmax<<")="<<(result-exact)/exact;
            xmax=2.0;
            result = math::integrateGL(tpl, xmin, xmax, n);
            exact  = tpl.exactValue(xmin, xmax);
            std::cout << ", error("<<xmax<<")="<<(result-exact)/exact<<"\n";
        }
    }
#endif

    if(ok)
        std::cout << "ALL TESTS PASSED\n";
    return 0;
}
