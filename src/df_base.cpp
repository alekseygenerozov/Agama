#include "df_base.h"
#include "math_core.h"
#include "math_sample.h"
#include <cmath>

namespace df{

/// convert from scaled variables to the actual actions to be passed to DF
/// if jac!=NULL, store the value of jacobian of transformation in this variable
actions::Actions unscaleActions(const double vars[], double* jac)
{
    // scaled variables p, q and s lie in the range [0:1];
    const double s = vars[0], p = vars[1], q = vars[2];
    const double Jsum = exp( 1/(1-s) - 1/s );  // = Jr+Jz+|Jphi|
#if 0
    if(jac)
        *jac  = math::withinReasonableRange(Jsum) ?   // if near J=0 or infinity, set jacobian to zero
            2*(1-p) * pow_3(Jsum) * (1/pow_2(1-s) + 1/pow_2(s)) : 0;
    actions::Actions acts;
    acts.Jr   = Jsum * p;
    acts.Jphi = Jsum * (1-p) * (2*q-1);
    acts.Jz   = Jsum * (1-p) * (1-fabs(2*q-1));
#else
    if(jac)
        *jac  = math::withinReasonableRange(Jsum) ?   // if near J=0 or infinity, set jacobian to zero
            pow_3(Jsum) * (1/pow_2(1-s) + 1/pow_2(s)) : 0;
    actions::Actions acts;
    bool up   = q>=p;
    acts.Jr   = Jsum * (up? p : q);
    acts.Jz   = Jsum * (up? 1-q : 1-p);
    acts.Jphi = Jsum * (q-p);
#endif
    return acts;
}

/// helper class for computing the integral of distribution function f
/// or f * ln(f)  if LogTerm==true, in scaled coords in action space.
template <bool LogTerm>
class DFIntegrandNdim: public math::IFunctionNdim {
public:
    DFIntegrandNdim(const BaseDistributionFunction& _df) :
        df(_df) {};

    /// compute the value of DF, taking into accound the scaling transformation for actions:
    /// input array of length 3 contains the three actions, scaled as described above;
    /// output a single value (DF multiplied by the jacobian of scaling transformation)
    virtual void eval(const double vars[], double values[]) const
    {
        double jac;  // will be initialized by the following call
        const actions::Actions act = unscaleActions(vars, &jac);
        double val = 0;
        if(jac!=0) {
            double dfval = df.value(act);
            if(LogTerm && dfval>0)
                dfval *= log(dfval);
            val = dfval * jac * TWO_PI_CUBE;   // integral over three angles
        } else {
            // we're (almost) at zero or infinity in terms of magnitude of J
            // at infinity we expect that f(J) tends to zero,
            // while at J->0 the jacobian of transformation is exponentially small.
        }            
        values[0] = val;
    }

    /// number of variables (3 actions)
    virtual unsigned int numVars()   const { return 3; }
    /// number of values to compute (1 value of DF)
    virtual unsigned int numValues() const { return 1; }
private:
    const BaseDistributionFunction& df;  ///< the instance of DF
};

double BaseDistributionFunction::totalMass(const double reqRelError, const int maxNumEval,
    double* error, int* numEval) const
{
    double xlower[3] = {0, 0, 0};  // boundaries of integration region in scaled coordinates
    double xupper[3] = {1, 1, 1};
    double result;  // store the value of integral
    math::integrateNdim(DFIntegrandNdim<false>(*this), xlower, xupper, 
        reqRelError, maxNumEval, &result, error, numEval);
    return result;
}

double totalEntropy(const BaseDistributionFunction& DF, const double reqRelError, const int maxNumEval)
{
    double xlower[3] = {0, 0, 0};
    double xupper[3] = {1, 1, 1};
    double result;
    math::integrateNdim(DFIntegrandNdim<true>(DF), xlower, xupper, reqRelError, maxNumEval, &result);
    return result;
}

void sampleActions(const BaseDistributionFunction& DF, const int numSamples,
    std::vector<actions::Actions>& samples, double* totalMass, double* totalMassErr)
{
    double xlower[3] = {0, 0, 0};  // boundaries of integration region in scaled coordinates
    double xupper[3] = {1, 1, 1};
    math::Matrix<double> result;   // the result array of actions
    DFIntegrandNdim<false> fnc(DF);
    math::sampleNdim(fnc, xlower, xupper, numSamples, result, 0, totalMass, totalMassErr);
    samples.resize(result.rows());
    for(unsigned int i=0; i<result.rows(); i++) {
        const double point[3] = {result(i,0), result(i,1), result(i,2)};
        samples[i] = unscaleActions(point);  // transform from scaled vars to actions
    }
}

}
