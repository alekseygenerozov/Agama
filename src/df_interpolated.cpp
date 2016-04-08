#include "df_interpolated.h"
#include "math_core.h"
#include <cmath>
#include <stdexcept>

namespace df{

namespace {
// old vs new method for scaling coord transform
#define COORDCONV 1

/// converts the actions into the internal scaled coordinates on the interpolation grid
static void scaleJ(const actions::Actions &J, const double J0, double gridCoords[3])
{
    double Ltot   = J.Jz + fabs(J.Jphi);
    gridCoords[0] = log(1 + (J.Jr + Ltot) / J0);
    if(gridCoords[0]==0) {
        gridCoords[1] = gridCoords[2] = 0;
        return;
    }
#if COORDCONV==0
    gridCoords[1] = J.Jr / (J.Jr + Ltot);
    gridCoords[2] = (Ltot>0 ? 0.5 * J.Jphi / Ltot : 0) + 0.5;
#else
    gridCoords[1] = J.Jphi>=0 ? J.Jr / (J.Jr + Ltot) : (J.Jr - J.Jphi) / (J.Jr + Ltot);
    gridCoords[2] = J.Jphi>=0 ? (J.Jr + J.Jphi) / (J.Jr + Ltot) : J.Jr / (J.Jr + Ltot);
#endif
}

/// the inverse of scaling procedure implemented in scaleJ
static actions::Actions unscaleJ(const double vars[3], const double J0)
{
    const double s    = vars[0], p = vars[1], q = vars[2];
    const double Jsum = (exp(s) - 1) * J0;
    actions::Actions ac;
#if COORDCONV==0
    ac.Jr   = Jsum * p;
    ac.Jz   = Jsum * (1-p) * (1-fabs(2*q-1));
    ac.Jphi = Jsum * (1-p) * (2*q-1);
#else
    bool up = q>=p;
    ac.Jr   = Jsum * (up? p : q);
    ac.Jz   = Jsum * (up? 1-q : 1-p);
    ac.Jphi = Jsum * (q-p);
#endif
    return ac;
}

/// the jacobian of transformation of actions to scaled coords on the interpolation grid
static double jac(const double vars[3], const double J0)
{
    double Jsum = (exp(vars[0]) - 1) * J0;
#if COORDCONV==0
    return 2 * (1-vars[1]) * Jsum * Jsum * (Jsum+J0);
#else
    return Jsum * Jsum * (Jsum+J0);
#endif
}


/// auxiliary class for computing the phase volume associated with a single component of interpolated DF
template<int N>
class InterpolatedDFintegrand: public math::IFunctionNdim{
    const math::KernelInterpolator3d<N> &interp;
    const double J0;
    const unsigned int indComp;
public:
    InterpolatedDFintegrand(const math::KernelInterpolator3d<N> &_interp,
        const double _J0, const unsigned int _indComp) :
        interp(_interp), J0(_J0), indComp(_indComp) {}
    virtual void eval(const double vars[], double values[]) const {
        // (2pi)^3 comes from integration over angles
        values[0] = TWO_PI_CUBE * interp.valueOfComponent(vars, indComp) * jac(vars, J0);
    }
    virtual unsigned int numVars()   const { return 3; }
    virtual unsigned int numValues() const { return 1; }
};

/// auxiliary class for collecting the values of source DF at grid points in scaled action space
class DFscaled: public math::IFunctionNdim{
    const BaseDistributionFunction &df;
    const double J0;
public:
    DFscaled(const BaseDistributionFunction &_df, const double _J0) :
        df(_df), J0(_J0) {}
    virtual void eval(const double vars[], double values[]) const {
        values[0] = df.value(unscaleJ(vars, J0));
    }
    virtual unsigned int numVars()   const { return 3; }
    virtual unsigned int numValues() const { return 1; }
};
}  // internal namespace

template<int N>
InterpolatedDF<N>::InterpolatedDF(const InterpolatedDFParam &params) :
    interp(params.gridJsum, params.gridJrrel, params.gridJphirel),
    amplitudes(params.amplitudes), J0(params.J0)
{
    if(J0<=0)
        throw std::invalid_argument("InterpolatedDF: J0 must be positive");
    if(amplitudes.size() != interp.numValues())
        throw std::invalid_argument("InterpolatedDF: invalid array size");
    for(unsigned int i=0; i<amplitudes.size(); i++)
        if(amplitudes[i] < 0 || !math::isFinite(amplitudes[i]))
            throw std::invalid_argument("InterpolatedDF: amplitudes must be non-negative");
}
    
template<int N>
double InterpolatedDF<N>::value(const actions::Actions &J) const
{
    double gridC[3];
    scaleJ(J, J0, gridC);
    return interp.interpolate(gridC, amplitudes);
}

template<int N>
double InterpolatedDF<N>::valueOfComponent(const actions::Actions &J, unsigned int indComp) const
{
    double gridC[3];
    scaleJ(J, J0, gridC);
    return interp.valueOfComponent(gridC, indComp) * amplitudes.at(indComp);
}

template<int N>
void InterpolatedDF<N>::valuesOfAllComponents(const actions::Actions &J, double val[]) const
{
    double gridC[3];
    scaleJ(J, J0, gridC);
    interp.eval(gridC, val);
    for(unsigned int i=0; i<amplitudes.size(); i++)
        val[i] *= amplitudes[i];
}

template<int N>
double InterpolatedDF<N>::computePhaseVolume(const unsigned int indComp, const double reqRelError) const
{
    if(indComp >= amplitudes.size())
        throw std::range_error("InterpolatedDF: component index out of range");
    double xlower[3], xupper[3];
    interp.nonzeroDomain(indComp, xlower, xupper);
    double result, error;
    const int maxNumEval = 10000;
    math::integrateNdim(InterpolatedDFintegrand<N>(interp, J0, indComp), xlower, xupper, 
        reqRelError, maxNumEval, &result, &error);
    return result;
}

template<int N>
InterpolatedDFParam createInterpolatedDFParam(
    const BaseDistributionFunction& df, unsigned int gridSize[3], double J0, double Jmax)
{
    if(Jmax<=0 || J0<=0)
        throw std::invalid_argument("createInterpolatedDFParams: incorrect J");
    InterpolatedDFParam param;
    param.J0 = J0;
    param.gridJsum    = math::createUniformGrid(gridSize[0], 0, log(1 + Jmax/J0));
    param.gridJrrel   = math::createUniformGrid(gridSize[1], 0, 1);
    param.gridJphirel = math::createUniformGrid(gridSize[2], 0, 1);
    param.amplitudes  = math::createInterpolator3dArray<N>(DFscaled(df, J0), 
        param.gridJsum, param.gridJrrel, param.gridJphirel);
    return param;
}

template<int N>
InterpolatedDFParam createInterpolatedDFParamFromActionSamples(
    const std::vector<actions::Actions>& actions, const std::vector<double>& masses,
    unsigned int gridSize[3], double J0, double Jmax)
{
    if(Jmax<=0 || J0<=0)
        throw std::invalid_argument("createInterpolatedDFParamsFromActionSamples: incorrect J");
    if(actions.size() != masses.size())
        throw std::invalid_argument("createInterpolatedDFParamsFromActionSamples: "
            "incorrect size of input arrays");
    math::Matrix<double> points(masses.size(), 3);
    std::vector<double> weights(masses.size());
    for(unsigned int i=0; i<masses.size(); i++) {
        scaleJ(actions[i], J0, &points(i, 0));
        weights[i] = 1 / jac(&(points(i, 0)), J0);
    }
    InterpolatedDFParam param;
    param.J0 = J0;
    param.gridJsum    = math::createUniformGrid(gridSize[0], 0, log(1 + Jmax/J0));
    param.gridJrrel   = math::createUniformGrid(gridSize[1], 0, 1);
    param.gridJphirel = math::createUniformGrid(gridSize[2], 0, 1);
    param.amplitudes  = math::createInterpolator3dArrayFromSamples<N>(points, weights, 
        param.gridJsum, param.gridJrrel, param.gridJphirel);
    return param;
}

// force the compilation of template instantiations
template class InterpolatedDF<1>;
template class InterpolatedDF<3>;

template InterpolatedDFParam createInterpolatedDFParam<1>(
    const BaseDistributionFunction& df, unsigned int gridSize[3], double J0, double Jmax);
template InterpolatedDFParam createInterpolatedDFParam<3>(
    const BaseDistributionFunction& df, unsigned int gridSize[3], double J0, double Jmax);

template InterpolatedDFParam createInterpolatedDFParamFromActionSamples<1>(
    const std::vector<actions::Actions>& actions, const std::vector<double>& masses,
    unsigned int gridSize[3], double J0, double Jmax);
template InterpolatedDFParam createInterpolatedDFParamFromActionSamples<3>(
    const std::vector<actions::Actions>& actions, const std::vector<double>& masses,
    unsigned int gridSize[3], double J0, double Jmax);

}  // namespace df
