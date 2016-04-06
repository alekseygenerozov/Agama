#include "df_interpolated.h"
#include "math_core.h"
#include <cmath>
#include <stdexcept>

namespace df{

namespace {

/// converts the actions into the internal scaled coordinates on the interpolation grid
static void scaleJ(const actions::Actions &J, const double J0, double gridCoords[3])
{
    double Ltot   = J.Jz + fabs(J.Jphi);
    gridCoords[0] = log(1 + (J.Jr + Ltot) / J0);
    if(gridCoords[0]==0) {
        gridCoords[1] = gridCoords[2] = 0;
        return;
    }
    gridCoords[1] = J.Jr / (J.Jr + Ltot);
    gridCoords[2] = (Ltot>0 ? 0.5 * J.Jphi / Ltot : 0) + 0.5;
}

/// the inverse of scaling procedure implemented in scaleJ
static actions::Actions unscaleJ(const double vars[3], const double J0)
{
    const double s    = vars[0], p = vars[1], q = vars[2];
    const double Jsum = (exp(s) - 1) * J0;
    actions::Actions ac;
    ac.Jr   = Jsum * p;
    ac.Jphi = Jsum * (1-p) * (2*q-1);
    ac.Jz   = Jsum * (1-p) * (1-fabs(2*q-1));
    return ac;
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
        /// Note: these expressions must correspond to the ones used in scaleJ/unscaleJ
        const double s = vars[0], p = vars[1],
        Jsum = (exp(s) - 1) * J0,
        jac  = TWO_PI_CUBE * 2 * (1-p) * Jsum * Jsum * (Jsum+J0);
        // (2pi)^3 comes from integration over angles
        values[0] = interp.valueOfComponent(vars, indComp) * jac;
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
    param.gridJsum    = math::createUniformGrid(gridSize[0], 0, log(1 + Jmax/J0));
    param.gridJrrel   = math::createUniformGrid(gridSize[1], 0, 1);
    param.gridJphirel = math::createUniformGrid(gridSize[2], 0, 1);
    param.amplitudes  = math::createInterpolator3dArray<N>(DFscaled(df, J0), 
        param.gridJsum, param.gridJrrel, param.gridJphirel);
    param.J0 = J0;
    return param;
}
    
// force the compilation of template instantiations
template class InterpolatedDF<1>;
template class InterpolatedDF<3>;

template
InterpolatedDFParam createInterpolatedDFParam<1>(
    const BaseDistributionFunction& df, unsigned int gridSize[3], double J0, double Jmax);
template
InterpolatedDFParam createInterpolatedDFParam<3>(
    const BaseDistributionFunction& df, unsigned int gridSize[3], double J0, double Jmax);

}  // namespace df
