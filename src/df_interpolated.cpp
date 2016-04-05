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
}

template<int N>
InterpolatedDF<N>::InterpolatedDF(const InterpolatedDFParam &params) :
    interp(params.gridJsum, params.gridJrrel, params.gridJphirel),
    values(params.values), J0(params.J0)
{
    if(J0<=0)
        throw std::invalid_argument("InterpolatedDF: J0 must be positive");
    if(values.size() != interp.numValues())
        throw std::invalid_argument("InterpolatedDF: invalid array size");
}
    
template<int N>
double InterpolatedDF<N>::value(const actions::Actions &J) const
{
    double gridC[3];
    scaleJ(J, J0, gridC);
    return interp.interpolate(gridC, values);
}

template<int N>
double InterpolatedDF<N>::valueOfComponent(const actions::Actions &J, unsigned int indComp) const
{
    double gridC[3];
    scaleJ(J, J0, gridC);
    return interp.valueOfComponent(gridC, indComp);
}

template<int N>
void InterpolatedDF<N>::valuesOfAllComponents(const actions::Actions &J, double val[]) const
{
    double gridC[3];
    scaleJ(J, J0, gridC);
    interp.eval(gridC, val);
    for(unsigned int i=0; i<values.size(); i++)
        val[i] *= values[i];
}

template<int N>
double InterpolatedDF<N>::computePhaseVolume(const unsigned int indComp) const
{
    if(indComp>=values.size())
        throw std::range_error("InterpolatedDF: component index out of range");
    return NAN;  // not yet implemented
}

// force the compilation of template instantiations
template class InterpolatedDF<1>;
template class InterpolatedDF<3>;

InterpolatedDFParam createInterpolatedDFParam(
    const BaseDistributionFunction& df, unsigned int gridSize[3], double J0, double Jmax)
{
    if(Jmax<=0 || J0<=0)
        throw std::invalid_argument("createInterpolatedDFParams: incorrect J");
    InterpolatedDFParam param;
    param.gridJsum    = math::createUniformGrid(gridSize[0], 0, log(1 + Jmax/J0));
    param.gridJrrel   = math::createUniformGrid(gridSize[1], 0, 1);
    param.gridJphirel = math::createUniformGrid(gridSize[2], 0, 1);
    math::KernelInterpolator3d<1> interp(param.gridJsum, param.gridJrrel, param.gridJphirel);
    param.values.assign(interp.numValues(), 0);
    param.J0 = J0;
    for(unsigned int i=0; i<gridSize[0]; i++)
        for(unsigned int j=0; j<gridSize[1]; j++)
            for(unsigned int k=0; k<gridSize[2]; k++) {
                double coord[3] = { param.gridJsum[i], param.gridJrrel[j], param.gridJphirel[k] };
                actions::Actions J = unscaleJ(coord, J0);
                param.values[interp.indComp(i, j, k)] = df.value(J);
            }
    return param;
}

}  // namespace df
