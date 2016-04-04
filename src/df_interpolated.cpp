#include "df_interpolated.h"
#include "math_core.h"
#include <cmath>
#include <stdexcept>

namespace df{

namespace {
/// the inverse of scaling procedure implemented in InterpolatedDF::scaleJ
static actions::Actions unscaleJ(const double vars[3], const double Jscale)
{
    const double s  = vars[0], p = vars[1], q = vars[2];
    const double J0 = (exp(s) - 1) * Jscale;
    actions::Actions ac;
    ac.Jr   = J0 * p;
    ac.Jphi = J0 * (1-p) * (2*q-1);
    ac.Jz   = J0 * (1-p) * (1-fabs(2*q-1));
    return ac;
}
}

template<int N>
void InterpolatedDF<N>::scaleJ(const actions::Actions &J, double gridCoords[3]) const
{
    double Ltot   = J.Jz + fabs(J.Jphi);
    gridCoords[0] = log(1 + (J.Jr + Ltot) / Jscale);
    if(gridCoords[0]==0) {
        gridCoords[1] = gridCoords[2] = 0;
        return;
    }
    gridCoords[1] = J.Jr / (J.Jr + Ltot);
    gridCoords[2] = (Ltot>0 ? 0.5 * J.Jphi / Ltot : 0) + 0.5;
}

template<int N>
double InterpolatedDF<N>::value(const actions::Actions &J) const
{
    double gridC[3];
    scaleJ(J, gridC);
    double val;
    interp.eval(gridC, &val);
    return math::isFinite(val) ? val : 0;  // val=NAN if outside the grid, replaced by zero
}

template<int N>
double InterpolatedDF<N>::valueOfComponent(const actions::Actions &J, unsigned int index) const
{
    double gridC[3];
    scaleJ(J, gridC);
    unsigned int leftInd[3];
    double weights[(N+1)*(N+1)*(N+1)];
    interp.components(gridC, leftInd, weights);
    if(!math::isFinite(weights[0]))  // J is outside the grid
        return 0;
    const unsigned int
        ysize = interp.yvalues().size(),
        zsize = interp.zvalues().size(),
        ind_k = index % zsize,
        ind_j = index / zsize % ysize,
        ind_i = index / zsize / ysize;
    if( ind_i>=leftInd[0] && ind_i<=leftInd[0]+N &&
        ind_j>=leftInd[1] && ind_j<=leftInd[1]+N &&
        ind_k>=leftInd[2] && ind_k<=leftInd[2]+N )
        return weights[(ind_i * (N+1) + ind_j) * (N+1) + ind_k] * interp.fncvalues()[index];
    else
        return 0;
}

template<int N>
void InterpolatedDF<N>::valuesOfAllComponents(const actions::Actions &J, double values[]) const
{
    for(unsigned int i=0; i<numComp; i++)
        values[i] = 0;
    double gridC[3];
    scaleJ(J, gridC);
    const unsigned int ysize = interp.yvalues().size();
    const unsigned int zsize = interp.zvalues().size();
    unsigned int leftInd[3];
    double weights[(N+1)*(N+1)*(N+1)];
    interp.components(gridC, leftInd, weights);
    if(!math::isFinite(weights[0]))  // J is outside the grid - all components remain zero
        return;
    for(int i=0; i<=N; i++)
        for(int j=0; j<=N; j++)
            for(int k=0; k<=N; k++)
            {
                int index = ((i+leftInd[0]) * ysize + j+leftInd[1]) * zsize + k+leftInd[2];
                values[index] += weights[(i * (N+1) + j) * (N+1) + k] * interp.fncvalues()[index];
            }
}

// force the compilation of template instantiations
template class InterpolatedDF<1>;
template class InterpolatedDF<3>;

InterpolatedDFParam createInterpolatedDFParam(
    const BaseDistributionFunction& df, unsigned int gridSize[3], double Jscale, double Jmax)
{
    if(Jmax<=0 || Jscale<=0)
        throw std::invalid_argument("createInterpolatedDFParams: incorrect J");
    InterpolatedDFParam param;
    param.gridJsum    = math::createUniformGrid(gridSize[0], 0, log(1 + Jmax/Jscale));
    param.gridJrrel   = math::createUniformGrid(gridSize[1], 0, 1);
    param.gridJphirel = math::createUniformGrid(gridSize[2], 0, 1);
    param.values.resize(gridSize[0] * gridSize[1] * gridSize[2]);
    param.Jscale = Jscale;
    for(unsigned int i=0; i<gridSize[0]; i++)
        for(unsigned int j=0; j<gridSize[1]; j++)
            for(unsigned int k=0; k<gridSize[2]; k++) {
                double coord[3] = {param.gridJsum[i], param.gridJrrel[j], param.gridJphirel[k]};
                actions::Actions J = unscaleJ(coord, Jscale);
                unsigned int index = (i * gridSize[1] + j) * gridSize[2] + k;
                param.values[index] = df.value(J);
            }
    return param;
}

}  // namespace df
