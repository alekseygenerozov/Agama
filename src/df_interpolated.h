/** \file    df_interpolated.h
    \brief   Distribution function specified in the interpolated form
    \date    2016
    \author  Eugene Vasiliev
*/
#pragma once
#include "df_base.h"
#include "math_spline.h"

namespace df{

/// Parameters that describe an interpolated distribution function.
struct InterpolatedDFParam{
    double Jscale;
    std::vector<double> gridJsum;
    std::vector<double> gridJrrel;
    std::vector<double> gridJphirel;
    std::vector<double> values;
};

template<int N>
class InterpolatedDF: public BaseMulticomponentDF{
public:
    /** Create an instance of interpolated distribution function with given parameters
        \param[in] params  are the parameters of DF
        \throws std::invalid_argument exception if parameters are nonsense
    */
    InterpolatedDF(const InterpolatedDFParam &params) :
        numComp(params.gridJsum.size() * params.gridJrrel.size() * params.gridJphirel.size()),
        Jscale(params.Jscale),
        interp(params.gridJsum, params.gridJrrel, params.gridJphirel, params.values)
    {};

    virtual double value(const actions::Actions &J) const;

    virtual unsigned int size() const { return numComp; }

    virtual double valueOfComponent(const actions::Actions &J, unsigned int index) const;

    virtual void valuesOfAllComponents(const actions::Actions &J, double values[]) const;

private:
    /// the total number of nodes on the 3d interpolation grid (stored for convenience)
    const unsigned int numComp;
    
    const double Jscale;

    /// the interpolated function defined on the scaled grid
    const math::BaseInterpolator3d<N> interp;

    /// converts the actions into the internal scaled coordinates on the interpolation grid
    void scaleJ(const actions::Actions &J, double gridCoords[3]) const;
};

InterpolatedDFParam createInterpolatedDFParam(
    const BaseDistributionFunction& df, unsigned int gridSize[3], double Jscale, double Jmax);

}  // namespace df
