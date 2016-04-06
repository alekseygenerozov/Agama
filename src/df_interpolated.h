/** \file    df_interpolated.h
    \brief   Distribution function specified in the interpolated form
    \date    2016
    \author  Eugene Vasiliev
*/
#pragma once
#include "df_base.h"
#include "math_spline.h"

namespace df{

/** Parameters that describe an interpolated distribution function.
    It is represented on a 3d grid in scaled coordinates in the action space,
    where the first variable corresponds to the total action (sum of three actions)
    with a logarithmic transformation:  x = log(1 + (Jr+Jz+|Jphi|) / J0),
    and the other two variables range from 0 to 1 and determine the ratio between actions.
    The array of amplitudes defines the value of interpolated DF;
    in the case of linear interpolation, the amplitudes correspond to the values of DF at grid nodes,
    but in the case of cubic interpolation there is no straightforward correspondence between them;
    moreover, in the latter case the number of components (elements in the amplitudes array)
    is larger than the number of nodes in the 3d grid.
*/
struct InterpolatedDFParam{
    double J0;                      ///< scale action used for log transformation of the grid
    std::vector<double> gridJsum;   ///< grid in the log-transformed magnitude of action Jsum
    std::vector<double> gridJrrel;  ///< grid in the direction Jr/Jsum
    std::vector<double> gridJphirel;///< grid in the direction Jphi/L
    std::vector<double> amplitudes; ///< the amplitudes of interpolation kernels
};

template<int N>
class InterpolatedDF: public BaseMulticomponentDF{
public:
    /** Create an instance of interpolated distribution function with given parameters
        \param[in] params  are the parameters of DF
        \throws std::invalid_argument exception if parameters are nonsense
    */
    InterpolatedDF(const InterpolatedDFParam &params);

    /// the value of interpolated DF at the given actions
    virtual double value(const actions::Actions &J) const;

    /// the number of components in the interpolation array
    virtual unsigned int size() const { return amplitudes.size(); }

    /// the value of a single component at the given actions
    virtual double valueOfComponent(const actions::Actions &J, unsigned int indComp) const;

    /// values of all components at the given actions reported separately
    virtual void valuesOfAllComponents(const actions::Actions &J, double values[]) const;

    /** Compute the phase volume associated with the given component.
        The volume is given by the integral of interpolation kernel associated with this
        component over actions, multiplied by (2pi)^3 which is the integral over angles.
        The sum of products of component amplitudes times their phase volumes is equal
        to the integral of the DF over the entire action/angle space, i.e. the total mass.
        \param[in]  indComp  is the index of component, 0 <= indComp < size();
        \param[in]  reqRelError is the required accuracy (relative error);
        \return  the phase volume;
        \throw   std::range_error if the index is out of range.
    */
    double computePhaseVolume(const unsigned int indComp, const double reqRelError=1e-3) const;

private:
    /// the interpolator defined on the scaled grid in action space
    const math::KernelInterpolator3d<N> interp;

    /// the amplitudes of 3d interpolation kernels
    const std::vector<double> amplitudes;

    /// characteristic value of actions used for scaling
    const double J0;
};

/** Initialize the parameters used to create an interpolated DF by collecting the values
    of the provided source DF at the nodes of a 3d grid in action space.
*/
template<int N>
InterpolatedDFParam createInterpolatedDFParam(
    const BaseDistributionFunction& df, unsigned int gridSize[3], double J0, double Jmax);

}  // namespace df
