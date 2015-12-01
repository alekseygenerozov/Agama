/** \file    df_factory.h
    \brief   Creation of DistributionFunction instances
    \author  EV
    \date    2015
*/

#pragma once
#include "df_base.h"
#include "units.h"
#include "smart.h"

// forward declaration
namespace utils { class KeyValueMap; }

namespace df {

/** A trivial collection of several distribution functions */
class CompositeDF: public BaseDistributionFunction{
public:
    CompositeDF(const std::vector<PtrDistributionFunction> &comps) :
        components(comps) {};
    unsigned int size() const { return components.size(); }
    PtrDistributionFunction component(unsigned int index) const {
        return components.at(index); }
    virtual double value(const actions::Actions &J) const {
        double sum=0;
        for(unsigned int i=0; i<components.size(); i++) 
            sum += components[i]->value(J); 
        return sum;
    }    
private:
    std::vector<PtrDistributionFunction> components;
};

/** Create an instance of distribution function according to the parameters contained in the key-value map.
    \param[in] params     is the list of parameters (pairs of "key" and "value" strings).
    \param[in] potential  is the instance of global potential, required by some of the DF classes
    at construction (it is not stored in the DF but only used to initialize internal interpolation tables);
    may pass an empty pointer if the DF does not need it.
    \param[in] converter  is the unit converter for transforming the dimensional quantities 
    in parameters into internal units; can be a trivial converter.
    \return    a new instance of shared pointer to BaseDistributionFunction on success.
    \throws    std::invalid_argument or std::runtime_error or other df-specific exception on failure.
*/
PtrDistributionFunction createDistributionFunction(
    const utils::KeyValueMap& params,
    const potential::BasePotential* potential = NULL,
    const units::ExternalUnits& converter = units::ExternalUnits());

}; // namespace
