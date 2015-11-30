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

/** Create an instance of distribution function according to the parameters contained in the key-value map.
    \param[in] params is the list of parameters;
    \param[in] converter is the unit converter for transforming the dimensional quantities 
    in parameters into internal units; can be a trivial converter.
    \return    a new instance of BaseDistributionFunction* on success.
    \throws    std::invalid_argument or std::runtime_error or other df-specific exception on failure.
*/
PtrDistributionFunction createDistributionFunction(
    const utils::KeyValueMap& params,
    const units::ExternalUnits& converter = units::ExternalUnits());

}; // namespace
