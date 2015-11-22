#include "df_factory.h"
#include "df_disk.h"
#include "df_halo.h"
#include "utils.h"
#include "utils_config.h"
#include <cassert>
#include <stdexcept>

namespace df {

static DoublePowerLawParam parseDoublePowerLawParams(
    const utils::KeyValueMap& kvmap,
    const units::ExternalUnits& conv)
{
    DoublePowerLawParam par;
    par.norm  = kvmap.getDouble("norm")  * conv.massUnit;
    par.j0    = kvmap.getDouble("j0")    * conv.lengthUnit * conv.velocityUnit;
    par.jcore = kvmap.getDouble("jcore") * conv.lengthUnit * conv.velocityUnit;
    par.alpha = kvmap.getDouble("alpha");
    par.beta  = kvmap.getDouble("beta");
    par.ar    = kvmap.getDouble("ar", 1.);
    par.az    = kvmap.getDouble("az", 1.);
    par.aphi  = kvmap.getDouble("aphi", 1.);
    par.br    = kvmap.getDouble("br", 1.);
    par.bz    = kvmap.getDouble("bz", 1.);
    par.bphi  = kvmap.getDouble("bphi", 1.);
    return par;
}

/*static PseudoIsothermalParam parsePseudoIsothermalParams(
    const utils::KeyValueMap& kvmap,
    const units::ExternalUnits& conv)
{
    PseudoIsothermalParam par;
    return par;
}*/

PtrDistributionFunction createDistributionFunction(
    const utils::KeyValueMap& kvmap,
    const units::ExternalUnits& converter)
{
    std::string type = kvmap.getString("type");
    if(utils::stringsEqual(type, "DoublePowerLaw")) {
        DoublePowerLawParam params = parseDoublePowerLawParams(kvmap, converter);
        return PtrDistributionFunction(new DoublePowerLaw(params));
    }
    /*else if(utils::stringsEqual(type, "PseudoIsothermal")) {
        PseudoIsothermalParam params = parsePseudoIsothermalParams(kvmap, converter);
        return PtrDistributionFunction(new PseudoIsothermal(params));
    }*/
    else
        throw std::invalid_argument("Unknown type of distribution function");
}

}; // namespace
