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
    par.jmax  = kvmap.getDouble("jmax")  * conv.lengthUnit * conv.velocityUnit;
    par.alpha = kvmap.getDouble("alpha", par.alpha);
    par.beta  = kvmap.getDouble("beta",  par.beta);
    par.ar    = kvmap.getDouble("ar",    par.ar);
    par.az    = kvmap.getDouble("az",    par.az);
    par.aphi  = kvmap.getDouble("aphi",  par.aphi);
    par.br    = kvmap.getDouble("br",    par.br);
    par.bz    = kvmap.getDouble("bz",    par.bz);
    par.bphi  = kvmap.getDouble("bphi",  par.bphi);
    par.b     = kvmap.getDouble("b",     par.b);
    return par;
}

static PseudoIsothermalParam parsePseudoIsothermalParams(
    const utils::KeyValueMap& kvmap,
    const units::ExternalUnits& conv)
{
    PseudoIsothermalParam par;
    par.Sigma0  = kvmap.getDouble("Sigma0")  * conv.massUnit / pow_2(conv.lengthUnit);
    par.Rdisk   = kvmap.getDouble("Rdisk")   * conv.lengthUnit;
    par.Jphimin = kvmap.getDouble("Jphimin") * conv.lengthUnit * conv.velocityUnit;
    par.Jphi0   = kvmap.getDouble("Jphi0")   * conv.lengthUnit * conv.velocityUnit;
    par.sigmar0 = kvmap.getDouble("sigmar0") * conv.velocityUnit;
    par.sigmaz0 = kvmap.getDouble("sigmaz0") * conv.velocityUnit;
    par.sigmamin= kvmap.getDouble("sigmamin")* conv.velocityUnit;
    par.Rsigmar = kvmap.getDouble("Rsigmar", 2*par.Rdisk) * conv.lengthUnit;
    par.Rsigmaz = kvmap.getDouble("Rsigmaz", 2*par.Rdisk) * conv.lengthUnit;
    par.beta    = kvmap.getDouble("beta", par.beta);
    par.Tsfr    = kvmap.getDouble("Tsfr", par.Tsfr);  // dimensionless! in units of Hubble time
    par.sigmabirth = kvmap.getDouble("sigmabirth", par.sigmabirth);  // dimensionless ratio
    return par;
}

static void checkNonzero(const potential::BasePotential* potential, const std::string& type)
{
    if(potential == NULL)
        throw std::invalid_argument("Need an instance of potential to initialize "+type+" DF");
}

PtrDistributionFunction createDistributionFunction(
    const utils::KeyValueMap& kvmap,
    const potential::BasePotential* potential,
    const units::ExternalUnits& converter)
{
    std::string type = kvmap.getString("type");
    if(utils::stringsEqual(type, "DoublePowerLaw")) {
        DoublePowerLawParam params = parseDoublePowerLawParams(kvmap, converter);
        return PtrDistributionFunction(new DoublePowerLaw(params));
    }
    else if(utils::stringsEqual(type, "DoublePowerLawSph")) {
        checkNonzero(potential, type);
        DoublePowerLawParam params = parseDoublePowerLawParams(kvmap, converter);
        return PtrDistributionFunction(new DoublePowerLawSph(
            params, potential::Interpolator(*potential)));
    }
    else if(utils::stringsEqual(type, "PseudoIsothermal")) {
        checkNonzero(potential, type);
        PseudoIsothermalParam params = parsePseudoIsothermalParams(kvmap, converter);
        return PtrDistributionFunction(new PseudoIsothermal(
            params, potential::Interpolator(*potential)));
    }
    else
        throw std::invalid_argument("Unknown type of distribution function");
}

}; // namespace
