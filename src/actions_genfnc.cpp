#include "actions_genfnc.h"
#include <cmath>

namespace actions{

ActionAngles ActionMap::map(const ActionAngles& actAng) const
{
    ActionAngles aa(actAng);
    for(unsigned int i=0; i<indices.size(); i++) {
        double val = values[i] * cos(indices[i].mr * aa.thetar +
            indices[i].mz * aa.thetaz + indices[i].mphi * aa.thetaphi);
        aa.Jr  += val * indices[i].mr;
        aa.Jz  += val * indices[i].mz;
        aa.Jphi+= val * indices[i].mphi;
    }
    return aa;
}

GenFncFit::GenFncFit(const GenFncIndices& _indices,
    const Actions& _acts, const std::vector<Angles>& _angs) :
    indices(_indices), acts(_acts), angs(_angs), coefs(angs.size(), indices.size())
{
    for(unsigned int indexAngle=0; indexAngle<angs.size(); indexAngle++)
        for(unsigned int indexCoef=0; indexCoef<indices.size(); indexCoef++)
            coefs(indexAngle, indexCoef) = cos(
                indices[indexCoef].mr * angs[indexAngle].thetar +
                indices[indexCoef].mz * angs[indexAngle].thetaz +
                indices[indexCoef].mphi * angs[indexAngle].thetaphi);
}

ActionAngles GenFncFit::toyActionAngles(unsigned int indexAngle, const double values[]) const
{
    ActionAngles aa(acts, angs[indexAngle]);
    for(unsigned int indexCoef=0; indexCoef<indices.size(); indexCoef++) {
        double val = values[indexCoef] * coefs(indexAngle, indexCoef);
        aa.Jr  += val * indices[indexCoef].mr;
        aa.Jz  += val * indices[indexCoef].mz;
        aa.Jphi+= val * indices[indexCoef].mphi;
    }
    return aa;
}

}  // namespace actions
