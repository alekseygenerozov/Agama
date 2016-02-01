#include "actions_isochrone.h"
#include "math_core.h"
#include <stdexcept>
#include <cmath>

namespace actions{

Actions actionsIsochrone(
    const double M, const double b,
    const coord::PosVelCyl& point)
{
    double L = Ltotal(point);
    double E = -M / (b + sqrt(b*b + pow_2(point.R) + pow_2(point.z))) +
        0.5 * (pow_2(point.vR) + pow_2(point.vz) + pow_2(point.vphi));
    Actions acts;
    acts.Jphi = Lz(point);
    acts.Jz   = L - fabs(acts.Jphi);
    acts.Jr   = M / sqrt(-2*E) - 0.5 * (L + sqrt(L*L + 4*M*b));
    return acts;
}

ActionAngles actionAnglesIsochrone(
    const double M, const double b,
    const coord::PosVelCyl& pointCyl,
    Frequencies* freq)
{
    coord::PosVelSph point(toPosVelSph(pointCyl));
    ActionAngles aa;
    double rb   = sqrt(b*b + pow_2(point.r));
    double L    = Ltotal(point);
    double Lb   = sqrt(L*L + 4*M*b);
    double LLb  = 0.5 * (1 + L/Lb);
    double E    = -M / (b+rb) + 0.5 * (pow_2(point.vr) + pow_2(point.vtheta) + pow_2(point.vphi));
    double sq2E = sqrt(-2*E);
    aa.Jphi     = coord::Lz(pointCyl);
    aa.Jz       = L - fabs(aa.Jphi);
    aa.Jr       = M / sq2E - Lb * LLb;
    double M2Eb = M + 2*E * b;
    double ecc  = sqrt(1 + 2*E * pow_2(L / M2Eb)); // eccentricity
    double fac1 =  M2Eb * (1+ecc) / (sq2E * L);    // sqrt( (1+ecc) / (1-ecc) );
    double fac2 = (M2Eb * (1+ecc) - 4*E*b) / (sq2E * Lb);
    // below are quantities that depend on position along the orbit
    double k1   = sq2E * point.r * point.vr;
    double k2   = M + 2*E * rb;
    double eta  = atan2(k1, k2);
    double sineta     = k1 / sqrt(k1*k1 + k2*k2);  // sin(eta)
    double tanhalfeta = -k2/k1 + 1/sineta;         // tan(eta/2)
    double psi  = atan2(pointCyl.z * L,  -pointCyl.R * point.vtheta * point.r);
    double chi  = atan2(pointCyl.z * point.vphi, point.vtheta * point.r) + M_PI;
    aa.thetar   = math::wrapAngle(eta - sineta * ecc * M2Eb/M);
    aa.thetaz   = math::wrapAngle(psi + LLb * (aa.thetar - (eta<0 ? 2*M_PI : 0))
                - atan(fac1 * tanhalfeta) - atan(fac2 * tanhalfeta) * L/Lb );
    aa.thetaphi = math::wrapAngle(point.phi + chi + math::sign(aa.Jphi) * aa.thetaz);
    if(aa.Jz == 0)
        aa.thetaz = 0;
    if(freq) {
        freq->Omegar   = -2*E * sq2E / M;
        freq->Omegaz   = freq->Omegar * LLb;
        freq->Omegaphi = math::sign(aa.Jphi) * freq->Omegaz;
    }
    return aa;
}

}  // namespace actions