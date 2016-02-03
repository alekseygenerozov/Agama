#include "actions_isochrone.h"
#include "math_core.h"
#include <stdexcept>
#include <cmath>

namespace actions{

/// solve  phase = eta - ecc * sin(eta)  for eta (eccentric anomaly)
/// store eta, its sin and cos in output arguments
static void solveKepler(double ecc, double phase, double &eta, double &sineta, double &coseta)
{
    const double EPS = 1e-10;  // accuracy
    if(ecc < EPS) {
        eta    = phase;
        sineta = sin(eta);
        coseta = cos(eta);
        return;
    }
    bool signeta = phase > M_PI;
    // initial guess
    eta = 0.5*M_PI + (M_PI/8)/ecc * 
        (sqrt(M_PI*M_PI + (ecc + (signeta ? 1.5*M_PI-phase : phase-0.5*M_PI)) * 16*ecc ) - M_PI);
    if(signeta)
        eta = 2*M_PI - eta;
    double deltaeta = 0;
    int niter = 0;
    do {  // Newton's method
        sineta    = sin(eta);
        coseta    = cos(eta);
        double f  = eta - ecc * sineta - phase;
        double df = 1.  - ecc * coseta;
        deltaeta  = -f/df;
        // refinement using second derivative (thanks to A.Gurkan)
        deltaeta  = -f / (df + 0.5 * deltaeta * ecc * sineta);
        eta      += deltaeta;
        niter++;
    } while(fabs(deltaeta) > EPS && niter<42);
}
    
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
    double sq2E = sqrt(2*M / (b+rb) - pow_2(point.vr) - pow_2(point.vtheta) - pow_2(point.vphi)); // sqrt(-2*E)
    aa.Jphi     = coord::Lz(pointCyl);
    aa.Jz       = L - fabs(aa.Jphi);
    aa.Jr       = M / sq2E - Lb * LLb;
    double Lcir = M / sq2E - b * sq2E;      // ang.mom. of a circular orbit with energy E
    double ecc  = sqrt(1 - pow_2(L/Lcir));  // eccentricity
    double W    = 2 * M / (sq2E*Lcir) - 1;  // (M - 2 E b) / (M + 2 E b)
    double fac1 = (1+ecc) * Lcir / L;       // sqrt( (1+ecc) / (1-ecc) )
    double fac2 = (W+ecc) * Lcir / Lb;      // sqrt( (W+ecc) / (W-ecc) )
    // below are quantities that depend on position along the orbit
    double k1   = point.r * point.vr;
    double k2   = M / sq2E - sq2E * rb;
    double eta  = atan2(k1, k2);  // eccentric anomaly
    double sineta     = k1 / sqrt(k1*k1 + k2*k2);  // sin(eta)
    double tanhalfeta = -k2/k1 + 1/sineta;         // tan(eta/2)
    double psi  = atan2(pointCyl.z * L,  -pointCyl.R * point.vtheta * point.r);
    double chi  = atan2(pointCyl.z * point.vphi, point.vtheta * point.r) + M_PI;
    aa.thetar   = math::wrapAngle(eta - sineta * ecc * 2 / (1+W) );
    aa.thetaz   = math::wrapAngle(psi + LLb * (aa.thetar - (eta<0 ? 2*M_PI : 0))
                - atan(fac1 * tanhalfeta) - atan(fac2 * tanhalfeta) * L/Lb );
    aa.thetaphi = math::wrapAngle(point.phi + chi + math::sign(aa.Jphi) * aa.thetaz);
    if(aa.Jz == 0)  // in this case the value of theta_z is meaningless
        aa.thetaz = 0;
    if(freq) {
        freq->Omegar   = pow_3(sq2E) / M;
        freq->Omegaz   = freq->Omegar * LLb;
        freq->Omegaphi = math::sign(aa.Jphi) * freq->Omegaz;
    }
    return aa;
}

coord::PosVelCyl mapIsochrone(
    const double M, const double b,
    const ActionAngles& aa, Frequencies* freq)
{
    return ToyMapIsochrone(M, b).map(aa, freq);
}

coord::PosVelCyl ToyMapIsochrone::mapDeriv(
    const ActionAngles& aa,
    Frequencies* freq,
    DerivAct* derivAct,
    DerivAng* derivAng,
    coord::PosVelCyl* derivParam) const
{
    double L    = aa.Jz + fabs(aa.Jphi);    // total angular momentum
    double Lb   = sqrt(L*L + 4*M*b);        // and related quantities
    double LLb  = 0.5 * (1 + L/Lb);
    double sini = sqrt(1 - pow_2(aa.Jphi / L));  // l in A9 - sin of inclination angle
    double sq2E = M / (aa.Jr + Lb*LLb);     // sqrt(-2*E)
    double Lcir = M / sq2E - b * sq2E;      // ang.mom. of a circular orbit with energy E
    double ecc  = sqrt(1 - pow_2(L/Lcir));  // eccentricity
    double W    = 2 * M / (sq2E*Lcir) - 1;  // (M - 2 E b) / (M + 2 E b)
    double fac1 = (1+ecc) * Lcir / L;       // sqrt( (1+ecc) / (1-ecc) )
    double fac2 = (W+ecc) * Lcir / Lb;      // sqrt( (W+ecc) / (W-ecc) )
    // quantities below depend on angles
    double eta, sineta, coseta;  // psi in A12; will be computed by the following routine
    solveKepler(ecc * 2 / (1+W), aa.thetar, eta, sineta, coseta);
    double r    = Lcir / sq2E * sqrt( (1 - ecc * coseta) * (W - ecc * coseta) );
    double vr   = Lcir / r * ecc * sineta;
    double tanhalfeta = sineta / (1 + coseta);
    double psi  = aa.thetaz - LLb * (aa.thetar - (eta>M_PI ? 2*M_PI : 0))
                + atan(fac1 * tanhalfeta) + atan(fac2 * tanhalfeta) * L/Lb;  // chi in A14
    double tanhalfpsi = tan(0.5*psi);
    double sinpsi     = 2 * tanhalfpsi / (1 + pow_2(tanhalfpsi));
    double costheta   = sini * sinpsi;              // z/r
    double sintheta   = sqrt(1 - pow_2(costheta));  // R/r is always non-negative
    coord::PosVelCyl point;
    point.R    = r * sintheta;
    point.z    = r * costheta;
    double vtheta = -L * sini * (1 - sinpsi * tanhalfpsi) / point.R;
    point.vR   = vr * sintheta + vtheta * costheta;
    point.vz   = vr * costheta - vtheta * sintheta;
    point.vphi = aa.Jphi / point.R;
    double chi = atan2(costheta * fabs(point.vphi), vtheta) + M_PI;
    point.phi  = math::wrapAngle(aa.thetaphi -
        (aa.Jz != 0 ? aa.thetaz+chi : aa.thetaz-psi) * math::sign(aa.Jphi) );
    if(freq) {
        freq->Omegar   = pow_3(sq2E) / M;
        freq->Omegaz   = freq->Omegar * LLb;
        freq->Omegaphi = math::sign(aa.Jphi) * freq->Omegaz;
    }
    if(derivParam) {
    }
    return point;
}

}  // namespace actions