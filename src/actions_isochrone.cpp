#include "actions_isochrone.h"
#include "math_core.h"
#include <stdexcept>
#include <cmath>

namespace actions{

/// solve  phase = eta - ecc * sin(eta)  for eta (eccentric anomaly)
/// store eta, its sin and cos in output arguments
static void solveKepler(double ecc, double phase, double &eta, double &sineta, double &coseta)
{
    if(phase==0 || phase==M_PI) {
        eta    = phase;
        sineta = 0;
        coseta = phase==0 ? 1 : -1;
        return;
    }
    bool signeta = phase > M_PI;
    // initial guess - TODO!!! needs to be improved
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
    } while(fabs(deltaeta) > 1e-12 && niter<42);
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
    double rb   = hypot(b, point.r);
    double L    = Ltotal(point);
    double L1   = sqrt(L*L + 4*M*b);
    double J0   = M / sqrt(2*M / (b+rb) - pow_2(point.vr) - pow_2(point.vtheta) - pow_2(point.vphi));    
    double j0invsq = M*b / pow_2(J0);
    // J0 is related to total energy via  J0 = M / sqrt(-2*E)
    aa.Jphi     = Lz(pointCyl);
    aa.Jz       = L - fabs(aa.Jphi);
    aa.Jr       = J0 - 0.5 * (L + L1);
    double ecc  = sqrt(pow_2(1-j0invsq) - pow_2(L/J0));
    double fac1 = (1 + ecc - j0invsq) * J0 / L;
    double fac2 = (1 + ecc + j0invsq) * J0 / L1;
    // below are quantities that depend on position along the orbit
    double k1   = point.r * point.vr;
    double k2   = J0 - M * rb / J0;
    double eta  = atan2(k1, k2);    // eccentric anomaly
    double sineta     = k1 / hypot(k1, k2);              // sin(eta)
    double tanhalfeta = eta==0 ? 0 : -k2/k1 + 1/sineta;  // tan(eta/2)
    double psi  = atan2(pointCyl.z * L,  -pointCyl.R * point.vtheta * point.r);
    double chi  = atan2(pointCyl.z * point.vphi, -point.vtheta * point.r);
    aa.thetar   = math::wrapAngle(eta - sineta * ecc );
    aa.thetaz   = math::wrapAngle(psi + 0.5 * (1 + L/L1) * (aa.thetar - (eta<0 ? 2*M_PI : 0))
                - atan(fac1 * tanhalfeta) - atan(fac2 * tanhalfeta) * L/L1 );
    aa.thetaphi = math::wrapAngle(point.phi - chi + math::sign(aa.Jphi) * aa.thetaz);
    if(aa.Jz == 0)  // in this case the value of theta_z is meaningless
        aa.thetaz = 0;
    if(freq) {
        freq->Omegar   = pow_2(M) / pow_3(J0);
        freq->Omegaz   = freq->Omegar * 0.5 * (1 + L/L1);
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
    double L  = aa.Jz + fabs(aa.Jphi);
    double L1 = sqrt(L*L + 4*M*b);
    double J0 = aa.Jr + 0.5 * (L + L1);  // combined magnitude of actions
    double j0invsq = M*b / pow_2(J0);
    // x1,x2 are roots of equation  x^2 - 2*(x-1)/j0^2 + (L/J0)^2-1 = 0:
    // x1 = j0invsq - det, x2 = j0invsq + det;  -1 <= x1 <= x2 <= 1.
    double ecc  = sqrt(pow_2(1-j0invsq) - pow_2(L/J0));  // determinant of the eqn
    double fac1 = (1 + ecc - j0invsq) * J0 / L;   // sqrt( (1-x1) / (1-x2) )
    double fac2 = (1 + ecc + j0invsq) * J0 / L1;  // sqrt( (1+x2) / (1+x1) )
    // quantities below depend on angles
    double eta, sineta, coseta;  // psi in A12; will be computed by the following routine
    solveKepler(ecc, aa.thetar, eta, sineta, coseta); // thetar = eta - det * sin(eta)
    double tanhalfeta = coseta==-1 ? INFINITY : sineta / (1 + coseta);
    double psi = aa.thetaz - 0.5 * (1 + L/L1) * (aa.thetar - (eta>M_PI ? 2*M_PI : 0))
               + atan(fac1 * tanhalfeta) + atan(fac2 * tanhalfeta) * L/L1;  // chi in A14
    //double tanhalfpsi = tan(0.5*psi);
    double sinpsi   = sin(psi);  // 2 / (1/tanhalfpsi + tanhalfpsi);
    double cospsi   = cos(psi);  // 1 - sinpsi * tanhalfpsi;
    double chi      = aa.Jz != 0 ? atan2(fabs(aa.Jphi) * sinpsi, L * cospsi) : psi;
    double sini     = sqrt(1 - pow_2(aa.Jphi / L)); // inclination angle of the orbital plane
    double costheta = sini * sinpsi;                // z/r
    double sintheta = sqrt(1 - pow_2(costheta));    // R/r is always non-negative
    coord::PosVelCyl point;
    double r   = b * sqrt(pow_2((ecc * coseta - 1) / j0invsq) - 1);
    double vr  = J0 * ecc * sineta / r;
    point.R    = r * sintheta;
    point.z    = r * costheta;
    double vtheta = -L * sini * cospsi / point.R;
    point.vR   = vr * sintheta + vtheta * costheta;
    point.vz   = vr * costheta - vtheta * sintheta;
    point.vphi = aa.Jphi / point.R;
    point.phi  = math::wrapAngle(aa.thetaphi + (chi-aa.thetaz) * math::sign(aa.Jphi) );
    if(freq) {
        freq->Omegar   = pow_2(M) / pow_3(J0);
        freq->Omegaz   = freq->Omegar * 0.5 * (1 + L/L1);
        freq->Omegaphi = math::sign(aa.Jphi) * freq->Omegaz;
    }
    if(derivAct || derivAng || derivParam) {
    }
    return point;
}

}  // namespace actions