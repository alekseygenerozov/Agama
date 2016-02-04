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
    if(ecc < EPS || phase==0 || phase==M_PI) {
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
    double J0   = M / sqrt(2*M / (b+rb) - pow_2(point.vr) - pow_2(point.vtheta) - pow_2(point.vphi));
    // J0 is related to total energy via  J0 = M / sqrt(-2*E)
    aa.Jphi     = Lz(pointCyl);
    aa.Jz       = L - fabs(aa.Jphi);
    aa.Jr       = J0 - Lb * LLb;
    double Lcir = J0 - M*b / J0;            // ang.mom. of a circular orbit with energy E
    double ecc  = sqrt(1 - pow_2(L/Lcir));  // eccentricity
    double W    = 2 * J0 / Lcir - 1;        // (J0^2 + M b) / (J0^2 - M b)
    double fac1 = (1+ecc) * Lcir / L;       // sqrt( (1+ecc) / (1-ecc) )
    double fac2 = (W+ecc) * Lcir / Lb;      // sqrt( (W+ecc) / (W-ecc) )
    // below are quantities that depend on position along the orbit
    double k1   = point.r * point.vr;
    double k2   = J0 - M * rb / J0;
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
        freq->Omegar   = pow_2(M) / pow_3(J0);
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
    double Jscale = sqrt(M * b);      // dimensional scaling factor for actions
    double jr     = aa.Jr / Jscale;
    double jphi   = aa.Jphi / (2*Jscale);
    double jt     = (aa.Jz + fabs(aa.Jphi)) / (2*Jscale);  // dimensionless L^2
    double jt1    = sqrt(jt*jt + 1);
    double j0     = jr + jt + jt1;      // combined dimensionless magnitude of actions
    double j0invsq= 1 / pow_2(j0);
    // x1,x2 are roots of equation  x^2 - 2*x/j0^2 + (2+4*jt^2)/j0^2-1 = 0
    double det  = sqrt(pow_2(1-j0invsq) - pow_2(2*jt) * j0invsq);
    double x1   = j0invsq - det;
    double x2   = j0invsq + det;
    double fac1 = (1-x1) * j0 / (2*jt);   // sqrt( (1-x1) / (1-x2) )
    double fac2 = (1+x2) * j0 / (2*jt1);  // sqrt( (1+x2) / (1+x1) )
    // quantities below depend on angles
    double eta, sineta, coseta;  // psi in A12; will be computed by the following routine
    solveKepler(det, aa.thetar, eta, sineta, coseta);  // thetar = eta - det * sin(eta)
    double tanhalfeta = sineta / (1 + coseta);
    double psi  = aa.thetaz - 0.5 * (1 + jt/jt1) * (aa.thetar - (eta>M_PI ? 2*M_PI : 0))
                + atan(fac1 * tanhalfeta) + atan(fac2 * tanhalfeta) * jt/jt1;  // chi in A14
    double tanhalfpsi = tan(0.5*psi);
    double sinpsi     = 2 * tanhalfpsi / (1 + pow_2(tanhalfpsi));
    double sini       = sqrt(1 - pow_2(jphi / jt)); // inclination angle of the orbital plane
    double costheta   = sini * sinpsi;              // z/r
    double sintheta   = sqrt(1 - pow_2(costheta));  // R/r is always non-negative
    coord::PosVelCyl point;
    double rvr = j0 * det * sineta;           // r * v_r
    double Mrb = j0*j0 * (det * coseta - 1);  // sqrt(1 + (r/b)^2)
    double r   = b * sqrt(pow_2(Mrb) - 1);
    double vr  = Jscale/2/b * rvr / r;
    point.R    = r * sintheta;
    point.z    = r * costheta;
    double vtheta = -2*Jscale*jt * sini * (1 - sinpsi * tanhalfpsi) / point.R;
    point.vR   = vr * sintheta + vtheta * costheta;
    point.vz   = vr * costheta - vtheta * sintheta;
    point.vphi = aa.Jphi / point.R;
    double chi = atan2(costheta * fabs(point.vphi), vtheta) + M_PI;   //!!?? better expr?
    point.phi  = math::wrapAngle(aa.thetaphi -
        (aa.Jz != 0 ? aa.thetaz+chi : aa.thetaz-psi) * math::sign(aa.Jphi) );
    if(freq) {
        freq->Omegar   = Jscale / (b*b * pow_3(j0));
        freq->Omegaz   = freq->Omegar * 0.5 * (1 + jt/jt1);
        freq->Omegaphi = math::sign(aa.Jphi) * freq->Omegaz;
    }
    if(derivAct || derivAng || derivParam) {
    }
    return point;
}

}  // namespace actions