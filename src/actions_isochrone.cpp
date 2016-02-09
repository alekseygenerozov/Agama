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
    double signJphi = math::sign(aa.Jphi);
    double absJphi  = signJphi * aa.Jphi;
    double L    = aa.Jz + absJphi;
    double L1   = sqrt(L*L + 4*M*b);
    double LL1  = 0.5 + 0.5 * L/L1;
    double J0   = aa.Jr + L1 * LL1;  // combined magnitude of actions
    double j0invsq = M*b / pow_2(J0);
    // x1,x2 are roots of equation  x^2 - 2*(x-1)/j0^2 + (L/J0)^2-1 = 0:
    // x1 = j0invsq - ecc, x2 = j0invsq + ecc;  -1 <= x1 <= x2 <= 1.
    double ecc  = sqrt(pow_2(1-j0invsq) - pow_2(L/J0));  // determinant of the eqn
    // or  ecc  = sqrt(pow_2(1+j0invsq) - pow_2(L1/J0))
    double fac1 = (1 + ecc - j0invsq) * J0 / L;   // sqrt( (1-x1) / (1-x2) )
    double fac2 = (1 + ecc + j0invsq) * J0 / L1;  // sqrt( (1+x2) / (1+x1) )
    // quantities below depend on angles
    double eta, sineta, coseta;     // will be computed by the following routine:
    solveKepler(ecc, aa.thetar, eta, sineta, coseta); // thetar = eta - ecc * sin(eta)
    double ra = 1 - ecc * coseta;   // Kepler problem:  r / a = 1 - e cos(eta)
    double tanhalfeta = coseta==-1 ? INFINITY : sineta / (1 + coseta);
    double thetar   = aa.thetar - (eta>M_PI ? 2*M_PI : 0);
    double psi1     = atan(fac1 * tanhalfeta);
    double psi2     = atan(fac2 * tanhalfeta);
    double psi      = aa.thetaz - LL1 * thetar + psi1 + psi2 * L/L1;
    double sinpsi   = sin(psi);
    double cospsi   = cos(psi);
    double chi      = aa.Jz != 0 ? atan2(absJphi * sinpsi, L * cospsi) : psi;
    double sini     = sqrt(1 - pow_2(aa.Jphi / L)); // inclination angle of the orbital plane
    double costheta = sini * sinpsi;                // z/r
    double sintheta = sqrt(1 - pow_2(costheta));    // R/r is always non-negative
    coord::PosVelCyl point;
    double r   = b * sqrt(pow_2(ra / j0invsq) - 1);
    double vr  = J0 * ecc * sineta / r;
    point.R    = r * sintheta;
    point.z    = r * costheta;
    double vtheta = -L * sini * cospsi / point.R;
    point.vR   = vr * sintheta + vtheta * costheta;
    point.vz   = vr * costheta - vtheta * sintheta;
    point.vphi = aa.Jphi / point.R;
    point.phi  = math::wrapAngle(aa.thetaphi + (chi-aa.thetaz) * signJphi);
    if(freq) {
        freq->Omegar   = pow_2(M) / pow_3(J0);
        freq->Omegaz   = freq->Omegar * LL1;
        freq->Omegaphi = freq->Omegaz * signJphi;
    }
    if(!derivParam && !derivAct && !derivAng)
        return point;
    double dtan1 = 1 / (1/tanhalfeta + tanhalfeta*pow_2(fac1));
    double dtan2 = 1 / (1/tanhalfeta + tanhalfeta*pow_2(fac2)) * L/L1;
    double mult_chi  = 1 / (pow_2(aa.Jphi * sinpsi) + pow_2(L * cospsi));
    if(derivParam) {
        // common terms - derivs w.r.t. (M*b)
        double tmp       = 1 + j0invsq - L1/J0;
        double  decc_dMb = (tmp * (1-j0invsq) / ecc - ecc) / (J0*L1);
        double dfac1_dMb = tmp * ((1-j0invsq) / ecc + 1)   / (L *L1);
        double dfac2_dMb = (ecc/(1+j0invsq) + 1) * (J0 * decc_dMb + (1-2*J0/L1)/L1 * ecc) / L1;
        double  dpsi_dMb = (thetar - 2*psi2) * L/pow_3(L1) + 
            (dfac1_dMb + fac1 * decc_dMb / ra) * dtan1 +
            (dfac2_dMb + fac2 * decc_dMb / ra) * dtan2;
        double  dchi_dMb = aa.Jz == 0 ? dpsi_dMb :
            absJphi * L * dpsi_dMb * mult_chi;
        double dcostheta_dMb = sini * cospsi * dpsi_dMb;
        double dsintheta_dMb = -costheta / sintheta * dcostheta_dMb;
        double  drvtheta_dMb = L * sini * (1-pow_2(sini)) * sinpsi * dpsi_dMb / pow_3(sintheta);
        double drvr_dMb = sineta * (ecc/L1 + J0/ra * decc_dMb);         // d(r*vr) / d(M*b)
        double  drb_dMb = b/r / pow_2(j0invsq) * 
            (decc_dMb * (ecc-coseta) + (2/(J0*L1) - 1/(M*b)) * ra*ra);  // d(r/b)  / d(M*b)
        double drvR_dMb = drvr_dMb * sintheta + drvtheta_dMb * costheta -
            r*point.vz * dcostheta_dMb / sintheta;
        double drvz_dMb = drvr_dMb * costheta - drvtheta_dMb * sintheta +
            r*point.vR * dcostheta_dMb / sintheta;
        // derivs w.r.t. M: dX/dM = b * dX/d(M*b)
        double dr_dM = b*b * drb_dMb;
        derivParam[0].R    = r * dsintheta_dMb * b + dr_dM * sintheta;
        derivParam[0].z    = r * dcostheta_dMb * b + dr_dM * costheta;
        derivParam[0].phi  = b * dchi_dMb * signJphi;
        derivParam[0].vR   = b/r * drvR_dMb - point.vR/r * dr_dM;
        derivParam[0].vz   = b/r * drvz_dMb - point.vz/r * dr_dM;
        derivParam[0].vphi = -point.vphi / point.R * derivParam[0].R;
        // derivs w.r.t. b
        double dr_db = M*b * drb_dMb + r/b;
        derivParam[1].R    = r * dsintheta_dMb * M + dr_db * sintheta;
        derivParam[1].z    = r * dcostheta_dMb * M + dr_db * costheta;
        derivParam[1].phi  = M * dchi_dMb * signJphi;
        derivParam[1].vR   = M/r * drvR_dMb - point.vR/r * dr_db;
        derivParam[1].vz   = M/r * drvz_dMb - point.vz/r * dr_db;
        derivParam[1].vphi = -point.vphi / point.R * derivParam[1].R;
    }
    if(derivAct) {
        double  decc_dJr = ( (1 - pow_2(j0invsq)) / ecc - ecc) / J0;
        double  decc_add = -L / (J0*J0*ecc);
        double  decc_dL  = decc_dJr * LL1 + decc_add;
        double dfac1_dJr = fac1 / J0 + (decc_dJr*J0 + 2*j0invsq) / L;
        double dfac1_dL  = dfac1_dJr * LL1 -  fac1 / L  - 1 / (J0*ecc);
        double dfac2_dJr = fac2 / J0 + (decc_dJr*J0 - 2*j0invsq) / L1;
        double dfac2_dL  = dfac2_dJr * LL1 - (fac2 / L1 + 1 / (J0*ecc)) * L/L1;
        // derivs of intermediate angle vars
        double  dpsi_dJr = 
            (dfac1_dJr + fac1 * decc_dJr / ra) * dtan1 +
            (dfac2_dJr + fac2 * decc_dJr / ra) * dtan2;
        double  dpsi_dL  = (2*psi2 - thetar) * 2*M*b/pow_3(L1) +
            (dfac1_dL  + fac1 * decc_dL  / ra) * dtan1 +
            (dfac2_dL  + fac2 * decc_dL  / ra) * dtan2;
        double  dchi_dJr = aa.Jz == 0 ? dpsi_dJr :
            absJphi * L * dpsi_dJr * mult_chi;
        double  dchi_dJz = aa.Jz == 0 ? dpsi_dL  :
            absJphi * (L * dpsi_dL - sinpsi * cospsi) * mult_chi;
        double dchi_dJphi= aa.Jz == 0 ? dpsi_dL  :
            (absJphi * L * dpsi_dL + sinpsi * cospsi * (L - absJphi)) * mult_chi;
        // derivs of spherical coords (r,theta,vr,vtheta)
        double dcostheta_dJr  = sini * cospsi * dpsi_dJr;
        double dcostheta_dL   = sini * cospsi * dpsi_dL + pow_2(aa.Jphi) * sinpsi / (pow_3(L) * sini);
        double dcostheta_dJphi= sini * cospsi * dpsi_dL - absJphi * (L - absJphi) * sinpsi / (pow_3(L) * sini);
        double dsintheta_dJr  = -costheta / sintheta * dcostheta_dJr;
        double dsintheta_dL   = -costheta / sintheta * dcostheta_dL;
        double dsintheta_dJphi= -costheta / sintheta * dcostheta_dJphi;
        double A = pow_2(b/j0invsq) / r;
        double   dr_dJr = A * (decc_dJr * (ecc-coseta) + 2*ra*ra / J0);
        double   dr_dL  = dr_dJr * LL1 + A * (ecc-coseta) * decc_add;
        double drvr_dJr = sineta * (J0 * decc_dJr / ra + ecc);
        double drvr_dL  = sineta * (J0 * decc_dL  / ra + ecc * LL1);
        double drvtheta_dJr   = L * sini / sintheta *
            (sinpsi * dpsi_dJr + cospsi / sintheta * dsintheta_dJr);
        double drvtheta_dL    = L * sini / sintheta *
            (sinpsi * dpsi_dL  + cospsi / sintheta * dsintheta_dL) - cospsi / (sintheta * sini);
        double drvtheta_dJphi = L * sini / sintheta *
            (sinpsi * dpsi_dL  + cospsi / sintheta * dsintheta_dJphi) - cospsi * (L - absJphi) / (L * sintheta * sini);
        // d/dJr
        derivAct->dbyJr.R  = dr_dJr * sintheta + r * dsintheta_dJr;
        derivAct->dbyJr.z  = dr_dJr * costheta + r * dcostheta_dJr;
        derivAct->dbyJr.phi= dchi_dJr * signJphi;
        derivAct->dbyJr.vR = vr * dsintheta_dJr + vtheta * dcostheta_dJr +
            (drvr_dJr * sintheta + drvtheta_dJr * costheta - point.vR * dr_dJr) / r;
        derivAct->dbyJr.vz = vr * dcostheta_dJr - vtheta * dsintheta_dJr +
            (drvr_dJr * costheta - drvtheta_dJr * sintheta - point.vz * dr_dJr) / r;
        derivAct->dbyJr.vphi = -point.vphi / point.R * derivAct->dbyJr.R;
        // d/dJz
        derivAct->dbyJz.R  = dr_dL * sintheta + r * dsintheta_dL;
        derivAct->dbyJz.z  = dr_dL * costheta + r * dcostheta_dL;
        derivAct->dbyJz.phi= dchi_dJz * signJphi;
        derivAct->dbyJz.vR = vr * dsintheta_dL + vtheta * dcostheta_dL +
            (drvr_dL * sintheta + drvtheta_dL * costheta - point.vR * dr_dL) / r;
        derivAct->dbyJz.vz = vr * dcostheta_dL - vtheta * dsintheta_dL +
            (drvr_dL * costheta - drvtheta_dL * sintheta - point.vz * dr_dL) / r;
        derivAct->dbyJz.vphi = -point.vphi / point.R * derivAct->dbyJz.R;
        // d/dJphi
        derivAct->dbyJphi.R  = (dr_dL * sintheta + r * dsintheta_dJphi) * signJphi;
        derivAct->dbyJphi.z  = (dr_dL * costheta + r * dcostheta_dJphi) * signJphi;
        derivAct->dbyJphi.phi= dchi_dJphi;
        derivAct->dbyJphi.vR = (vr * dsintheta_dJphi + vtheta * dcostheta_dJphi +
            (drvr_dL * sintheta + drvtheta_dJphi * costheta - point.vR * dr_dL) / r) * signJphi;
        derivAct->dbyJphi.vz = (vr * dcostheta_dJphi - vtheta * dsintheta_dJphi +
            (drvr_dL * costheta - drvtheta_dJphi * sintheta - point.vz * dr_dL) / r) * signJphi;
        derivAct->dbyJphi.vphi = (1 - point.vphi * derivAct->dbyJphi.R) / point.R;
    }
    return point;
}

}  // namespace actions