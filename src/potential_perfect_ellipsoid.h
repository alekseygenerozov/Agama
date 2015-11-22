/** \file    potential_perfect_ellipsoid.h
    \brief   Potential for Oblate Perfect Ellipsoid model
    \author  Eugene Vasiliev
    \date    2015
*/
#pragma once
#include "potential_base.h"

namespace potential{

/** Axisymmetric Stackel potential with oblate perfect ellipsoidal density.
    Potential is computed in prolate spheroidal coordinate system
    through an auxiliary function  \f$  G(\tau)  \f$  as
    \f$  \Phi = -[ (\lambda+\gamma)G(\lambda) - (\nu+\gamma)G(\nu) ] / (\lambda-\nu)  \f$.
    The parameters of the internal prolate spheroidal coordinate system are 
    \f$  \alpha=-a^2, \gamma=-c^2  \f$, where a and c are the major and minor axes, 
    and the coordinates  \f$  \lambda, \nu  \f$  in this system satisfy
    \f$  -\gamma \le \nu \le -\alpha \le \lambda < \infty  \f$.
*/
class OblatePerfectEllipsoid: public BasePotential, 
    public coord::IScalarFunction<coord::ProlSph>, public math::IFunction {
public:
    OblatePerfectEllipsoid(double _mass, double major_axis, double minor_axis);

    virtual SymmetryType symmetry() const { return ST_AXISYMMETRIC; }

    const coord::ProlSph& coordsys() const { return coordSys; }
    
    virtual const char* name() const { return myName(); };
    static const char* myName() { return "OblatePerfectEllipsoid"; };
    virtual double totalMass() const { return mass; }

    /** evaluates the function G(tau) and up to two its derivatives,
        if the supplied output arguments are not NULL 
        (implements the math::IFunction interface) */
    virtual void evalDeriv(double tau, double* G=0, double* Gderiv=0, double* Gderiv2=0) const;

private:
    const double mass;
    /** prolate spheroidal coordinate system corresponding to the oblate density profile */
    const coord::ProlSph coordSys;
    const double minorAxis;

    /** implementations of the standard triad of coordinate transformations */
    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const {
        coord::evalAndConvertTwoStep<coord::ProlSph, coord::Cyl, coord::Car>
            (*this, pos, coordSys, potential, deriv, deriv2);  // no direct conversion exists, use two-step
    }
    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const {
        coord::evalAndConvert<coord::ProlSph, coord::Cyl>
            (*this, pos, coordSys, potential, deriv, deriv2);
    }
    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const {
        coord::evalAndConvertTwoStep<coord::ProlSph, coord::Cyl, coord::Sph>
            (*this, pos, coordSys, potential, deriv, deriv2);  // use two-step conversion
    }

    /** the function that does the actual computation in prolate spheroidal coordinates 
        (implements the coord::IScalarFunction<ProlSph> interface) */
    virtual void evalScalar(const coord::PosProlSph& pos,
        double* value=0, coord::GradProlSph* deriv=0, coord::HessProlSph* deriv2=0) const;

    virtual unsigned int numDerivs() const { return 2; }
};

/// specialization of a shared pointer to this potential (to be used in ActionFinderAxisymStaeckel)
#ifdef HAVE_CXX11
typedef std::shared_ptr<const OblatePerfectEllipsoid> PtrOblatePerfectEllipsoid;
#else
typedef std::tr1::shared_ptr<const OblatePerfectEllipsoid> PtrOblatePerfectEllipsoid;
#endif

}  // namespace potential