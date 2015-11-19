/** \file    potential_composite.h
    \brief   Composite density and potential classes
    \author  Eugene Vasiliev
    \date    2014-2015
*/
#pragma once
#include "potential_base.h"
#include <vector>

namespace potential{

/** A trivial collection of several density objects */
class CompositeDensity: public BaseDensity{
public:
    /** construct from the provided array of components by making copies of them */
    CompositeDensity(const std::vector<const BaseDensity*>& _components);

    /** destroy all internally created copies of components */
    virtual ~CompositeDensity() {
        for(unsigned int i=0; i<components.size(); i++) delete components[i]; }

    /** provides the 'least common denominator' for the symmetry degree */
    virtual SymmetryType symmetry() const;

    virtual const char* name() const { return myName(); };
    static const char* myName() { return "CompositeDensity"; };

    /** clone the density by cloning each of its components individually */
    virtual BaseDensity* clone() const { return new CompositeDensity(components); }

private:
    std::vector<const BaseDensity*> components;
    virtual double densityCar(const coord::PosCar &pos) const;
    virtual double densityCyl(const coord::PosCyl &pos) const;
    virtual double densitySph(const coord::PosSph &pos) const;
};

/** A trivial collection of several potential objects, evaluated in cylindrical coordinates */
class CompositeCyl: public BasePotentialCyl{
public:
    /** construct from the provided array of components by making copies of them */ 
    CompositeCyl(const std::vector<const BasePotential*>& _components);

    /** delete the internally copied sub-components */
    virtual ~CompositeCyl() {
        for(unsigned int i=0; i<components.size(); i++) delete components[i]; }

    /** provides the 'least common denominator' for the symmetry degree */
    virtual SymmetryType symmetry() const;

    virtual const char* name() const { return myName(); };
    static const char* myName() { return "CompositePotential"; };

    /** clone the potential by cloning each of its components individually */
    virtual BasePotential* clone() const { return new CompositeCyl(components); }

private:
    std::vector<const BasePotential*> components;
    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;
};

}  // namespace potential