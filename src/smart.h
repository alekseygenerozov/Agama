/** \file    smart.h
    \brief   Forward declarations of fundamental classes and smart pointers to them
    \author  Eugene Vasiliev
    \date    2015

This file teaches how to be smart and use modern practices in C++ programming.
Smart pointers are a convenient and robust approach for automated memory management.
One does not need to worry about keeping track of dynamically created objects,
does not risk memory leaks or dangling pointers.
In short, instead of using ordinary pointers for handling dynamically allocated objects,
we wrap them into special objects that behave almost like pointers, but are smarter.
The key point is the following C++ feature: when a local variable gets out of scope,
and if it was an instance of some class, its destructor is called automatically,
no matter how the control flow goes (whether we return from anywhere inside a routine,
or even throw an exception which propagates up the call stack).
Ordinary pointers are not objects, but smart pointer wrappers are, and they take care
of deleting the ordinary pointer when they get out of scope or are manually released.
There are two main types of smart pointers that correspond to different ownership rules:
exclusive and shared.
The first model applies to local variables that should not leave the scope of a routine,
or to class members that are created in the constructor and are owned by the instance of
this class, therefore must be disposed of when the owner object is destroyed.
This is represented by 'std::unique_ptr', which is only available in C++11;
its incomplete analog for older compiler versions is 'std::auto_ptr'.
The second model allows shared ownership of an object by several pointers, ensuring that
the object under control stays alive as long as there is at least one smart pointer
that keeps track of it (in other words, it implements reference counting approach).
This is used to pass around wrapped objects between different parts of code that do not
have a predetermined execution order or lifetime. For example, an instance of Potential
can be passed to an ActionFinder that makes a copy of the pointer and keeps it as long
as the action finder itself stays alive, even though the original smart pointer might
have been deleted long ago.
This second model is represented by 'std::shared_ptr' (or its pre-C++11 namesake
'std::tr1::shared_ptr'), and is used for the common classes like Density, Potential,
DistributionFunction or ActionFinder.
*/
#pragma once
#include <tr1/memory>

namespace math{

class IFunction;
class IFunctionNdim;

/// pointer to a function class
typedef std::tr1::shared_ptr<const IFunction> PtrFunction;
typedef std::tr1::shared_ptr<const IFunctionNdim> PtrFunctionNdim;

}  // namespace math


namespace potential{

class BaseDensity;
class BasePotential;
class OblatePerfectEllipsoid;

/// Shared pointers to density and potential classes
typedef std::tr1::shared_ptr<const BaseDensity>    PtrDensity;
typedef std::tr1::shared_ptr<const BasePotential>  PtrPotential;
typedef std::tr1::shared_ptr<const OblatePerfectEllipsoid> PtrOblatePerfectEllipsoid;

}  // namespace potential


namespace actions{

class BaseActionFinder;

typedef std::tr1::shared_ptr<const BaseActionFinder> PtrActionFinder;

}  // namespace actions


namespace df{

class BaseDistributionFunction;

typedef std::tr1::shared_ptr<const BaseDistributionFunction> PtrDistributionFunction;

}  // namespace df