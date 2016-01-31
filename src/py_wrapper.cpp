/** \file   py_wrapper.cpp
    \brief  Python wrapper for the library
    \author Eugene Vasiliev
    \date   2014-2015

    This is a Python extension module that provides the interface to
    some of the classes and functions from the C++ library.
    It needs to be compiled into a dynamic library and placed in a folder
    that Python is aware of (e.g., through the PYTHONPATH= environment variable).

    Currently this module provides access to potential classes, orbit integration
    routine, action finders, and smoothing splines.
    Unit conversion is also part of the calling convention: the quantities 
    received from Python are assumed to be in some physical units and converted
    into internal units inside this module, and the output from the library 
    routines is converted back to physical units. The physical units are assigned
    by `set_units` and `reset_units` functions.

    Type `help(agama)` in Python to get a list of exported routines and classes,
    and `help(agama.whatever)` to get the usage syntax for each of them.
*/
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// note: for some versions of NumPy, it seems necessary to replace constants
// starting with NPY_ARRAY_*** by NPY_***
#include <numpy/arrayobject.h>
#include <stdexcept>
#include "units.h"
#include "potential_factory.h"
#include "potential_composite.h"
#include "actions_spherical.h"
#include "actions_staeckel.h"
#include "df_factory.h"
#include "galaxymodel.h"
#include "orbit.h"
#include "math_core.h"
#include "math_sample.h"
#include "math_spline.h"
#include "utils_config.h"

namespace{  // private namespace

/// \name  ----- Some general definitions -----
///@{

/// return a string representation of a Python object
static std::string toString(PyObject* obj)
{
    if(PyString_Check(obj))
        return std::string(PyString_AsString(obj));
    PyObject* s = PyObject_Str(obj);
    std::string str = PyString_AsString(s);
    Py_DECREF(s);
    return str;
}

/// convert a Python dictionary to its c++ analog
static utils::KeyValueMap convertPyDictToKeyValueMap(PyObject* args)
{
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    utils::KeyValueMap params;
    while (PyDict_Next(args, &pos, &key, &value))
        params.set(toString(key), toString(value));
    return params;
}

///@}
/// \name  ------- Unit handling routines --------
///@{

/// internal working units
static const units::InternalUnits unit(units::Kpc, units::Myr);

/// external units that are used in the calling code
static const units::ExternalUnits* conv;

/// description of set_units function
static const char* docstringSetUnits = 
    "Inform the library about the physical units that are used in Python code\n"
    "Arguments should be any three independent physical quantities that define "
    "'mass', 'length', 'velocity' or 'time' scales "
    "(note that the latter three are not all independent).\n"
    "Their values specify the units in terms of "
    "'Solar mass', 'Kiloparsec', 'km/s' and 'Megayear', correspondingly.\n"
    "Example: standard GADGET units are defined as\n"
    "    setUnits(mass=1e10, length=1, velocity=1)\n";

/// define the unit conversion
static PyObject* set_units(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"mass", "length", "velocity", "time", NULL};
    double mass = 0, length = 0, velocity = 0, time = 0;
    if(!PyArg_ParseTupleAndKeywords(
        args, namedArgs, "|dddd", const_cast<char**>(keywords),
        &mass, &length, &velocity, &time) ||
        mass<0 || length<0 || velocity<0 || time<0)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to set_units()");
        return NULL;
    }
    if(length>0 && velocity>0 && time>0) {
        PyErr_SetString(PyExc_ValueError, 
            "You may not assign length, velocity and time units simultaneously");
        return NULL;
    }
    if(mass==0) {
        PyErr_SetString(PyExc_ValueError, "You must specify mass unit");
        return NULL;
    }
    const units::ExternalUnits* newConv = NULL;
    if(length>0 && time>0)
        newConv = new units::ExternalUnits(unit,
            length*units::Kpc, length/time * units::Kpc/units::Myr, mass*units::Msun);
    else if(length>0 && velocity>0)
        newConv = new units::ExternalUnits(unit,
            length*units::Kpc, velocity*units::kms, mass*units::Msun);
    else if(time>0 && velocity>0)
        newConv = new units::ExternalUnits(unit,
            velocity*time * units::kms*units::Myr, velocity*units::kms, mass*units::Msun);
    else {
        PyErr_SetString(PyExc_ValueError,
            "You must specify exactly two out of three units: length, time and velocity");
        return NULL;
    }
    delete conv;
    conv = newConv;
    Py_INCREF(Py_None);
    return Py_None;
}

/// description of reset_units function
static const char* docstringResetUnits = 
    "Reset the unit conversion system to a trivial one "
    "(i.e., no conversion involved and all quantities are assumed to be in N-body units, "
    "with the gravitational constant equal to 1\n";

/// reset the unit conversion
static PyObject* reset_units(PyObject* /*self*/, PyObject* /*args*/)
{
    delete conv;
    conv = new units::ExternalUnits();
    Py_INCREF(Py_None);
    return Py_None;
}

/// helper function for converting position to internal units
static coord::PosCar convertPos(const double input[]) {
    return coord::PosCar(
        input[0] * conv->lengthUnit, 
        input[1] * conv->lengthUnit, 
        input[2] * conv->lengthUnit);
}
/// helper function for converting position/velocity to internal units
coord::PosVelCar convertPosVel(const double input[]) {
    return coord::PosVelCar(
        input[0] * conv->lengthUnit,
        input[1] * conv->lengthUnit,
        input[2] * conv->lengthUnit,
        input[3] * conv->velocityUnit,
        input[4] * conv->velocityUnit,
        input[5] * conv->velocityUnit);
}
/// helper function for converting actions to internal units
actions::Actions convertActions(const double input[]) {
    return actions::Actions(
        input[0] * conv->lengthUnit * conv->velocityUnit, 
        input[1] * conv->lengthUnit * conv->velocityUnit, 
        input[2] * conv->lengthUnit * conv->velocityUnit);
}
/// helper function to convert position/velocity from internal units back to user units
void unconvertPosVel(const coord::PosVelCar& point, double dest[])
{
    dest[0] = point.x / conv->lengthUnit;
    dest[1] = point.y / conv->lengthUnit;
    dest[2] = point.z / conv->lengthUnit;
    dest[3] = point.vx / conv->velocityUnit;
    dest[4] = point.vy / conv->velocityUnit;
    dest[5] = point.vz / conv->velocityUnit;
}

///@}
/// \name ----- a truly general interface for evaluating some function
///             for some input data and storing its output somewhere -----
///@{

/// any function that evaluates something for a given object and an `input` array of floats,
/// and stores one or more values in the `result` array of floats
typedef void (*anyFunction) 
    (void* obj, const double input[], double *result);

/// anyFunction input type
enum INPUT_VALUE {
    INPUT_VALUE_SINGLE = 1,  ///< a single number
    INPUT_VALUE_TRIPLET= 3,  ///< three numbers
    INPUT_VALUE_SEXTET = 6   ///< six numbers
};

/// anyFunction output type; numerical value is arbitrary
enum OUTPUT_VALUE {
    OUTPUT_VALUE_SINGLE              = 1,  ///< scalar value
    OUTPUT_VALUE_TRIPLET             = 3,  ///< a triplet of numbers
    OUTPUT_VALUE_SEXTET              = 6,  ///< a sextet of numbers
    OUTPUT_VALUE_SINGLE_AND_SINGLE   = 11, ///< a single number and another single number
    OUTPUT_VALUE_SINGLE_AND_TRIPLET  = 13, ///< a single number and a triplet
    OUTPUT_VALUE_SINGLE_AND_SEXTET   = 16, ///< a single number and a sextet
    OUTPUT_VALUE_TRIPLET_AND_TRIPLET = 33, ///< a triplet and another triplet -- two separate arrays
    OUTPUT_VALUE_TRIPLET_AND_SEXTET  = 36, ///< a triplet and a sextet
    OUTPUT_VALUE_SINGLE_AND_TRIPLET_AND_SEXTET = 136 ///< all wonders at once
};

/// size of input array for a single point
template<int numArgs>
static size_t inputLength();

/// parse a list of numArgs floating-point arguments for a Python function, 
/// and store them in inputArray[]; return 1 on success, 0 on failure 
template<int numArgs>
int parseTuple(PyObject* args, double inputArray[]);

/// check that the input array is of right dimensions, and return its length
template<int numArgs>
int parseArray(PyArrayObject* arr)
{
    if(PyArray_NDIM(arr) == 2 && PyArray_DIM(arr, 1) == numArgs)
        return PyArray_DIM(arr, 0);
    else
        return 0;
}

/// error message for an input array of incorrect size
template<int numArgs>
const char* errStrInvalidArrayDim();

/// error message for an input array of incorrect size or an invalid list of arguments
template<int numArgs>
const char* errStrInvalidInput();

/// size of output array for a single point
template<int numOutput>
static size_t outputLength();

/// construct an output tuple containing the given result data computed for a single input point
template<int numOutput>
PyObject* formatTuple(const double result[]);

/// construct an output array, or several arrays, that will store the output for many input points
template<int numOutput>
PyObject* allocOutputArr(int size);

/// store the 'result' data computed for a single input point in an output array 'resultObj' at 'index'
template<int numOutput>
void formatOutputArr(const double result[], const int index, PyObject* resultObj);

// ---- template instantiations for input parameters ----

template<> inline size_t inputLength<INPUT_VALUE_SINGLE>()  {return 1;}
template<> inline size_t inputLength<INPUT_VALUE_TRIPLET>() {return 3;}
template<> inline size_t inputLength<INPUT_VALUE_SEXTET>()  {return 6;}

template<> int parseTuple<INPUT_VALUE_SINGLE>(PyObject* args, double input[]) {
    input[0] = PyFloat_AsDouble(args);
    return PyErr_Occurred() ? 0 : 1;
}
template<> int parseTuple<INPUT_VALUE_TRIPLET>(PyObject* args, double input[]) {
    return PyArg_ParseTuple(args, "ddd", &input[0], &input[1], &input[2]);
}
template<> int parseTuple<INPUT_VALUE_SEXTET>(PyObject* args, double input[]) {
    return PyArg_ParseTuple(args, "dddddd",
        &input[0], &input[1], &input[2], &input[3], &input[4], &input[5]);
}

template<>
int parseArray<INPUT_VALUE_SINGLE>(PyArrayObject* arr)
{
    if(PyArray_NDIM(arr) == 1)
        return PyArray_DIM(arr, 0);
    else
        return 0;
}

template<> const char* errStrInvalidArrayDim<INPUT_VALUE_SINGLE>() {
    return "Input does not contain a valid one-dimensional array";
}
template<> const char* errStrInvalidArrayDim<INPUT_VALUE_TRIPLET>() {
    return "Input does not contain a valid Nx3 array";
}
template<> const char* errStrInvalidArrayDim<INPUT_VALUE_SEXTET>() {
    return "Input does not contain a valid Nx6 array";
}

template<> const char* errStrInvalidInput<INPUT_VALUE_SINGLE>() {
    return "Input does not contain valid data (either a single number or a one-dimensional array)";
}
template<> const char* errStrInvalidInput<INPUT_VALUE_TRIPLET>() {
    return "Input does not contain valid data (either 3 numbers for a single point or a Nx3 array)";
}
template<> const char* errStrInvalidInput<INPUT_VALUE_SEXTET>() {
    return "Input does not contain valid data (either 6 numbers for a single point or a Nx6 array)";
}

// ---- template instantiations for output parameters ----

template<> inline size_t outputLength<OUTPUT_VALUE_SINGLE>()  {return 1;}
template<> inline size_t outputLength<OUTPUT_VALUE_TRIPLET>() {return 3;}
template<> inline size_t outputLength<OUTPUT_VALUE_SEXTET>()  {return 6;}
template<> inline size_t outputLength<OUTPUT_VALUE_SINGLE_AND_SINGLE>()   {return 2;}
template<> inline size_t outputLength<OUTPUT_VALUE_SINGLE_AND_TRIPLET>()  {return 4;}
template<> inline size_t outputLength<OUTPUT_VALUE_SINGLE_AND_SEXTET>()   {return 7;}
template<> inline size_t outputLength<OUTPUT_VALUE_TRIPLET_AND_TRIPLET>() {return 6;}
template<> inline size_t outputLength<OUTPUT_VALUE_TRIPLET_AND_SEXTET>()  {return 9;}
template<> inline size_t outputLength<OUTPUT_VALUE_SINGLE_AND_TRIPLET_AND_SEXTET>() {return 10;}

template<> PyObject* formatTuple<OUTPUT_VALUE_SINGLE>(const double result[]) {
    return Py_BuildValue("d", result[0]);
}
template<> PyObject* formatTuple<OUTPUT_VALUE_TRIPLET>(const double result[]) {
    return Py_BuildValue("ddd", result[0], result[1], result[2]);
}
template<> PyObject* formatTuple<OUTPUT_VALUE_SEXTET>(const double result[]) {
    return Py_BuildValue("dddddd",
        result[0], result[1], result[2], result[3], result[4], result[5]);
}
template<> PyObject* formatTuple<OUTPUT_VALUE_SINGLE_AND_SINGLE>(const double result[]) {
    return Py_BuildValue("(dd)", result[0], result[1]);
}
template<> PyObject* formatTuple<OUTPUT_VALUE_SINGLE_AND_TRIPLET>(const double result[]) {
    return Py_BuildValue("d(ddd)", result[0], result[1], result[2], result[3]);
}
template<> PyObject* formatTuple<OUTPUT_VALUE_SINGLE_AND_SEXTET>(const double result[]) {
    return Py_BuildValue("d(dddddd)", result[0],
        result[1], result[2], result[3], result[4], result[5], result[6]);
}
template<> PyObject* formatTuple<OUTPUT_VALUE_TRIPLET_AND_TRIPLET>(const double result[]) {
    return Py_BuildValue("(ddd)(ddd)", result[0], result[1], result[2],
        result[3], result[4], result[5]);
}
template<> PyObject* formatTuple<OUTPUT_VALUE_TRIPLET_AND_SEXTET>(const double result[]) {
    return Py_BuildValue("(ddd)(dddddd)", result[0], result[1], result[2],
        result[3], result[4], result[5], result[6], result[7], result[8]);
}
template<> PyObject* formatTuple<OUTPUT_VALUE_SINGLE_AND_TRIPLET_AND_SEXTET>(const double result[]) {
    return Py_BuildValue("d(ddd)(dddddd)", result[0], result[1], result[2], result[3],
        result[4], result[5], result[6], result[7], result[8], result[9]);
}

template<> PyObject* allocOutputArr<OUTPUT_VALUE_SINGLE>(int size) {
    npy_intp dims[] = {size};
    return PyArray_SimpleNew(1, dims, NPY_DOUBLE);
}
template<> PyObject* allocOutputArr<OUTPUT_VALUE_TRIPLET>(int size) {
    npy_intp dims[] = {size, 3};
    return PyArray_SimpleNew(2, dims, NPY_DOUBLE);
}
template<> PyObject* allocOutputArr<OUTPUT_VALUE_SEXTET>(int size) {
    npy_intp dims[] = {size, 6};
    return PyArray_SimpleNew(2, dims, NPY_DOUBLE);
}
template<> PyObject* allocOutputArr<OUTPUT_VALUE_SINGLE_AND_SINGLE>(int size) {
    npy_intp dims[] = {size};
    PyObject* arr1 = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyObject* arr2 = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    return Py_BuildValue("NN", arr1, arr2);
}
template<> PyObject* allocOutputArr<OUTPUT_VALUE_SINGLE_AND_TRIPLET>(int size) {
    npy_intp dims1[] = {size};
    npy_intp dims2[] = {size, 3};
    PyObject* arr1 = PyArray_SimpleNew(1, dims1, NPY_DOUBLE);
    PyObject* arr2 = PyArray_SimpleNew(2, dims2, NPY_DOUBLE);
    return Py_BuildValue("NN", arr1, arr2);
}
template<> PyObject* allocOutputArr<OUTPUT_VALUE_SINGLE_AND_SEXTET>(int size) {
    npy_intp dims1[] = {size};
    npy_intp dims2[] = {size, 6};
    PyObject* arr1 = PyArray_SimpleNew(1, dims1, NPY_DOUBLE);
    PyObject* arr2 = PyArray_SimpleNew(2, dims2, NPY_DOUBLE);
    return Py_BuildValue("NN", arr1, arr2);
}
template<> PyObject* allocOutputArr<OUTPUT_VALUE_TRIPLET_AND_TRIPLET>(int size) {
    npy_intp dims[] = {size, 3};
    PyObject* arr1 = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    PyObject* arr2 = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    return Py_BuildValue("NN", arr1, arr2);
}
template<> PyObject* allocOutputArr<OUTPUT_VALUE_TRIPLET_AND_SEXTET>(int size) {
    npy_intp dims1[] = {size, 3};
    npy_intp dims2[] = {size, 6};
    PyObject* arr1 = PyArray_SimpleNew(2, dims1, NPY_DOUBLE);
    PyObject* arr2 = PyArray_SimpleNew(2, dims2, NPY_DOUBLE);
    return Py_BuildValue("NN", arr1, arr2);
}
template<> PyObject* allocOutputArr<OUTPUT_VALUE_SINGLE_AND_TRIPLET_AND_SEXTET>(int size) {
    npy_intp dims1[] = {size};
    npy_intp dims2[] = {size, 3};
    npy_intp dims3[] = {size, 6};
    PyObject* arr1 = PyArray_SimpleNew(1, dims1, NPY_DOUBLE);
    PyObject* arr2 = PyArray_SimpleNew(2, dims2, NPY_DOUBLE);
    PyObject* arr3 = PyArray_SimpleNew(2, dims3, NPY_DOUBLE);
    return Py_BuildValue("NNN", arr1, arr2, arr3);
}

template<> void formatOutputArr<OUTPUT_VALUE_SINGLE>(
    const double result[], const int index, PyObject* resultObj) 
{
    ((double*)PyArray_DATA((PyArrayObject*)resultObj))[index] = result[0];
}
template<> void formatOutputArr<OUTPUT_VALUE_TRIPLET>(
    const double result[], const int index, PyObject* resultObj) 
{
    for(int d=0; d<3; d++)
        ((double*)PyArray_DATA((PyArrayObject*)resultObj))[index*3+d] = result[d];
}
template<> void formatOutputArr<OUTPUT_VALUE_SEXTET>(
    const double result[], const int index, PyObject* resultObj) 
{
    for(int d=0; d<6; d++)
        ((double*)PyArray_DATA((PyArrayObject*)resultObj))[index*6+d] = result[d];
}
template<> void formatOutputArr<OUTPUT_VALUE_SINGLE_AND_SINGLE>(
    const double result[], const int index, PyObject* resultObj) 
{
    PyArrayObject* arr1 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 0);
    PyArrayObject* arr2 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 1);
    ((double*)PyArray_DATA(arr1))[index] = result[0];
    ((double*)PyArray_DATA(arr2))[index] = result[1];
}
template<> void formatOutputArr<OUTPUT_VALUE_SINGLE_AND_TRIPLET>(
    const double result[], const int index, PyObject* resultObj) 
{
    PyArrayObject* arr1 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 0);
    PyArrayObject* arr2 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 1);
    ((double*)PyArray_DATA(arr1))[index] = result[0];
    for(int d=0; d<3; d++)
        ((double*)PyArray_DATA(arr2))[index*3+d] = result[d+1];
}
template<> void formatOutputArr<OUTPUT_VALUE_SINGLE_AND_SEXTET>(
    const double result[], const int index, PyObject* resultObj) 
{
    PyArrayObject* arr1 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 0);
    PyArrayObject* arr2 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 1);
    ((double*)PyArray_DATA(arr1))[index] = result[0];
    for(int d=0; d<6; d++)
        ((double*)PyArray_DATA(arr2))[index*6+d] = result[d+1];
}
template<> void formatOutputArr<OUTPUT_VALUE_TRIPLET_AND_TRIPLET>(
    const double result[], const int index, PyObject* resultObj) 
{
    PyArrayObject* arr1 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 0);
    PyArrayObject* arr2 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 1);
    for(int d=0; d<3; d++) {
        ((double*)PyArray_DATA(arr1))[index*3+d] = result[d];
        ((double*)PyArray_DATA(arr2))[index*3+d] = result[d+3];
    }
}
template<> void formatOutputArr<OUTPUT_VALUE_TRIPLET_AND_SEXTET>(
    const double result[], const int index, PyObject* resultObj) 
{
    PyArrayObject* arr1 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 0);
    PyArrayObject* arr2 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 1);
    for(int d=0; d<3; d++)
        ((double*)PyArray_DATA(arr1))[index*3+d] = result[d];
    for(int d=0; d<6; d++)
        ((double*)PyArray_DATA(arr2))[index*6+d] = result[d+3];
}
template<> void formatOutputArr<OUTPUT_VALUE_SINGLE_AND_TRIPLET_AND_SEXTET>(
    const double result[], const int index, PyObject* resultObj) 
{
    PyArrayObject* arr1 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 0);
    PyArrayObject* arr2 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 1);
    PyArrayObject* arr3 = (PyArrayObject*) PyTuple_GET_ITEM(resultObj, 2);
    ((double*)PyArray_DATA(arr1))[index] = result[0];
    for(int d=0; d<3; d++)
        ((double*)PyArray_DATA(arr2))[index*3+d] = result[d+1];
    for(int d=0; d<6; d++)
        ((double*)PyArray_DATA(arr3))[index*6+d] = result[d+4];
}

/** A general function that computes something for one or many input points.
    \tparam numArgs  is the size of array that contains the value of a single input point.
    \tparam numOutput is the identifier (not literally the size) of output data format 
    for a single input point: it may be a single number, an array of floats, or even several arrays.
    \param[in]  fnc  is the function pointer to the routine that actually computes something,
    taking a pointer to an instance of Python object, an array of floats as the input point,
    and producing another array of floats as the output.
    \param[in] params  is the pointer to auxiliary parameters that is passed to the 'fnc' routine
    \param[in] args  is the arguments of the function call: it may be a sequence of numArg floats 
    that represents a single input point, or a 1d array of the same length and same meaning,
    or a 2d array of dimensions N * numArgs, representing N input points.
    \returns  the result of applying 'fnc' to one or many input points, in the form determined 
    both by the number of input points, and the output data format. 
    The output for a single point may be a sequence of numbers (tuple or 1d array), 
    or several such arrays forming a tuple (e.g., [ [1,2,3], [1,2,3,4,5,6] ]). 
    The output for an array of input points would be one or several 2d arrays of length N and 
    shape determined by the output format, i.e., for the above example it would be ([N,3], [N,6]).
*/
template<int numArgs, int numOutput>
static PyObject* callAnyFunctionOnArray(void* params, PyObject* args, anyFunction fnc)
{
    if(args==NULL) {
        PyErr_SetString(PyExc_ValueError, "No input data provided");
        return NULL;
    }
    double input [inputLength<numArgs>()];
    double result[outputLength<numOutput>()];
    try{
        if(parseTuple<numArgs>(args, input)) {  // one point
            fnc(params, input, result);
            return formatTuple<numOutput>(result);
        }
        PyErr_Clear();  // clear error if the argument list is not a tuple of a proper type
        PyObject* obj=NULL;
        if(PyArray_Check(args))
            obj = args;
        else if(PyTuple_Check(args) && PyTuple_Size(args)==1)
            obj = PyTuple_GET_ITEM(args, 0);
        if(obj) {
            PyArrayObject *arr  = (PyArrayObject*) PyArray_FROM_OTF(obj,  NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
            if(arr == NULL) {
                PyErr_SetString(PyExc_ValueError, "Input does not contain a valid array");
                return NULL;
            }
            int numpt = 0;
            if(PyArray_NDIM(arr) == 1 && PyArray_DIM(arr, 0) == numArgs) 
            {   // 1d array of length numArgs - a single point
                fnc(params, static_cast<double*>(PyArray_GETPTR1(arr, 0)), result);
                Py_DECREF(arr);
                return formatTuple<numOutput>(result);
            }
            // check the shape of input array
            numpt = parseArray<numArgs>(arr);
            if(numpt == 0) {
                PyErr_SetString(PyExc_ValueError, errStrInvalidArrayDim<numArgs>());
                Py_DECREF(arr);
                return NULL;
            }
            // allocate an appropriate output object
            PyObject* resultObj = allocOutputArr<numOutput>(numpt);
            // loop over input array
            for(int i=0; i<numpt; i++) {
                fnc(params, static_cast<double*>(PyArray_GETPTR2(arr, i, 0)), result);
                formatOutputArr<numOutput>(result, i, resultObj);
            }
            Py_DECREF(arr);
            return resultObj;
        }
        PyErr_SetString(PyExc_ValueError, errStrInvalidInput<numArgs>());
        return NULL;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, (std::string("Exception occured: ")+e.what()).c_str());
        return NULL;
    }
}

///@}
/// \name  ---------- Potential class and related data ------------
///@{

/// \cond INTERNAL_DOCS
/// Python type corresponding to Potential class
typedef struct {
    PyObject_HEAD
    potential::PtrPotential pot;
} PotentialObject;
/// \endcond

static PyObject* Potential_new(PyTypeObject *type, PyObject*, PyObject*)
{
    PotentialObject *self = (PotentialObject*)type->tp_alloc(type, 0);
    return (PyObject*)self;
}

static void Potential_dealloc(PyObject* self)
{
    ((PotentialObject*)self)->pot.reset();
    self->ob_type->tp_free(self);
}

/// pointer to the Potential type object (will be initialized below)
static PyTypeObject* PotentialTypePtr;

/// description of Potential class
static const char* docstringPotential = 
    "Potential is a class that represents a wide range of gravitational potentials\n"
    "There are several ways of initializing the potential instance:\n"
    "  - from a list of key=value arguments that specify an elementary potential class;\n"
    "  - from a tuple of dictionary objects that contain the same list of possible "
    "key/value pairs for each component of a composite potential;\n"
    "  - from an INI file with these parameters for one or several components;\n"
    "  - from a tuple of existing Potential objects created previously: "
    "in this case a composite potential is created from these components, "
    "but the original objects cannot be used anymore.\n"
    "Note that all keywords and their values are not case-sensitive.\n\n"
    "List of possible keywords for a single component:\n"
    "  type='...'   the type of potential, can be one of the following 'basic' types:\n"
    "    Harmonic, Logarithmic, Plummer, MiyamotoNagai, NFW, Ferrers, Dehnen, "
    "OblatePerfectEllipsoid, DiskAnsatz, SpheroidDensity;\n"
    "    or one of the expansion types:  BasisSetExp, SplineExp, CylSplineExp - "
    "in these cases, one should provide either a density model, file name, "
    "or an array of points.\n"
    "  mass=...   total mass of the model, if applicable.\n"
    "  scaleRadius=...   scale radius of the model (if applicable).\n"
    "  scaleHeight=...   scale height of the model (currently applicable to "
    "Dehnen, MiyamotoNagai and DiskAnsatz).\n"
    "  axisRatio=...   axis ratio z/R for SpheroidDensity density profiles.\n"
    "  q=...   axis ratio y/x, i.e., intermediate to long axis (applicable to triaxial "
    "potential models such as Dehnen and Ferrers).\n"
    "  p=...   axis ratio z/x, i.e., short to long axis (if applicable, same as axisRatio).\n"
    "  gamma=...   central cusp slope (applicable for Dehnen and SpheroidDensity).\n"
    "  beta=...   outer density slope (SpheroidDensity).\n"
    "  innerCutoffRadius=...   radius of inner hole (DiskAnsatz).\n"
    "  outerCutoffRadius=...   radius of outer exponential cutoff (SpheroidDensity).\n"
    "  surfaceDensity=...   central surface density (or its value if no inner cutoff exists), "
    "for DiskAnsatz.\n"
    "  densityNorm=...   normalization of density profile for SpheroidDensity (the value "
    "at scaleRadius).\n"
    "Parameters for potential expansions:\n"
    "  density='...'   the density model for a potential expansion "
    "(most of the above elementary potentials can be used as density models, "
    "except those with infinite mass; in addition, a selection of density models"
    "without a corresponding potential will be available soon).\n"
    "  file='...'   the name of a file with potential coefficients for a potential "
    "expansion (an alternative to density='...'), or with an N-body snapshot that "
    "will be used to compute the coefficients.\n"
    "  points=(coords, mass)   array of point masses to be used in construction "
    "of a potential expansion (an alternative to density='...' or file='...' options): "
    "should be a tuple with two arrays - coordinates and mass, where the first one is "
    "a two-dimensional Nx3 array and the second one is a one-dimensional array of length N.\n"
    "  symmetry='...'   assumed symmetry for potential expansion constructed from "
    "an N-body snapshot (possible options, in order of decreasing symmetry: "
    "'Spherical', 'Axisymmetric', 'Triaxial', 'Reflection', 'None').\n"
    "  numCoefsRadial=...   number of radial terms in BasisSetExp or grid points in spline potentials.\n"
    "  numCoefsAngular=...   order of spherical-harmonic expansion (max.index of angular harmonic coefficient).\n"
    "  numCoefsVertical=...   number of coefficients in z-direction for CylSplineExp potential.\n"
    "  alpha=...   parameter that determines functional form of BasisSetExp potential.\n"
    "  splineSmoothfactor=...   amount of smoothing in SplineExp initialized from an N-body snapshot.\n"
    "  splineRmin=...   if nonzero, specifies the innermost grid node radius for SplineExp and CylSplineExp.\n"
    "  splineRmax=...   if nonzero, specifies the outermost grid node radius for SplineExp and CylSplineExp.\n"
    "  splineZmin=...   if nonzero, specifies the z-value of the innermost grid node in CylSplineExp.\n"
    "  splineZmax=...   if nonzero, specifies the z-value of the outermost grid node in CylSplineExp.\n"
    "\nMost of these parameters have reasonable default values; the only necessary ones are "
    "`type`, and for a potential expansion, `density` or `file` or `points`.\n\n"
    "Examples:\n\n"
    ">>> pot_halo = Potential(type='Dehnen', mass=1e12, gamma=1, scaleRadius=100, q=0.8, p=0.6)\n"
    ">>> pot_disk = Potential(type='MiyamotoNagai', mass=5e10, scaleRadius=5, scaleHeight=0.5)\n"
    ">>> pot_from_ini = Potential('my_potential.ini')\n"
    ">>> pot_composite = Potential(pot_halo, pot_disk)\n"
    ">>> disk_par = dict(type='DiskAnsatz', surfaceDensity=1e9, scaleRadius=3, scaleHeight=0.4)\n"
    ">>> halo_par = dict(type='SpheroidDensity', densityNorm=2e7, scaleRadius=15, gamma=1, beta=3, "
    "outerCutoffRadius=150, axisRatio=0.8)\n"
    ">>> pot_galpot = Potential(disk_par, halo_par)\n"
    "\nThe latter example illustrates the use of GalPot components (exponential disks and spheroids) "
    "from Dehnen&Binney 1998; these are internally implemented using another variant of potential expansion, "
    "but may also be combined with any other component if needed.\n"
    "The numerical values in the above examples are given in solar masses and kiloparsecs; "
    "a call to `set_units` should precede the construction of potentials in this approach. "
    "Alternatively, one may provide no units at all, and use the `N-body` convention G=1 "
    "(this is the default regime and is restored by `reset_units`).\n";

/// attempt to construct potential from an array of particles
static potential::PtrPotential Potential_initFromParticles(
    const utils::KeyValueMap& params, PyObject* points)
{
    if(params.contains("file"))
        throw std::invalid_argument("Cannot provide both 'points' and 'file' arguments");
    PyObject *pointCoordObj, *pointMassObj;
    if(!PyArg_ParseTuple(points, "OO", &pointCoordObj, &pointMassObj)) {
        throw std::invalid_argument("'points' must be a tuple with two arrays - "
            "coordinates and mass, where the first one is a two-dimensional Nx3 array "
            "and the second one is a one-dimensional array of length N");
    }
    PyArrayObject *pointCoordArr = (PyArrayObject*)
        PyArray_FROM_OTF(pointCoordObj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *pointMassArr  = (PyArrayObject*)
        PyArray_FROM_OTF(pointMassObj,  NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(pointCoordArr == NULL || pointMassArr == NULL) {
        Py_XDECREF(pointCoordArr);
        Py_XDECREF(pointMassArr);
        throw std::invalid_argument("'points' does not contain valid arrays");
    }
    int numpt = 0;
    if(PyArray_NDIM(pointMassArr) == 1)
        numpt = PyArray_DIM(pointMassArr, 0);
    if(numpt == 0 || PyArray_NDIM(pointCoordArr) != 2 || 
        PyArray_DIM(pointCoordArr, 0) != numpt || PyArray_DIM(pointCoordArr, 1) != 3)
    {
        Py_DECREF(pointCoordArr);
        Py_DECREF(pointMassArr);
        throw std::invalid_argument("'points' does not contain valid arrays "
            "(the first one must be 2d array of shape Nx3 and the second one must be 1d array of length N)");
    }
    particles::PointMassArray<coord::PosCar> pointArray;
    pointArray.data.reserve(numpt);
    for(int i=0; i<numpt; i++) {
        pointArray.add(convertPos((double*)PyArray_GETPTR2(pointCoordArr, i, 0)), 
            *((double*)PyArray_GETPTR1(pointMassArr, i)) * conv->massUnit);
    }
    Py_DECREF(pointCoordArr);
    Py_DECREF(pointMassArr);
    return potential::createPotentialFromPoints(params, *conv, pointArray);
}

/// attempt to construct an elementary potential from the parameters provided in dictionary
static potential::PtrPotential Potential_initFromDict(PyObject* args)
{
    utils::KeyValueMap params = convertPyDictToKeyValueMap(args);
    // check if the list of arguments contains points
    PyObject* points = PyDict_GetItemString(args, "points");
    if(points) {
        params.unset("points");
        return Potential_initFromParticles(params, points);
    }
    return potential::createPotential(params, *conv);
}

/// attempt to construct a composite potential from a tuple of Potential objects
/// or dictionaries with potential parameters
static potential::PtrPotential Potential_initFromTuple(PyObject* tuple)
{
    if(PyTuple_Size(tuple) == 1 && PyString_Check(PyTuple_GET_ITEM(tuple, 0)))
    {   // assuming that we have one parameter which is the INI file name
        return potential::createPotential(PyString_AsString(PyTuple_GET_ITEM(tuple, 0)), *conv);
    }
    bool onlyPot = true, onlyDict = true;
    // first check the types of tuple elements
    for(Py_ssize_t i=0; i<PyTuple_Size(tuple); i++) {
        onlyPot &= PyObject_TypeCheck(PyTuple_GET_ITEM(tuple, i), PotentialTypePtr) &&
             ((PotentialObject*)PyTuple_GET_ITEM(tuple, i))->pot;  // an existing Potential object
        onlyDict &= PyDict_Check(PyTuple_GET_ITEM(tuple, i));      // a dictionary with param=value pairs
    }
    if(onlyPot) {
        std::vector<potential::PtrPotential> components;
        for(Py_ssize_t i=0; i<PyTuple_Size(tuple); i++) {
            components.push_back(((PotentialObject*)PyTuple_GET_ITEM(tuple, i))->pot);
        }
        return potential::PtrPotential(new potential::CompositeCyl(components));
    } else if(onlyDict) {
        std::vector<utils::KeyValueMap> paramsArr;
        for(Py_ssize_t i=0; i<PyTuple_Size(tuple); i++) {
            paramsArr.push_back(convertPyDictToKeyValueMap(PyTuple_GET_ITEM(tuple, i)));
        }
        return potential::createPotential(paramsArr, *conv);
    } else
        throw std::invalid_argument(
            "The tuple should contain either Potential objects or dictionaries with potential parameters");
}

/// the generic constructor of Potential object
static int Potential_init(PyObject* self, PyObject* args, PyObject* namedArgs)
{
    potential::PtrPotential pot;
    try{
        // check if we have only a tuple of potential components as arguments
        if(args!=NULL && PyTuple_Check(args) && PyTuple_Size(args)>0 && 
            (namedArgs==NULL || PyDict_Size(namedArgs)==0))
            ((PotentialObject*)self)->pot = Potential_initFromTuple(args);
        else if(namedArgs!=NULL && PyDict_Check(namedArgs) && PyDict_Size(namedArgs)>0)
            ((PotentialObject*)self)->pot = Potential_initFromDict(namedArgs);
        else {
            printf("Received %d positional arguments", (int)PyTuple_Size(args));
            if(namedArgs==NULL)
                printf(" and no named arguments\n");
            else
                printf(" and %d named arguments\n", (int)PyDict_Size(namedArgs));
            throw std::invalid_argument(
                "Invalid parameters passed to the constructor, type help(Potential) for details");
        }
        assert(((PotentialObject*)self)->pot);
#ifdef DEBUGPRINT
        printf("Created an instance of %s potential\n", pot->name());
#endif
        return 0;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, (std::string("Error in creating potential: ")+e.what()).c_str());
        return -1;
    }
}

static bool Potential_isCorrect(PyObject* self)
{
    if(self==NULL) {
        PyErr_SetString(PyExc_ValueError, "Should be called as method of Potential object");
        return false;
    }
    if(!((PotentialObject*)self)->pot) {
        PyErr_SetString(PyExc_ValueError, "Potential is not initialized properly");
        return false;
    }
    return true;
}

// function that do actually compute something from the potential object,
// applying appropriate unit conversions

static void fncPotential(void* obj, const double input[], double *result) {
    const coord::PosCar point = convertPos(input);
    result[0] = ((PotentialObject*)obj)->pot->value(point)
        / pow_2(conv->velocityUnit);   // unit of potential is V^2
}
static PyObject* Potential_potential(PyObject* self, PyObject* args) {
    if(!Potential_isCorrect(self))
        return NULL;
    return callAnyFunctionOnArray<INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SINGLE>
        (self, args, fncPotential);
}
static PyObject* Potential_value(PyObject* self, PyObject* args, PyObject* /*namedArgs*/) {
    return Potential_potential(self, args);
}

static void fncDensity(void* obj, const double input[], double *result) {
    const coord::PosCar point = convertPos(input);
    result[0] = ((PotentialObject*)obj)->pot->density(point)
        / (conv->massUnit / pow_2(conv->lengthUnit));  // unit of density is M/L^3
}
static PyObject* Potential_density(PyObject* self, PyObject* args) {
    if(!Potential_isCorrect(self))
        return NULL;
    return callAnyFunctionOnArray<INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SINGLE>
        (self, args, fncDensity);
}

static void fncForce(void* obj, const double input[], double *result) {
    const coord::PosCar point = convertPos(input);
    coord::GradCar grad;
    ((PotentialObject*)obj)->pot->eval(point, NULL, &grad);
    // unit of force per unit mass is V/T
    const double convF = 1 / (conv->velocityUnit/conv->timeUnit);
    result[0] = -grad.dx * convF;
    result[1] = -grad.dy * convF;
    result[2] = -grad.dz * convF;
}
static PyObject* Potential_force(PyObject* self, PyObject* args) {
    if(!Potential_isCorrect(self))
        return NULL;
    return callAnyFunctionOnArray<INPUT_VALUE_TRIPLET, OUTPUT_VALUE_TRIPLET>
        (self, args, fncForce);
}

static void fncForceDeriv(void* obj, const double input[], double *result) {
    const coord::PosCar point = convertPos(input);
    coord::GradCar grad;
    coord::HessCar hess;
    ((PotentialObject*)obj)->pot->eval(point, NULL, &grad, &hess);
    // unit of force per unit mass is V/T
    const double convF = 1 / (conv->velocityUnit/conv->timeUnit);
    // unit of force deriv per unit mass is V/T^2
    const double convD = 1 / (conv->velocityUnit/pow_2(conv->timeUnit));
    result[0] = -grad.dx * convF;
    result[1] = -grad.dy * convF;
    result[2] = -grad.dz * convF;
    result[3] = -hess.dx2  * convD;
    result[4] = -hess.dy2  * convD;
    result[5] = -hess.dz2  * convD;
    result[6] = -hess.dxdy * convD;
    result[7] = -hess.dydz * convD;
    result[8] = -hess.dxdz * convD;
}
static PyObject* Potential_force_deriv(PyObject* self, PyObject* args) {
    if(!Potential_isCorrect(self))
        return NULL;
    return callAnyFunctionOnArray<INPUT_VALUE_TRIPLET, OUTPUT_VALUE_TRIPLET_AND_SEXTET>
        (self, args, fncForceDeriv);
}

static PyObject* Potential_name(PyObject* self)
{
    if(!Potential_isCorrect(self))
        return NULL;
    return Py_BuildValue("s", ((PotentialObject*)self)->pot->name());
}

static PyObject* Potential_export(PyObject* self, PyObject* args)
{
    const char* filename=NULL;
    if(!Potential_isCorrect(self) || !PyArg_ParseTuple(args, "s", &filename))
        return NULL;
    try{
        writePotential(filename, *((PotentialObject*)self)->pot);
        Py_INCREF(Py_None);
        return Py_None;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, (std::string("Error writing file: ")+e.what()).c_str());
        return NULL;
    }
}

static PyObject* Potential_totalMass(PyObject* self)
{
    if(!Potential_isCorrect(self))
        return NULL;
    return Py_BuildValue("d", ((PotentialObject*)self)->pot->totalMass() / conv->massUnit);
}

static PyMethodDef Potential_methods[] = {
    { "name", (PyCFunction)Potential_name, METH_NOARGS, 
      "Return the name of the potential\n"
      "No arguments\n"
      "Returns: string" },
    { "potential", Potential_potential, METH_VARARGS, 
      "Compute potential at a given point or array of points\n"
      "Arguments: a triplet of floats (x,y,z) or array of such triplets\n"
      "Returns: float or array of floats" },
    { "density", Potential_density, METH_VARARGS, 
      "Compute density at a given point or array of points\n"
      "Arguments: a triplet of floats (x,y,z) or array of such triplets\n"
      "Returns: float or array of floats" },
    { "force", Potential_force, METH_VARARGS, 
      "Compute force at a given point or array of points\n"
      "Arguments: a triplet of floats (x,y,z) or array of such triplets\n"
      "Returns: float[3] - x,y,z components of force, or array of such triplets" },
    { "force_deriv", Potential_force_deriv, METH_VARARGS, 
      "Compute force and its derivatives at a given point or array of points\n"
      "Arguments: a triplet of floats (x,y,z) or array of such triplets\n"
      "Returns: (float[3],float[6]) - x,y,z components of force, "
      "and the matrix of force derivatives stored as dFx/dx,dFy/dy,dFz/dz,dFx/dy,dFy/dz,dFz/dx; "
      "or if the input was an array of N points, then both items in the tuple are 2d arrays "
      "with sizes Nx3 and Nx6, respectively"},
    { "export", Potential_export, METH_VARARGS, 
      "Export potential expansion coefficients to a text file\n"
      "Arguments: filename (string)\n"
      "Returns: none" },
    { "total_mass", (PyCFunction)Potential_totalMass, METH_NOARGS, 
      "Return the total mass of the density model\n"
      "No arguments\n"
      "Returns: float number" },
    { NULL, NULL, 0, NULL }
};

static PyTypeObject PotentialType = {
    PyObject_HEAD_INIT(NULL)
    0, "agama.Potential",
    sizeof(PotentialObject), 0, Potential_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, Potential_value, Potential_name, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringPotential, 
    0, 0, 0, 0, 0, 0, Potential_methods, 0, 0, 0, 0, 0, 0, 0,
    Potential_init, 0, Potential_new
};

///@}
/// \name  ---------- ActionFinder class and related data ------------
///@{

/// create a spherical or non-spherical action finder
actions::PtrActionFinder createActionFinder(const potential::PtrPotential& pot)
{
    if(isSpherical(*pot))
        return actions::PtrActionFinder(new actions::ActionFinderSpherical(pot));
    else
        return actions::PtrActionFinder(new actions::ActionFinderAxisymFudge(pot));
}

/// \cond INTERNAL_DOCS
/// Python type corresponding to ActionFinder class
typedef struct {
    PyObject_HEAD
    actions::PtrActionFinder af;  // C++ object for action finder
} ActionFinderObject;
/// \endcond

static PyObject* ActionFinder_new(PyTypeObject *type, PyObject*, PyObject*)
{
    ActionFinderObject *self = (ActionFinderObject*)type->tp_alloc(type, 0);
    return (PyObject*)self;
}

static void ActionFinder_dealloc(PyObject* self)
{
    ((ActionFinderObject*)self)->af.reset();
    self->ob_type->tp_free(self);
}

static const char* docstringActionFinder =
    "ActionFinder object is created for a given potential, and its () operator "
    "computes actions for a given position/velocity point, or array of points\n"
    "Arguments: a sextet of floats (x,y,z,vx,vy,vz) or array of such sextets\n"
    "Returns: float or array of floats (for each point: Jr, Jz, Jphi)";

static int ActionFinder_init(PyObject* self, PyObject* args, PyObject* /*namedArgs*/)
{
    PyObject* objPot=NULL;
    if(!PyArg_ParseTuple(args, "O", &objPot)) {
        PyErr_SetString(PyExc_ValueError, "Incorrect parameters for ActionFinder constructor: "
            "must provide an instance of Potential to work with.");
        return -1;
    }
    if(!PyObject_TypeCheck(objPot, &PotentialType) || 
        ((PotentialObject*)objPot)->pot==NULL ) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a valid instance of Potential class");
        return -1;
    }
    try{
        ((ActionFinderObject*)self)->af = createActionFinder(((PotentialObject*)objPot)->pot);
        return 0;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, 
            (std::string("Error in ActionFinder initialization: ")+e.what()).c_str());
        return -1;
    }
}

static void fncActions(void* obj, const double input[], double *result) {
    try{
        const coord::PosVelCyl point = coord::toPosVelCyl(convertPosVel(input));
        actions::Actions acts = ((ActionFinderObject*)obj)->af->actions(point);
        // unit of action is V*L
        const double convA = 1 / (conv->velocityUnit * conv->lengthUnit);
        result[0] = acts.Jr   * convA;
        result[1] = acts.Jz   * convA;
        result[2] = acts.Jphi * convA;
    }
    catch(std::exception& ) {  // indicates an error, e.g., positive value of energy
        result[0] = result[1] = result[2] = NAN;
    }
}
static PyObject* ActionFinder_value(PyObject* self, PyObject* args, PyObject* /*namedArgs*/)
{
    if(!((ActionFinderObject*)self)->af)
        return NULL;
    return callAnyFunctionOnArray<INPUT_VALUE_SEXTET, OUTPUT_VALUE_TRIPLET>
        (self, args, fncActions);
}


static PyTypeObject ActionFinderType = {
    PyObject_HEAD_INIT(NULL)
    0, "agama.ActionFinder",
    sizeof(ActionFinderObject), 0, ActionFinder_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, ActionFinder_value, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringActionFinder, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ActionFinder_init, 0, ActionFinder_new
};

/// \cond INTERNAL_DOCS
/// standalone action finder
typedef struct {
    potential::PtrPotential pot;
    double ifd;
} ActionFinderParams;
/// \endcond

static void fncActionsStandalone(void* obj, const double input[], double *result) {
    try{
        const coord::PosVelCyl point = coord::toPosVelCyl(convertPosVel(input));
        const ActionFinderParams* params = static_cast<const ActionFinderParams*>(obj);
        double ifd = params->ifd * conv->lengthUnit;
        actions::Actions acts = isSpherical(*params->pot) ?
            actions::actionsSpherical  (*params->pot, point) :
            actions::actionsAxisymFudge(*params->pot, point, ifd);
        // unit of action is V*L
        const double convA = 1 / (conv->velocityUnit * conv->lengthUnit);
        result[0] = acts.Jr   * convA;
        result[1] = acts.Jz   * convA;
        result[2] = acts.Jphi * convA;
    }
    catch(std::exception& ) {  // indicates an error, e.g., positive value of energy
        result[0] = result[1] = result[2] = NAN;
    }
}

static const char* docstringActions = 
    "Compute actions for a given position/velocity point, or array of points\n"
    "Arguments: \n"
    "    point - a sextet of floats (x,y,z,vx,vy,vz) or array of such sextets;\n"
    "    pot - Potential object that defines the gravitational potential;\n"
    "    ifd (float) - interfocal distance for the prolate spheroidal coordinate system.\n"
    "Returns: float or array of floats (for each point: Jr, Jz, Jphi)";
static PyObject* find_actions(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"point", "pot", "ifd", NULL};
    double ifd = 0;
    PyObject *points_obj = NULL, *pot_obj = NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "|OOd", const_cast<char**>(keywords),
        &points_obj, &pot_obj, &ifd) || ifd<=0)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to actions()");
        return NULL;
    }
    if(!PyObject_TypeCheck(pot_obj, &PotentialType) || ((PotentialObject*)pot_obj)->pot==NULL) {
        PyErr_SetString(PyExc_TypeError, "Argument 'pot' must be a valid instance of Potential class");
        return NULL;
    }
    ActionFinderParams params;
    params.pot = ((PotentialObject*)pot_obj)->pot;
    params.ifd = ifd;
    return callAnyFunctionOnArray<INPUT_VALUE_SEXTET, OUTPUT_VALUE_TRIPLET>
        (&params, points_obj, fncActionsStandalone);
}

///@}
/// \name  --------- DistributionFunction class -----------
///@{

/// \cond INTERNAL_DOCS
/// Python type corresponding to DistributionFunction class
typedef struct {
    PyObject_HEAD
    df::PtrDistributionFunction df;
} DistributionFunctionObject;
/// \endcond

static PyObject* DistributionFunction_new(PyTypeObject *type, PyObject*, PyObject*)
{
    DistributionFunctionObject *self = (DistributionFunctionObject*)type->tp_alloc(type, 0);
    return (PyObject*)self;
}

static void DistributionFunction_dealloc(PyObject* self)
{
    ((DistributionFunctionObject*)self)->df.reset();
    self->ob_type->tp_free(self);
}

static const char* docstringDistributionFunction =
    "DistributionFunction class represents an action-based distribution function.\n\n"
    "The constructor accepts several key=value arguments that describe the parameters "
    "of distribution function.\n"
    "Required parameter is type='...', specifying the type of DF: currently available types are "
    "'DoublePowerLaw' (for the halo) and 'PseudoIsothermal' (for the disk component).\n"
    "For the latter, one also needs to provide the potential to initialize the table of "
    "epicyclic frequencies (pot=... argument).\n"
    "Other parameters are specific to each DF type.\n\n"
    "The () operator computes the value of distribution function for the given triplet of actions.\n"
    "The total_mass() function computes the total mass in the entire phase space.\n";

static int DistributionFunction_init(PyObject* self, PyObject* args, PyObject* namedArgs)
{
    if(namedArgs==NULL || !PyDict_Check(namedArgs) || PyDict_Size(namedArgs)==0 ||
        (args!=0 && PyTuple_Check(args) && PyTuple_Size(args)>0) )
    {
        PyErr_SetString(PyExc_ValueError,
            "Should provide a list of key=value arguments and no positional arguments");
        return -1;
    }
    PyObject *pot_obj = PyDict_GetItemString(namedArgs, "pot");  // borrowed reference or NULL
    const potential::BasePotential* pot = NULL;
    if(pot_obj!=NULL) {
        if(!PyObject_TypeCheck(pot_obj, &PotentialType) || ((PotentialObject*)pot_obj)->pot==NULL) {
            PyErr_SetString(PyExc_TypeError, "Argument 'pot' must be a valid instance of Potential class");
            return NULL;
        }
        pot = ((PotentialObject*)pot_obj)->pot.get();
        PyDict_DelItemString(namedArgs, "pot");
    }
    try{
        utils::KeyValueMap params = convertPyDictToKeyValueMap(namedArgs);
        ((DistributionFunctionObject*)self)->df = df::createDistributionFunction(params, pot, *conv);
        assert(((DistributionFunctionObject*)self)->df);
        return 0;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError,
            (std::string("Error in creating distribution function: ")+e.what()).c_str());
        return -1;
    }
}

static void fncDistributionFunction(void* obj, const double input[], double *result) {
    const actions::Actions acts = convertActions(input);
    // dimension of distribution function is M L^-3 V^-3
    const double dim = pow_3(conv->velocityUnit * conv->lengthUnit) / conv->massUnit;
    try{
        result[0] = ((DistributionFunctionObject*)obj)->df->value(acts) * dim;
    }
    catch(std::exception& ) {
        result[0] = NAN;
    }
}

static PyObject* DistributionFunction_value(PyObject* self, PyObject* args, PyObject* /*namedArgs*/)
{
    if(((DistributionFunctionObject*)self)->df==NULL)
        return NULL;
    return callAnyFunctionOnArray<INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SINGLE>
        (self, args, fncDistributionFunction);
}

static PyObject* DistributionFunction_totalMass(PyObject* self)
{
    if(((DistributionFunctionObject*)self)->df==NULL)
        return NULL;
    double val = ((DistributionFunctionObject*)self)->df->totalMass(1e-5,1e7);
    return Py_BuildValue("d", val / conv->massUnit);
}

static PyMethodDef DistributionFunction_methods[] = {
    { "total_mass", (PyCFunction)DistributionFunction_totalMass, METH_NOARGS,
      "Return the total mass of the model (integral of the distribution function "
      "over the entire phase space of actions\n"
      "No arguments\n"
      "Returns: float number" },
    { NULL, NULL, 0, NULL }
};

static PyTypeObject DistributionFunctionType = {
    PyObject_HEAD_INIT(NULL)
    0, "agama.DistributionFunction",
    sizeof(DistributionFunctionObject), 0, DistributionFunction_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, DistributionFunction_value, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringDistributionFunction, 
    0, 0, 0, 0, 0, 0, DistributionFunction_methods, 0, 0, 0, 0, 0, 0, 0,
    DistributionFunction_init, 0, DistributionFunction_new
};

///@}
/// \name  ----- GalaxyModel class -----
///@{

/// \cond INTERNAL_DOCS
/// Python type corresponding to ActionFinder class
typedef struct {
    PyObject_HEAD
    potential::PtrPotential pot;
    df::PtrDistributionFunction df;
    actions::PtrActionFinder af;
} GalaxyModelObject;
/// \endcond

static PyObject* GalaxyModel_new(PyTypeObject *type, PyObject*, PyObject*)
{
    GalaxyModelObject *self = (GalaxyModelObject*)type->tp_alloc(type, 0);
    return (PyObject*)self;
}

static void GalaxyModel_dealloc(GalaxyModelObject* self)
{
    self->pot.reset();
    self->df.reset();
    self->af.reset();
    self->ob_type->tp_free((PyObject*)self);
}

static const char* docstringGalaxyModel =
    "GalaxyModel is a class that takes together a Potential, "
    "a DistributionFunction, and an ActionFinder objects, "
    "and provides methods to compute moments and projections of the distribution function "
    "at a given point in the ordinary phase space (coordinate/velocity), as well as "
    "methods for drawing samples from the distribution function in the given potential.\n"
    "The constructor takes the following arguments:\n"
    "  pot - a Potential object;\n"
    "  df  - a DistributionFunction object;\n"
    "  af (optional) - an ActionFinder object; if not provided then the action finder is created internally.\n";

static int GalaxyModel_init(PyObject* self, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"pot", "df", "af", NULL};
    PyObject *pot_obj = NULL, *df_obj = NULL, *af_obj = NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "OO|O", const_cast<char**>(keywords),
        &pot_obj, &df_obj, &af_obj))
    {
        PyErr_SetString(PyExc_ValueError, "GalaxyModel constructor takes two or three arguments: pot, df, [af]");
        return -1;
    }
    if(pot_obj==NULL || !PyObject_TypeCheck(pot_obj, &PotentialType) ||
       ((PotentialObject*)pot_obj)->pot==NULL )
    {
        PyErr_SetString(PyExc_TypeError,
            "Argument 'pot' must be a valid instance of Potential class");
        return -1;
    }
    ((GalaxyModelObject*)self)->pot = ((PotentialObject*)pot_obj)->pot;
    if(df_obj==NULL || !PyObject_TypeCheck(df_obj, &DistributionFunctionType) ||
       ((DistributionFunctionObject*)df_obj)->df==NULL )
    {
        PyErr_SetString(PyExc_TypeError,
            "Argument 'df' must be a valid instance of DistributionFunction class");
        return -1;
    }
    ((GalaxyModelObject*)self)->df = ((DistributionFunctionObject*)df_obj)->df;
    // af_obj might be NULL; if not NULL then check its validity
    if(af_obj!=NULL && (!PyObject_TypeCheck(af_obj, &ActionFinderType) ||
       ((ActionFinderObject*)af_obj)->af==NULL))
    {
        PyErr_SetString(PyExc_TypeError,
            "Argument 'af' must be a valid instance of ActionFinder class "
            "corresponding to the given potential");
        return -1;
    }
    if(af_obj==NULL) {  // no action finder provided - create one internally
        try{
            ((GalaxyModelObject*)self)->af = createActionFinder(((PotentialObject*)pot_obj)->pot);
        }
        catch(std::exception& e) {
            PyErr_SetString(PyExc_ValueError, 
                (std::string("Error in constructing action finder: ")+e.what()).c_str());
            return -1;
        }
    } else {
        ((GalaxyModelObject*)self)->af = ((ActionFinderObject*)af_obj)->af;
    }
    return 0;
}

/// generate samples in position/velocity space
static PyObject* GalaxyModel_sample_posvel(GalaxyModelObject* self, PyObject* args)
{
    int numPoints=0;
    if(!PyArg_ParseTuple(args, "i", &numPoints) || numPoints<=0)
    {
        PyErr_SetString(PyExc_ValueError, "sample() takes one integer argument - the number of points");
        return NULL;
    }
    try{
        // do the sampling
        galaxymodel::GalaxyModel galmod(*self->pot, *self->af, *self->df);
        particles::PointMassArrayCar points;
        galaxymodel::generatePosVelSamples(galmod, numPoints, points);

        // convert output to NumPy array
        numPoints = points.size();
        npy_intp dims[] = {numPoints, 6};
        PyArrayObject* posvel_arr = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        PyArrayObject* mass_arr   = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        for(int i=0; i<numPoints; i++) {
            unconvertPosVel(points.point(i), ((double*)PyArray_DATA(posvel_arr))+i*6);
            ((double*)PyArray_DATA(mass_arr))[i] = points.mass(i) / conv->massUnit;
        }
        return Py_BuildValue("NN", posvel_arr, mass_arr);
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, 
            (std::string("Error in sample(): ")+e.what()).c_str());
        return NULL;
    }
}

/// \cond INTERNAL_DOCS
struct GalaxyModelParams{
    const galaxymodel::GalaxyModel model;
    bool needDens;
    bool needVel;
    bool needVel2;
    double accuracy;
    int maxNumEval;
    double vz_error;
    GalaxyModelParams(
        const potential::BasePotential& pot,
        const actions::BaseActionFinder& af,
        const df::BaseDistributionFunction& df) :
        model(pot, af, df) {};
};
/// \endcond

static void fncGalaxyModelMoments(void* obj, const double input[], double *result) {
    const coord::PosCar point = convertPos(input);
    GalaxyModelParams* params = static_cast<GalaxyModelParams*>(obj);
    double dens;
    coord::VelCyl vel;
    coord::Vel2Cyl vel2;
    computeMoments(params->model, coord::toPosCyl(point), params->accuracy, params->maxNumEval,
        params->needDens ? &dens : NULL,
        params->needVel  ? &vel  : NULL,
        params->needVel2 ? &vel2 : NULL, NULL, NULL, NULL);
    unsigned int offset=0;
    if(params->needDens) {
        result[offset] = dens * pow_3(conv->lengthUnit) / conv->massUnit;  // dimension of density is M L^-3
        offset += 1;
    }
    if(params->needVel) {
        result[offset  ] = vel.vR   / conv->velocityUnit;
        result[offset+1] = vel.vz   / conv->velocityUnit;
        result[offset+2] = vel.vphi / conv->velocityUnit;
        offset += 3;
    }
    if(params->needVel2) {
        result[offset  ] = vel2.vR2    / pow_2(conv->velocityUnit);
        result[offset+1] = vel2.vz2    / pow_2(conv->velocityUnit);
        result[offset+2] = vel2.vphi2  / pow_2(conv->velocityUnit);
        result[offset+3] = vel2.vRvz   / pow_2(conv->velocityUnit);
        result[offset+4] = vel2.vRvphi / pow_2(conv->velocityUnit);
        result[offset+5] = vel2.vzvphi / pow_2(conv->velocityUnit);
    }
}

/// compute moments of DF at a given 3d point
static PyObject* GalaxyModel_moments(GalaxyModelObject* self, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"point","dens", "vel", "vel2", NULL};
    PyObject *points_obj = NULL, *dens_flag = NULL, *vel_flag = NULL, *vel2_flag = NULL;
    if(!PyArg_ParseTupleAndKeywords(
        args, namedArgs, "O|OOO", const_cast<char**>(keywords),
        &points_obj, &dens_flag, &vel_flag, &vel2_flag))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to moments()");
        return NULL;
    }
    try{
        GalaxyModelParams params(*self->pot, *self->af, *self->df);
        params.accuracy = 1e-3;
        params.maxNumEval = 1e5;
        params.needDens = dens_flag==NULL || PyObject_IsTrue(dens_flag);
        params.needVel  = vel_flag !=NULL && PyObject_IsTrue(vel_flag);
        params.needVel2 = vel2_flag==NULL || PyObject_IsTrue(vel2_flag);
        if(params.needDens) {
            if(params.needVel) {
                if(params.needVel2)
                    return callAnyFunctionOnArray
                    <INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SINGLE_AND_TRIPLET_AND_SEXTET>
                    (&params, points_obj, fncGalaxyModelMoments);
                else
                    return callAnyFunctionOnArray
                    <INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SINGLE_AND_TRIPLET>
                    (&params, points_obj, fncGalaxyModelMoments);
            } else {
                if(params.needVel2)
                    return callAnyFunctionOnArray
                    <INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SINGLE_AND_SEXTET>
                    (&params, points_obj, fncGalaxyModelMoments);
                else
                    return callAnyFunctionOnArray
                    <INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SINGLE>
                    (&params, points_obj, fncGalaxyModelMoments);
            }
        } else {
            if(params.needVel) {
                if(params.needVel2)
                    return callAnyFunctionOnArray
                    <INPUT_VALUE_TRIPLET, OUTPUT_VALUE_TRIPLET_AND_SEXTET>
                    (&params, points_obj, fncGalaxyModelMoments);
                else
                    return callAnyFunctionOnArray
                    <INPUT_VALUE_TRIPLET, OUTPUT_VALUE_TRIPLET>
                    (&params, points_obj, fncGalaxyModelMoments);
            } else {
                if(params.needVel2)
                    return callAnyFunctionOnArray
                    <INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SEXTET>
                    (&params, points_obj, fncGalaxyModelMoments);
                else {
                    PyErr_SetString(PyExc_ValueError, "Nothing to compute!");
                    return NULL;
                }
            }
        }
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError,
            (std::string("Error in moments(): ")+e.what()).c_str());
        return NULL;
    }
}

static void fncGalaxyModelProjectedMoments(void* obj, const double input[], double *result) {
    GalaxyModelParams* params = static_cast<GalaxyModelParams*>(obj);
    try{
        double surfaceDensity, losvdisp;
        computeProjectedMoments(params->model, input[0] * conv->lengthUnit,
            params->accuracy, params->maxNumEval, surfaceDensity, losvdisp);
        result[0] = surfaceDensity * pow_2(conv->lengthUnit) / conv->massUnit;
        result[1] = losvdisp / pow_2(conv->velocityUnit);
    }
    catch(std::exception& ) {
        result[0] = NAN;
        result[1] = NAN;
    }
}

/// compute projected moments of distribution function
static PyObject* GalaxyModel_projected_moments(GalaxyModelObject* self, PyObject* args)
{
    PyObject *points_obj = NULL;
    if(!PyArg_ParseTuple(args, "O", &points_obj))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to projected_moments()");
        return NULL;
    }
    try{
        GalaxyModelParams params(*self->pot, *self->af, *self->df);
        params.accuracy = 1e-3;
        params.maxNumEval = 1e5;
        return callAnyFunctionOnArray<INPUT_VALUE_SINGLE, OUTPUT_VALUE_SINGLE_AND_SINGLE>
            (&params, points_obj, fncGalaxyModelProjectedMoments);
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError,
            (std::string("Error in projected_moments(): ")+e.what()).c_str());
        return NULL;
    }
}

static double err=0.;
static int numCalls=0;
static void fncGalaxyModelProjectedDF(void* obj, const double input[], double *result) {
    const double R = sqrt(pow_2(input[0]) + pow_2(input[1])) * conv->lengthUnit;
    const double vz = input[2] * conv->velocityUnit;
    // dimension of projected distribution function is M L^-2 V^-1
    const double dim = conv->velocityUnit * pow_2(conv->lengthUnit) / conv->massUnit;
    GalaxyModelParams* params = static_cast<GalaxyModelParams*>(obj);
    try{
        double error;
        int numEval;
        result[0] = computeProjectedDF(params->model, R, vz, params->vz_error,
            params->accuracy, params->maxNumEval, &error, &numEval) * dim;
        err +=error/result[0]*dim;
        numCalls += numEval;
        //printf("R=%g vz=%g => df=%g * (1+-%g) in %i calls\n", 
        //       R/conv->lengthUnit, input[2], result[0], error/result[0]*dim, numEval);
    }
    catch(std::exception& ) {
        result[0] = NAN;
    }
}

/// compute projected distribution function
static PyObject* GalaxyModel_projected_df(GalaxyModelObject* self, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"point","vz_error", NULL};
    PyObject *points_obj = NULL;
    double vz_error = 0;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "O|d", const_cast<char**>(keywords),
        &points_obj, &vz_error))
    {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to projected_df()");
        return NULL;
    }
    try{
        GalaxyModelParams params(*self->pot, *self->af, *self->df);
        params.accuracy = 1e-4;
        params.maxNumEval = 1e5;
        params.vz_error = vz_error * conv->velocityUnit;
        err=0;
        numCalls=0;
        PyObject* result = callAnyFunctionOnArray<INPUT_VALUE_TRIPLET, OUTPUT_VALUE_SINGLE>
            (&params, points_obj, fncGalaxyModelProjectedDF);
        printf("Sum rel err=%g, numCalls=%i\n",err,numCalls);
        return result;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError,
            (std::string("Error in projected_df(): ")+e.what()).c_str());
        return NULL;
    }
}

static PyMethodDef GalaxyModel_methods[] = {
    { "sample", (PyCFunction)GalaxyModel_sample_posvel, METH_VARARGS,
      "Sample distribution function in the given potential by N points\n"
      "Arguments:\n"
      "  Number of points to sample.\n"
      "Returns:\n"
      "  A tuple of two arrays: position/velocity (2d array of size Nx6) and mass (1d array of length N)." },
    { "moments", (PyCFunction)GalaxyModel_moments, METH_VARARGS | METH_KEYWORDS,
      "Compute moments of distribution function in the given potential\n"
      "Arguments:\n"
      "  point -- a single point or an array of points specifying the position "
      "in cartesian coordinates at which the moments need to be computed "
      "(a triplet of numbers or an Nx3 array);\n"
      "  dens (boolean, default True)  -- flag telling whether the density (0th moment) needs to be computed;\n"
      "  vel  (boolean, default False) -- same for streaming velocity (1st moment);\n"
      "  vel2 (boolean, default True)  -- same for 2nd moment of velocity.\n"
      "Returns:\n"
      "  For each input point, return the requested moments (one value for density, "
      "a triplet for velocity, and 6 components of the 2nd moment tensor)." },
    { "projected_moments", (PyCFunction)GalaxyModel_projected_moments, METH_VARARGS,
      "Compute projected moments of distribution function in the given potential\n"
      "Arguments:\n"
      "  A single value or an array of values of cylindrical radius at which to compute moments.\n"
      "Returns:\n"
      "  A tuple of two floats or arrays: surface density and line-of-sight velocity dispersion at each input radius.\n" },
    { "projected_df", (PyCFunction)GalaxyModel_projected_df, METH_VARARGS | METH_KEYWORDS,
      "Compute projected distribution function (integrated over z-coordinate and x- and y-velocities)\n"
      "Named arguments:\n"
      "  point -- a single point or an array of points specifying the x,y- components of position "
      "in cartesian coordinates and z-component of velocity "
      "(a triplet of numbers or an Nx3 array);\n"
      "  vz_error -- optional error on z-component of velocity "
      "(DF will be convolved with a Gaussian if this error is non-zero)\n"
      "Returns:\n"
      "  The value of projected DF (integrated over the missing components of position and velocity) at each point." },
    { NULL, NULL, 0, NULL }
};

static PyTypeObject GalaxyModelType = {
    PyObject_HEAD_INIT(NULL)
    0, "agama.GalaxyModel",
    sizeof(GalaxyModelObject), 0, (destructor)GalaxyModel_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringGalaxyModel,
    0, 0, 0, 0, 0, 0, GalaxyModel_methods, 0, 0, 0, 0, 0, 0, 0,
    GalaxyModel_init, 0, GalaxyModel_new
};

///@}
/// \name  --------- SplineApprox class -----------
///@{

/// \cond INTERNAL_DOCS
/// Python type corresponding to SplineApprox class
typedef struct {
    PyObject_HEAD
    math::CubicSpline* spl;
} SplineApproxObject;
/// \endcond

static PyObject* SplineApprox_new(PyTypeObject *type, PyObject*, PyObject*)
{
    SplineApproxObject *self = (SplineApproxObject*)type->tp_alloc(type, 0);
    if(self)
        self->spl=NULL;
    return (PyObject*)self;
}

static void SplineApprox_dealloc(SplineApproxObject* self)
{
    if(self->spl)
        delete self->spl;
    self->ob_type->tp_free((PyObject*)self);
}

static const char* docstringSplineApprox = 
    "SplineApprox is a class that deals with smoothing splines.\n"
    "It approximates a large set of (x,y) points by a smooth curve with "
    "a rather small number of knots, which should encompass the entire range "
    "of input x values, but preferrably in such a way that each interval "
    "between knots contains at least one x-value from the set of input points.\n"
    "The smoothness of the approximating spline is adjusted by an optional "
    "input parameter `smooth`, which determines the tradeoff between smoothness "
    "and approximation error; zero means no additional smoothing (beyond the one "
    "resulting from discreteness of the spacing of knots), and values around "
    "unity usually yield a reasonable smoothing of noise without sacrificing "
    "too much of accuracy.\n"
    "Values of the spline and up to its second derivative are computed using "
    "the () operator with the first argument being a single x-point or an array "
    "of points, and optional second argument being the derivative index (0, 1, or 2).";

static int SplineApprox_init(PyObject* self, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"x","y","knots","smooth",NULL};
    PyObject* objx=NULL;
    PyObject* objy=NULL;
    PyObject* objk=NULL;
    double smoothfactor=0;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "OOO|d", const_cast<char **>(keywords),
        &objx, &objy, &objk, &smoothfactor)) {
        PyErr_SetString(PyExc_ValueError, "Incorrect parameters passed to the SplineApprox constructor: "
            "must provide two arrays of equal length (input x and y points), "
            "a third array of spline knots, and optionally a float (smooth factor)");
        return -1;
    }
    PyArrayObject *arrx = (PyArrayObject*) PyArray_FROM_OTF(objx, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arry = (PyArrayObject*) PyArray_FROM_OTF(objy, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *arrk = (PyArrayObject*) PyArray_FROM_OTF(objk, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(arrx == NULL || arry == NULL || arrk == NULL) {
        Py_XDECREF(arrx);
        Py_XDECREF(arry);
        Py_XDECREF(arrk);
        PyErr_SetString(PyExc_ValueError, "Input does not contain valid arrays");
        return -1;
    }
    int numpt = 0;
    if(PyArray_NDIM(arrx) == 1)
        numpt = PyArray_DIM(arrx, 0);
    int numknots = 0;
    if(PyArray_NDIM(arrk) == 1)
        numknots = PyArray_DIM(arrk, 0);
    if(numpt <= 0 || numknots < 4|| PyArray_NDIM(arry) != 1 || PyArray_DIM(arry, 0) != numpt) {
        Py_DECREF(arrx);
        Py_DECREF(arry);
        Py_DECREF(arrk);
        PyErr_SetString(PyExc_ValueError, 
            "Arguments must be two arrays of equal length (x and y) and a third array (knots, at least 4)");
        return -1;
    }
    std::vector<double> xvalues((double*)PyArray_DATA(arrx), (double*)PyArray_DATA(arrx) + numpt);
    std::vector<double> yvalues((double*)PyArray_DATA(arry), (double*)PyArray_DATA(arry) + numpt);
    std::vector<double> knots((double*)PyArray_DATA(arrk), (double*)PyArray_DATA(arrk) + numknots);
    try{
        math::SplineApprox spl(xvalues, knots);
        std::vector<double> splinevals;
        double der1, der2;
        if(smoothfactor>0)
            spl.fitDataOversmooth(yvalues, smoothfactor, splinevals, der1, der2);
        else
            spl.fitData(yvalues, -smoothfactor, splinevals, der1, der2);
        if(((SplineApproxObject*)self)->spl)  // check if this is not the first time that constructor is called
            delete ((SplineApproxObject*)self)->spl;
        ((SplineApproxObject*)self)->spl = new math::CubicSpline(knots, splinevals, der1, der2);
        return 0;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, 
            (std::string("Error in SplineApprox initialization: ")+e.what()).c_str());
        return -1;
    }
}

static double spl_eval(const math::CubicSpline* spl, double x, int der=0)
{
    double result;
    switch(der) {
        case 0: return spl->value(x);
        case 1: spl->evalDeriv(x, NULL, &result); return result;
        case 2: spl->evalDeriv(x, NULL, NULL, &result); return result;
        default: return NAN;
    }
}

static PyObject* SplineApprox_value(PyObject* self, PyObject* args, PyObject* /*kw*/)
{
    PyObject* ptx=NULL;
    int der=0;
    if(self==NULL || ((SplineApproxObject*)self)->spl==NULL || !PyArg_ParseTuple(args, "O|i", &ptx, &der))
        return NULL;
    if(der>2) {
        PyErr_SetString(PyExc_ValueError, "Can only compute derivatives up to 2nd");
        return NULL;
    }
    if(PyFloat_Check(ptx))  // one value
        return Py_BuildValue("d", spl_eval(((SplineApproxObject*)self)->spl, PyFloat_AsDouble(ptx), der) );
    // else an array of values
    PyArrayObject *arr = (PyArrayObject*) 
        PyArray_FROM_OTF(ptx, NPY_DOUBLE, NPY_ARRAY_OUT_ARRAY | NPY_ARRAY_ENSURECOPY);
    if(arr == NULL) {
        PyErr_SetString(PyExc_ValueError, "Argument must be either float, list or numpy array");
        return NULL;
    }
    // replace elements of the copy of input array with computed values
    for(int i=0; i<PyArray_SIZE(arr); i++)
        ((double*)PyArray_DATA(arr))[i] = 
            spl_eval(((SplineApproxObject*)self)->spl, ((double*)PyArray_DATA(arr))[i], der);
    return PyArray_Return(arr);
}

static PyMethodDef SplineApprox_methods[] = {
    { NULL, NULL, 0, NULL }  // no named methods
};

static PyTypeObject SplineApproxType = {
    PyObject_HEAD_INIT(NULL)
    0, "agama.SplineApprox",
    sizeof(SplineApproxObject), 0, (destructor)SplineApprox_dealloc,
    0, 0, 0, 0, 0, 0, 0, 0, 0, SplineApprox_value, 0, 0, 0, 0,
    Py_TPFLAGS_DEFAULT, docstringSplineApprox, 
    0, 0, 0, 0, 0, 0, SplineApprox_methods, 0, 0, 0, 0, 0, 0, 0,
    SplineApprox_init, 0, SplineApprox_new
};


///@}
/// \name  ----- Orbit integration -----
///@{

/// description of orbit function
static const char* docstringOrbit = 
    "Compute an orbit starting from the given initial conditions in the given potential\n"
    "Arguments:\n"
    "    ic=float[6] : initial conditions - an array of 6 numbers "
    "(3 positions and 3 velocities in Cartesian coordinates);\n"
    "    pot=Potential object that defines the gravitational potential;\n"
    "    time=float : total integration time;\n"
    "    step=float : output timestep (does not affect the integration accuracy);\n"
    "    acc=float, optional : relative accuracy parameter (default 1e-10).\n"
    "Returns: an array of Nx6 numbers, where N=time/step is the number of output points "
    "in the trajectory, and each point consists of position and velocity in Cartesian coordinates.";

/// orbit integration
static PyObject* integrate_orbit(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"ic", "pot", "time", "step", "acc", NULL};
    double time = 0, step = 0, acc = 1e-10;
    PyObject *ic_obj = NULL, *pot_obj = NULL;
    if(!PyArg_ParseTupleAndKeywords(
        args, namedArgs, "|OOddd", const_cast<char**>(keywords),
        &ic_obj, &pot_obj, &time, &step, &acc) ||
        time<=0 || step<=0 || acc<=0)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid arguments passed to orbit()");
        return NULL;
    }
    if(!PyObject_TypeCheck(pot_obj, &PotentialType) || 
        ((PotentialObject*)pot_obj)->pot==NULL ) {
        PyErr_SetString(PyExc_TypeError, "Argument 'pot' must be a valid instance of Potential class");
        return NULL;
    }
    PyArrayObject *ic_arr  = (PyArrayObject*) PyArray_FROM_OTF(ic_obj,  NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(ic_arr == NULL || PyArray_NDIM(ic_arr) != 1 || PyArray_DIM(ic_arr, 0) != 6) {
        Py_XDECREF(ic_arr);
        PyErr_SetString(PyExc_ValueError, "Argument 'ic' does not contain a valid array of length 6");
        return NULL;
    }
    // initialize
    const coord::PosVelCar ic_point(
        ((double*)PyArray_DATA(ic_arr))[0] * conv->lengthUnit, 
        ((double*)PyArray_DATA(ic_arr))[1] * conv->lengthUnit, 
        ((double*)PyArray_DATA(ic_arr))[2] * conv->lengthUnit,
        ((double*)PyArray_DATA(ic_arr))[3] * conv->velocityUnit, 
        ((double*)PyArray_DATA(ic_arr))[4] * conv->velocityUnit, 
        ((double*)PyArray_DATA(ic_arr))[5] * conv->velocityUnit);
    std::vector<coord::PosVelCar> traj;
    Py_DECREF(ic_arr);
    // integrate
    try{
        orbit::integrate( *((PotentialObject*)pot_obj)->pot, ic_point, 
            time * conv->timeUnit, step * conv->timeUnit, traj, acc);
        // build an appropriate output array
        const unsigned int size = traj.size();
        npy_intp dims[] = {size, 6};
        PyArrayObject* result = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
        for(unsigned int index=0; index<size; index++) {
            ((double*)PyArray_DATA(result))[index*6  ] = traj[index].x / conv->lengthUnit;
            ((double*)PyArray_DATA(result))[index*6+1] = traj[index].y / conv->lengthUnit;
            ((double*)PyArray_DATA(result))[index*6+2] = traj[index].z / conv->lengthUnit;
            ((double*)PyArray_DATA(result))[index*6+3] = traj[index].vx / conv->velocityUnit;
            ((double*)PyArray_DATA(result))[index*6+4] = traj[index].vy / conv->velocityUnit;
            ((double*)PyArray_DATA(result))[index*6+5] = traj[index].vz / conv->velocityUnit;
        }
        return (PyObject*)result;
    }
    catch(std::exception& e) {
        PyErr_SetString(PyExc_ValueError, 
            (std::string("Error in orbit computation: ")+e.what()).c_str());
        return NULL;
    }
}
///@}
/// \name  ----- Math routines -----
///@{

class FncWrapper: public math::IFunctionNdim {
public:
    FncWrapper(unsigned int _nvars, PyObject* _fnc): nvars(_nvars), fnc(_fnc) {}
    virtual void eval(const double vars[], double values[]) const {
        npy_intp dim   = nvars;
        PyObject* arr  = PyArray_SimpleNewFromData(1, &dim, NPY_DOUBLE, const_cast<double*>(vars));
        PyObject* args = Py_BuildValue("(O)", arr);
        Py_DECREF(arr);
        PyObject* result = PyObject_CallObject(fnc, args);
        Py_DECREF(args);
        values[0] = PyFloat_AsDouble(result);
        Py_XDECREF(result);
        if(PyErr_Occurred())
            throw std::runtime_error("Exception occured inside integrand");
    }
    virtual unsigned int numVars()   const { return nvars; }
    virtual unsigned int numValues() const { return 1; }
private:
    const unsigned int nvars;
    PyObject* fnc;
};

static bool parseLowerUpperBounds(PyObject* lower_obj, PyObject* upper_obj,
    std::vector<double> &xlow, std::vector<double> &xupp)
{
    if(!lower_obj) {   // this should always be provided - either # of dimensions, or lower boundary
        PyErr_SetString(PyExc_ValueError,
            "Either integration region or number of dimensions must be provided");
        return false;
    }
    int ndim = -1;
    if(PyInt_Check(lower_obj)) {
        ndim = PyInt_AsLong(lower_obj);
        if(ndim<1) {
            PyErr_SetString(PyExc_ValueError, "Number of dimensions is invalid");
            return false;
        }
        if(upper_obj) {
            PyErr_Format(PyExc_ValueError,
                "May not provide 'upper' argument if 'lower' specifies the number of dimensions (%i)", ndim);
            return false;
        }
        xlow.assign(ndim, 0.);  // default integration region
        xupp.assign(ndim, 1.);
        return true;
    }
    // if the first parameter is not the number of dimensions, then it must be the lower boundary,
    // and the second one must be the upper boundary
    PyArrayObject *lower_arr = (PyArrayObject*) PyArray_FROM_OTF(lower_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(lower_arr == NULL || PyArray_NDIM(lower_arr) != 1) {
        Py_XDECREF(lower_arr);
        PyErr_SetString(PyExc_ValueError,
            "Argument 'lower' does not contain a valid array");
        return false;
    }
    ndim = PyArray_DIM(lower_arr, 0);
    if(!upper_obj) {
        PyErr_SetString(PyExc_ValueError, "Must provide both 'lower' and 'upper' arguments if both are arrays");
        return false;
    }
    PyArrayObject *upper_arr = (PyArrayObject*) PyArray_FROM_OTF(upper_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if(upper_arr == NULL || PyArray_NDIM(upper_arr) != 1 || PyArray_DIM(upper_arr, 0) != ndim) {
        Py_XDECREF(upper_arr);
        PyErr_Format(PyExc_ValueError,
            "Argument 'upper' does not contain a valid array of length %i", ndim);
        return false;
    }
    xlow.resize(ndim);
    xupp.resize(ndim);
    for(int d=0; d<ndim; d++) {
        xlow[d] = static_cast<double*>(PyArray_DATA(lower_arr))[d];
        xupp[d] = static_cast<double*>(PyArray_DATA(upper_arr))[d];
    }
    Py_DECREF(lower_arr);
    Py_DECREF(upper_arr);
    return true;
}

/// description of integration function
static const char* docstringIntegrateNdim =
    "Integrate an N-dimensional function\n"
    "Arguments:\n"
    "    fnc - a callable object that must accept a single argument (array of coordinates) "
    "and return a single numeric value;\n"
    "    lower, upper - two arrays of the same length (equal to the number of dimensions) "
    "that specify the lower and upper boundaries of integration hypercube; "
    "alternatively, a single value - the number of dimensions - may be passed instead of 'lower', "
    "in which case the default interval [0:1] is used for each dimension;\n"
    "    toler - relative error tolerance (default is 1e-4);\n"
    "    maxeval - maximum number of function evaluations (will not exceed it even if "
    "the required tolerance cannot be reached, default is 1e5).\n"
    "Returns: a tuple consisting of integral value, error estimate, "
    "and the actual number of function evaluations performed.\n"
    "Example:\n"
    "    integrateNdim(fnc, [0,-1,0], [3.14,1,100])   "
    "# three-dimensional integral over the region [0:pi] x [-1:1] x [0:100]\n"
    "    integrateNdim(fnc, 2)   # two-dimensional integral over default region [0:1] x [0:1]\n"
    "    integrateNdim(fnc, 4, toler=1e-3, maxeval=1e6)   "
    "# non-default values for tolerance and number of evaluations must be passed as named arguments\n";

/// N-dimensional integration
static PyObject* integrateNdim(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"fnc", "lower", "upper", "toler", "maxeval", NULL};
    double eps=1e-4;
    int maxNumEval=100000, numEval=-1;
    PyObject *callback=NULL, *lower_obj=NULL, *upper_obj=NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "O|OOdi", const_cast<char**>(keywords),
        &callback, &lower_obj, &upper_obj, &eps, &maxNumEval) ||
        !PyCallable_Check(callback) || eps<=0 || maxNumEval<=0)
    {
        PyErr_SetString(PyExc_ValueError, "Incorrect arguments for integrateNdim");
        return NULL;
    }
    std::vector<double> xlow, xupp;
    if(!parseLowerUpperBounds(lower_obj, upper_obj, xlow, xupp))
        return NULL;
    double result, error;
    try{
        FncWrapper fnc(xlow.size(), callback);
        math::integrateNdim(fnc, &xlow.front(), &xupp.front(), eps, maxNumEval, &result, &error, &numEval);
    }
    catch(std::exception& e) {
        if(!PyErr_Occurred())    // set our own error string if it hadn't been set by Python
            PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
    return Py_BuildValue("ddi", result, error, numEval);
}

/// description of sampling function
static const char* docstringSampleNdim =
    "Sample from a non-negative N-dimensional function.\n"
    "Draw a requested number of points from the hypercube in such a way that "
    "the density of points at any location is proportional to the value of function.\n"
    "Arguments:\n"
    "    fnc - a callable object that must accept a single argument (array of coordinates) "
    "and return a single non-negative numeric value (interpreted as probability density);\n"
    "    nsamples - the required number of samples drawn from this function;\n"
    "    lower, upper - two arrays of the same length (equal to the number of dimensions) "
    "that specify the lower and upper boundaries of the region (hypercube) to be sampled; "
    "alternatively, a single value - the number of dimensions - may be passed instead of 'lower', "
    "in which case the default interval [0:1] is used for each dimension;\n"
    "Returns: a tuple consisting of the array of samples with shape (nsamples,ndim), "
    "the integral of the function over the given region estimated in a Monte Carlo way from the samples, "
    "error estimate of the integral, and the actual number of function evaluations performed "
    "(which is typically a factor of few larger than the number of output samples).\n"
    "Example:\n"
    "    samples,integr,error,_ = sampleNdim(fnc, 10000, [0,-1,0], [10,1,3.14])\n";

/// N-dimensional sampling
static PyObject* sampleNdim(PyObject* /*self*/, PyObject* args, PyObject* namedArgs)
{
    static const char* keywords[] = {"fnc", "nsamples", "lower", "upper", NULL};
    int numSamples=-1, numEval=-1;
    PyObject *callback=NULL, *lower_obj=NULL, *upper_obj=NULL;
    if(!PyArg_ParseTupleAndKeywords(args, namedArgs, "Oi|OO", const_cast<char**>(keywords),
        &callback, &numSamples, &lower_obj, &upper_obj) ||
        !PyCallable_Check(callback) || numSamples<=0)
    {
        PyErr_SetString(PyExc_ValueError, "Incorrect arguments for sampleNdim");
        return NULL;
    }
    std::vector<double> xlow, xupp;
    if(!parseLowerUpperBounds(lower_obj, upper_obj, xlow, xupp))
        return NULL;
    double result, error;
    math::Matrix<double> samples;
    try{
        FncWrapper fnc(xlow.size(), callback);
        math::sampleNdim(fnc, &xlow[0], &xupp[0], numSamples, samples, &numEval, &result, &error, false);
        npy_intp dim[] = {numSamples, xlow.size()};
        PyObject* arr  = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, const_cast<double*>(samples.getData()));
        return Py_BuildValue("Oddi", arr, result, error, numEval);
    }
    catch(std::exception& e) {
        if(!PyErr_Occurred())    // set our own error string if it hadn't been set by Python
            PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
}

///@}

static PyMethodDef module_methods[] = {
    {"set_units", (PyCFunction)set_units, METH_VARARGS | METH_KEYWORDS, docstringSetUnits},
    {"reset_units", reset_units, METH_NOARGS, docstringResetUnits},
    {"orbit", (PyCFunction)integrate_orbit, METH_VARARGS | METH_KEYWORDS, docstringOrbit},
    {"actions", (PyCFunction)find_actions, METH_VARARGS | METH_KEYWORDS, docstringActions},
    {"integrateNdim", (PyCFunction)integrateNdim, METH_VARARGS | METH_KEYWORDS, docstringIntegrateNdim},
    {"sampleNdim", (PyCFunction)sampleNdim, METH_VARARGS | METH_KEYWORDS, docstringSampleNdim},
    {NULL}
};

} // end internal namespace

PyMODINIT_FUNC
initagama(void)
{
    PyObject* mod = Py_InitModule("agama", module_methods);
    if(!mod) return;
    conv = new units::ExternalUnits();

    PotentialTypePtr = &PotentialType;
    if (PyType_Ready(&PotentialType) < 0) return;
    Py_INCREF(&PotentialType);
    PyModule_AddObject(mod, "Potential", (PyObject *)&PotentialType);

    if (PyType_Ready(&ActionFinderType) < 0) return;
    Py_INCREF(&ActionFinderType);
    PyModule_AddObject(mod, "ActionFinder", (PyObject *)&ActionFinderType);

    if (PyType_Ready(&DistributionFunctionType) < 0) return;
    Py_INCREF(&DistributionFunctionType);
    PyModule_AddObject(mod, "DistributionFunction", (PyObject *)&DistributionFunctionType);

    if (PyType_Ready(&GalaxyModelType) < 0) return;
    Py_INCREF(&GalaxyModelType);
    PyModule_AddObject(mod, "GalaxyModel", (PyObject *)&GalaxyModelType);

    if (PyType_Ready(&SplineApproxType) < 0) return;
    Py_INCREF(&SplineApproxType);
    PyModule_AddObject(mod, "SplineApprox", (PyObject *)&SplineApproxType);

    import_array();  // needed for NumPy to work properly
}
