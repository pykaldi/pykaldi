/*
 * Copyright 2017 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef CLIF_PYTHON_TYPES_H_
#define CLIF_PYTHON_TYPES_H_

/* "Standard" types known to CLIF. */

/*
From .../python-2.7.3-docs-html/c-api/intro.html#include-files:
Since Python may define some pre-processor definitions which affect the
standard headers on some systems, you must include Python.h before any standard
headers are included.
*/
#include <Python.h>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <type_traits>
//
// CLIF use `::proto2::Message` as proto2_Message
/** NOTE (DC):
  Commented out because we don't need CLIF's protobuf wrapping functionality at the moment
  and we don't want PyKaldi to depend on the availability of protobuf headers.
**/
// #include "clif/python/pyproto.h"
#include "clif/python/postconv.h"
#include "clif/python/runtime.h"
#if PY_MAJOR_VERSION >= 3
#define PyInt_Check PyLong_Check
#define PyInt_AsLong PyLong_AsLong
#define PyInt_FromLong PyLong_FromLong
#define PyInt_AsSize_t PyLong_AsSize_t
#define PyInt_FromSize_t PyLong_FromSize_t
#define PyInt_AsSsize_t PyLong_AsSsize_t
#define PyInt_FromSsize_t PyLong_FromSsize_t
#define PyString_FromStringAndSize PyUnicode_FromStringAndSize
#define PyString_FromString PyUnicode_FromString
#endif

namespace clif {
using std::swap;

//
// To Python conversions.
//

// CLIF use `PyObject*` as object
inline PyObject* Clif_PyObjFrom(PyObject* c, py::PostConv)  {
  // Ignore postconversion for object output.
  if (c == nullptr && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_SystemError,
                    "When returning the NULL object, exception must be set");
  }
  return c;
}

// int (long)
// pyport.h should define Py_ssize_t as either int or long
static_assert(std::is_same<Py_ssize_t, int>::value ||
              std::is_same<Py_ssize_t, long>::value,  //NOLINT runtime/int
              "The world is strange");
// CLIF use `int` as int
inline PyObject* Clif_PyObjFrom(int c, py::PostConv pc)  {
  return pc.Apply(PyInt_FromLong(c));
}
// CLIF use `unsigned int` as uint
inline PyObject* Clif_PyObjFrom(unsigned int c, py::PostConv pc) {
  return pc.Apply(PyInt_FromSize_t(c));
}
#ifdef uint32_t
// CLIF use `uint32` as uint32
inline PyObject* Clif_PyObjFrom(uint32_t c, py::PostConv pc) {
  return pc.Apply(PyInt_FromSize_t(c));
}
#endif
// CLIF use `long` as long
inline PyObject* Clif_PyObjFrom(long c, py::PostConv pc) {  //NOLINT runtime/int
  return pc.Apply(PyInt_FromLong(c));
}
// CLIF use `ulong` as ulong
inline PyObject* Clif_PyObjFrom(unsigned long c, py::PostConv pc) {  //NOLINT runtime/int
  return pc.Apply(PyInt_FromSize_t(c));
}
// CLIF use `int64` as int64
#ifdef HAVE_LONG_LONG
inline PyObject* Clif_PyObjFrom(long long c, py::PostConv pc) {  //NOLINT runtime/int
  return pc.Apply(PyLong_FromLongLong(c));
}
// CLIF use `uint64` as uint64
inline PyObject* Clif_PyObjFrom(unsigned long long c, py::PostConv pc) {  //NOLINT runtime/int
  return pc.Apply(PyLong_FromUnsignedLongLong(c));
}
#endif
// CLIF use `unsigned char` as uint8
inline PyObject* Clif_PyObjFrom(unsigned char c, py::PostConv pc) {
  return pc.Apply(PyInt_FromLong(c));
}
// CLIF use `char` as int8
inline PyObject* Clif_PyObjFrom(char c, py::PostConv pc) {
  return pc.Apply(PyInt_FromLong(c));
}

// float (double)
// CLIF use `float` as float
// CLIF use `double` as float
inline PyObject* Clif_PyObjFrom(double c, py::PostConv pc) {
  return pc.Apply(PyFloat_FromDouble(c));
}

// CLIF use `bool` as bool
inline PyObject* Clif_PyObjFrom(bool c, py::PostConv pc) {
  return pc.Apply(PyBool_FromLong(c));
}

// CLIF use `std::string` as bytes
PyObject* Clif_PyObjFrom(const std::string&, py::PostConv);
typedef const char* char_ptr;  // A distinct type for constexpr CONST string.
inline PyObject* Clif_PyObjFrom(const char_ptr c, py::PostConv unused) {
  // Always use native str, ignore postconversion.
  return PyString_FromString(c);
}

//
// From Python conversions.
//

inline bool Clif_PyObjAs(PyObject* py, PyObject** c) {
  assert(c != nullptr);
  assert(py != nullptr);
  *c = py;  // Borrow reference from Python for C++ processing.
  return true;
}

// int (long)
bool Clif_PyObjAs(PyObject*, unsigned char*);
bool Clif_PyObjAs(PyObject*, unsigned short*);      //NOLINT runtime/int
bool Clif_PyObjAs(PyObject*, unsigned int*);
bool Clif_PyObjAs(PyObject*, unsigned long*);       //NOLINT runtime/int
#ifdef HAVE_LONG_LONG
bool Clif_PyObjAs(PyObject*, unsigned long long*);  //NOLINT runtime/int
#endif
bool Clif_PyObjAs(PyObject*, char*);
bool Clif_PyObjAs(PyObject*, short*);               //NOLINT runtime/int
bool Clif_PyObjAs(PyObject*, int*);
bool Clif_PyObjAs(PyObject*, long*);                //NOLINT runtime/int // Py_ssize_t on x64
#ifdef HAVE_LONG_LONG
bool Clif_PyObjAs(PyObject*, long long*);           //NOLINT runtime/int
#endif

// float (double)
bool Clif_PyObjAs(PyObject*, double*);
bool Clif_PyObjAs(PyObject*, float*);

// bool
bool Clif_PyObjAs(PyObject*, bool*);

// bytes
bool Clif_PyObjAs(PyObject*, std::string*);

PyObject* UnicodeFromBytes(PyObject*);

//
// Containers
//
// CLIF use `std::array` as list
// CLIF use `std::list` as list
// CLIF use `std::queue` as list
// CLIF use `std::priority_queue` as list
// CLIF use `std::stack` as list
// CLIF use `std::deque` as list
// CLIF use `std::vector` as list

template<typename T, typename... Args>
PyObject* Clif_PyObjFrom(const std::vector<T, Args...>& c, py::PostConv);
template<typename... Args>
PyObject* Clif_PyObjFrom(const std::vector<bool, Args...>& c, py::PostConv);
template<typename T, typename... Args>
PyObject* Clif_PyObjFrom(std::vector<T, Args...>&& c, py::PostConv);
template<typename... Args>
PyObject* Clif_PyObjFrom(std::vector<bool, Args...>&& c, py::PostConv);

template<typename T, typename... Args>
bool Clif_PyObjAs(PyObject* py, std::vector<T, Args...>* c);

// CLIF use `std::pair` as tuple
template<typename T, typename U>
PyObject* Clif_PyObjFrom(const std::pair<T, U>& c, py::PostConv);
template<typename T, typename U>
bool Clif_PyObjAs(PyObject* py, std::pair<T, U>* c);
template<typename... T>
PyObject* Clif_PyObjFrom(const std::tuple<T...>& c, py::PostConv);
template<typename... T>
bool Clif_PyObjAs(PyObject* py, std::tuple<T...>* c);


// CLIF use `std::map` as dict
// CLIF use `std::unordered_map` as dict
template<typename T, typename U, typename... Args>
PyObject* Clif_PyObjFrom(const std::unordered_map<T, U, Args...>& c,
                         py::PostConv pc);
template<typename T, typename U, typename... Args>
bool Clif_PyObjAs(PyObject* py, std::unordered_map<T, U, Args...>* c);

template<typename T, typename U, typename... Args>
PyObject* Clif_PyObjFrom(const std::map<T, U, Args...>& c, py::PostConv);
template<typename T, typename U, typename... Args>
bool Clif_PyObjAs(PyObject* py, std::map<T, U, Args...>* c);

// CLIF use `std::set` as set
// CLIF use `std::unordered_set` as set
template<typename T, typename... Args>
PyObject* Clif_PyObjFrom(const std::unordered_set<T, Args...>& c, py::PostConv);
template<typename T, typename... Args>
bool Clif_PyObjAs(PyObject* py, std::unordered_set<T, Args...>* c);
template<typename T, typename... Args>
PyObject* Clif_PyObjFrom(const std::set<T, Args...>& c, py::PostConv);
template<typename T, typename... Args>
bool Clif_PyObjAs(PyObject* py, std::set<T, Args...>* c);

// ---------------------------------------------------------------------
// Fill in extra overloads for copyable types.

// Just is_copy_assignable<T> is not enough here as From(T*) will engage in
// T*&& resolution instead of generated (for capsule) const T*, so implement a
// direct check if From(T) is available.
template<typename T, typename = decltype(Clif_PyObjFrom(std::declval<T>(), {}))>
inline PyObject* Clif_PyObjFrom(T* c, py::PostConv pc) {
  if (c) return Clif_PyObjFrom(*c, pc);
  Py_RETURN_NONE;
}
template<typename T>
typename std::enable_if<std::is_copy_assignable<T>::value, PyObject*>::type
inline Clif_PyObjFrom(const std::unique_ptr<T>& c, py::PostConv pc) {
  if (c) return Clif_PyObjFrom(*c, pc);
  Py_RETURN_NONE;
}

namespace callback {
// This class is used to convert return values from callbacks and virtual
// functions implemented in Python to C++. It deals with converting Python
// objects returned under normal conditions, as well as error conditions
// expressed by raising exceptions in Python, into appropriate C++ return
// values of type R.
//
// A generic version of this class is defined in
// clif/python/stltypes.h.
//
// There are a certain rules when defining specializations of this class.
// 1. It specialized class should have a valid implicit or explicit default
//    constructor.
// 2. It should define a method 'FromPyValue' with the following signature.
//       R FromPyValue(PyObject* r);
// The method 'FromPyValue' is called to get a suitable C++ value of type R
// corresponding to the Python object |r|. The implementations should call
// PyErr_Occurred to determine if a Python exception is to be converted to a
// value of type R. |r| can be equal to nullptr when PyErr_Occurred returns
// true.
template <typename R>
class ReturnValue;

}  // namespace callback
}  // namespace clif

#endif  // CLIF_PYTHON_TYPES_H_
