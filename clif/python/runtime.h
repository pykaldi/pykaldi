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
#ifndef CLIF_PYTHON_RUNTIME_H_
#define CLIF_PYTHON_RUNTIME_H_

/*
From .../python-2.7.3-docs-html/c-api/intro.html#include-files:
Since Python may define some pre-processor definitions which affect the
standard headers on some systems, you must include Python.h before any standard
headers are included.
*/
#include <Python.h>
#include <string>
#include "clif/python/pyobj.h"
#include "clif/python/shared_ptr.h"
// CHECK_NOTNULL used in generated code, so it belongs here.
// NOLINTNEXTLINE(whitespace/line_length) because of MOE, result is within 80.
#define CHECK_NOTNULL(condition) (assert((condition) != nullptr),(condition))
using std::string;

extern "C" void Clif_PyType_GenericFree(PyObject* self);
extern "C" int Clif_PyType_Inconstructible(PyObject*, PyObject*, PyObject*);

namespace clif {
namespace python {

string ExcStr(bool add_type = true);

template <typename T>
T* Get(clif::SharedPtr<T> sp, bool set_err = true) {
  T* d = sp.get();
  if (set_err && d == nullptr) {
    PyErr_SetString(PyExc_ValueError,
                    "Value invalidated due to capture by std::unique_ptr.");
  }
  return d;
}
}  // namespace python

// Returns py.__class__.__name__ (needed for PY2 old style classes).
const char* ClassName(PyObject* py);
const char* ClassType(PyObject* py);

// Load the base Python class.
PyObject* ImportFQName(const string& full_class_name);

// Ensure we have enough args for callable.
bool CallableNeedsNarguments(PyObject* callable, int nargs);

// Convert string literal "like this" to (char*) for C compatibility.
inline constexpr char* C(const char c[]) { return const_cast<char*>(&c[0]); }

// Format function argument missed error.
PyObject* DefaultArgMissedError(const char func[], char* argname);

// Format function argument [conversion] error.
PyObject* ArgError(const char func[], char* argname, const char ctype[],
                   PyObject* arg);

// PyObject* "self" storage mixin for virtual method overrides.
struct PyObj {
  py::Object pythis;
  void Init(PyObject* self) { pythis = self; }
};

// RAII GIL management for virtual override methods.
class SafeGetAttrString {
  PyGILState_STATE state_;
  PyObject* meth_;

 public:
  SafeGetAttrString(PyObject* pyobj, const char* name);  // Gets GIL.
  ~SafeGetAttrString();                                  // Releases GIL.
  PyObject* get() const { return meth_; }
};
}  // namespace clif

#endif  // CLIF_PYTHON_RUNTIME_H_
