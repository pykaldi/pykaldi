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
#ifndef CLIF_PYTHON_SLOTS_H_
#define CLIF_PYTHON_SLOTS_H_

// Given different functions with the same signature
//   static PyObject* F1(PyObject*, PyObject*, PyObject*);
//   static PyObject* F2(PyObject*, PyObject*, PyObject*);
//   ...
// we need a bunch of adapters below to pass them into some structures:
//   static struct A = {
//      (adapter1) &dapter_instance_with_F1,
//      (adapter2) &dapter_instance_with_F5,
//   };
//   static struct B = {
//      (adapter1) &dapter_instance_with_F18,
//      (adapter2) &dapter_instance_with_F54,
//   };
//   ...
//
// We use those templates as
//      (adapter1) &adapter<return_type, convert_result, F, O...>,
// where O... is PyObject* repeated 0..N times depending on adapter1 sigmature
// (ie. for adapter1 = long (*)(PyObject* self, PyObject* x, PyObject* y)
//  we use &adapter<long, &some_func_here, &wrapper_CFunc, O, O>;


#include <Python.h>

namespace clif {
namespace slot {

Py_ssize_t item_index(PyObject* self, Py_ssize_t idx);
Py_ssize_t as_size(PyObject* res);
int as_bool(PyObject* res);
long as_hash(PyObject* res);    //NOLINT: runtime/int
int as_cmp(PyObject* res);
int ignore(PyObject* res);

template<PyObject* (*Wrapper)(PyObject*, PyObject*, PyObject*)>
PyObject* getitem(PyObject* self, Py_ssize_t idx) {
  idx = item_index(self, idx);
  if (idx < 0) return nullptr;
#if PY_MAJOR_VERSION >= 3
  PyObject* i = PyLong_FromSize_t(idx);
#else
  PyObject* i = PyInt_FromSize_t(idx);
#endif
  if (i == nullptr) return nullptr;

  PyObject* args = PyTuple_Pack(1, i);
  if (args == nullptr) {
    Py_DECREF(i);
    return nullptr;
  }

  PyObject* res = Wrapper(self, args, nullptr);
  Py_DECREF(args);
  Py_DECREF(i);

  return res;
}

template<typename R> R error_value();  // Common error value for type R.
template<> inline int error_value() { return -1; }
template<> inline long error_value() { return -1; }  //NOLINT: runtime/int
template<> inline PyObject* error_value() { return nullptr; }

// METH_NOARGS
template<typename R, R (*AdaptOut)(PyObject*), PyObject* (*Wrapper)(PyObject*)>
R adapter(PyObject* self) {
  PyObject* res = Wrapper(self);
  if (res == nullptr) return error_value<R>();
  return (*AdaptOut)(res);
}
// R=PyObject* specialization.
template<PyObject* (*Wrapper)(PyObject*)>
PyObject* adapter(PyObject* self) { return Wrapper(self); }

// METH_VARARGS | METH_KEYWORDS
template<typename R, R (*AdaptOut)(PyObject*),
         PyObject* (*Wrapper)(PyObject*, PyObject*, PyObject*), typename... T>
R adapter(PyObject* self, T... in_args) {
  PyObject* args = PyTuple_Pack(sizeof...(T), in_args...);
  if (args == nullptr) return error_value<R>();
  PyObject* res = Wrapper(self, args, nullptr);
  Py_DECREF(args);
  if (res == nullptr) return error_value<R>();
  return (*AdaptOut)(res);
}

// R=PyObject* specialization.
template<PyObject* (*Wrapper)(PyObject*, PyObject*, PyObject*), typename... T>
PyObject* adapter(PyObject* self, T... in_args) {
  PyObject* args = PyTuple_Pack(sizeof...(T), in_args...);
  if (args == nullptr) return nullptr;
  PyObject* res = Wrapper(self, args, nullptr);
  Py_DECREF(args);
  return res;
}
}  // namespace slot
}  // namespace clif

#endif  // CLIF_PYTHON_SLOTS_H_
