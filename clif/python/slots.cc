// Copyright 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "clif/python/slots.h"

namespace clif {
namespace slot {

Py_ssize_t item_index(PyObject* self, Py_ssize_t idx) {
  Py_ssize_t len = -1;
  PySequenceMethods *sq = Py_TYPE(self)->tp_as_sequence;
  if (sq && sq->sq_length) {
    len = (*sq->sq_length)(self);
    if (len < 0) return -1;
  } else {
    PyErr_SetString(PyExc_TypeError, "not a sequential object");
    return -1;
  }
  if (idx < 0 || idx >= len) {
    PyErr_SetNone(PyExc_IndexError);
    return -1;
  }
  return idx;
}

Py_ssize_t as_size(PyObject* res) {
#if PY_MAJOR_VERSION < 3
  Py_ssize_t size = PyInt_AsSsize_t(res);
#else
  Py_ssize_t size = PyLong_AsSsize_t(res);
#endif
  Py_DECREF(res);
  if (size < 0 && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_ValueError, "__len__ returned a negative value");
  }
  return size;
}

int as_bool(PyObject* res) {
#if PY_MAJOR_VERSION < 3
  if (PyInt_CheckExact(res) || PyBool_Check(res)) {
#else
  if (PyLong_CheckExact(res) || PyBool_Check(res)) {
#endif
    int i = PyObject_IsTrue(res);
    Py_DECREF(res);
    return i;
  }
  Py_DECREF(res);
  // Renamed to bool in Python 3.
  PyErr_SetString(PyExc_ValueError, "__nonzero__ must return int or bool");
  return -1;
}

#if PY_MAJOR_VERSION >= 3
#define PyInt_AsLong PyLong_AsLong
#endif

long as_hash(PyObject* res) {    //NOLINT: runtime/int
  long i = PyLong_Check(res) ? PyLong_Type.tp_hash(res) : PyInt_AsLong(res);  //NOLINT: runtime/int
  Py_DECREF(res);
  if (i == -1) {
    if (PyErr_Occurred()) {
      PyErr_SetString(PyExc_ValueError, "__hash__ must return int");
    } else {
      return -2;
    }
  }
  return i;
}

int as_cmp(PyObject* res) {
  long i = PyInt_AsLong(res);  //NOLINT: runtime/int
  Py_DECREF(res);
  if (i == -1 && PyErr_Occurred()) {
    PyErr_SetString(PyExc_ValueError, "__cmp__ must return int");
    return -2;
  }
  return static_cast<int>(i);
}

int ignore(PyObject* res) {
  Py_XDECREF(res);
  return (res == nullptr || PyErr_Occurred()) ? -1 : 0;
}
}  // namespace slot
}  // namespace clif
