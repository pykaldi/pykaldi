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

#include "clif/python/types.h"
#include <climits>

namespace clif {

//// To Python conversions.

// bytes
PyObject* Clif_PyObjFrom(const std::string& c, py::PostConv pc) {
  return pc.Apply(PyBytes_FromStringAndSize(c.data(), c.size()));
}
PyObject* UnicodeFromBytes(PyObject* b) {
  if (!b || PyUnicode_Check(b)) return b;
  if (!PyBytes_Check(b)) {
    PyErr_Format(PyExc_TypeError, "expecting bytes, got %s %s",
                 ClassName(b), ClassType(b));
    Py_DECREF(b);
    return nullptr;
  }
  PyObject* u = PyUnicode_FromStringAndSize(PyBytes_AS_STRING(b),
                                            PyBytes_GET_SIZE(b));
  Py_DECREF(b);
  return u;
}


//// From Python conversions.

// int (long)

bool Clif_PyObjAs(PyObject* py, int* c) {
  assert(c != nullptr);
  long i;  //NOLINT: runtime/int
  if (PyLong_Check(py)) {
    i = PyLong_AsLong(py);
    if (i == -1 && PyErr_Occurred()) return false;
#if SIZEOF_INT < SIZEOF_LONG
    if (i > INT_MAX || i < INT_MIN) {
      PyErr_SetString(PyExc_ValueError, "value too large for int");
      return false;
    }
#endif
#if PY_MAJOR_VERSION < 3
  } else if (PyInt_Check(py)) {
    i = PyInt_AS_LONG(py);
#endif
  } else {
    PyErr_SetString(PyExc_TypeError, "expecting int");
    return false;
  }
  *c = i;
  return true;
}

bool Clif_PyObjAs(PyObject* py, short* c) {  //NOLINT: runtime/int
  assert(c != nullptr);
  long i;  // NOLINT: runtime/int
  if (!Clif_PyObjAs(py, &i)) {
    return false;
  }
  if (i > SHRT_MAX || i < SHRT_MIN) {
    PyErr_SetString(PyExc_ValueError, "value too large for short int");
    return false;
  }
  *c = i;
  return true;
}

// int8
bool Clif_PyObjAs(PyObject* py, char* c) {
  assert(c != nullptr);
  long i;  // NOLINT: runtime/int
  if (!Clif_PyObjAs(py, &i)) {
    return false;
  }
  if (i > CHAR_MAX || i < CHAR_MIN) {
    PyErr_SetString(PyExc_ValueError, "value too large for char");
    return false;
  }
  *c = i;
  return true;
}

// uint8
bool Clif_PyObjAs(PyObject* py, unsigned char* c) {
  assert(c != nullptr);
  unsigned long i;  // NOLINT: runtime/int
  if (!Clif_PyObjAs(py, &i)) {
    return false;
  }
  if (i > UCHAR_MAX) {
    PyErr_SetString(PyExc_ValueError, "value too large for unsigned char");
    return false;
  }
  *c = i;
  return true;
}

bool Clif_PyObjAs(PyObject* py, unsigned short* c) {  //NOLINT: runtime/int
  assert(c != nullptr);
  unsigned long i;  //NOLINT: runtime/int
  if (PyLong_Check(py)) i = PyLong_AsUnsignedLong(py);
  else  //NOLINT readability/braces
#if PY_MAJOR_VERSION < 3
  if (PyInt_Check(py)) {
    long d = PyInt_AS_LONG(py);  //NOLINT: runtime/int
    if (d < 0) {
      PyErr_SetString(PyExc_ValueError, "expecting non-negative number");
      return false;
    }
    i = d;
  }
  else  //NOLINT readability/braces
#endif
  {
    PyErr_SetString(PyExc_TypeError, "expecting int");
    return false;
  }
  if (PyErr_Occurred()) return false;
  if (i > USHRT_MAX) {
    PyErr_SetString(PyExc_ValueError, "value too large for unsigned short");
    return false;
  }
  *c = i;
  return true;
}

bool Clif_PyObjAs(PyObject* py, unsigned int* c) {
  assert(c != nullptr);
  unsigned long i;  //NOLINT: runtime/int
  if (PyLong_Check(py)) i = PyLong_AsUnsignedLong(py);
  else  //NOLINT readability/braces
#if PY_MAJOR_VERSION < 3
  if (PyInt_Check(py)) {
    long d = PyInt_AS_LONG(py);  //NOLINT: runtime/int
    if (d < 0) {
      PyErr_SetString(PyExc_ValueError, "expecting non-negative number");
      return false;
    }
    i = d;
  }
  else  //NOLINT readability/braces
#endif
  {
    PyErr_SetString(PyExc_TypeError, "expecting int");
    return false;
  }
  if (PyErr_Occurred()) return false;
  if (i > UINT_MAX) {
    PyErr_SetString(PyExc_ValueError, "value too large for unsigned int");
    return false;
  }
  *c = i;
  return true;
}

bool Clif_PyObjAs(PyObject* py, unsigned long* c) {  //NOLINT: runtime/int
  assert(c != nullptr);
  if (PyLong_Check(py)) *c = PyLong_AsUnsignedLong(py);
  else  //NOLINT readability/braces
#if PY_MAJOR_VERSION < 3
  if (PyInt_Check(py)) {
    long i = PyInt_AS_LONG(py);  //NOLINT: runtime/int
    if (i < 0) {
      PyErr_SetString(PyExc_ValueError, "expecting non-negative number");
      return false;
    }
    *c = i;
  }
  else  //NOLINT readability/braces
#endif
  {
    PyErr_SetString(PyExc_TypeError, "expecting int");
    return false;
  }
  return !PyErr_Occurred();
}

bool Clif_PyObjAs(PyObject* py, long* c) {  //NOLINT: runtime/int
  assert(c != nullptr);
  if (PyLong_Check(py)) *c = PyLong_AsSsize_t(py);
  else
#if PY_MAJOR_VERSION < 3
  if (PyInt_Check(py)) *c = PyInt_AsSsize_t(py);
  else  //NOLINT readability/braces
#endif
  {
    PyErr_SetString(PyExc_TypeError, "expecting int");
    return false;
  }
  return !PyErr_Occurred();
}

// int64
#ifdef HAVE_LONG_LONG
bool Clif_PyObjAs(PyObject* py, long long* c) {  //NOLINT: runtime/int
  assert(c != nullptr);
#if PY_MAJOR_VERSION < 3
  if (PyInt_Check(py)) {
    long i = PyInt_AS_LONG(py);  //NOLINT: runtime/int
    *c = i;
    return true;
  }
#endif
  if (!PyLong_Check(py)) {
    PyErr_SetString(PyExc_TypeError, "expecting int");
    return false;
  }
  *c = PyLong_AsLongLong(py);
  return !PyErr_Occurred();
}

// uint64
bool Clif_PyObjAs(PyObject* py, unsigned long long* c) {  //NOLINT: runtime/int
  assert(c != nullptr);
  if (PyLong_Check(py)) *c = PyLong_AsUnsignedLongLong(py);
  else
#if PY_MAJOR_VERSION < 3
  if (PyInt_Check(py)) *c = PyInt_AsUnsignedLongLongMask(py);
  else  //NOLINT readability/braces
#endif
  {
    PyErr_SetString(PyExc_TypeError, "expecting int");
    return false;
  }
  return !PyErr_Occurred();
}
#endif  // HAVE_LONG_LONG

// float (double)
bool Clif_PyObjAs(PyObject* py, double* c) {
  assert(c != nullptr);
  double f = PyFloat_AsDouble(py);
  if (f == -1.0 && PyErr_Occurred()) return false;
  *c = f;
  return true;
}

bool Clif_PyObjAs(PyObject* py, float* c) {
  assert(c != nullptr);
  double f = PyFloat_AsDouble(py);
  if (f == -1.0 && PyErr_Occurred()) return false;
  *c = static_cast<float>(f);
  return true;
}

// bool
bool Clif_PyObjAs(PyObject* py, bool* c) {
  assert(c != nullptr);
  if (!PyBool_Check(py)) {
    PyErr_SetString(PyExc_TypeError, "expecting bool");
    return false;
  }
  *c = (py == Py_True);
  return true;
}

namespace py {

// bytes/unicode
template<typename C>
bool ObjToStr(PyObject* py, C copy ) {
  bool decref = false;
  if (PyUnicode_Check(py)) {
    py = PyUnicode_AsUTF8String(py);
    if (!py) return false;
    decref = true;
  } else if (!PyBytes_Check(py)) {
    PyErr_SetString(PyExc_TypeError, "expecting str");
    return false;
  }
  copy(PyBytes_AS_STRING(py), PyBytes_GET_SIZE(py));
  if (decref) Py_DECREF(py);
  return true;
}
}  // namespace py

bool Clif_PyObjAs(PyObject* p, std::string* c) {
  assert(c != nullptr);
  return py::ObjToStr(p,
      [c](const char* data, size_t length) { c->assign(data, length); });
}
}  // namespace clif
