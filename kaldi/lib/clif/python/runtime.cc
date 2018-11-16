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

#include "clif/python/runtime.h"

extern "C"
int Clif_PyType_Inconstructible(PyObject* self, PyObject* a, PyObject* kw) {
  PyErr_Format(PyExc_ValueError, "Class %s has no default constructor",
               Py_TYPE(self)->tp_name);
  return -1;
}

namespace clif {

void PyObjRef::Init(PyObject* self) {
  self_ = PyWeakref_NewRef(self, nullptr);
  // We don't care about non-weakrefencable objects (most likely the
  // CLIF-generated wrapper object), so nullptr is fine, just clear the error.
  PyErr_Clear();
}

void PyObjRef::HoldPyObj(PyObject* self) {
    pyowner_ = self;
  Py_INCREF(pyowner_);
}
void PyObjRef::DropPyObj() {
  Py_CLEAR(pyowner_);
}

PyObject* PyObjRef::PoisonPill() const {
  // Memory pattern mnemonic is _______CallInit.
  return reinterpret_cast<PyObject*>(0xCA771417);
}

PyObject* PyObjRef::self() const {
    if (self_ == nullptr) return nullptr;
  PyGILState_STATE threadstate = PyGILState_Ensure();
  PyObject* py = PyWeakref_GetObject(self_);
  if (py == Py_None) py = nullptr;
  Py_XINCREF(py);
  PyGILState_Release(threadstate);
  return py;
}

SafePyObject::~SafePyObject() {
  if (py_) {
    PyGILState_STATE threadstate = PyGILState_Ensure();
    Py_DECREF(py_);
    PyGILState_Release(threadstate);
  }
}

// Resource management for virtual override methods.
SafeAttr::SafeAttr(PyObject* pyobj, const char* name) {
  state_ = PyGILState_Ensure();
  meth_ = pyobj ? PyObject_GetAttrString(pyobj, name) : nullptr;
  Py_XDECREF(pyobj);  // Assume that method descriptor keeps the obj alive.
  if (meth_ == nullptr) {
    if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
      PyErr_Clear();
    } else if (PyErr_Occurred()) {
      PyErr_PrintEx(0);
    }
    PyGILState_Release(state_);
  }
}
SafeAttr::~SafeAttr() {
  if (meth_ != nullptr) {
    Py_DECREF(meth_);
    PyGILState_Release(state_);
  }
}

// Given full.path.to.a.module.Name import module and return Name from module.
PyObject* ImportFQName(const string& full_class_name) {
  // Split full_class_name at the last dot.
  auto last_dot = full_class_name.find_last_of('.');
  if (last_dot == string::npos) {
    PyErr_Format(PyExc_ValueError, "No dot in full_class_name '%s'",
                 full_class_name.c_str());
    return nullptr;
  }
  PyObject* module = PyImport_ImportModule(
      full_class_name.substr(0, last_dot).c_str());
  if (!module) return nullptr;
  PyObject* py = PyObject_GetAttrString(
      module, full_class_name.substr(last_dot+1).c_str());
  Py_DECREF(module);
  return py;
}

// py.__class__.__name__
const char* ClassName(PyObject* py) {
  /* PyPy doesn't have a separate C API for old-style classes. */
#if PY_MAJOR_VERSION < 3 && !defined(PYPY_VERSION)
  if (PyClass_Check(py)) return PyString_AS_STRING(CHECK_NOTNULL(
      reinterpret_cast<PyClassObject*>(py)->cl_name));
  if (PyInstance_Check(py)) return PyString_AS_STRING(CHECK_NOTNULL(
      reinterpret_cast<PyInstanceObject*>(py)->in_class->cl_name));
#endif
  if (Py_TYPE(py) == &PyType_Type) {
    return reinterpret_cast<PyTypeObject*>(py)->tp_name;
  }
  return Py_TYPE(py)->tp_name;
}

// type(py) renamed from {classobj, instance, type, class X}
const char* ClassType(PyObject* py) {
  /* PyPy doesn't have a separate C API for old-style classes. */
#if PY_MAJOR_VERSION < 3 && !defined(PYPY_VERSION)
  if (PyClass_Check(py)) return "old class";
  if (PyInstance_Check(py)) return "old class instance";
#endif
  if (Py_TYPE(py) == &PyType_Type) {
    return "class";
  }
  return "instance";
}

bool CallableNeedsNarguments(PyObject* callable, int nargs) {
  PyObject* getcallargs = ImportFQName("inspect.getcallargs");
  if (!getcallargs) return false;
  PyObject* args = PyTuple_New(nargs+1);
  Py_INCREF(callable);
  PyTuple_SET_ITEM(args, 0, callable);
  for (int i=1; i <= nargs; ++i) {
    Py_INCREF(Py_None);
    PyTuple_SET_ITEM(args, i, Py_None);
  }
  PyObject* binded = PyObject_CallObject(getcallargs, args);
  Py_DECREF(getcallargs);
  Py_DECREF(args);
  if (!binded) return false;  // PyExc_TypeError is set.
  // explicitly clear references to give to garbage collector more information
  // this potentially can cause faster collecting of unused objects
  PyDict_Clear(binded);
  Py_DECREF(binded);
  return true;
}

PyObject* DefaultArgMissedError(const char func[], char* argname) {
  PyErr_Format(PyExc_ValueError, "%s() argument %s needs a non-default value",
               func, argname);
  return nullptr;
}

PyObject* ArgError(const char func[],
                          char* argname,
                          const char ctype[],
                          PyObject* arg) {
  PyObject* exc = PyErr_Occurred();
  if (exc == nullptr) {
    PyErr_Format(
        PyExc_TypeError, "%s() argument %s is not valid for %s (%s %s given)",
        func, argname, ctype, ClassName(arg), ClassType(arg));
  } else if (exc == PyExc_TypeError) {
    PyErr_Format(
        exc, "%s() argument %s is not valid for %s (%s %s given): %s",
        func, argname, ctype, ClassName(arg), ClassType(arg),
        python::ExcStr(false).c_str());
  } else {
    PyErr_Format(
        exc, "%s() argument %s is not valid: %s",
        func, argname, python::ExcStr(false).c_str());
  }
  return nullptr;
}

namespace python {

string ExcStr(bool add_type) {
  PyObject* exc, *val, *tb;
  PyErr_Fetch(&exc, &val, &tb);
  if (!exc) return "";
  PyErr_NormalizeException(&exc, &val, &tb);
  string err;
  if (add_type) err = string(ClassName(exc)) + ": ";
  Py_DECREF(exc);
  if (val) {
    PyObject* val_str = PyObject_Str(val);
    Py_DECREF(val);
    if (val_str) {
#if PY_MAJOR_VERSION < 3
      err += PyString_AS_STRING(val_str);
#else
      err += PyUnicode_AsUTF8(val_str);
#endif
      Py_DECREF(val_str);
    }
  }
  Py_XDECREF(tb);
  return err;
}
}  // namespace python
}  // namespace clif
