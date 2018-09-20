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
#ifndef CLIF_PYTHON_PYOBJ_H_
#define CLIF_PYTHON_PYOBJ_H_
/*
From .../python-2.7.3-docs-html/c-api/intro.html#include-files:
Since Python may define some pre-processor definitions which affect the
standard headers on some systems, you must include Python.h before any standard
headers are included.
*/
#include "Python.h"
#include <utility>

namespace clif {

/* Intended use inside clif namespace is py::Object which is a nice reminiscent
 * of C API PyObject.
 */
namespace py {

// Copyable scoped PyObject* (does refcounting).
class Object {
 public:
  Object() : p_(nullptr) {}
  // Unlike ownerhip transfer in Object(PyObject* p), init from NULL is
  // safe and happen eg. in (Object == nullptr) if. Thus implicit & nolint.
  Object(std::nullptr_t) : p_(nullptr) {}  //NOLINT
  explicit Object(PyObject* p) : p_(p) { Py_XINCREF(p_); }
  Object(const Object& p) : p_(p.get()) { Py_XINCREF(p_); }
  Object(Object&& p) : p_(p.release()) {}

  Object& operator=(Object p) {
    swap(*this, p);
    return *this;
  }

  Object& operator=(PyObject* p) {
    PyGILState_STATE state = PyGILState_Ensure();
    Py_XDECREF(p_);
    p_ = p;
    Py_XINCREF(p_);
    PyGILState_Release(state);
    return *this;
  }

  ~Object() {
    if (p_) {
      PyGILState_STATE state = PyGILState_Ensure();
      Py_DECREF(p_);
      PyGILState_Release(state);
    }
  }

  friend void swap(Object& p, Object& q) {
    using std::swap;
    swap(p.p_, q.p_);
  }

  PyObject* get() const { return p_; }

  PyObject* release() {
    PyObject* p = p_;
    p_ = nullptr;
    return p;
  }

  bool operator!() { return p_ == nullptr; }
  explicit operator bool() const { return p_ != nullptr; }

  friend bool operator==(const Object& p, const Object& q) {
    return p.p_ == q.p_;
  }
  friend bool operator!=(const Object& p, const Object& q) {
    return p.p_ != q.p_;
  }
  friend bool operator==(const Object& p, PyObject* q) {
    return p.p_ == q;
  }
  friend bool operator!=(const Object& p, PyObject* q) {
    return p.p_ != q;
  }
  friend bool operator==(PyObject* p, const Object& q) {
    return p == q.p_;
  }
  friend bool operator!=(PyObject* p, const Object& q) {
    return p != q.p_;
  }

 private:
  PyObject* p_;
};
}  // namespace py
}  // namespace clif

#endif  // CLIF_PYTHON_PYOBJ_H_
