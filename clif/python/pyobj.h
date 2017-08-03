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

  Object& operator=(PyObject* p) {
    Py_XDECREF(p_);
    p_ = p;
    Py_XINCREF(p_);
    return *this;
  }

  Object& operator=(Object p) {
    swap(*this, p);
    return *this;
  }

  ~Object() { Py_XDECREF(p_); }

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
