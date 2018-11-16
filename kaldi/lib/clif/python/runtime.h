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
#include "clif/python/instance.h"
// CHECK_NOTNULL used in generated code, so it belongs here.
// NOLINTNEXTLINE(whitespace/line_length) because of MOE, result is within 80.
#define CHECK_NOTNULL(condition) (assert((condition) != nullptr),(condition))
using std::string;

extern "C" int Clif_PyType_Inconstructible(PyObject*, PyObject*, PyObject*);

namespace clif {
namespace python {

string ExcStr(bool add_type = true);

template <typename T>
T* Get(const clif::Instance<T>& cpp, bool set_err = true) {
  T* d = cpp.get();
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

// RAII pyobj attribute holder with GIL management for virtual override methods.
class SafeAttr {
 public:
  SafeAttr(PyObject* pyobj, const char* name);  // Acquires GIL.
  ~SafeAttr();                                  // Releases GIL.
  SafeAttr(const SafeAttr& other) = delete;
  SafeAttr& operator=(const SafeAttr& other) = delete;
  SafeAttr(SafeAttr&& other) = delete;
  SafeAttr& operator=(SafeAttr&& other) = delete;

  PyObject* get() const { return meth_; }

 private:
  PyGILState_STATE state_;
  PyObject* meth_;
};

// RAII PyObject ownership and GIL safe to call from a C++ object.
class SafePyObject {
 public:
  // SafePyObject is implicitly convertible from PyObject*.
  explicit SafePyObject(PyObject* py) : py_(py) { Py_XINCREF(py); }
  ~SafePyObject();  // Grab GIL to do XDECREF.
  // Copy/move disabled to avoid extra inc/dec-refs (decref can be costly).
  SafePyObject(const SafePyObject& other) = delete;
  SafePyObject& operator=(const SafePyObject& other) = delete;
  SafePyObject(SafePyObject&& other) = delete;
  SafePyObject& operator=(SafePyObject&& other) = delete;
 private:
  PyObject* py_;
};

// When we share a C++ instance to a shared_ptr<T> C++ consumer, we need to make
// sure that its owner implementing virual functions (Python object) will not
// go away or leak and keep ownership while we use it.
template<typename T>
struct SharedVirtual {
  // If Instance has 2+ owners it can't renounce ownership to an unique_ptr.
  Instance<T> prevent_ownership_renouncing;
  SafePyObject owner;

  SharedVirtual() = default;
  SharedVirtual(Instance<T> shared, PyObject* py)
      : prevent_ownership_renouncing(std::move(shared)), owner(py) {}
};

// Return the shared U* as a T*, which may be a different type.
template<typename T, typename U>
std::shared_ptr<T> MakeSharedVirtual(Instance<U> cpp, PyObject* py) {
  T* ptr = cpp.get();
  auto helper = std::make_shared<SharedVirtual<U>>(std::move(cpp), py);
  return std::shared_ptr<T>(helper, ptr);
}

// PyObject* "self" storage mixin for virtual method overrides.
// It weakrefs back to "self" to get the virtual method object.
// Also if C++ instance memory ownership moved to C++ code, it owns the "self"
// object.
// 
// the C++ pointer after the normal 'self' renounced ownership, for potential
// use in virtual method run. Currently 'self' can't access C++ class content
// after it renounced ownership.
class PyObjRef {
 public:
#ifndef NDEBUG
  // Create a poison pill to check that Init() called for all instances.
  PyObjRef() : self_(PoisonPill()), pyowner_(nullptr) {}
#else
  PyObjRef() = default;
#endif
  // This class is copy/movable but never used alone, only with some user class:
  //   class Overrider : public PyObjRef, public UserClass { ... }
  // and only instantiated from Python so those defs are not very useful.
  PyObjRef(const PyObjRef& other) = default;
  PyObjRef& operator=(const PyObjRef& other) = default;
  PyObjRef(PyObjRef&& other) = default;
  PyObjRef& operator=(PyObjRef&& other) = default;

  // Keep a (weak) reference to 'self'.
  void Init(PyObject* self);
  // Get Python version of "this" to access virtual function implementation:
  //     ::clif::SafeAttr vfunc(self(), "v_func_name");
  PyObject* self() const;

  // Take/drop 'self' ownership.
  void HoldPyObj(PyObject* self);
  void DropPyObj();

 private:
  PyObject* PoisonPill() const;

  // A weak reference to 'self' for C++ callers into Python-holded instance.
  // Initialized by Init(py) call after the ctor (postcall).
  PyObject* self_;
  // Transfer python ownership here when 'self' released to an unique_ptr.
  PyObject* pyowner_;
};
}  // namespace clif

#endif  // CLIF_PYTHON_RUNTIME_H_
