
#include <Python.h>
#include "clif/python/ptr_util.h"
#include "clif/python/optional.h"
#include "clif/python/types.h"
#define PY_ARRAY_UNIQUE_SYMBOL __numpy_array_api
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "matrix/kaldi-vector-clifwrap.h"
#include "matrix/kaldi-matrix-clifwrap.h"
#include "matrix/matrix-ext.h"
#include "clif/python/stltypes.h"
#include "clif/python/slots.h"

namespace __matrix__ext {
using namespace clif;

#define _0 py::postconv::PASS
#define _1 UnicodeFromBytes

namespace pySubVector {

struct wrapper {
  PyObject_HEAD
  ::clif::Instance<::kaldi::SubVector<float>> cpp;
  PyObject* base;
};
static ::kaldi::SubVector<float>* ThisPtr(PyObject*);

// __init__(t:VectorBase, start:int, length:int)
static PyObject* wrapSubVector_float_as___init__(PyObject* self,
                                                 PyObject* args,
                                                 PyObject* kw) {
  PyObject* a[3];
  char* names[] = { C("src"), C("start"), C("length"), nullptr };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO:__init__", names, &a[0], &a[1], &a[2]))
    return nullptr;
  ::kaldi::MatrixIndexT arg2;
  if (!Clif_PyObjAs(a[1], &arg2))
    return ArgError("__init__", names[1], "::kaldi::MatrixIndexT", a[1]);
  ::kaldi::MatrixIndexT arg3;
  if (!Clif_PyObjAs(a[2], &arg3))
    return ArgError("__init__", names[2], "::kaldi::MatrixIndexT", a[2]);
  if (PyArray_Check(a[0])) {
    if (PyArray_NDIM((PyArrayObject*)a[0]) != 1) {
      PyErr_SetString(PyExc_RuntimeError, "Input ndarray is not 1-dimensional.");
      return nullptr;
    }
    int dtype = PyArray_TYPE((PyArrayObject*)a[0]);
    if (dtype == NPY_FLOAT) {
      // Kaldi requires each vector to be a contiguous chunk of well behaved
      // memory. No gaps are allowed between the items. If this requirement is
      // not satisfied by the input array, a new array will be allocated.
      PyObject *array = PyArray_FromArray((PyArrayObject*)a[0], nullptr,
                                          NPY_ARRAY_DEFAULT);
      PyObject* err_type = nullptr;
      string err_msg{"C++ exception"};
      try {
        reinterpret_cast<wrapper*>(self)->cpp = ::clif::MakeShared<::kaldi::SubVector<float>>((float*)PyArray_DATA((PyArrayObject*)array) + arg2, std::move(arg3));
      } catch(const std::exception& e) {
        err_type = PyExc_RuntimeError;
        err_msg += string(": ") + e.what();
      } catch (...) {
        err_type = PyExc_RuntimeError;
      }
      if (err_type) {
        Py_DECREF(array);
        PyErr_SetString(err_type, err_msg.c_str());
        return nullptr;
      }
      // Reference count of array will be decremented when self is deallocated.
      reinterpret_cast<wrapper*>(self)->base = array;
    } else {
      PyErr_SetString(PyExc_RuntimeError,
                      "Cannot convert given ndarray to a SubVector since "
                      "it has an invalid dtype. Supported dtypes: np.float32.");
      return nullptr;
    }
  } else {
    ::kaldi::VectorBase<float>* arg1;
    if (!Clif_PyObjAs(a[0], &arg1))
      return ArgError("__init__", names[0], "PyArray_Type or ::kaldi::VectorBase<float>", a[0]);
    // Call actual C++ method.
    PyObject* err_type = nullptr;
    string err_msg{"C++ exception"};
    try {
      reinterpret_cast<wrapper*>(self)->cpp = ::clif::MakeShared<::kaldi::SubVector<float>>(*arg1, std::move(arg2), std::move(arg3));
    } catch(const std::exception& e) {
      err_type = PyExc_RuntimeError;
      err_msg += string(": ") + e.what();
    } catch (...) {
      err_type = PyExc_RuntimeError;
    }
    if (err_type) {
      PyErr_SetString(err_type, err_msg.c_str());
      return nullptr;
    }
    // Reference count of a[0] will be decremented when self is deallocated.
    Py_INCREF(a[0]);
    reinterpret_cast<wrapper*>(self)->base = a[0];
  }
  Py_RETURN_NONE;
}

// Range(start:int, length:int) -> SubVector
// NOTE: Range method is implemented in Python by initializing a new
// kaldi.matrix.SubVector object since the user facing SubVector class
// (kaldi.matrix.Vector) is defined in Python and is a child class of the
// SubVector type provided by this wrapper.

// Implicit cast this as ::kaldi::VectorBase<float>*
static PyObject* as_kaldi_VectorBase_float(PyObject* self) {
  ::kaldi::VectorBase<float>* p = ::clif::python::Get(reinterpret_cast<wrapper*>(self)->cpp);
  if (p == nullptr) return nullptr;
  return PyCapsule_New(p, C("::kaldi::VectorBase<float>"), nullptr);
}

static PyMethodDef Methods[] = {
  {C("__init__"), (PyCFunction)wrapSubVector_float_as___init__, METH_VARARGS | METH_KEYWORDS, C("__init__(t:VectorBase, start:int, length:int)\n  Calls C++ function\n  void ::kaldi::SubVector<float>::SubVector(::kaldi::VectorBase<float>, ::kaldi::MatrixIndexT, ::kaldi::MatrixIndexT)")},
  {C("as_kaldi_VectorBase_float"), (PyCFunction)as_kaldi_VectorBase_float, METH_NOARGS, C("Upcast to ::kaldi::VectorBase<float>*")},
  {}
};

// SubVector __new__
static PyObject* _allocator(PyTypeObject* type, Py_ssize_t nitems);
// SubVector __init__
static int _ctor(PyObject* self, PyObject* args, PyObject* kw);

static void _deallocator(PyObject* self) {
  reinterpret_cast<wrapper*>(self)->cpp.Destruct();
  Py_XDECREF(reinterpret_cast<wrapper*>(self)->base);
  Py_TYPE(self)->tp_free(self);
}

static void _dtor(void* self) {
  delete reinterpret_cast<wrapper*>(self);
}

PyTypeObject wrapper_Type = {
  PyVarObject_HEAD_INIT(&PyType_Type, 0)
  "kaldi.matrix._matrix_ext.SubVector", // tp_name
  sizeof(wrapper),                     // tp_basicsize
  0,                                   // tp_itemsize
  _deallocator,                        // tp_dealloc
  nullptr,                             // tp_print
  nullptr,                             // tp_getattr
  nullptr,                             // tp_setattr
  nullptr,                             // tp_compare
  nullptr,                             // tp_repr
  nullptr,                             // tp_as_number
  nullptr,                             // tp_as_sequence
  nullptr,                             // tp_as_mapping
  nullptr,                             // tp_hash
  nullptr,                             // tp_call
  nullptr,                             // tp_str
  nullptr,                             // tp_getattro
  nullptr,                             // tp_setattro
  nullptr,                             // tp_as_buffer
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_TYPE_SUBCLASS | Py_TPFLAGS_BASETYPE, // tp_flags
  "CLIF wrapper for ::kaldi::SubVector<float>", // tp_doc
  nullptr,                             // tp_traverse
  nullptr,                             // tp_clear
  nullptr,                             // tp_richcompare
  0,                                   // tp_weaklistoffset
  nullptr,                             // tp_iter
  nullptr,                             // tp_iternext
  Methods,                             // tp_methods
  nullptr,                             // tp_members
  nullptr,                             // tp_getset
  nullptr,                             // tp_base
  nullptr,                             // tp_dict
  nullptr,                             // tp_descr_get
  nullptr,                             // tp_descr_set
  0,                                   // tp_dictoffset
  _ctor,                               // tp_init
  _allocator,                          // tp_alloc
  PyType_GenericNew,                   // tp_new
  _dtor,                               // tp_free
  nullptr,                             // tp_is_gc
  nullptr,                             // tp_bases
  nullptr,                             // tp_mro
  nullptr,                             // tp_cache
  nullptr,                             // tp_subclasses
  nullptr,                             // tp_weaklist
  nullptr,                             // tp_del
  0,                                   // tp_version_tag
};

static int _ctor(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* init = wrapSubVector_float_as___init__(self, args, kw);
  Py_XDECREF(init);
  return init? 0: -1;
}

static PyObject* _allocator(PyTypeObject* type, Py_ssize_t nitems) {
  assert(nitems == 0);
  PyObject* self = reinterpret_cast<PyObject*>(new wrapper);
  return PyObject_Init(self, &wrapper_Type);
}

static ::kaldi::SubVector<float>* ThisPtr(PyObject* py) {
  if (Py_TYPE(py) == &wrapper_Type) {
    return ::clif::python::Get(reinterpret_cast<wrapper*>(py)->cpp);
  }
  PyObject* base = PyObject_CallMethod(py, C("as_kaldi_SubVector_float"),
                                       nullptr);
  if (base) {
    if (PyCapsule_CheckExact(base)) {
      void* p = PyCapsule_GetPointer(base, C("::kaldi::SubVector<float>"));
      if (!PyErr_Occurred()) {
        ::kaldi::SubVector<float>* c = static_cast<::kaldi::SubVector<float>*>(p);
        Py_DECREF(base);
        return c;
      }
    }
    Py_DECREF(base);
  }
  if (PyObject_IsInstance(py, reinterpret_cast<PyObject*>(&wrapper_Type))) {
    if (!base) {
      PyErr_Clear();
      return ::clif::python::Get(reinterpret_cast<wrapper*>(py)->cpp);
    }
    PyErr_Format(PyExc_ValueError,
                 "can't convert %s %s to ::kaldi::SubVector<float>*",
                 ClassName(py), ClassType(py));
  } else {
    PyErr_Format(PyExc_TypeError, "expecting %s instance, got %s %s",
                 wrapper_Type.tp_name, ClassName(py), ClassType(py));
  }
  return nullptr;
}
}  // namespace pySubVector

namespace pySubMatrix {

struct wrapper {
  PyObject_HEAD
  ::clif::Instance<::kaldi::SubMatrix<float>> cpp;
  PyObject* base;
};
static ::kaldi::SubMatrix<float>* ThisPtr(PyObject*);

// __init__(T:MatrixBase, ro:int, r:int, co:int, c:int)
static PyObject* wrapSubMatrix_float_as___init__(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* a[5];
  char* names[] = { C("src"), C("row_start"), C("num_rows"), C("col_start"), C("num_cols"), nullptr };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOOO:__init__", names, &a[0], &a[1], &a[2], &a[3], &a[4])) return nullptr;
  ::kaldi::MatrixIndexT arg2;
  if (!Clif_PyObjAs(a[1], &arg2)) return ArgError("__init__", names[1], "::kaldi::MatrixIndexT", a[1]);
  ::kaldi::MatrixIndexT arg3;
  if (!Clif_PyObjAs(a[2], &arg3)) return ArgError("__init__", names[2], "::kaldi::MatrixIndexT", a[2]);
  ::kaldi::MatrixIndexT arg4;
  if (!Clif_PyObjAs(a[3], &arg4)) return ArgError("__init__", names[3], "::kaldi::MatrixIndexT", a[3]);
  ::kaldi::MatrixIndexT arg5;
  if (!Clif_PyObjAs(a[4], &arg5)) return ArgError("__init__", names[4], "::kaldi::MatrixIndexT", a[4]);
  if (PyArray_Check(a[0])) {
    if (PyArray_NDIM((PyArrayObject*)a[0]) != 2) {
      PyErr_SetString(PyExc_RuntimeError, "Input ndarray should be 2-dimensional.");
      return nullptr;
    }
    int dtype = PyArray_TYPE((PyArrayObject*)a[0]);
    if (dtype == NPY_FLOAT) {
      // Kaldi requires each matrix row to be a contiguous chunk of well behaved
      // memory. There can be gaps between rows but no gaps are allowed between
      // items in a row. Also, the row stride can not be smaller than the size
      // of a row. If any of these requirements are not satisfied by the input
      // array, a new array will be allocated.
      int requirements = NPY_ARRAY_BEHAVED;  // aligned and writeable
      npy_intp dim1 = PyArray_DIM((PyArrayObject*)a[0], 1);
      npy_intp stride0 = PyArray_STRIDE((PyArrayObject*)a[0], 0);
      npy_intp stride1 = PyArray_STRIDE((PyArrayObject*)a[0], 1);
      long item_size = ((long)sizeof(float));
      // We do not ask for a contiguous memory region, if we can do without one.
      if (((dim1 > 1) and (stride1 != item_size)) || (stride0 < item_size * dim1)) {
        requirements |= NPY_ARRAY_C_CONTIGUOUS;
      }
      PyObject *array = PyArray_FromArray((PyArrayObject*)a[0], nullptr,
                                          requirements);
      PyObject* err_type = nullptr;
      string err_msg{"C++ exception"};
      try {
        ::kaldi::MatrixIndexT stride = PyArray_STRIDE((PyArrayObject*)array, 0) / ((long)sizeof(float));
        reinterpret_cast<wrapper*>(self)->cpp = ::clif::MakeShared<::kaldi::SubMatrix<float>>((float*)PyArray_DATA((PyArrayObject*)array) + arg2 * stride + arg4, std::move(arg3), std::move(arg5), std::move(stride));
      } catch(const std::exception& e) {
        err_type = PyExc_RuntimeError;
        err_msg += string(": ") + e.what();
      } catch (...) {
        err_type = PyExc_RuntimeError;
      }
      if (err_type) {
        Py_DECREF(array);
        PyErr_SetString(err_type, err_msg.c_str());
        return nullptr;
      }
      // Reference count of array will be decremented when self is deallocated.
      reinterpret_cast<wrapper*>(self)->base = array;
    } else {
      PyErr_SetString(PyExc_RuntimeError,
                      "Cannot convert given ndarray to a SubMatrix since "
                      "it has an invalid dtype. Supported dtypes: np.float32.");
      return nullptr;
    }
  } else {
    ::kaldi::MatrixBase<float>* arg1;
    if (!Clif_PyObjAs(a[0], &arg1))
      return ArgError("__init__", names[0], "PyArray_Type or ::kaldi::MatrixBase<float>", a[0]);
    // Call actual C++ method.
    PyObject* err_type = nullptr;
    string err_msg{"C++ exception"};
    try {
      reinterpret_cast<wrapper*>(self)->cpp = ::clif::MakeShared<::kaldi::SubMatrix<float>>(*arg1, std::move(arg2), std::move(arg3), std::move(arg4), std::move(arg5));
    } catch(const std::exception& e) {
      err_type = PyExc_RuntimeError;
      err_msg += string(": ") + e.what();
    } catch (...) {
      err_type = PyExc_RuntimeError;
    }
    if (err_type) {
      PyErr_SetString(err_type, err_msg.c_str());
      return nullptr;
    }
    // Reference count of a[0] will be decremented when self is deallocated.
    Py_INCREF(a[0]);
    reinterpret_cast<wrapper*>(self)->base = a[0];
  }
  Py_RETURN_NONE;
}

// Range(row_start:int, num_rows:int, col_start:int, num_cols:int) -> SubMatrix
// NOTE: Range method is implemented in Python by initializing a new
// kaldi.matrix.SubMatrix object since the user facing SubMatrix class
// (kaldi.matrix.SubMatrix) is defined in Python and is a child class of the
// SubMatrix type provided by this wrapper.

// Implicit cast this as ::kaldi::MatrixBase<float>*
static PyObject* as_kaldi_MatrixBase_float(PyObject* self) {
  ::kaldi::MatrixBase<float>* p = ::clif::python::Get(reinterpret_cast<wrapper*>(self)->cpp);
  if (p == nullptr) return nullptr;
  return PyCapsule_New(p, C("::kaldi::MatrixBase<float>"), nullptr);
}

static PyMethodDef Methods[] = {
  {C("__init__"), (PyCFunction)wrapSubMatrix_float_as___init__, METH_VARARGS | METH_KEYWORDS, C("__init__(T:MatrixBase, ro:int, r:int, co:int, c:int)\n  Calls C++ function\n  void ::kaldi::SubMatrix<float>::SubMatrix(::kaldi::MatrixBase<float>, ::kaldi::MatrixIndexT, ::kaldi::MatrixIndexT, ::kaldi::MatrixIndexT, ::kaldi::MatrixIndexT)")},
  {C("as_kaldi_MatrixBase_float"), (PyCFunction)as_kaldi_MatrixBase_float, METH_NOARGS, C("Upcast to ::kaldi::MatrixBase<float>*")},
  {}
};

// SubMatrix __new__
static PyObject* _allocator(PyTypeObject* type, Py_ssize_t nitems);
// SubMatrix __init__
static int _ctor(PyObject* self, PyObject* args, PyObject* kw);

static void _deallocator(PyObject* self) {
  reinterpret_cast<wrapper*>(self)->cpp.Destruct();
  Py_XDECREF(reinterpret_cast<wrapper*>(self)->base);
  Py_TYPE(self)->tp_free(self);
}

static void _dtor(void* self) {
  delete reinterpret_cast<wrapper*>(self);
}

PyTypeObject wrapper_Type = {
  PyVarObject_HEAD_INIT(&PyType_Type, 0)
  "kaldi.matrix._matrix_ext.SubMatrix", // tp_name
  sizeof(wrapper),                     // tp_basicsize
  0,                                   // tp_itemsize
  _deallocator,                        // tp_dealloc
  nullptr,                             // tp_print
  nullptr,                             // tp_getattr
  nullptr,                             // tp_setattr
  nullptr,                             // tp_compare
  nullptr,                             // tp_repr
  nullptr,                             // tp_as_number
  nullptr,                             // tp_as_sequence
  nullptr,                             // tp_as_mapping
  nullptr,                             // tp_hash
  nullptr,                             // tp_call
  nullptr,                             // tp_str
  nullptr,                             // tp_getattro
  nullptr,                             // tp_setattro
  nullptr,                             // tp_as_buffer
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_TYPE_SUBCLASS | Py_TPFLAGS_BASETYPE, // tp_flags
  "CLIF wrapper for ::kaldi::SubMatrix<float>", // tp_doc
  nullptr,                             // tp_traverse
  nullptr,                             // tp_clear
  nullptr,                             // tp_richcompare
  0,                                   // tp_weaklistoffset
  nullptr,                             // tp_iter
  nullptr,                             // tp_iternext
  Methods,                             // tp_methods
  nullptr,                             // tp_members
  nullptr,                             // tp_getset
  nullptr,                             // tp_base
  nullptr,                             // tp_dict
  nullptr,                             // tp_descr_get
  nullptr,                             // tp_descr_set
  0,                                   // tp_dictoffset
  _ctor,                               // tp_init
  _allocator,                          // tp_alloc
  PyType_GenericNew,                   // tp_new
  _dtor,                               // tp_free
  nullptr,                             // tp_is_gc
  nullptr,                             // tp_bases
  nullptr,                             // tp_mro
  nullptr,                             // tp_cache
  nullptr,                             // tp_subclasses
  nullptr,                             // tp_weaklist
  nullptr,                             // tp_del
  0,                                   // tp_version_tag
};

static int _ctor(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* init = wrapSubMatrix_float_as___init__(self, args, kw);
  Py_XDECREF(init);
  return init? 0: -1;
}

static PyObject* _allocator(PyTypeObject* type, Py_ssize_t nitems) {
  assert(nitems == 0);
  PyObject* self = reinterpret_cast<PyObject*>(new wrapper);
  return PyObject_Init(self, &wrapper_Type);
}

static ::kaldi::SubMatrix<float>* ThisPtr(PyObject* py) {
  if (Py_TYPE(py) == &wrapper_Type) {
    return ::clif::python::Get(reinterpret_cast<wrapper*>(py)->cpp);
  }
  PyObject* base = PyObject_CallMethod(py, C("as_kaldi_SubMatrix_float"), nullptr);
  if (base) {
    if (PyCapsule_CheckExact(base)) {
      void* p = PyCapsule_GetPointer(base, C("::kaldi::SubMatrix<float>"));
      if (!PyErr_Occurred()) {
        ::kaldi::SubMatrix<float>* c = static_cast<::kaldi::SubMatrix<float>*>(p);
        Py_DECREF(base);
        return c;
      }
    }
    Py_DECREF(base);
  }
  if (PyObject_IsInstance(py, reinterpret_cast<PyObject*>(&wrapper_Type))) {
    if (!base) {
      PyErr_Clear();
      return ::clif::python::Get(reinterpret_cast<wrapper*>(py)->cpp);
    }
    PyErr_Format(PyExc_ValueError, "can't convert %s %s to ::kaldi::SubMatrix<float>*", ClassName(py), ClassType(py));
  } else {
    PyErr_Format(PyExc_TypeError, "expecting %s instance, got %s %s", wrapper_Type.tp_name, ClassName(py), ClassType(py));
  }
  return nullptr;
}
}  // namespace pySubMatrix

namespace pyDoubleSubVector {

struct wrapper {
  PyObject_HEAD
  ::clif::Instance<::kaldi::SubVector<double>> cpp;
  PyObject* base;
};
static ::kaldi::SubVector<double>* ThisPtr(PyObject*);

// __init__(t:VectorBase, start:int, length:int)
static PyObject* wrapSubVector_double_as___init__(PyObject* self,
                                                 PyObject* args,
                                                 PyObject* kw) {
  PyObject* a[3];
  char* names[] = { C("src"), C("start"), C("length"), nullptr };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO:__init__", names, &a[0], &a[1], &a[2]))
    return nullptr;
  ::kaldi::MatrixIndexT arg2;
  if (!Clif_PyObjAs(a[1], &arg2))
    return ArgError("__init__", names[1], "::kaldi::MatrixIndexT", a[1]);
  ::kaldi::MatrixIndexT arg3;
  if (!Clif_PyObjAs(a[2], &arg3))
    return ArgError("__init__", names[2], "::kaldi::MatrixIndexT", a[2]);
  if (PyArray_Check(a[0])) {
    if (PyArray_NDIM((PyArrayObject*)a[0]) != 1) {
      PyErr_SetString(PyExc_RuntimeError, "Input ndarray is not 1-dimensional.");
      return nullptr;
    }
    int dtype = PyArray_TYPE((PyArrayObject*)a[0]);
    if (dtype == NPY_DOUBLE) {
      // Kaldi requires each vector to be a contiguous chunk of well behaved
      // memory. No gaps are allowed between the items. If this requirement is
      // not satisfied by the input array, a new array will be allocated.
      PyObject *array = PyArray_FromArray((PyArrayObject*)a[0], nullptr,
                                          NPY_ARRAY_DEFAULT);
      PyObject* err_type = nullptr;
      string err_msg{"C++ exception"};
      try {
        reinterpret_cast<wrapper*>(self)->cpp = ::clif::MakeShared<::kaldi::SubVector<double>>((double*)PyArray_DATA((PyArrayObject*)array) + arg2, std::move(arg3));
      } catch(const std::exception& e) {
        err_type = PyExc_RuntimeError;
        err_msg += string(": ") + e.what();
      } catch (...) {
        err_type = PyExc_RuntimeError;
      }
      if (err_type) {
        Py_DECREF(array);
        PyErr_SetString(err_type, err_msg.c_str());
        return nullptr;
      }
      // Reference count of array will be decremented when self is deallocated.
      reinterpret_cast<wrapper*>(self)->base = array;
    } else {
      PyErr_SetString(PyExc_RuntimeError,
                      "Cannot convert given ndarray to a DoubleSubVector since "
                      "it has an invalid dtype. Supported dtypes: np.float64.");
      return nullptr;
    }
  } else {
    ::kaldi::VectorBase<double>* arg1;
    if (!Clif_PyObjAs(a[0], &arg1))
      return ArgError("__init__", names[0], "PyArray_Type or ::kaldi::VectorBase<double>", a[0]);
    // Call actual C++ method.
    PyObject* err_type = nullptr;
    string err_msg{"C++ exception"};
    try {
      reinterpret_cast<wrapper*>(self)->cpp = ::clif::MakeShared<::kaldi::SubVector<double>>(*arg1, std::move(arg2), std::move(arg3));
    } catch(const std::exception& e) {
      err_type = PyExc_RuntimeError;
      err_msg += string(": ") + e.what();
    } catch (...) {
      err_type = PyExc_RuntimeError;
    }
    if (err_type) {
      PyErr_SetString(err_type, err_msg.c_str());
      return nullptr;
    }
    // Reference count of a[0] will be decremented when self is deallocated.
    Py_INCREF(a[0]);
    reinterpret_cast<wrapper*>(self)->base = a[0];
  }
  Py_RETURN_NONE;
}

// Range(start:int, length:int) -> DoubleSubVector
// NOTE: Range method is implemented in Python by initializing a new
// kaldi.matrix.DoubleSubVector object since the user facing DoubleSubVector
// class (kaldi.matrix.DoubleSubVector) is defined in Python and is a child
// class of the DoubleSubVector type provided by this wrapper.

// Implicit cast this as ::kaldi::VectorBase<double>*
static PyObject* as_kaldi_VectorBase_double(PyObject* self) {
  ::kaldi::VectorBase<double>* p = ::clif::python::Get(reinterpret_cast<wrapper*>(self)->cpp);
  if (p == nullptr) return nullptr;
  return PyCapsule_New(p, C("::kaldi::VectorBase<double>"), nullptr);
}

static PyMethodDef Methods[] = {
  {C("__init__"), (PyCFunction)wrapSubVector_double_as___init__, METH_VARARGS | METH_KEYWORDS, C("__init__(t:VectorBase, start:int, length:int)\n  Calls C++ function\n  void ::kaldi::SubVector<double>::SubVector(::kaldi::VectorBase<double>, ::kaldi::MatrixIndexT, ::kaldi::MatrixIndexT)")},
  {C("as_kaldi_VectorBase_double"), (PyCFunction)as_kaldi_VectorBase_double, METH_NOARGS, C("Upcast to ::kaldi::VectorBase<double>*")},
  {}
};

// DoubleSubVector __new__
static PyObject* _allocator(PyTypeObject* type, Py_ssize_t nitems);
// DoubleSubVector __init__
static int _ctor(PyObject* self, PyObject* args, PyObject* kw);

static void _deallocator(PyObject* self) {
  reinterpret_cast<wrapper*>(self)->cpp.Destruct();
  Py_XDECREF(reinterpret_cast<wrapper*>(self)->base);
  Py_TYPE(self)->tp_free(self);
}

static void _dtor(void* self) {
  delete reinterpret_cast<wrapper*>(self);
}

PyTypeObject wrapper_Type = {
  PyVarObject_HEAD_INIT(&PyType_Type, 0)
  "kaldi.matrix._matrix_ext.DoubleSubVector", // tp_name
  sizeof(wrapper),                     // tp_basicsize
  0,                                   // tp_itemsize
  _deallocator,                        // tp_dealloc
  nullptr,                             // tp_print
  nullptr,                             // tp_getattr
  nullptr,                             // tp_setattr
  nullptr,                             // tp_compare
  nullptr,                             // tp_repr
  nullptr,                             // tp_as_number
  nullptr,                             // tp_as_sequence
  nullptr,                             // tp_as_mapping
  nullptr,                             // tp_hash
  nullptr,                             // tp_call
  nullptr,                             // tp_str
  nullptr,                             // tp_getattro
  nullptr,                             // tp_setattro
  nullptr,                             // tp_as_buffer
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_TYPE_SUBCLASS | Py_TPFLAGS_BASETYPE, // tp_flags
  "CLIF wrapper for ::kaldi::SubVector<double>", // tp_doc
  nullptr,                             // tp_traverse
  nullptr,                             // tp_clear
  nullptr,                             // tp_richcompare
  0,                                   // tp_weaklistoffset
  nullptr,                             // tp_iter
  nullptr,                             // tp_iternext
  Methods,                             // tp_methods
  nullptr,                             // tp_members
  nullptr,                             // tp_getset
  nullptr,                             // tp_base
  nullptr,                             // tp_dict
  nullptr,                             // tp_descr_get
  nullptr,                             // tp_descr_set
  0,                                   // tp_dictoffset
  _ctor,                               // tp_init
  _allocator,                          // tp_alloc
  PyType_GenericNew,                   // tp_new
  _dtor,                               // tp_free
  nullptr,                             // tp_is_gc
  nullptr,                             // tp_bases
  nullptr,                             // tp_mro
  nullptr,                             // tp_cache
  nullptr,                             // tp_subclasses
  nullptr,                             // tp_weaklist
  nullptr,                             // tp_del
  0,                                   // tp_version_tag
};

static int _ctor(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* init = wrapSubVector_double_as___init__(self, args, kw);
  Py_XDECREF(init);
  return init? 0: -1;
}

static PyObject* _allocator(PyTypeObject* type, Py_ssize_t nitems) {
  assert(nitems == 0);
  PyObject* self = reinterpret_cast<PyObject*>(new wrapper);
  return PyObject_Init(self, &wrapper_Type);
}

static ::kaldi::SubVector<double>* ThisPtr(PyObject* py) {
  if (Py_TYPE(py) == &wrapper_Type) {
    return ::clif::python::Get(reinterpret_cast<wrapper*>(py)->cpp);
  }
  PyObject* base = PyObject_CallMethod(py, C("as_kaldi_SubVector_double"),
                                       nullptr);
  if (base) {
    if (PyCapsule_CheckExact(base)) {
      void* p = PyCapsule_GetPointer(base, C("::kaldi::SubVector<double>"));
      if (!PyErr_Occurred()) {
        ::kaldi::SubVector<double>* c = static_cast<::kaldi::SubVector<double>*>(p);
        Py_DECREF(base);
        return c;
      }
    }
    Py_DECREF(base);
  }
  if (PyObject_IsInstance(py, reinterpret_cast<PyObject*>(&wrapper_Type))) {
    if (!base) {
      PyErr_Clear();
      return ::clif::python::Get(reinterpret_cast<wrapper*>(py)->cpp);
    }
    PyErr_Format(PyExc_ValueError,
                 "can't convert %s %s to ::kaldi::SubVector<double>*",
                 ClassName(py), ClassType(py));
  } else {
    PyErr_Format(PyExc_TypeError, "expecting %s instance, got %s %s",
                 wrapper_Type.tp_name, ClassName(py), ClassType(py));
  }
  return nullptr;
}
}  // namespace pyDoubleSubVector

namespace pyDoubleSubMatrix {

struct wrapper {
  PyObject_HEAD
  ::clif::Instance<::kaldi::SubMatrix<double>> cpp;
  PyObject* base;
};
static ::kaldi::SubMatrix<double>* ThisPtr(PyObject*);

// __init__(T:MatrixBase, ro:int, r:int, co:int, c:int)
static PyObject* wrapSubMatrix_double_as___init__(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* a[5];
  char* names[] = { C("src"), C("row_start"), C("num_rows"), C("col_start"), C("num_cols"), nullptr };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOOO:__init__", names, &a[0], &a[1], &a[2], &a[3], &a[4])) return nullptr;
  ::kaldi::MatrixIndexT arg2;
  if (!Clif_PyObjAs(a[1], &arg2)) return ArgError("__init__", names[1], "::kaldi::MatrixIndexT", a[1]);
  ::kaldi::MatrixIndexT arg3;
  if (!Clif_PyObjAs(a[2], &arg3)) return ArgError("__init__", names[2], "::kaldi::MatrixIndexT", a[2]);
  ::kaldi::MatrixIndexT arg4;
  if (!Clif_PyObjAs(a[3], &arg4)) return ArgError("__init__", names[3], "::kaldi::MatrixIndexT", a[3]);
  ::kaldi::MatrixIndexT arg5;
  if (!Clif_PyObjAs(a[4], &arg5)) return ArgError("__init__", names[4], "::kaldi::MatrixIndexT", a[4]);
  if (PyArray_Check(a[0])) {
    if (PyArray_NDIM((PyArrayObject*)a[0]) != 2) {
      PyErr_SetString(PyExc_RuntimeError, "Input ndarray should be 2-dimensional.");
      return nullptr;
    }
    int dtype = PyArray_TYPE((PyArrayObject*)a[0]);
    if (dtype == NPY_DOUBLE) {
      // Kaldi requires each matrix row to be a contiguous chunk of well behaved
      // memory. There can be gaps between rows but no gaps are allowed between
      // items in a row. Also, the row stride can not be smaller than the size
      // of a row. If any of these requirements are not satisfied by the input
      // array, a new array will be allocated.
      int requirements = NPY_ARRAY_BEHAVED;  // aligned and writeable
      npy_intp dim1 = PyArray_DIM((PyArrayObject*)a[0], 1);
      npy_intp stride0 = PyArray_STRIDE((PyArrayObject*)a[0], 0);
      npy_intp stride1 = PyArray_STRIDE((PyArrayObject*)a[0], 1);
      long item_size = ((long)sizeof(double));
      // We do not ask for a contiguous memory region, if we can do without one.
      if (((dim1 > 1) and (stride1 != item_size)) || (stride0 < item_size * dim1)) {
        requirements |= NPY_ARRAY_C_CONTIGUOUS;
      }
      PyObject *array = PyArray_FromArray((PyArrayObject*)a[0], nullptr,
                                          requirements);
      PyObject* err_type = nullptr;
      string err_msg{"C++ exception"};
      try {
        ::kaldi::MatrixIndexT stride = PyArray_STRIDE((PyArrayObject*)array, 0) / ((long)sizeof(double));
        reinterpret_cast<wrapper*>(self)->cpp = ::clif::MakeShared<::kaldi::SubMatrix<double>>((double*)PyArray_DATA((PyArrayObject*)array) + arg2 * stride + arg4, std::move(arg3), std::move(arg5), std::move(stride));
      } catch(const std::exception& e) {
        err_type = PyExc_RuntimeError;
        err_msg += string(": ") + e.what();
      } catch (...) {
        err_type = PyExc_RuntimeError;
      }
      if (err_type) {
        Py_DECREF(array);
        PyErr_SetString(err_type, err_msg.c_str());
        return nullptr;
      }
      // Reference count of array will be decremented when self is deallocated.
      reinterpret_cast<wrapper*>(self)->base = array;
    } else {
      PyErr_SetString(PyExc_RuntimeError,
                      "Cannot convert given ndarray to a DoubleSubMatrix since "
                      "it has an invalid dtype. Supported dtypes: np.float64.");
      return nullptr;
    }
  } else {
    ::kaldi::MatrixBase<double>* arg1;
    if (!Clif_PyObjAs(a[0], &arg1))
      return ArgError("__init__", names[0], "PyArray_Type or ::kaldi::MatrixBase<double>", a[0]);
    // Call actual C++ method.
    PyObject* err_type = nullptr;
    string err_msg{"C++ exception"};
    try {
      reinterpret_cast<wrapper*>(self)->cpp = ::clif::MakeShared<::kaldi::SubMatrix<double>>(*arg1, std::move(arg2), std::move(arg3), std::move(arg4), std::move(arg5));
    } catch(const std::exception& e) {
      err_type = PyExc_RuntimeError;
      err_msg += string(": ") + e.what();
    } catch (...) {
      err_type = PyExc_RuntimeError;
    }
    if (err_type) {
      PyErr_SetString(err_type, err_msg.c_str());
      return nullptr;
    }
    // Reference count of a[0] will be decremented when self is deallocated.
    Py_INCREF(a[0]);
    reinterpret_cast<wrapper*>(self)->base = a[0];
  }
  Py_RETURN_NONE;
}

// Range(row_start:int, num_rows:int, col_start:int, num_cols:int) -> DoubleSubMatrix
// NOTE: Range method is implemented in Python by initializing a new
// kaldi.matrix.DoubleSubMatrix object since the user facing DoubleSubMatrix
// class (kaldi.matrix.DoubleSubMatrix) is defined in Python and is a child
// class of the DoubleSubMatrix type provided by this wrapper.

// Implicit cast this as ::kaldi::MatrixBase<double>*
static PyObject* as_kaldi_MatrixBase_double(PyObject* self) {
  ::kaldi::MatrixBase<double>* p = ::clif::python::Get(reinterpret_cast<wrapper*>(self)->cpp);
  if (p == nullptr) return nullptr;
  return PyCapsule_New(p, C("::kaldi::MatrixBase<double>"), nullptr);
}

static PyMethodDef Methods[] = {
  {C("__init__"), (PyCFunction)wrapSubMatrix_double_as___init__, METH_VARARGS | METH_KEYWORDS, C("__init__(T:MatrixBase, ro:int, r:int, co:int, c:int)\n  Calls C++ function\n  void ::kaldi::SubMatrix<double>::SubMatrix(::kaldi::MatrixBase<double>, ::kaldi::MatrixIndexT, ::kaldi::MatrixIndexT, ::kaldi::MatrixIndexT, ::kaldi::MatrixIndexT)")},
  {C("as_kaldi_MatrixBase_double"), (PyCFunction)as_kaldi_MatrixBase_double, METH_NOARGS, C("Upcast to ::kaldi::MatrixBase<double>*")},
  {}
};

// DoubleSubMatrix __new__
static PyObject* _allocator(PyTypeObject* type, Py_ssize_t nitems);
// DoubleSubMatrix __init__
static int _ctor(PyObject* self, PyObject* args, PyObject* kw);

static void _deallocator(PyObject* self) {
  reinterpret_cast<wrapper*>(self)->cpp.Destruct();
  Py_XDECREF(reinterpret_cast<wrapper*>(self)->base);
  Py_TYPE(self)->tp_free(self);
}

static void _dtor(void* self) {
  delete reinterpret_cast<wrapper*>(self);
}

PyTypeObject wrapper_Type = {
  PyVarObject_HEAD_INIT(&PyType_Type, 0)
  "kaldi.matrix._matrix_ext.DoubleSubMatrix", // tp_name
  sizeof(wrapper),                     // tp_basicsize
  0,                                   // tp_itemsize
  _deallocator,                        // tp_dealloc
  nullptr,                             // tp_print
  nullptr,                             // tp_getattr
  nullptr,                             // tp_setattr
  nullptr,                             // tp_compare
  nullptr,                             // tp_repr
  nullptr,                             // tp_as_number
  nullptr,                             // tp_as_sequence
  nullptr,                             // tp_as_mapping
  nullptr,                             // tp_hash
  nullptr,                             // tp_call
  nullptr,                             // tp_str
  nullptr,                             // tp_getattro
  nullptr,                             // tp_setattro
  nullptr,                             // tp_as_buffer
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_TYPE_SUBCLASS | Py_TPFLAGS_BASETYPE, // tp_flags
  "CLIF wrapper for ::kaldi::SubMatrix<double>", // tp_doc
  nullptr,                             // tp_traverse
  nullptr,                             // tp_clear
  nullptr,                             // tp_richcompare
  0,                                   // tp_weaklistoffset
  nullptr,                             // tp_iter
  nullptr,                             // tp_iternext
  Methods,                             // tp_methods
  nullptr,                             // tp_members
  nullptr,                             // tp_getset
  nullptr,                             // tp_base
  nullptr,                             // tp_dict
  nullptr,                             // tp_descr_get
  nullptr,                             // tp_descr_set
  0,                                   // tp_dictoffset
  _ctor,                               // tp_init
  _allocator,                          // tp_alloc
  PyType_GenericNew,                   // tp_new
  _dtor,                               // tp_free
  nullptr,                             // tp_is_gc
  nullptr,                             // tp_bases
  nullptr,                             // tp_mro
  nullptr,                             // tp_cache
  nullptr,                             // tp_subclasses
  nullptr,                             // tp_weaklist
  nullptr,                             // tp_del
  0,                                   // tp_version_tag
};

static int _ctor(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* init = wrapSubMatrix_double_as___init__(self, args, kw);
  Py_XDECREF(init);
  return init? 0: -1;
}

static PyObject* _allocator(PyTypeObject* type, Py_ssize_t nitems) {
  assert(nitems == 0);
  PyObject* self = reinterpret_cast<PyObject*>(new wrapper);
  return PyObject_Init(self, &wrapper_Type);
}

static ::kaldi::SubMatrix<double>* ThisPtr(PyObject* py) {
  if (Py_TYPE(py) == &wrapper_Type) {
    return ::clif::python::Get(reinterpret_cast<wrapper*>(py)->cpp);
  }
  PyObject* base = PyObject_CallMethod(py, C("as_kaldi_SubMatrix_double"), nullptr);
  if (base) {
    if (PyCapsule_CheckExact(base)) {
      void* p = PyCapsule_GetPointer(base, C("::kaldi::SubMatrix<double>"));
      if (!PyErr_Occurred()) {
        ::kaldi::SubMatrix<double>* c = static_cast<::kaldi::SubMatrix<double>*>(p);
        Py_DECREF(base);
        return c;
      }
    }
    Py_DECREF(base);
  }
  if (PyObject_IsInstance(py, reinterpret_cast<PyObject*>(&wrapper_Type))) {
    if (!base) {
      PyErr_Clear();
      return ::clif::python::Get(reinterpret_cast<wrapper*>(py)->cpp);
    }
    PyErr_Format(PyExc_ValueError, "can't convert %s %s to ::kaldi::SubMatrix<double>*", ClassName(py), ClassType(py));
  } else {
    PyErr_Format(PyExc_TypeError, "expecting %s instance, got %s %s", wrapper_Type.tp_name, ClassName(py), ClassType(py));
  }
  return nullptr;
}
}  // namespace pyDoubleSubMatrix

namespace numpy {

// vector_to_numpy(v: VectorBase) -> ndarray
static PyObject* VectorToNumpy(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* obj;
  char* names[] = { C("vector"), nullptr };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:vector_to_numpy",
                                   names, &obj)) {
    return nullptr;
  }
  ::kaldi::VectorBase<float>* vector;
  if (!Clif_PyObjAs(obj, &vector)) {
    return ArgError("vector_to_numpy", names[0],
                    "::kaldi::VectorBase<float>", obj);
  }
  // Construct ndarray
  npy_intp size = vector->Dim();
  PyObject* array = PyArray_New(
      &PyArray_Type, 1, &size, NPY_FLOAT, nullptr, vector->Data(), 0,
      NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE | NPY_ARRAY_C_CONTIGUOUS,
      nullptr);
  if (!array) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot convert to ndarray.");
    return nullptr;
  }
  // From NumPy C-API Docs:
  // If data is passed to PyArray_New, this memory must not be deallocated
  // until the new array is deleted. If this data came from another Python
  // object, this can be accomplished using Py_INCREF on that object and
  // setting the base member of the new array to point to that object.
  // PyArray_SetBaseObject(PyArrayObject* array, PyObject* obj) steals a
  // reference to obj and sets it as the base property of array.
  Py_INCREF(obj);
  if (PyArray_SetBaseObject((PyArrayObject*)(array), obj) == -1) {
    Py_DECREF(array);
    PyErr_SetString(PyExc_RuntimeError, "Cannot convert to ndarray.");
    return nullptr;
  }
  return array;
}

// matrix_to_numpy(matrix:MatrixBase) -> ndarray
static PyObject* MatrixToNumpy(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* obj;
  char* names[] = { C("matrix"), nullptr };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:matrix_to_numpy",
                                   names, &obj)) {
    return nullptr;
  }
  ::kaldi::MatrixBase<float>* matrix;
  if (!Clif_PyObjAs(obj, &matrix)) {
    return ArgError("matrix_to_numpy", names[0],
                    "::kaldi::MatrixBase<float>", obj);
  }
  // Construct ndarray
  npy_intp sizes[2] = { matrix->NumRows(), matrix->NumCols() };
  npy_intp strides[2] = { matrix->Stride() * ((long)sizeof(float)),
                          ((long)sizeof(float)) };
  PyObject* array = PyArray_New(
      &PyArray_Type, 2, sizes, NPY_FLOAT, strides, matrix->Data(), 0,
      NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE | NPY_ARRAY_C_CONTIGUOUS,
      nullptr);
  if (!array) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot convert to ndarray.");
    return nullptr;
  }
  // From NumPy C-API Docs:
  // If data is passed to PyArray_New, this memory must not be deallocated
  // until the new array is deleted. If this data came from another Python
  // object, this can be accomplished using Py_INCREF on that object and
  // setting the base member of the new array to point to that object.
  // PyArray_SetBaseObject(PyArrayObject* array, PyObject* obj) steals a
  // reference to obj and sets it as the base property of array.
  Py_INCREF(obj);
  if (PyArray_SetBaseObject((PyArrayObject*)(array), obj) == -1) {
    Py_DECREF(array);
    PyErr_SetString(PyExc_RuntimeError, "Cannot convert to ndarray.");
    return nullptr;
  }
  return array;
}

// double_vector_to_numpy(v:DoubleVectorBase) -> ndarray
static PyObject* DoubleVectorToNumpy(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* obj;
  char* names[] = { C("vector"), nullptr };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:double_vector_to_numpy",
                                   names, &obj)) {
    return nullptr;
  }
  ::kaldi::VectorBase<double>* vector;
  if (!Clif_PyObjAs(obj, &vector)) {
    return ArgError("double_vector_to_numpy", names[0],
                    "::kaldi::VectorBase<double>", obj);
  }
  // Construct ndarray
  npy_intp size = vector->Dim();
  PyObject* array = PyArray_New(
      &PyArray_Type, 1, &size, NPY_DOUBLE, nullptr, vector->Data(), 0,
      NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE | NPY_ARRAY_C_CONTIGUOUS,
      nullptr);
  if (!array) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot convert to ndarray.");
    return nullptr;
  }
  // From NumPy C-API Docs:
  // If data is passed to PyArray_New, this memory must not be deallocated
  // until the new array is deleted. If this data came from another Python
  // object, this can be accomplished using Py_INCREF on that object and
  // setting the base member of the new array to point to that object.
  // PyArray_SetBaseObject(PyArrayObject* array, PyObject* obj) steals a
  // reference to obj and sets it as the base property of array.
  Py_INCREF(obj);
  if (PyArray_SetBaseObject((PyArrayObject*)(array), obj) == -1) {
    Py_DECREF(array);
    PyErr_SetString(PyExc_RuntimeError, "Cannot convert to ndarray.");
    return nullptr;
  }
  return array;
}

// double_matrix_to_numpy(matrix:DoubleMatrixBase) -> ndarray
static PyObject* DoubleMatrixToNumpy(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* obj;
  char* names[] = { C("matrix"), nullptr };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:double_matrix_to_numpy",
                                   names, &obj)) {
    return nullptr;
  }
  ::kaldi::MatrixBase<double>* matrix;
  if (!Clif_PyObjAs(obj, &matrix)) {
    return ArgError("double_matrix_to_numpy", names[0],
                    "::kaldi::MatrixBase<double>", obj);
  }
  // Construct ndarray
  npy_intp sizes[2] = { matrix->NumRows(), matrix->NumCols() };
  npy_intp strides[2] = { matrix->Stride() * ((long)sizeof(double)),
                          ((long)sizeof(double)) };
  PyObject* array = PyArray_New(
      &PyArray_Type, 2, sizes, NPY_DOUBLE, strides, matrix->Data(), 0,
      NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE | NPY_ARRAY_C_CONTIGUOUS,
      nullptr);
  if (!array) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot convert to ndarray.");
    return nullptr;
  }
  // From NumPy C-API Docs:
  // If data is passed to PyArray_New, this memory must not be deallocated
  // until the new array is deleted. If this data came from another Python
  // object, this can be accomplished using Py_INCREF on that object and
  // setting the base member of the new array to point to that object.
  // PyArray_SetBaseObject(PyArrayObject* array, PyObject* obj) steals a
  // reference to obj and sets it as the base property of array.
  Py_INCREF(obj);
  if (PyArray_SetBaseObject((PyArrayObject*)(array), obj) == -1) {
    Py_DECREF(array);
    PyErr_SetString(PyExc_RuntimeError, "Cannot convert to ndarray.");
    return nullptr;
  }
  return array;
}

}  // namespace numpy

static PyMethodDef Methods[] = {
  { C("vector_to_numpy"), (PyCFunction)numpy::VectorToNumpy,
    METH_VARARGS | METH_KEYWORDS,
    C("vector_to_numpy(vector:VectorBase) -> ndarray\n Converts a single precision vector to a 1-D NumPy array.") },
  { C("matrix_to_numpy"), (PyCFunction)numpy::MatrixToNumpy,
    METH_VARARGS | METH_KEYWORDS,
    C("matrix_to_numpy(matrix:MatrixBase) -> ndarray\n Converts a single precision matrix to a 2-D NumPy array.") },
  { C("double_vector_to_numpy"), (PyCFunction)numpy::DoubleVectorToNumpy,
    METH_VARARGS | METH_KEYWORDS,
    C("double_vector_to_numpy(vector:DoubleVectorBase) -> ndarray\n Converts a double precision vector to a 1-D NumPy array.") },
  { C("double_matrix_to_numpy"), (PyCFunction)numpy::DoubleMatrixToNumpy,
    METH_VARARGS | METH_KEYWORDS,
    C("double_matrix_to_numpy(matrix:DoubleMatrixBase) -> ndarray\n Converts a double precision matrix to a 2-D NumPy array.") },
  {}
};

bool Ready() {
  PyObject* base_cls = ImportFQName("kaldi.matrix._kaldi_vector.VectorBase");
  if (base_cls == nullptr) return false;
  pySubVector::wrapper_Type.tp_base = reinterpret_cast<PyTypeObject*>(base_cls);
  if (PyType_Ready(&pySubVector::wrapper_Type) < 0) return false;
  Py_INCREF(&pySubVector::wrapper_Type);  // For PyModule_AddObject to steal.
  base_cls = ImportFQName("kaldi.matrix._kaldi_matrix.MatrixBase");
  if (base_cls == nullptr) return false;
  pySubMatrix::wrapper_Type.tp_base = reinterpret_cast<PyTypeObject*>(base_cls);
  if (PyType_Ready(&pySubMatrix::wrapper_Type) < 0) return false;
  Py_INCREF(&pySubMatrix::wrapper_Type);  // For PyModule_AddObject to steal.
  base_cls = ImportFQName("kaldi.matrix._kaldi_vector.DoubleVectorBase");
  if (base_cls == nullptr) return false;
  pyDoubleSubVector::wrapper_Type.tp_base = reinterpret_cast<PyTypeObject*>(base_cls);
  if (PyType_Ready(&pyDoubleSubVector::wrapper_Type) < 0) return false;
  Py_INCREF(&pyDoubleSubVector::wrapper_Type);  // For PyModule_AddObject to steal.
  base_cls = ImportFQName("kaldi.matrix._kaldi_matrix.DoubleMatrixBase");
  if (base_cls == nullptr) return false;
  pyDoubleSubMatrix::wrapper_Type.tp_base = reinterpret_cast<PyTypeObject*>(base_cls);
  if (PyType_Ready(&pyDoubleSubMatrix::wrapper_Type) < 0) return false;
  Py_INCREF(&pyDoubleSubMatrix::wrapper_Type);  // For PyModule_AddObject to steal.
  return true;
}

#if PY_MAJOR_VERSION == 2
PyObject* Init() {
  PyObject* module = Py_InitModule3("kaldi.matrix._matrix_ext", Methods,
                                    "kaldi matrix extension module");
  if (!module) {
    PyErr_SetString(PyExc_ImportError,
                    "Cannot initialize kaldi.matrix._matrix_ext module.");
    return nullptr;
  }
  if (PyObject* m = PyImport_ImportModule("kaldi.matrix._matrix_common")) Py_DECREF(m);
  else return nullptr;
  if (PyObject* m = PyImport_ImportModule("kaldi.matrix._kaldi_vector")) Py_DECREF(m);
  else return nullptr;
  if (PyObject* m = PyImport_ImportModule("kaldi.matrix._kaldi_matrix")) Py_DECREF(m);
  else return nullptr;
  if (PyModule_AddObject(module, "SubVector", reinterpret_cast<PyObject*>(&pySubVector::wrapper_Type)) < 0) return nullptr;
  if (PyModule_AddObject(module, "SubMatrix", reinterpret_cast<PyObject*>(&pySubMatrix::wrapper_Type)) < 0) return nullptr;
  if (PyModule_AddObject(module, "DoubleSubVector", reinterpret_cast<PyObject*>(&pyDoubleSubVector::wrapper_Type)) < 0) return nullptr;
  if (PyModule_AddObject(module, "DoubleSubMatrix", reinterpret_cast<PyObject*>(&pyDoubleSubMatrix::wrapper_Type)) < 0) return nullptr;
  if (_import_array() < 0) {
    PyErr_Print();
    PyErr_SetString(PyExc_ImportError,
                    "numpy.core.multiarray failed to import");
    return nullptr;
  }
  return module;
}
#else
static struct PyModuleDef Module = {
  PyModuleDef_HEAD_INIT,
  "kaldi.matrix._matrix_ext",
  "kaldi matrix extension module",
  -1,  // module keeps state in global variables
  Methods
};

PyObject* Init() {
  PyObject* module = PyModule_Create(&Module);
  if (!module) return nullptr;
  if (PyObject* m = PyImport_ImportModule("kaldi.matrix._matrix_common")) Py_DECREF(m);
  else goto err;
  if (PyObject* m = PyImport_ImportModule("kaldi.matrix._kaldi_vector")) Py_DECREF(m);
  else goto err;
  if (PyObject* m = PyImport_ImportModule("kaldi.matrix._kaldi_matrix")) Py_DECREF(m);
  else goto err;
  if (PyModule_AddObject(module, "SubVector", reinterpret_cast<PyObject*>(&pySubVector::wrapper_Type)) < 0) goto err;
  if (PyModule_AddObject(module, "SubMatrix", reinterpret_cast<PyObject*>(&pySubMatrix::wrapper_Type)) < 0) goto err;
  if (PyModule_AddObject(module, "DoubleSubVector", reinterpret_cast<PyObject*>(&pyDoubleSubVector::wrapper_Type)) < 0) goto err;
  if (PyModule_AddObject(module, "DoubleSubMatrix", reinterpret_cast<PyObject*>(&pyDoubleSubMatrix::wrapper_Type)) < 0) goto err;
  if (_import_array() < 0) {
    PyErr_Print();
    PyErr_SetString(PyExc_ImportError,
                    "numpy.core.multiarray failed to import");
    goto err;
  }
  return module;
err:
  Py_DECREF(module);
  return nullptr;
}
#endif

}  // namespace __matrix__ext

// Initialize module

#if PY_MAJOR_VERSION == 2
PyMODINIT_FUNC init_matrix_ext(void) {
  __matrix__ext::Ready() &&
  __matrix__ext::Init();
}
#else
PyMODINIT_FUNC PyInit__matrix_ext(void) {
  if (!__matrix__ext::Ready()) return nullptr;
  return __matrix__ext::Init();
}
#endif

namespace kaldi {
using namespace ::clif;
using ::clif::Clif_PyObjAs;
using ::clif::Clif_PyObjFrom;

// SubVector to/from ::kaldi::SubVector<float> conversion

bool Clif_PyObjAs(PyObject* py, ::kaldi::SubVector<float>** c) {
  assert(c != nullptr);
  if (Py_None == py) {
    *c = nullptr;
    return true;
  }
  ::kaldi::SubVector<float>* cpp = __matrix__ext::pySubVector::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = cpp;
  return true;
}

bool Clif_PyObjAs(PyObject* py, std::shared_ptr<::kaldi::SubVector<float>>* c) {
  assert(c != nullptr);
  ::kaldi::SubVector<float>* cpp = __matrix__ext::pySubVector::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = ::clif::MakeStdShared(reinterpret_cast<__matrix__ext::pySubVector::wrapper*>(py)->cpp, cpp);
  return true;
}

bool Clif_PyObjAs(PyObject* py, std::unique_ptr<::kaldi::SubVector<float>>* c) {
  assert(c != nullptr);
  ::kaldi::SubVector<float>* cpp = __matrix__ext::pySubVector::ThisPtr(py);
  if (cpp == nullptr) return false;
  if (!reinterpret_cast<__matrix__ext::pySubVector::wrapper*>(py)->cpp.Detach()) {
    PyErr_SetString(PyExc_ValueError,
                    "Cannot convert SubVector instance to std::unique_ptr.");
    return false;
  }
  c->reset(cpp);
  return true;
}

bool Clif_PyObjAs(PyObject* py, ::kaldi::SubVector<float>* c) {
  assert(c != nullptr);
  ::kaldi::SubVector<float>* cpp = __matrix__ext::pySubVector::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = *cpp;
  return true;
}

bool Clif_PyObjAs(PyObject* py, ::gtl::optional<::kaldi::SubVector<float>>* c) {
  assert(c != nullptr);
  ::kaldi::SubVector<float>* cpp = __matrix__ext::pySubVector::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = *cpp;
  return true;
}

PyObject* Clif_PyObjFrom(::kaldi::SubVector<float>* c, py::PostConv unused) {
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(&__matrix__ext::pySubVector::wrapper_Type, NULL, NULL);
  reinterpret_cast<__matrix__ext::pySubVector::wrapper*>(py)->cpp = ::clif::Instance<::kaldi::SubVector<float>>(c, ::clif::UnOwnedResource());
  return py;
}

PyObject* Clif_PyObjFrom(std::unique_ptr<::kaldi::SubVector<float>> c, py::PostConv unused) {
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(&__matrix__ext::pySubVector::wrapper_Type, NULL, NULL);
  reinterpret_cast<__matrix__ext::pySubVector::wrapper*>(py)->cpp = ::clif::Instance<::kaldi::SubVector<float>>(std::move(c));
  return py;
}

PyObject* Clif_PyObjFrom(std::shared_ptr<::kaldi::SubVector<float>> c, py::PostConv unused) {
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(&__matrix__ext::pySubVector::wrapper_Type, NULL, NULL);
  reinterpret_cast<__matrix__ext::pySubVector::wrapper*>(py)->cpp = ::clif::Instance<::kaldi::SubVector<float>>(c);
  return py;
}

PyObject* Clif_PyObjFrom(const ::kaldi::SubVector<float>& c, py::PostConv unused) {
  PyObject* py = PyType_GenericNew(&__matrix__ext::pySubVector::wrapper_Type, NULL, NULL);
  reinterpret_cast<__matrix__ext::pySubVector::wrapper*>(py)->cpp = ::clif::MakeShared<::kaldi::SubVector<float>>(c);
  return py;
}

// SubMatrix to/from ::kaldi::SubMatrix<float> conversion

bool Clif_PyObjAs(PyObject* py, ::kaldi::SubMatrix<float>** c) {
  assert(c != nullptr);
  if (Py_None == py) {
    *c = nullptr;
    return true;
  }
  ::kaldi::SubMatrix<float>* cpp = __matrix__ext::pySubMatrix::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = cpp;
  return true;
}

bool Clif_PyObjAs(PyObject* py, std::shared_ptr<::kaldi::SubMatrix<float>>* c) {
  assert(c != nullptr);
  ::kaldi::SubMatrix<float>* cpp = __matrix__ext::pySubMatrix::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = ::clif::MakeStdShared(reinterpret_cast<__matrix__ext::pySubMatrix::wrapper*>(py)->cpp, cpp);
  return true;
}

bool Clif_PyObjAs(PyObject* py, std::unique_ptr<::kaldi::SubMatrix<float>>* c) {
  assert(c != nullptr);
  ::kaldi::SubMatrix<float>* cpp = __matrix__ext::pySubMatrix::ThisPtr(py);
  if (cpp == nullptr) return false;
  if (!reinterpret_cast<__matrix__ext::pySubMatrix::wrapper*>(py)->cpp.Detach()) {
    PyErr_SetString(PyExc_ValueError, "Cannot convert SubMatrix instance to std::unique_ptr.");
    return false;
  }
  c->reset(cpp);
  return true;
}

bool Clif_PyObjAs(PyObject* py, ::kaldi::SubMatrix<float>* c) {
  assert(c != nullptr);
  ::kaldi::SubMatrix<float>* cpp = __matrix__ext::pySubMatrix::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = *cpp;
  return true;
}

bool Clif_PyObjAs(PyObject* py, ::gtl::optional<::kaldi::SubMatrix<float>>* c) {
  assert(c != nullptr);
  ::kaldi::SubMatrix<float>* cpp = __matrix__ext::pySubMatrix::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = *cpp;
  return true;
}

PyObject* Clif_PyObjFrom(::kaldi::SubMatrix<float>* c, py::PostConv unused) {
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(&__matrix__ext::pySubMatrix::wrapper_Type, NULL, NULL);
  reinterpret_cast<__matrix__ext::pySubMatrix::wrapper*>(py)->cpp = ::clif::Instance<::kaldi::SubMatrix<float>>(c, ::clif::UnOwnedResource());
  return py;
}

PyObject* Clif_PyObjFrom(std::unique_ptr<::kaldi::SubMatrix<float>> c, py::PostConv unused) {
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(&__matrix__ext::pySubMatrix::wrapper_Type, NULL, NULL);
  reinterpret_cast<__matrix__ext::pySubMatrix::wrapper*>(py)->cpp = ::clif::Instance<::kaldi::SubMatrix<float>>(std::move(c));
  return py;
}

PyObject* Clif_PyObjFrom(std::shared_ptr<::kaldi::SubMatrix<float>> c, py::PostConv unused) {
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(&__matrix__ext::pySubMatrix::wrapper_Type, NULL, NULL);
  reinterpret_cast<__matrix__ext::pySubMatrix::wrapper*>(py)->cpp = ::clif::Instance<::kaldi::SubMatrix<float>>(c);
  return py;
}

PyObject* Clif_PyObjFrom(const ::kaldi::SubMatrix<float>& c, py::PostConv unused) {
  PyObject* py = PyType_GenericNew(&__matrix__ext::pySubMatrix::wrapper_Type, NULL, NULL);
  reinterpret_cast<__matrix__ext::pySubMatrix::wrapper*>(py)->cpp = ::clif::MakeShared<::kaldi::SubMatrix<float>>(c);
  return py;
}

// DoubleSubVector to/from ::kaldi::SubVector<double> conversion

bool Clif_PyObjAs(PyObject* py, ::kaldi::SubVector<double>** c) {
  assert(c != nullptr);
  if (Py_None == py) {
    *c = nullptr;
    return true;
  }
  ::kaldi::SubVector<double>* cpp = __matrix__ext::pyDoubleSubVector::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = cpp;
  return true;
}

bool Clif_PyObjAs(PyObject* py, std::shared_ptr<::kaldi::SubVector<double>>* c) {
  assert(c != nullptr);
  ::kaldi::SubVector<double>* cpp = __matrix__ext::pyDoubleSubVector::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = ::clif::MakeStdShared(reinterpret_cast<__matrix__ext::pyDoubleSubVector::wrapper*>(py)->cpp, cpp);
  return true;
}

bool Clif_PyObjAs(PyObject* py, std::unique_ptr<::kaldi::SubVector<double>>* c) {
  assert(c != nullptr);
  ::kaldi::SubVector<double>* cpp = __matrix__ext::pyDoubleSubVector::ThisPtr(py);
  if (cpp == nullptr) return false;
  if (!reinterpret_cast<__matrix__ext::pyDoubleSubVector::wrapper*>(py)->cpp.Detach()) {
    PyErr_SetString(PyExc_ValueError,
                    "Cannot convert DoubleSubVector instance to std::unique_ptr.");
    return false;
  }
  c->reset(cpp);
  return true;
}

bool Clif_PyObjAs(PyObject* py, ::kaldi::SubVector<double>* c) {
  assert(c != nullptr);
  ::kaldi::SubVector<double>* cpp = __matrix__ext::pyDoubleSubVector::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = *cpp;
  return true;
}

bool Clif_PyObjAs(PyObject* py, ::gtl::optional<::kaldi::SubVector<double>>* c) {
  assert(c != nullptr);
  ::kaldi::SubVector<double>* cpp = __matrix__ext::pyDoubleSubVector::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = *cpp;
  return true;
}

PyObject* Clif_PyObjFrom(::kaldi::SubVector<double>* c, py::PostConv unused) {
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(&__matrix__ext::pyDoubleSubVector::wrapper_Type, NULL, NULL);
  reinterpret_cast<__matrix__ext::pyDoubleSubVector::wrapper*>(py)->cpp = ::clif::Instance<::kaldi::SubVector<double>>(c, ::clif::UnOwnedResource());
  return py;
}

PyObject* Clif_PyObjFrom(std::unique_ptr<::kaldi::SubVector<double>> c, py::PostConv unused) {
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(&__matrix__ext::pyDoubleSubVector::wrapper_Type, NULL, NULL);
  reinterpret_cast<__matrix__ext::pyDoubleSubVector::wrapper*>(py)->cpp = ::clif::Instance<::kaldi::SubVector<double>>(std::move(c));
  return py;
}

PyObject* Clif_PyObjFrom(std::shared_ptr<::kaldi::SubVector<double>> c, py::PostConv unused) {
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(&__matrix__ext::pyDoubleSubVector::wrapper_Type, NULL, NULL);
  reinterpret_cast<__matrix__ext::pyDoubleSubVector::wrapper*>(py)->cpp = ::clif::Instance<::kaldi::SubVector<double>>(c);
  return py;
}

PyObject* Clif_PyObjFrom(const ::kaldi::SubVector<double>& c, py::PostConv unused) {
  PyObject* py = PyType_GenericNew(&__matrix__ext::pyDoubleSubVector::wrapper_Type, NULL, NULL);
  reinterpret_cast<__matrix__ext::pyDoubleSubVector::wrapper*>(py)->cpp = ::clif::MakeShared<::kaldi::SubVector<double>>(c);
  return py;
}

// DoubleSubMatrix to/from ::kaldi::SubMatrix<double> conversion

bool Clif_PyObjAs(PyObject* py, ::kaldi::SubMatrix<double>** c) {
  assert(c != nullptr);
  if (Py_None == py) {
    *c = nullptr;
    return true;
  }
  ::kaldi::SubMatrix<double>* cpp = __matrix__ext::pyDoubleSubMatrix::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = cpp;
  return true;
}

bool Clif_PyObjAs(PyObject* py, std::shared_ptr<::kaldi::SubMatrix<double>>* c) {
  assert(c != nullptr);
  ::kaldi::SubMatrix<double>* cpp = __matrix__ext::pyDoubleSubMatrix::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = ::clif::MakeStdShared(reinterpret_cast<__matrix__ext::pyDoubleSubMatrix::wrapper*>(py)->cpp, cpp);
  return true;
}

bool Clif_PyObjAs(PyObject* py, std::unique_ptr<::kaldi::SubMatrix<double>>* c) {
  assert(c != nullptr);
  ::kaldi::SubMatrix<double>* cpp = __matrix__ext::pyDoubleSubMatrix::ThisPtr(py);
  if (cpp == nullptr) return false;
  if (!reinterpret_cast<__matrix__ext::pyDoubleSubMatrix::wrapper*>(py)->cpp.Detach()) {
    PyErr_SetString(PyExc_ValueError, "Cannot convert DoubleSubMatrix instance to std::unique_ptr.");
    return false;
  }
  c->reset(cpp);
  return true;
}

bool Clif_PyObjAs(PyObject* py, ::kaldi::SubMatrix<double>* c) {
  assert(c != nullptr);
  ::kaldi::SubMatrix<double>* cpp = __matrix__ext::pyDoubleSubMatrix::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = *cpp;
  return true;
}

bool Clif_PyObjAs(PyObject* py, ::gtl::optional<::kaldi::SubMatrix<double>>* c) {
  assert(c != nullptr);
  ::kaldi::SubMatrix<double>* cpp = __matrix__ext::pyDoubleSubMatrix::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = *cpp;
  return true;
}

PyObject* Clif_PyObjFrom(::kaldi::SubMatrix<double>* c, py::PostConv unused) {
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(&__matrix__ext::pyDoubleSubMatrix::wrapper_Type, NULL, NULL);
  reinterpret_cast<__matrix__ext::pyDoubleSubMatrix::wrapper*>(py)->cpp = ::clif::Instance<::kaldi::SubMatrix<double>>(c, ::clif::UnOwnedResource());
  return py;
}

PyObject* Clif_PyObjFrom(std::unique_ptr<::kaldi::SubMatrix<double>> c, py::PostConv unused) {
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(&__matrix__ext::pyDoubleSubMatrix::wrapper_Type, NULL, NULL);
  reinterpret_cast<__matrix__ext::pyDoubleSubMatrix::wrapper*>(py)->cpp = ::clif::Instance<::kaldi::SubMatrix<double>>(std::move(c));
  return py;
}

PyObject* Clif_PyObjFrom(std::shared_ptr<::kaldi::SubMatrix<double>> c, py::PostConv unused) {
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(&__matrix__ext::pyDoubleSubMatrix::wrapper_Type, NULL, NULL);
  reinterpret_cast<__matrix__ext::pyDoubleSubMatrix::wrapper*>(py)->cpp = ::clif::Instance<::kaldi::SubMatrix<double>>(c);
  return py;
}

PyObject* Clif_PyObjFrom(const ::kaldi::SubMatrix<double>& c, py::PostConv unused) {
  PyObject* py = PyType_GenericNew(&__matrix__ext::pyDoubleSubMatrix::wrapper_Type, NULL, NULL);
  reinterpret_cast<__matrix__ext::pyDoubleSubMatrix::wrapper*>(py)->cpp = ::clif::MakeShared<::kaldi::SubMatrix<double>>(c);
  return py;
}

}  // namespace kaldi
