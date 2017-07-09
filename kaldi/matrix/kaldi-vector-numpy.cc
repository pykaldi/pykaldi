
#include <Python.h>
// #include "clif/python/ptr_util.h"
// #include "clif/python/optional.h"
#include "/home/dogan/anaconda2/envs/clif/python/types.h"
#define PY_ARRAY_UNIQUE_SYMBOL __numpy_array_api
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "matrix/kaldi-vector_clifwrap.h"
// #include "clif/python/stltypes.h"
// #include "clif/python/slots.h"

namespace kaldi__vector__numpy {
using namespace clif;

// vector_to_numpy(v: VectorBase) -> ndarray
static PyObject* VectorToNumpy(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* obj;
  char* names[] = { C("v"), nullptr };
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
  PyObject* arr = PyArray_New(
      &PyArray_Type, 1, &size, NPY_FLOAT, nullptr, vector->Data(), 0,
      NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE | NPY_ARRAY_C_CONTIGUOUS,
      nullptr);
  if (!arr) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot convert to ndarray.");
    return nullptr;
  }
  // From NumPy C-API Docs:
  // If data is passed to PyArray_New, this memory must not be deallocated
  // until the new array is deleted. If this data came from another Python
  // object, this can be accomplished using Py_INCREF on that object and
  // setting the base member of the new array to point to that object.
  // PyArray_SetBaseObject(PyArrayObject* arr, PyObject* obj) steals a
  // reference to obj and sets it as the base property of arr.
  Py_INCREF(obj);
  if (PyArray_SetBaseObject((PyArrayObject*)(arr), obj) == -1) {
    Py_DECREF(obj);
    if (arr) Py_DECREF(arr);
    PyErr_SetString(PyExc_RuntimeError, "Cannot convert to ndarray.");
    return nullptr;
  }
  return arr;
}

static PyMethodDef Methods[] = {
  {C("vector_to_numpy"), (PyCFunction)VectorToNumpy,
   METH_VARARGS | METH_KEYWORDS, C("vector_to_numpy(v:VectorBase) -> ndarray")},
  // {C("numpy_to_vector"), (PyCFunction)NumpyToVector,
  //  METH_VARARGS | METH_KEYWORDS, C("numpy_to_vector(a:ndarray) -> Vector")},
  {}
};

}  // namespace kaldi__vector__numpy

// Initialize module

#if PY_MAJOR_VERSION == 2
PyMODINIT_FUNC initkaldi_vector_numpy(void) {
  PyObject* module = Py_InitModule3("kaldi_vector_numpy",
                                    kaldi__vector__numpy::Methods,
                                    "Vector/ndarray conversion module");
  if (!module) {
    PyErr_SetString(PyExc_ImportError,
                    "Cannot initialize kaldi_vector_numpy module.");
    return;
  }
  import_array();
}
#else
PyMODINIT_FUNC PyInit_kaldi_vector_numpy(void) {
  static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT,
    "kaldi_vector_numpy",  // module name
    "Vector/ndarray conversion module", // module doc
    -1,  // module keeps state in global variables
    kaldi__vector__numpy::Methods
  };
  PyObject* module = PyModule_Create(&Module);
  if (!module) return nullptr;
  import_array();
  return module;
}
#endif
