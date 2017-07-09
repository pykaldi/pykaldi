
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

// vector_to_numpy(v: Vector) -> ndarray
static PyObject* wrapVectorToNumpy(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* a;
  char* names[] = {
      C("v"),
      nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:vector_to_numpy", names, &a)) return nullptr;
  ::kaldi::Vector<float>* vector;
  if (!Clif_PyObjAs(a, &vector)) return ArgError("vector_to_numpy", names[0], "::kaldi::Vector<float>", a);
  // Construct ndarray
  npy_intp size = vector->Dim();
  PyObject* array = PyArray_New(
      &PyArray_Type, 1, &size, NPY_FLOAT, nullptr, vector->Data(), 0,
      NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE | NPY_ARRAY_C_CONTIGUOUS, nullptr);
  if (!array) {
    PyErr_SetString(PyExc_RuntimeError, "an error occured during conversion to numpy array");
    return nullptr;
  }
  // From NumPy C-API Docs: If data is passed to PyArray_New, this memory must
  // not be deallocated until the new array is deleted. If this data came from
  // another Python object, this can be accomplished using Py_INCREF on that
  // object and setting the base member of the new array to point to that object.
  Py_INCREF(a);
  if (PyArray_SetBaseObject((PyArrayObject*)(array), (PyObject*)a) == -1) {
    Py_DECREF(a);
    if (array) Py_DECREF(array);
    PyErr_SetString(PyExc_RuntimeError, "an error occured during conversion to numpy array");
    return nullptr;
  }
  return array;
}

// Initialize module

static PyMethodDef Methods[] = {
  {C("vector_to_numpy"), (PyCFunction)wrapVectorToNumpy, METH_VARARGS | METH_KEYWORDS, C("vector_to_numpy(v:Vector) -> ndarray")},
  // {C("numpy_to_vector"), (PyCFunction)wrapNumpyToVector, METH_VARARGS | METH_KEYWORDS, C("numpy_to_vector(a:ndarray) -> Vector")},
  {}
};

bool Ready() {
  return true;
}

PyObject* Init() {
  PyObject* module = Py_InitModule3("kaldi_vector_numpy", Methods, "Vector/ndarray conversion wrapper");
  if (!module) return nullptr;
  return module;
}

}  // namespace kaldi__vector__numpy

PyMODINIT_FUNC initkaldi_vector_numpy(void) {
  kaldi__vector__numpy::Ready() &&
  kaldi__vector__numpy::Init();
  import_array();
}
